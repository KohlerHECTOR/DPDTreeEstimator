// -----------------------------------------------------------------------------
// C++ core for Dynamic Programming Decision Trees (DPDT).
//
// This replaces the pure-Python MDP construction + dynamic program of
// DPDTreeClassifier / DPDTreeRegressor. Both estimators share the same solver;
// they only differ in their sufficient statistics ("stats" below):
//   * classification: K weighted class counts; leaf = majority class,
//     terminal reward r* = max_c w_c / w - 1, impurity = gini / entropy.
//   * regression: per-output weighted sums of y and y^2 (2*O doubles); leaf =
//     weighted mean vector, terminal reward r* = -MSE, impurity = MSE
//     (identical formula to sklearn's squared_error criterion).
//
// Each speed-up / bound trick is tagged where it is used:
//
//   [SPEED-1] In-house best-first CART split generator: no per-state sklearn
//             estimator construction, no input re-validation, and no
//             fancy-indexing copies of the dataset (`X[nz]`) for every state.
//   [SPEED-2] A DP state is a contiguous block of sample ids; children are
//             materialized by a single linear filtering pass. Work per state
//             is O(n_state) instead of the O(n_samples * n_splits)
//             boolean-mask arithmetic of the Python version.
//   [SPEED-3] The tree policy is stored in a flat "arena" of fixed-size records
//             (per-zeta chosen action + child record ids). This replaces the
//             Python dict keyed by tuples of 2*n_features floats, whose hashing
//             and boxing dominated memory traffic.
//   [SPEED-4] Parallelism over root candidate splits with raw std::thread on a
//             shared read-only dataset: no GIL, no joblib, and no deepcopy of
//             X/y per job (the Python version deep-copied the whole sub-MDP).
//   [SPEED-5] Duplicate candidate splits (same feature/threshold produced by
//             different CART nodes) are evaluated once instead of re-solving
//             an identical subtree.
//   [SPEED-6] Presort-once: every feature is argsorted ONCE at the root; each
//             state keeps one id-list per feature, already sorted by that
//             feature. Children inherit sortedness through stable filtering,
//             so the split search is a pure linear sweep -- the O(n log n)
//             per-feature per-node sort of a classic CART disappears from the
//             entire MDP construction.
//   [SPEED-7] X is transposed to column-major once, so the split sweep and the
//             partition predicate stream through a single contiguous column
//             instead of striding across rows.
//   [BOUND-1] The DFS depth is bounded by max_depth and the number of candidate
//             actions per state is bounded by cart_nodes_list[depth] - 1 (a
//             CART with L leaves has exactly L-1 internal splits). All
//             per-depth scratch (stats frames, Q frames, candidate lists,
//             child-value frames, child id-buffers) is preallocated / reused
//             from these upper bounds: no allocation churn inside the hot
//             recursion.
//   [BOUND-2] Branch & bound. Every reward is <= 0 (terminal reward r* <= 0
//             for both tasks, split reward zeta <= 0), hence every state value
//             satisfies V(s) <= 0. Therefore:
//               (a) Q(split) = zeta + p_l V_l + p_r V_r <= zeta:
//                   a candidate that cannot beat the incumbent for ANY zeta is
//                   skipped before anything is expanded;
//               (b) after solving the left child, Q <= zeta + p_l * V_l:
//                   this prunes the ENTIRE right subtree, which is then never
//                   even materialized.
//   [BOUND-3] The terminal action is evaluated first and CART candidates
//             arrive in decreasing split-quality order, so the branch & bound
//             incumbent is strong as early as possible. In sequential mode the
//             incumbent is shared across root candidates, so each solved
//             candidate strengthens pruning for the next one.
//   [WEIGHT-EQ] Fitting with integer sample_weight must be equivalent to
//             fitting on a dataset with repeated rows (sklearn's
//             check_sample_weight_equivalence). Zero-weight samples are
//             therefore invisible to the split search (boundaries, thresholds
//             and min_samples counts), and MDP transition probabilities are
//             weight fractions rather than row-count fractions.
// -----------------------------------------------------------------------------

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();
// Same minimal separation between feature values as sklearn's splitter.
constexpr double kFeatureThreshold = 1e-7;
// sklearn's EPSILON: a node with impurity <= eps is treated as pure. This
// matters for regression where floating-point MSE of a constant node may be a
// tiny positive number instead of exactly 0.
constexpr double kImpurityEps = 2.220446049250313e-16;

struct Data {
  const double* XT;   // [SPEED-7] column-major: XT[f * n + i]
  const int32_t* yc;  // classification labels 0..K-1 (nullptr for regression)
  const double* yr;   // regression targets, row-major (n, O) (nullptr else)
  const double* w;    // sample weights
  int n, F, K, O, Z;  // K = n_classes (0 -> regression), O = n_outputs,
                      // Z = number of zetas (max_nb_trees)
  bool classification() const { return K > 0; }
  // Size of the sufficient statistics of one node.
  int S() const { return K > 0 ? K : 2 * O; }
};

struct Config {
  int max_depth;
  std::vector<int> cart_nodes_list;
  // min_samples_split / min_samples_leaf keep sklearn's semantics: fractions
  // are resolved against the *state* the inner CART is fit on (see resolve()).
  double min_samples_split;
  bool mss_is_fraction;
  double min_samples_leaf;
  bool msl_is_fraction;
  double min_weight_fraction_leaf;
  int max_features;  // already resolved to an int in [1, F]
  double min_impurity_decrease;
  int criterion;  // 0 = gini, 1 = entropy / log_loss, 2 = squared_error
};

// Constraints of the inner CART resolved for one state (sklearn resolves
// fractional min_samples_* at each inner `fit`, i.e. per state, against that
// state's sample count / weight).
struct Resolved {
  int min_samples_split;
  int min_samples_leaf;
  double min_weight_leaf;
};

Resolved resolve(const Config& c, int n_state, double w_state) {
  Resolved r;
  r.min_samples_leaf =
      c.msl_is_fraction
          ? std::max(1, (int)std::ceil(c.min_samples_leaf * n_state))
          : (int)c.min_samples_leaf;
  r.min_samples_split =
      c.mss_is_fraction
          ? std::max(2, (int)std::ceil(c.min_samples_split * n_state))
          : (int)c.min_samples_split;
  r.min_samples_split = std::max(r.min_samples_split, 2 * r.min_samples_leaf);
  r.min_weight_leaf = c.min_weight_fraction_leaf * w_state;
  return r;
}

// Accumulate sample a into the sufficient statistics st (size D.S()).
inline void acc_stats(const Data& D, int a, double* st) {
  const double w = D.w[a];
  if (D.classification()) {
    st[D.yc[a]] += w;
  } else {
    const double* yv = D.yr + (size_t)a * D.O;
    for (int o = 0; o < D.O; ++o) {
      st[o] += w * yv[o];
      st[D.O + o] += w * yv[o] * yv[o];
    }
  }
}

// Node impurity from sufficient statistics.
double impurity(const Data& D, const double* st, double w, int criterion) {
  if (w <= 0.0) return 0.0;
  if (criterion == 2) {  // squared_error: mean over outputs of the weighted
                         // variance, computed exactly like sklearn's MSE
                         // criterion (sq_sum/w - mean^2).
    double s = 0.0;
    for (int o = 0; o < D.O; ++o) {
      const double m = st[o] / w;
      s += std::max(0.0, st[D.O + o] / w - m * m);
    }
    return s / D.O;
  }
  if (criterion == 0) {  // gini
    double sq = 0.0;
    for (int k = 0; k < D.K; ++k) {
      const double p = st[k] / w;
      sq += p * p;
    }
    return 1.0 - sq;
  }
  double h = 0.0;  // entropy / log_loss
  for (int k = 0; k < D.K; ++k) {
    if (st[k] > 0.0) {
      const double p = st[k] / w;
      h -= p * std::log2(p);
    }
  }
  return h;
}

// Per-zeta chosen action of an expanded state (one arena record = Z of these).
struct PZ {
  int32_t feat;   // -1 -> leaf
  int32_t leaf;   // majority class (classification) or index into the
                  // arena's leaf-value store (regression)
  int32_t left;   // child record ids (indices into the same arena)
  int32_t right;
  double thr;
};

// [SPEED-3] flat policy storage; `lv` holds O doubles per regression leaf.
struct Arena {
  std::vector<PZ> pz;
  std::vector<double> lv;
};

struct Split {
  int f;
  double t;
};

struct BestSplit {
  int f = -1;
  double t = 0.0;
  double improvement = kNegInf;
};

// [BOUND-1] All scratch a worker ever needs, preallocated or reused across
// states (n samples, F features, max_depth recursion levels, S stats size,
// Z zetas, cart_nodes_list[d] candidate actions at depth d).
struct WorkBuf {
  // [SPEED-6] `levels[d]` holds the F per-feature sorted id-lists (stride =
  // node size) of the state currently processed at DFS depth d. A parent
  // never mutates its own lists; it filters them into levels[d+1], so a
  // single buffer per depth is reused by every state visited at that depth.
  std::vector<std::vector<int32_t>> levels;
  std::vector<int32_t> cart;     // CART-local id-lists, stride = state size
  std::vector<int32_t> scratch;  // stable-partition scratch for CART rows
  std::vector<int32_t> pos;      // positive-weight ids scratch  [WEIGHT-EQ]
  std::vector<uint8_t> side;     // per-sample goes-left flag of current split
  std::vector<int> perm;         // feature permutation for max_features
  std::vector<double> stat_stack;   // (max_depth+2) * S stats frames
  std::vector<double> V_stack;      // (max_depth+2) * 2Z child-value frames
  std::vector<double> bestQ_stack;  // (max_depth+2) * Z incumbent Q frames
  std::vector<PZ> bestA_stack;      // (max_depth+2) * Z incumbent actions
  std::vector<std::vector<Split>> cand_stack;  // per-depth candidate lists
  std::vector<double> stat_left;  // S scratch: left stats in the sweep
  std::mt19937 rng;

  WorkBuf(const Data& d, const Config& c, uint32_t seed) : rng(seed) {
    const size_t frames = (size_t)c.max_depth + 2;
    levels.resize(frames);  // inner buffers grow lazily to F * node size
    cart.resize((size_t)d.F * d.n);
    scratch.resize(d.n);
    pos.resize(d.n);
    side.resize(d.n);
    perm.resize(d.F);
    std::iota(perm.begin(), perm.end(), 0);
    stat_stack.assign(frames * d.S(), 0.0);
    V_stack.assign(frames * 2 * d.Z, 0.0);
    bestQ_stack.assign(frames * d.Z, 0.0);
    bestA_stack.assign(frames * d.Z, PZ{-1, 0, -1, -1, 0.0});
    cand_stack.resize(frames);
    for (size_t dep = 0; dep < frames; ++dep) {
      // [BOUND-1] a CART with L leaves has exactly L-1 internal splits, so at
      // most max(2, cart_nodes_list[dep]) - 1 candidate actions exist at this
      // depth: reserve once, never reallocate.
      const int mln = dep < c.cart_nodes_list.size()
                          ? std::max(2, c.cart_nodes_list[dep])
                          : 2;
      cand_stack[dep].reserve((size_t)mln - 1);
    }
    stat_left.resize(d.S());
  }
};

// Best split of the CART node whose per-feature sorted id-lists are
// rows[f * stride + (s..e)]. node_st/node_w/node_imp: sufficient statistics,
// total weight and impurity of the node; w_root: total weight of the state
// the CART is fit on (sklearn normalizes improvement by the fit-time weight).
BestSplit best_split(const Data& D, const Config& C, const Resolved& R,
                     WorkBuf& wb, const int32_t* rows, int stride, int s,
                     int e, const double* node_st, double node_w,
                     double node_imp, double w_root) {
  BestSplit best;
  const int n = e - s;
  // [WEIGHT-EQ] only positive-weight samples exist for the split search;
  // n_pos is the same for every feature (same id multiset).
  int n_pos = 0;
  for (int i = 0; i < n; ++i)
    if (D.w[rows[(size_t)0 * stride + s + i]] > 0.0) ++n_pos;
  if (n_pos < R.min_samples_split || n_pos < 2 * R.min_samples_leaf ||
      node_imp <= kImpurityEps)
    return best;

  const int F = D.F, S = D.S(), O = D.O;
  double* lc = wb.stat_left.data();
  int32_t* pos = wb.pos.data();

  int visited = 0;
  for (int j = 0; j < F; ++j) {
    int f;
    if (C.max_features < F) {
      // Sample features without replacement (partial Fisher-Yates), like
      // sklearn; keep looking past max_features until at least one valid
      // split has been found.
      std::uniform_int_distribution<int> u(j, F - 1);
      std::swap(wb.perm[j], wb.perm[u(wb.rng)]);
      f = wb.perm[j];
    } else {
      f = j;
    }
    if (visited >= C.max_features && best.f >= 0) break;
    ++visited;

    // [SPEED-6] the id-list is already sorted by feature f: the whole split
    // search is one linear sweep, no sorting whatsoever.
    const int32_t* row = rows + (size_t)f * stride + s;
    const double* col = D.XT + (size_t)f * D.n;  // [SPEED-7]
    // [WEIGHT-EQ] drop zero-weight ids: boundaries, thresholds and
    // min_samples_leaf counts must be identical to a fit on repeated rows.
    int np = 0;
    for (int i = 0; i < n; ++i)
      if (D.w[row[i]] > 0.0) pos[np++] = row[i];
    if (col[pos[np - 1]] <= col[pos[0]] + kFeatureThreshold)
      continue;  // constant feature

    std::fill(lc, lc + S, 0.0);
    double wl = 0.0;
    for (int i = 0; i < np - 1; ++i) {
      const int a = pos[i];
      acc_stats(D, a, lc);
      wl += D.w[a];
      const double x0 = col[a];
      const double x1 = col[pos[i + 1]];
      if (x1 <= x0 + kFeatureThreshold) continue;  // not a value boundary
      const int nl = i + 1, nr = np - nl;
      if (nl < R.min_samples_leaf || nr < R.min_samples_leaf) continue;
      const double wr = node_w - wl;
      if (wl < R.min_weight_leaf || wr < R.min_weight_leaf) continue;

      double imp_l, imp_r;
      if (C.criterion == 0) {  // gini from left/right class counts
        double sl = 0.0, sr = 0.0;
        for (int k = 0; k < S; ++k) {
          sl += lc[k] * lc[k];
          const double rc = node_st[k] - lc[k];
          sr += rc * rc;
        }
        imp_l = 1.0 - sl / (wl * wl);
        imp_r = 1.0 - sr / (wr * wr);
      } else if (C.criterion == 1) {  // entropy
        imp_l = 0.0;
        imp_r = 0.0;
        for (int k = 0; k < S; ++k) {
          if (lc[k] > 0.0) {
            const double p = lc[k] / wl;
            imp_l -= p * std::log2(p);
          }
          const double rc = node_st[k] - lc[k];
          if (rc > 0.0) {
            const double p = rc / wr;
            imp_r -= p * std::log2(p);
          }
        }
      } else {  // squared_error from left/right weighted y, y^2 sums
        imp_l = 0.0;
        imp_r = 0.0;
        for (int o = 0; o < O; ++o) {
          const double ml = lc[o] / wl;
          imp_l += std::max(0.0, lc[O + o] / wl - ml * ml);
          const double rs = node_st[o] - lc[o];
          const double rq = node_st[O + o] - lc[O + o];
          const double mr = rs / wr;
          imp_r += std::max(0.0, rq / wr - mr * mr);
        }
        imp_l /= O;
        imp_r /= O;
      }
      // sklearn's weighted impurity improvement.
      const double improvement =
          (node_w / w_root) *
          (node_imp - (wl / node_w) * imp_l - (wr / node_w) * imp_r);
      if (improvement > best.improvement) {
        double t = (x0 + x1) * 0.5;
        if (!std::isfinite(t) || t == x1) t = x0;  // sklearn threshold rule
        best.f = f;
        best.t = t;
        best.improvement = improvement;
      }
    }
  }
  return best;
}

// Set wb.side[id] = (X[id, f] <= t) for the ids of one id-list; returns the
// number of ids going left and, via wl_out, their total sample weight.
int flag_sides(const Data& D, WorkBuf& wb, const int32_t* ids, int n, int f,
               double t, double* wl_out = nullptr) {
  const double* col = D.XT + (size_t)f * D.n;  // [SPEED-7]
  int nl = 0;
  double wl = 0.0;
  for (int i = 0; i < n; ++i) {
    const int a = ids[i];
    const uint8_t sd = col[a] <= t ? 1 : 0;
    wb.side[a] = sd;
    nl += sd;
    if (sd) wl += D.w[a];
  }
  if (wl_out) *wl_out = wl;
  return nl;
}

struct CartNode {
  int s, e;
  double w, imp;
  std::vector<double> st;
  BestSplit best;
};

struct CartCmp {
  bool operator()(const CartNode& a, const CartNode& b) const {
    return a.best.improvement < b.best.improvement;
  }
};

// [SPEED-1] Generate the candidate split actions of the DP state whose sorted
// id-lists are `rows` (stride = n) by growing a best-first CART with at most
// max_leaf_nodes leaves inside wb.cart (no dataset copy, no sklearn call).
// Splits are emitted in decreasing-quality (pop) order            [BOUND-3].
void cart_candidates(const Data& D, const Config& C, const Resolved& R,
                     WorkBuf& wb, const int32_t* rows, int n,
                     int max_leaf_nodes, const double* state_st,
                     double state_w, std::vector<Split>& out) {
  out.clear();
  const int F = D.F, S = D.S();
  // CART works on its own copy of the state's id-lists so the DP state's
  // lists stay intact for child materialization. One shared buffer per
  // worker suffices: a CART always completes before the DFS recurses.
  int32_t* cart = wb.cart.data();
  for (int f = 0; f < F; ++f)
    std::copy(rows + (size_t)f * n, rows + (size_t)f * n + n,
              cart + (size_t)f * n);

  std::priority_queue<CartNode, std::vector<CartNode>, CartCmp> pq;
  CartNode root;
  root.s = 0;
  root.e = n;
  root.w = state_w;
  root.st.assign(state_st, state_st + S);
  root.imp = impurity(D, state_st, state_w, C.criterion);
  root.best = best_split(D, C, R, wb, cart, n, 0, n, root.st.data(), root.w,
                         root.imp, state_w);
  if (root.best.f >= 0 &&
      root.best.improvement + 1e-12 >= C.min_impurity_decrease)
    pq.push(std::move(root));

  int n_leaves = 1;
  while (!pq.empty() && n_leaves < max_leaf_nodes) {
    CartNode nd = pq.top();
    pq.pop();
    // [SPEED-5] identical (feature, threshold) pairs from different CART
    // nodes would make the DP re-solve an identical subtree: emit once.
    bool dup = false;
    for (const Split& sp : out)
      if (sp.f == nd.best.f && sp.t == nd.best.t) {
        dup = true;
        break;
      }
    if (!dup) out.push_back({nd.best.f, nd.best.t});
    ++n_leaves;

    // [SPEED-6] split the CART node: stable-partition each per-feature
    // id-list in place, so both children remain sorted by every feature.
    const int m = nd.s + flag_sides(D, wb, cart + nd.s, nd.e - nd.s,
                                    nd.best.f, nd.best.t);
    for (int f = 0; f < F; ++f) {
      int32_t* row = cart + (size_t)f * n;
      int32_t* sc = wb.scratch.data();
      int wl = nd.s, wr = 0;
      for (int i = nd.s; i < nd.e; ++i) {
        const int32_t a = row[i];
        if (wb.side[a])
          row[wl++] = a;
        else
          sc[wr++] = a;
      }
      std::copy(sc, sc + wr, row + wl);
    }

    for (int child = 0; child < 2; ++child) {
      CartNode ch;
      ch.s = child == 0 ? nd.s : m;
      ch.e = child == 0 ? m : nd.e;
      if (ch.e - ch.s < 2) continue;  // cannot be split further
      ch.st.assign(S, 0.0);
      ch.w = 0.0;
      const int32_t* row0 = cart;  // any feature row: same id multiset
      for (int i = ch.s; i < ch.e; ++i) {
        acc_stats(D, row0[i], ch.st.data());
        ch.w += D.w[row0[i]];
      }
      ch.imp = impurity(D, ch.st.data(), ch.w, C.criterion);
      ch.best = best_split(D, C, R, wb, cart, n, ch.s, ch.e, ch.st.data(),
                           ch.w, ch.imp, state_w);
      if (ch.best.f >= 0 &&
          ch.best.improvement + 1e-12 >= C.min_impurity_decrease)
        pq.push(std::move(ch));
    }
  }
}

struct Ctx {
  const Data* D;
  const Config* C;
  const double* zetas;
  Arena* arena;
  WorkBuf* wb;
};

// [SPEED-6] Materialize a child state at DFS depth `depth`: filter the
// parent's per-feature sorted id-lists by wb.side (== keep), preserving
// order, into the reusable level buffer. Returns the child's id-lists.
const int32_t* materialize(const Data& D, WorkBuf& wb, const int32_t* rows,
                           int n_parent, int n_child, uint8_t keep,
                           int depth) {
  std::vector<int32_t>& buf = wb.levels[(size_t)depth];
  if ((int)buf.size() < D.F * n_child) buf.resize((size_t)D.F * n_child);
  int32_t* out = buf.data();
  for (int f = 0; f < D.F; ++f) {
    const int32_t* src = rows + (size_t)f * n_parent;
    int32_t* dst = out + (size_t)f * n_child;
    int k = 0;
    for (int i = 0; i < n_parent; ++i) {
      const int32_t a = src[i];
      if (wb.side[a] == keep) dst[k++] = a;
    }
  }
  return out;
}

// Terminal action of a state: fills st (size D.S()) and, for regression,
// appends the leaf mean vector to arena.lv. Returns the terminal reward r*
// and sets leaf_ref (class index or leaf-value id) and wsum.
double terminal_action(const Data& D, const int32_t* rows, int n, double* st,
                       Arena* arena, int32_t* leaf_ref, double* wsum_out) {
  std::fill(st, st + D.S(), 0.0);
  double wsum = 0.0;
  for (int i = 0; i < n; ++i) {
    const int a = rows[i];  // feature-0 id-list: same id multiset
    acc_stats(D, a, st);
    wsum += D.w[a];
  }
  *wsum_out = wsum;
  if (D.classification()) {
    int astar = 0;
    double cmax = st[0];
    for (int k = 1; k < D.K; ++k)
      if (st[k] > cmax) {
        cmax = st[k];
        astar = k;
      }
    *leaf_ref = astar;
    // Reward of stopping here; equals 0 iff the node is pure.
    return wsum > 0.0 ? cmax / wsum - 1.0 : 0.0;
  }
  // Regression: leaf predicts the weighted mean vector, reward is -MSE.
  const int32_t id = (int32_t)(arena->lv.size() / (size_t)D.O);
  for (int o = 0; o < D.O; ++o)
    arena->lv.push_back(wsum > 0.0 ? st[o] / wsum : 0.0);
  *leaf_ref = id;
  return -impurity(D, st, wsum, 2);
}

// A node is expandable when its terminal reward is strictly worse than a
// perfect fit. For regression, tolerate sklearn's EPSILON on the impurity so
// that numerically-constant nodes are leaves (sklearn does the same).
inline bool expandable(const Data& D, double rstar) {
  return D.classification() ? rstar < 0.0 : rstar < -kImpurityEps;
}

// Solve the DP for the state whose per-feature sorted id-lists are `rows`
// (n samples, stride n). Returns the arena record id; writes V(state) per
// zeta into Vout.
int32_t dfs(const Ctx& ctx, const int32_t* rows, int n, int depth,
            double* Vout) {
  const Data& D = *ctx.D;
  const Config& C = *ctx.C;
  WorkBuf& wb = *ctx.wb;
  const int Z = D.Z, S = D.S();

  // ---- terminal action ----------------------------------------------------
  // [BOUND-1] sufficient statistics live in the preallocated per-depth frame.
  double* st = wb.stat_stack.data() + (size_t)depth * S;
  int32_t leaf_ref = 0;
  double wsum = 0.0;
  const double rstar = terminal_action(D, rows, n, st, ctx.arena, &leaf_ref,
                                       &wsum);

  // [BOUND-3] incumbent = terminal action, evaluated before any split so the
  // branch & bound below always has a valid lower bound.
  double* bestQ = wb.bestQ_stack.data() + (size_t)depth * Z;
  PZ* bestA = wb.bestA_stack.data() + (size_t)depth * Z;
  for (int z = 0; z < Z; ++z) {
    bestQ[z] = rstar;
    bestA[z] = PZ{-1, leaf_ref, -1, -1, 0.0};
  }

  // A pure node or a max-depth node is a leaf. The min_samples check is a
  // short-circuit: the inner CART could not produce any split anyway.
  const Resolved R = resolve(C, n, wsum);
  if (expandable(D, rstar) && depth < C.max_depth &&
      n >= R.min_samples_split) {
    const int mln = depth < (int)C.cart_nodes_list.size()
                        ? std::max(2, C.cart_nodes_list[depth])
                        : 2;
    std::vector<Split>& cands = wb.cand_stack[depth];  // [BOUND-1]
    cart_candidates(D, C, R, wb, rows, n, mln, st, wsum, cands);

    double* Vl = wb.V_stack.data() + (size_t)depth * 2 * Z;  // [BOUND-1]
    double* Vr = Vl + Z;
    for (const Split& sp : cands) {
      // [BOUND-2a] Q(split) = zeta + p_l V_l + p_r V_r <= zeta because all
      // state values are <= 0: stop expanding candidates as soon as zeta
      // cannot beat the incumbent for any zeta (the bound is candidate-
      // independent and the incumbent only improves).
      bool can_win = false;
      for (int z = 0; z < Z; ++z)
        if (ctx.zetas[z] > bestQ[z]) {
          can_win = true;
          break;
        }
      if (!can_win) break;

      double wleft = 0.0;
      const int nl = flag_sides(D, wb, rows, n, sp.f, sp.t, &wleft);
      const int nr = n - nl;
      const double wright = wsum - wleft;
      // [WEIGHT-EQ] a side holding zero total weight does not exist in the
      // equivalent repeated-rows dataset: reject the split (this also covers
      // the empty-side filter of the Python version).
      if (nl == 0 || nr == 0 || wleft <= 0.0 || wright <= 0.0) continue;
      // [WEIGHT-EQ] transition probabilities are weight fractions (equal to
      // row-count fractions when sample_weight is uniform).
      const double pl = wleft / wsum, pr = 1.0 - pl;

      // [SPEED-2] children are linear filter passes over the parent's lists;
      // the parent's own lists are never mutated, so every candidate filters
      // from the same intact state.
      const int32_t* lrows = materialize(D, wb, rows, n, nl, 1, depth + 1);
      const int32_t recL = dfs(ctx, lrows, nl, depth + 1, Vl);

      // [BOUND-2b] V_r <= 0, hence Q <= zeta + p_l * V_l: if even that bound
      // cannot beat the incumbent for any zeta, the whole right subtree is
      // pruned before it is ever materialized.
      can_win = false;
      for (int z = 0; z < Z; ++z)
        if (ctx.zetas[z] + pl * Vl[z] > bestQ[z]) {
          can_win = true;
          break;
        }
      if (!can_win) continue;

      // The left subtree recycled wb.side at deeper levels: recompute the
      // flags before filtering the right child.
      flag_sides(D, wb, rows, n, sp.f, sp.t);
      const int32_t* rrows = materialize(D, wb, rows, n, nr, 0, depth + 1);
      const int32_t recR = dfs(ctx, rrows, nr, depth + 1, Vr);

      for (int z = 0; z < Z; ++z) {
        const double q = ctx.zetas[z] + pl * Vl[z] + pr * Vr[z];
        // Strict '>' reproduces np.argmax first-max tie-breaking with the
        // terminal action listed first.
        if (q > bestQ[z]) {
          bestQ[z] = q;
          bestA[z] = PZ{sp.f, -1, recL, recR, sp.t};
        }
      }
    }
  }

  // [SPEED-3] append this state's per-zeta optimal decision to the arena.
  const int32_t rid = (int32_t)(ctx.arena->pz.size() / (size_t)Z);
  ctx.arena->pz.insert(ctx.arena->pz.end(), bestA, bestA + Z);
  std::copy(bestQ, bestQ + Z, Vout);
  return rid;
}

// Flat sklearn-like tree representation returned to Python.
// Classification leaves carry a class index in `val`; regression leaves carry
// O doubles per node in `valr`.
struct TreeOut {
  int O = 0;  // 0 -> classification
  std::vector<int32_t> feat, left, right, val;
  std::vector<double> thr, valr;
  int add_node(int32_t f, double t, int32_t cls, const double* lv) {
    const int id = (int)feat.size();
    feat.push_back(f);
    thr.push_back(t);
    left.push_back(-1);
    right.push_back(-1);
    val.push_back(cls);
    if (O > 0) {
      if (lv)
        valr.insert(valr.end(), lv, lv + O);
      else
        valr.insert(valr.end(), (size_t)O, 0.0);
    }
    return id;
  }
  int add_leaf_cls(int32_t cls) { return add_node(-1, -2.0, cls, nullptr); }
  int add_leaf_reg(const double* lv) { return add_node(-1, -2.0, -1, lv); }
  int add_split(int32_t f, double t) { return add_node(f, t, -1, nullptr); }
};

// Extract the optimal tree for zeta index z by walking the arena records.
int emit(const Data& D, const Arena& A, int32_t rec, int z, TreeOut& T) {
  const PZ& p = A.pz[(size_t)rec * D.Z + z];
  if (p.feat < 0)
    return D.classification()
               ? T.add_leaf_cls(p.leaf)
               : T.add_leaf_reg(A.lv.data() + (size_t)p.leaf * D.O);
  const int id = T.add_split(p.feat, p.thr);
  T.left[id] = emit(D, A, p.left, z, T);
  T.right[id] = emit(D, A, p.right, z, T);
  return id;
}

struct RootCandResult {
  bool valid = false;
  double pl = 0.0;
  std::vector<double> Vl, Vr;
  int32_t recL = -1, recR = -1;
  int arena_id = 0;
};

// Shared solver for both tasks; exactly one of yc / yr is non-null.
py::list fit_impl(const double* Xrow, const int32_t* yc, const double* yr,
                  const double* wptr, const double* zt, int n, int F, int K,
                  int O, int Z, Config C, uint32_t seed, int n_jobs) {
  std::vector<TreeOut> trees((size_t)Z);
  for (auto& T : trees) T.O = O;
  {
    // [SPEED-4] the whole solve runs without the GIL.
    py::gil_scoped_release release;

    // [SPEED-7] transpose X once; every hot loop reads one contiguous column.
    std::vector<double> XT((size_t)n * F);
    for (int i = 0; i < n; ++i)
      for (int f = 0; f < F; ++f)
        XT[(size_t)f * n + i] = Xrow[(size_t)i * F + f];

    const Data D{XT.data(), yc, yr, wptr, n, F, K, O, Z};

    // [SPEED-6] argsort every feature exactly once. `root_rows` is the root
    // state's id-lists; it is shared read-only by all workers and stays
    // sorted forever (children are filtered copies, the root is never
    // partitioned in place).
    std::vector<int32_t> root_rows((size_t)F * n);
    for (int f = 0; f < F; ++f) {
      int32_t* row = root_rows.data() + (size_t)f * n;
      std::iota(row, row + n, 0);
      const double* col = XT.data() + (size_t)f * n;
      std::stable_sort(
          row, row + n, [col](int32_t a, int32_t b) { return col[a] < col[b]; });
    }

    // ---- root terminal action ----------------------------------------------
    Arena root_leaf_arena;  // holds the root leaf value for regression
    std::vector<double> root_st((size_t)D.S(), 0.0);
    int32_t root_leaf_ref = 0;
    double wsum = 0.0;
    const double rstar =
        terminal_action(D, root_rows.data(), n, root_st.data(),
                        &root_leaf_arena, &root_leaf_ref, &wsum);

    std::vector<double> rootQ((size_t)Z, rstar);
    std::vector<int> root_choice((size_t)Z, -1);  // -1 -> root is a leaf

    std::vector<Split> root_cands;
    std::vector<RootCandResult> results;
    std::vector<Arena> arenas;

    const Resolved R0 = resolve(C, n, wsum);
    if (expandable(D, rstar) && C.max_depth > 0 &&
        n >= R0.min_samples_split) {
      WorkBuf wb0(D, C, seed);
      const int mln = !C.cart_nodes_list.empty()
                          ? std::max(2, C.cart_nodes_list[0])
                          : 2;
      cart_candidates(D, C, R0, wb0, root_rows.data(), n, mln,
                      root_st.data(), wsum, root_cands);

      const int n_tasks = (int)root_cands.size();
      results.resize((size_t)n_tasks);
      int threads =
          n_jobs <= 0 ? (int)std::thread::hardware_concurrency() : n_jobs;
      threads = std::max(1, std::min(threads, n_tasks));

      // Evaluate root candidate i (materialize + solve both subtrees)
      // against the incumbent lower bound incQ.
      auto eval_cand = [&](int i, WorkBuf& wb, Arena& arena,
                           const double* incQ) {
        RootCandResult& r = results[(size_t)i];
        const Split sp = root_cands[(size_t)i];
        // [BOUND-2a] at the root.
        bool can_win = false;
        for (int z = 0; z < Z; ++z)
          if (zt[z] > incQ[z]) {
            can_win = true;
            break;
          }
        if (!can_win) return;
        // Reseeded per candidate so the result does not depend on which
        // thread runs which task (only matters when max_features < F).
        wb.rng.seed(seed + 1u + (uint32_t)i);
        double wleft = 0.0;
        const int nl =
            flag_sides(D, wb, root_rows.data(), n, sp.f, sp.t, &wleft);
        const int nr = n - nl;
        const double wright = wsum - wleft;
        if (nl == 0 || nr == 0 || wleft <= 0.0 || wright <= 0.0) return;
        r.pl = wleft / wsum;  // [WEIGHT-EQ]
        r.Vl.resize((size_t)Z);
        r.Vr.resize((size_t)Z);
        Ctx ctx{&D, &C, zt, &arena, &wb};
        const int32_t* lrows = materialize(D, wb, root_rows.data(), n, nl, 1, 1);
        r.recL = dfs(ctx, lrows, nl, 1, r.Vl.data());
        // [BOUND-2b] at the root: prune the whole right subtree.
        can_win = false;
        for (int z = 0; z < Z; ++z)
          if (zt[z] + r.pl * r.Vl[z] > incQ[z]) {
            can_win = true;
            break;
          }
        if (!can_win) return;
        flag_sides(D, wb, root_rows.data(), n, sp.f, sp.t);
        const int32_t* rrows = materialize(D, wb, root_rows.data(), n, nr, 0, 1);
        r.recR = dfs(ctx, rrows, nr, 1, r.Vr.data());
        r.valid = true;
      };

      if (threads <= 1) {
        // [BOUND-3] sequential mode: single arena and a LIVE incumbent; each
        // solved candidate immediately strengthens pruning for the next.
        arenas.resize(1);
        for (int i = 0; i < n_tasks; ++i) {
          results[(size_t)i].arena_id = 0;
          eval_cand(i, wb0, arenas[0], rootQ.data());
          const RootCandResult& r = results[(size_t)i];
          if (!r.valid) continue;
          for (int z = 0; z < Z; ++z) {
            const double q = zt[z] + r.pl * r.Vl[z] + (1.0 - r.pl) * r.Vr[z];
            if (q > rootQ[z]) {
              rootQ[z] = q;
              root_choice[(size_t)z] = i;
            }
          }
        }
      } else {
        // [SPEED-4] parallel mode: one arena per candidate, shared read-only
        // dataset and root id-lists, work-stealing over an atomic counter.
        // The incumbent is the root terminal value (a snapshot; candidates
        // cannot see each other's results, which only weakens pruning,
        // never correctness).
        arenas.resize((size_t)n_tasks);
        const std::vector<double> incQ0 = rootQ;
        std::atomic<int> next{0};
        std::vector<std::thread> pool;
        pool.reserve((size_t)threads);
        for (int tid = 0; tid < threads; ++tid) {
          pool.emplace_back([&]() {
            WorkBuf wb(D, C, seed);
            while (true) {
              const int i = next.fetch_add(1);
              if (i >= n_tasks) break;
              results[(size_t)i].arena_id = i;
              eval_cand(i, wb, arenas[(size_t)i], incQ0.data());
            }
          });
        }
        for (auto& th : pool) th.join();
        // Combine in candidate order (matches np.argmax tie-breaking).
        for (int i = 0; i < n_tasks; ++i) {
          const RootCandResult& r = results[(size_t)i];
          if (!r.valid) continue;
          for (int z = 0; z < Z; ++z) {
            const double q = zt[z] + r.pl * r.Vl[z] + (1.0 - r.pl) * r.Vr[z];
            if (q > rootQ[z]) {
              rootQ[z] = q;
              root_choice[(size_t)z] = i;
            }
          }
        }
      }
    }

    // ---- extract one flat tree per zeta --------------------------------------
    for (int z = 0; z < Z; ++z) {
      TreeOut& T = trees[(size_t)z];
      const int c = root_choice[(size_t)z];
      if (c < 0) {
        if (D.classification())
          T.add_leaf_cls(root_leaf_ref);
        else
          T.add_leaf_reg(root_leaf_arena.lv.data());
      } else {
        const RootCandResult& r = results[(size_t)c];
        const Arena& A = arenas[(size_t)r.arena_id];
        const int id =
            T.add_split(root_cands[(size_t)c].f, root_cands[(size_t)c].t);
        T.left[id] = emit(D, A, r.recL, z, T);
        T.right[id] = emit(D, A, r.recR, z, T);
      }
    }
  }  // GIL re-acquired

  py::list out;
  for (int z = 0; z < Z; ++z) {
    const TreeOut& T = trees[(size_t)z];
    py::dict d;
    d["feature"] =
        py::array_t<int32_t>((py::ssize_t)T.feat.size(), T.feat.data());
    d["threshold"] =
        py::array_t<double>((py::ssize_t)T.thr.size(), T.thr.data());
    d["children_left"] =
        py::array_t<int32_t>((py::ssize_t)T.left.size(), T.left.data());
    d["children_right"] =
        py::array_t<int32_t>((py::ssize_t)T.right.size(), T.right.data());
    if (O > 0) {
      py::array_t<double> v({(py::ssize_t)T.feat.size(), (py::ssize_t)O});
      std::copy(T.valr.begin(), T.valr.end(), v.mutable_data());
      d["value"] = v;
    } else {
      d["value"] =
          py::array_t<int32_t>((py::ssize_t)T.val.size(), T.val.data());
    }
    out.append(d);
  }
  return out;
}

py::list fit_classifier(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> w,
    py::array_t<double, py::array::c_style | py::array::forcecast> zetas,
    int n_classes, int max_depth, std::vector<int> cart_nodes_list,
    double min_samples_split, bool mss_is_fraction, double min_samples_leaf,
    bool msl_is_fraction, double min_weight_fraction_leaf, int max_features,
    double min_impurity_decrease, int criterion, uint32_t seed, int n_jobs) {
  if (X.ndim() != 2) throw std::invalid_argument("X must be 2D");
  const int n = (int)X.shape(0);
  const int F = (int)X.shape(1);
  const int Z = (int)zetas.shape(0);
  if ((int)y.shape(0) != n || (int)w.shape(0) != n)
    throw std::invalid_argument("X, y and sample_weight sizes disagree");
  const Config C{max_depth,
                 std::move(cart_nodes_list),
                 min_samples_split,
                 mss_is_fraction,
                 min_samples_leaf,
                 msl_is_fraction,
                 min_weight_fraction_leaf,
                 std::max(1, std::min(max_features, F)),
                 min_impurity_decrease,
                 criterion};
  return fit_impl(X.data(), y.data(), nullptr, w.data(), zetas.data(), n, F,
                  n_classes, 0, Z, C, seed, n_jobs);
}

py::list fit_regressor(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> w,
    py::array_t<double, py::array::c_style | py::array::forcecast> zetas,
    int max_depth, std::vector<int> cart_nodes_list, double min_samples_split,
    bool mss_is_fraction, double min_samples_leaf, bool msl_is_fraction,
    double min_weight_fraction_leaf, int max_features,
    double min_impurity_decrease, uint32_t seed, int n_jobs) {
  if (X.ndim() != 2) throw std::invalid_argument("X must be 2D");
  if (y.ndim() != 2) throw std::invalid_argument("y must be 2D (n, n_outputs)");
  const int n = (int)X.shape(0);
  const int F = (int)X.shape(1);
  const int O = (int)y.shape(1);
  const int Z = (int)zetas.shape(0);
  if ((int)y.shape(0) != n || (int)w.shape(0) != n)
    throw std::invalid_argument("X, y and sample_weight sizes disagree");
  const Config C{max_depth,
                 std::move(cart_nodes_list),
                 min_samples_split,
                 mss_is_fraction,
                 min_samples_leaf,
                 msl_is_fraction,
                 min_weight_fraction_leaf,
                 std::max(1, std::min(max_features, F)),
                 min_impurity_decrease,
                 2 /* squared_error */};
  return fit_impl(X.data(), nullptr, y.data(), w.data(), zetas.data(), n, F,
                  0, O, Z, C, seed, n_jobs);
}

// Route samples through one extracted flat classification tree.
// Returns (class indices, decision path lengths, #distinct internal nodes
// visited). Much faster than the Python per-leaf mask scan.
py::tuple predict_tree(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> feature,
    py::array_t<double, py::array::c_style | py::array::forcecast> threshold,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> left,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> right,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> value) {
  const int n = (int)X.shape(0);
  const int F = X.ndim() == 2 ? (int)X.shape(1) : 0;
  py::array_t<int32_t> yout((py::ssize_t)n);
  py::array_t<int32_t> lout((py::ssize_t)n);
  int32_t* yp = yout.mutable_data();
  int32_t* lp = lout.mutable_data();
  int visited_internal = 0;
  {
    py::gil_scoped_release release;
    const double* Xp = X.data();
    const int32_t* ft = feature.data();
    const double* th = threshold.data();
    const int32_t* lf = left.data();
    const int32_t* rt = right.data();
    const int32_t* vl = value.data();
    std::vector<char> seen((size_t)feature.shape(0), 0);
    for (int i = 0; i < n; ++i) {
      int node = 0, depth = 0;
      while (ft[node] >= 0) {
        if (!seen[(size_t)node]) {
          seen[(size_t)node] = 1;
          ++visited_internal;
        }
        node = Xp[(size_t)i * F + ft[node]] <= th[node] ? lf[node] : rt[node];
        ++depth;
      }
      yp[i] = vl[node];
      lp[i] = depth;
    }
  }
  return py::make_tuple(yout, lout, visited_internal);
}

// Same as predict_tree, but for regression trees whose `value` array has
// shape (n_nodes, n_outputs); returns (n, n_outputs) predictions.
py::tuple predict_tree_reg(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> feature,
    py::array_t<double, py::array::c_style | py::array::forcecast> threshold,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> left,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> right,
    py::array_t<double, py::array::c_style | py::array::forcecast> value) {
  const int n = (int)X.shape(0);
  const int F = X.ndim() == 2 ? (int)X.shape(1) : 0;
  if (value.ndim() != 2)
    throw std::invalid_argument("value must be 2D (n_nodes, n_outputs)");
  const int O = (int)value.shape(1);
  py::array_t<double> yout({(py::ssize_t)n, (py::ssize_t)O});
  py::array_t<int32_t> lout((py::ssize_t)n);
  double* yp = yout.mutable_data();
  int32_t* lp = lout.mutable_data();
  int visited_internal = 0;
  {
    py::gil_scoped_release release;
    const double* Xp = X.data();
    const int32_t* ft = feature.data();
    const double* th = threshold.data();
    const int32_t* lf = left.data();
    const int32_t* rt = right.data();
    const double* vl = value.data();
    std::vector<char> seen((size_t)feature.shape(0), 0);
    for (int i = 0; i < n; ++i) {
      int node = 0, depth = 0;
      while (ft[node] >= 0) {
        if (!seen[(size_t)node]) {
          seen[(size_t)node] = 1;
          ++visited_internal;
        }
        node = Xp[(size_t)i * F + ft[node]] <= th[node] ? lf[node] : rt[node];
        ++depth;
      }
      for (int o = 0; o < O; ++o)
        yp[(size_t)i * O + o] = vl[(size_t)node * O + o];
      lp[i] = depth;
    }
  }
  return py::make_tuple(yout, lout, visited_internal);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() =
      "C++ core for Dynamic Programming Decision Trees (MDP construction, "
      "dynamic program with branch & bound, and fast tree traversal).";
  m.def("fit_classifier", &fit_classifier, py::arg("X"), py::arg("y"),
        py::arg("sample_weight"), py::arg("zetas"), py::arg("n_classes"),
        py::arg("max_depth"), py::arg("cart_nodes_list"),
        py::arg("min_samples_split"), py::arg("mss_is_fraction"),
        py::arg("min_samples_leaf"), py::arg("msl_is_fraction"),
        py::arg("min_weight_fraction_leaf"), py::arg("max_features"),
        py::arg("min_impurity_decrease"), py::arg("criterion"),
        py::arg("seed"), py::arg("n_jobs"));
  m.def("fit_regressor", &fit_regressor, py::arg("X"), py::arg("y"),
        py::arg("sample_weight"), py::arg("zetas"), py::arg("max_depth"),
        py::arg("cart_nodes_list"), py::arg("min_samples_split"),
        py::arg("mss_is_fraction"), py::arg("min_samples_leaf"),
        py::arg("msl_is_fraction"), py::arg("min_weight_fraction_leaf"),
        py::arg("max_features"), py::arg("min_impurity_decrease"),
        py::arg("seed"), py::arg("n_jobs"));
  m.def("predict_tree", &predict_tree, py::arg("X"), py::arg("feature"),
        py::arg("threshold"), py::arg("children_left"),
        py::arg("children_right"), py::arg("value"));
  m.def("predict_tree_reg", &predict_tree_reg, py::arg("X"),
        py::arg("feature"), py::arg("threshold"), py::arg("children_left"),
        py::arg("children_right"), py::arg("value"));
}
