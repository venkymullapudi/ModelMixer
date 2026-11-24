#!/usr/bin/env python3
"""align_mlp.py: permute model B's hidden neurons to match model A before averaging (Git Re-Basin weight matching).

A starting point from ModelMixer, scoped on purpose:
  * Plain MLP stacks only: Linear -> elementwise activation -> Linear -> ... -> Linear.
    Hidden layer l has weight W_l [out, in] and optional bias b_l [out]; the next layer's W_{l+1} [out2, out] reads it.
    Reordering hidden units reorders the rows of W_l and b_l and the columns of W_{l+1}. The network computes exactly
    the same function, so B is unchanged as a model; it is only renumbered to line up with A.
  * Not for transformers, attention, convolutions or normalisation layers as-is: residual streams, heads and norms
    tie permutations together. Use model-specific tools, e.g. https://github.com/samuela/git-re-basin.
  * Only needed when the two models do NOT share a base. Fine-tunes of one base are already aligned.
  * Alignment assumes both models learned the same function (runs of one skill). For different skills it can hurt
    (in ModelMixer's toys: up to -17 pts at width 128, -8 to -35 when the same skill covers different inputs).

Usage:
  python3 align_mlp.py A.safetensors B.safetensors B_aligned.safetensors \\
      --layers fc1.weight,fc1.bias fc2.weight,fc2.bias fc3.weight,fc3.bias
  --layers lists every Linear layer in order as weight[,bias]; the last one is the output layer (its rows stay put).
  Files may be .safetensors (needs `pip install safetensors`) or .npz. Assignment uses scipy if installed, else a
  pure-numpy Hungarian solver (fine up to a few thousand units per layer).
  python3 align_mlp.py --self-test     # checks it recovers a known shuffle and preserves B's function
"""
import argparse
import sys

import numpy as np


# ---------- assignment (maximise total similarity) ----------
def _hungarian_min(cost):
    """Square min-cost assignment, O(n^3) with numpy-vectorised inner steps. Returns col index per row."""
    n = cost.shape[0]
    INF = np.inf
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)  # p[j] = row matched to column j (1-based, 0 = none)
    way = np.zeros(n + 1, dtype=int)
    c = np.zeros((n + 1, n + 1))
    c[1:, 1:] = cost
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = c[i0, 1:] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], INF)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    perm = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        perm[p[j] - 1] = j - 1
    return perm


def best_permutation(sim):
    """perm[i] = which unit of B goes to position i, maximising sum_i sim[i, perm[i]]."""
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(sim, maximize=True)
        perm = np.empty_like(cols)
        perm[rows] = cols
        return perm
    except ImportError:
        return _hungarian_min(-sim)


# ---------- weight matching ----------
def weight_matching(A, B, layers, rounds=30, seed=0):
    """A, B: dicts name -> array. layers: [(weight_name, bias_name or None), ...] in order.
    Returns one permutation per hidden layer (all layers but the last)."""
    rng = np.random.default_rng(seed)
    L = len(layers)
    sizes = [A[w].shape[0] for w, _ in layers[:-1]]
    perms = [np.arange(n) for n in sizes]
    for _ in range(rounds):
        changed = False
        for l in rng.permutation(L - 1):
            w, b = layers[l]
            w_next = layers[l + 1][0]
            # incoming weights of B's units, with B's previous hidden layer already permuted
            WB_in = B[w] if l == 0 else B[w][:, perms[l - 1]]
            sim = A[w] @ WB_in.T
            if b:
                sim += np.outer(A[b], B[b])
            # outgoing weights, with B's next hidden layer permuted (the output layer is never permuted)
            WB_out = B[w_next] if l + 1 == L - 1 else B[w_next][perms[l + 1], :]
            sim += A[w_next].T @ WB_out
            new = best_permutation(sim)
            if not np.array_equal(new, perms[l]):
                perms[l], changed = new, True
        if not changed:
            break
    return perms


def apply_permutations(B, layers, perms):
    """Renumber B's hidden units: rows of W_l / b_l and columns of W_{l+1}. Same function, new order."""
    out = {k: v.copy() for k, v in B.items()}
    for l, perm in enumerate(perms):
        w, b = layers[l]
        out[w] = out[w][perm, :]
        if b:
            out[b] = out[b][perm]
        w_next = layers[l + 1][0]
        out[w_next] = out[w_next][:, perm]
    return out


# ---------- io ----------
def load(path):
    if path.endswith(".npz"):
        return dict(np.load(path))
    from safetensors.numpy import load_file
    return load_file(path)


def save(tensors, path):
    if path.endswith(".npz"):
        np.savez(path, **tensors)
        return
    from safetensors.numpy import save_file
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, path)


def forward(T, layers, x):
    for i, (w, b) in enumerate(layers):
        x = x @ T[w].T + (T[b] if b else 0)
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    return x


def self_test():
    rng = np.random.default_rng(1)
    dims = [8, 64, 48, 3]
    layers = [(f"fc{i + 1}.weight", f"fc{i + 1}.bias") for i in range(len(dims) - 1)]
    A = {}
    for i in range(len(dims) - 1):
        A[f"fc{i + 1}.weight"] = rng.standard_normal((dims[i + 1], dims[i]))
        A[f"fc{i + 1}.bias"] = rng.standard_normal(dims[i + 1])
    # B = A with its hidden units shuffled (same function) plus a little noise, like a second run
    true = [rng.permutation(n) for n in dims[1:-1]]
    B = apply_permutations(A, layers, true)
    B = {k: v + 0.05 * rng.standard_normal(v.shape) for k, v in B.items()}
    perms = weight_matching(A, B, layers)
    aligned = apply_permutations(B, layers, perms)
    x = rng.standard_normal((32, dims[0]))
    same_fn = np.allclose(forward(aligned, layers, x), forward(B, layers, x))
    dist_before = sum(np.linalg.norm(A[k] - B[k]) for k in A)
    dist_after = sum(np.linalg.norm(A[k] - aligned[k]) for k in A)
    # the recovered order undoes the shuffle: B's unit perm[i] is A's unit i
    recovered = all(np.array_equal(np.argsort(t), p) for t, p in zip(true, perms))
    print(f"function preserved: {same_fn}; distance to A {dist_before:.1f} -> {dist_after:.1f}; shuffle recovered: {recovered}")
    # the pure-numpy solver agrees with brute force on a tiny problem
    from itertools import permutations
    sim = rng.standard_normal((6, 6))
    brute = max(permutations(range(6)), key=lambda q: sum(sim[i, q[i]] for i in range(6)))
    h = _hungarian_min(-sim)
    solver_ok = np.isclose(sum(sim[i, h[i]] for i in range(6)), sum(sim[i, brute[i]] for i in range(6)))
    print(f"numpy Hungarian matches brute force: {solver_ok}")
    ok = same_fn and recovered and dist_after < 0.2 * dist_before and solver_ok
    print("self-test", "passed" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--layers", nargs="+", help="weight[,bias] per Linear layer, in order (last = output layer)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not (args.a and args.b and args.out and args.layers and len(args.layers) >= 2):
        ap.error("need A, B, OUT and --layers with at least two layers (or --self-test)")
    layers = [(s.split(",")[0], (s.split(",") + [None])[1] or None) for s in args.layers]
    A, B = load(args.a), load(args.b)
    for w, b in layers:
        for name in filter(None, (w, b)):
            if name not in A or name not in B:
                sys.exit(f"tensor {name!r} missing from one of the models")
    perms = weight_matching(A, B, layers)
    out = apply_permutations(B, layers, perms)
    save(out, args.out)
    d0 = sum(np.linalg.norm(A[w] - B[w]) for w, _ in layers)
    d1 = sum(np.linalg.norm(A[w] - out[w]) for w, _ in layers)
    print(f"aligned {len(perms)} hidden layer(s); weight distance to A {d0:.2f} -> {d1:.2f}; wrote {args.out}")


if __name__ == "__main__":
    main()
