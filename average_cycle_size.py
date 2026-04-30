"""
Average Fundamental Cycle Length in K_n
========================================
Input  : the oriented vertex-edge incidence matrix of K_n
         rows = vertices (0..n-1)
         cols = edges {j,k} with j<k, ordered lexicographically
         M[i, {j,k}] = +1 if i==j, -1 if i==k, 0 otherwise
 
Process:
  1. Parse M  →  recover vertex/edge list of K_n
  2. Sample a uniformly-random spanning tree via Wilson's algorithm (Loop-Erased Random Walk)
  3. For every non-tree edge e=(u,v): find the unique fundamental cycle
     by tracing the tree path u→v (BFS on the tree) and adding e.
     Record its length (number of edges in the cycle).
  4. Repeat for `num_trials` random trees and average all cycle lengths.
 
Output : estimated average fundamental cycle length  E[|C|]
"""
 
import numpy as np
import sys
import csv
 
 
# ──────────────────────────────────────────────
# 1.  Parse the incidence matrix
# ──────────────────────────────────────────────
 
def parse_incidence_matrix(M: np.ndarray):
    """
    Recover the edge list of the graph encoded by the
    oriented vertex-edge incidence matrix M.
 
    Returns
    -------
    n_vertices : int
    edges      : list of (u, v) tuples with u < v,
                 in the same column order as M
    """
    n_vertices, n_edges = M.shape
    edges = []
    for col in range(n_edges):
        col_data = M[:, col]
        # +1 entry → smaller endpoint (u), -1 entry → larger endpoint (v)
        pos = np.where(col_data == 1)[0]
        neg = np.where(col_data == -1)[0]
        if len(pos) != 1 or len(neg) != 1:
            raise ValueError(f"Column {col} is not a valid oriented edge.")
        u, v = int(pos[0]), int(neg[0])
        edges.append((u, v))
    return n_vertices, edges
 
 
def build_adjacency(n: int, edges: list) -> dict:
    """Build adjacency list from edge list."""
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj
 
 
# ──────────────────────────────────────────────
# 2.  Wilson's algorithm (uniform spanning tree)
# ──────────────────────────────────────────────
 
def wilsons_algorithm(n: int, adj: dict, rng: np.random.Generator) -> set:
    """
    Generate a uniformly random spanning tree of the graph
    using Wilson's Loop-Erased Random Walk algorithm.
 
    Returns
    -------
    tree_edges : set of frozenset({u, v}) pairs
    """
    in_tree = [False] * n
    # Pick an arbitrary root
    root = int(rng.integers(0, n))
    in_tree[root] = True
 
    tree_edges = set()
 
    for start in range(n):
        if in_tree[start]:
            continue
 
        # Random walk from `start` until hitting a tree node
        path = [start]
        visited_pos = {start: 0}   # node → index in path (for loop erasure)
 
        current = start
        while not in_tree[current]:
            neighbours = adj[current]
            nxt = int(rng.choice(neighbours))
 
            if nxt in visited_pos:
                # Erase the loop
                loop_start = visited_pos[nxt]
                # Remove position info for erased nodes
                for node in path[loop_start + 1:]:
                    del visited_pos[node]
                path = path[: loop_start + 1]
            else:
                path.append(nxt)
                visited_pos[nxt] = len(path) - 1
 
            current = nxt
 
        # Commit path to tree
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            tree_edges.add(frozenset({u, v}))
            in_tree[u] = True
        in_tree[path[-1]] = True   # the node already in tree
 
    return tree_edges


def prufer_sequence_algorithm(n: int, rng: np.random.Generator) -> set:
    """
    Generate a uniformly random spanning tree of K_n using Prüfer sequences.
    
    A Prüfer sequence is a sequence of n-2 integers in [0, n-1] that uniquely
    encodes a labeled tree on n vertices. This algorithm:
    1. Generates a random Prüfer sequence
    2. Decodes it into a tree
    
    Returns
    -------
    tree_edges : set of frozenset({u, v}) pairs
    """
    if n == 1:
        return set()
    if n == 2:
        return {frozenset({0, 1})}
    
    # Generate random Prüfer sequence of length n-2
    prufer_seq = rng.integers(0, n, size=n - 2)
    
    # Decode Prüfer sequence to tree edges
    # Count the degree of each vertex (degree = count in sequence + 1)
    degree = np.ones(n, dtype=int)
    degree += np.bincount(prufer_seq, minlength=n)
    
    tree_edges = set()
    
    # Decode the sequence
    for vertex in prufer_seq:
        # Find the smallest vertex with degree 1
        for leaf in range(n):
            if degree[leaf] == 1:
                tree_edges.add(frozenset({leaf, vertex}))
                degree[leaf] -= 1
                degree[vertex] -= 1
                break
    
    # Connect the last two vertices with degree 1
    remaining = [i for i in range(n) if degree[i] == 1]
    if len(remaining) == 2:
        tree_edges.add(frozenset({remaining[0], remaining[1]}))
    
    return tree_edges

 
# ──────────────────────────────────────────────
# 3.  Fundamental cycle length via BFS on tree
# ──────────────────────────────────────────────
 
def tree_path_length(u: int, v: int, tree_adj: dict) -> int:
    """
    Return the number of edges on the unique path from u to v
    in the spanning tree (BFS).
    """
    if u == v:
        return 0
    visited = {u: None}
    queue = [u]
    head = 0
    while head < len(queue):
        node = queue[head]; head += 1
        for nb in tree_adj[node]:
            if nb not in visited:
                visited[nb] = node
                if nb == v:
                    # Trace back to count edges
                    length = 0
                    cur = v
                    while visited[cur] is not None:
                        cur = visited[cur]
                        length += 1
                    return length
                queue.append(nb)
    raise ValueError(f"No path found between {u} and {v} — tree disconnected?")
 
 
def fundamental_cycle_length(u: int, v: int, tree_adj: dict) -> int:
    """
    Length of the fundamental cycle created by adding edge (u,v)
    to the spanning tree = tree_path(u,v) + 1.
    """
    return tree_path_length(u, v, tree_adj) + 1
 # ──────────────────────────────────────────────
# 4.  Main simulation
# ──────────────────────────────────────────────
 
def average_fundamental_cycle_length(
    M: np.ndarray,
    num_trials: int = 1000,
    seed: int = 42,
    algorithm: str = "prufer",
) -> dict:
    """
    Estimate E[|fundamental cycle|] by Monte Carlo simulation.
 
    Parameters
    ----------
    M          : oriented vertex-edge incidence matrix of K_n
    num_trials : number of random spanning trees to sample
    seed       : RNG seed for reproducibility
    algorithm  : "wilson" or "prufer" for spanning tree generation method
 
    Returns
    -------
    dict with keys:
      'average'       – estimated E[|C|]
      'std'           – standard deviation of cycle lengths
      'all_lengths'   – numpy array of every recorded cycle length
      'n_vertices'    – number of vertices
      'n_edges'       – number of edges in the graph
      'n_tree_edges'  – n-1 (edges per spanning tree)
      'n_non_tree'    – non-tree edges per trial = n_edges - (n-1)
      'algorithm'     – which algorithm was used
    """
    if algorithm not in ["wilson", "prufer"]:
        raise ValueError("algorithm must be 'wilson' or 'prufer'")
    
    rng = np.random.default_rng(seed)
 
    n, edges = parse_incidence_matrix(M)
    adj = build_adjacency(n, edges)
    edge_set = {frozenset({u, v}) for u, v in edges}
 
    all_lengths = []
 
    for _ in range(num_trials):
        # Choose algorithm for generating spanning tree
        if algorithm == "wilson":
            tree_edges = wilsons_algorithm(n, adj, rng)
        elif algorithm == "prufer":
            tree_edges = prufer_sequence_algorithm(n, rng)
        else:
            raise ValueError("Invalid algorithm choice.")
        # Build tree adjacency list
        tree_adj = {i: [] for i in range(n)}
        for fe in tree_edges:
            u, v = tuple(fe)
            tree_adj[u].append(v)
            tree_adj[v].append(u)
 
        # Non-tree edges
        non_tree = edge_set - tree_edges
 
        for fe in non_tree:
            u, v = tuple(fe)
            length = fundamental_cycle_length(u, v, tree_adj)
            all_lengths.append(length)
 
    all_lengths = np.array(all_lengths, dtype=np.int32)
 
    return {
        "average":      float(np.mean(all_lengths)),
        "std":          float(np.std(all_lengths)),
        "all_lengths":  all_lengths,
        "n_vertices":   n,
        "n_edges":      len(edges),
        "n_tree_edges": n - 1,
        "n_non_tree":   len(edges) - (n - 1),
        "algorithm":    algorithm,
    }
 
 
# ──────────────────────────────────────────────
# 5.  Build K_n incidence matrix (helper)
# ──────────────────────────────────────────────
 
def build_kn_incidence_matrix(n: int) -> np.ndarray:
    """
    Build the oriented vertex-edge incidence matrix of K_n.
    Edges ordered lexicographically: (0,1),(0,2),...,(n-2,n-1).
    M[i, col({j,k})] = +1 if i==j<k, -1 if i==k>j, 0 otherwise.
    """
    edges = [(j, k) for j in range(n) for k in range(j + 1, n)]
    M = np.zeros((n, len(edges)), dtype=np.int8)
    for col, (j, k) in enumerate(edges):
        M[j, col] = 1
        M[k, col] = -1
    return M
 
 
# ──────────────────────────────────────────────
# 6.  Demo
# ──────────────────────────────────────────────
 
if __name__ == "__main__":
    # Default: run K_4 through K_70 
    ns = list(range(200,1000,100)))
    num_trials = 1000
    seed = 13
    algorithm = "prufer"  # "wilson" or "prufer"
    
    # Parse command-line arguments
    # Usage: python average_cycle_size.py [--algorithm {wilson|prufer}] [--num_trials N] [--seed S] [n1 n2 ...]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--algorithm":
            algorithm = sys.argv[i + 1]
            i += 2
        elif arg == "--num_trials":
            num_trials = int(sys.argv[i + 1])
            i += 2
        elif arg == "--seed":
            seed = int(sys.argv[i + 1])
            i += 2
        else:
            # Assume remaining args are vertex counts
            ns = [int(x) for x in sys.argv[i:]]
            break
    
    print(f"Running with algorithm: {algorithm}, num_trials: {num_trials}, seed: {seed}")
    
    # Write results to CSV file
    filename = f'results_{algorithm}_ns-{min(ns)}-{max(ns)}_trials-{num_trials}_seed-{seed}_generating{algorithm}.csv'
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['n', 'average_cycle_length', 'std_dev', 'num_trials', 'algorithm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for n in ns:
            M = build_kn_incidence_matrix(n)
            result = average_fundamental_cycle_length(M, num_trials, seed, algorithm=algorithm)
            writer.writerow({
                'n': n,
                'average_cycle_length': f"{result['average']:.4f}",
                'std_dev': f"{result['std']:.4f}",
                'num_trials': num_trials,
                'algorithm': algorithm
            })
            print(f"n={n}: average cycle length = {result['average']:.4f} (std: {result['std']:.4f})")
    
    print(f"Results written to {filename}")