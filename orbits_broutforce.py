import numpy as np
import itertools
from itertools import combinations
from scipy.linalg import qr, solve_triangular
from scipy.sparse import lil_matrix
import multiprocessing as mp
import time

# ─────────────────────────────────────────────────────────────────────────────
# 1. ORBIT INFRASTRUCTURE & ISOMORPHISM FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def get_orbit_representative(face: tuple[int, int, int], n: int = 8) -> tuple[int, int, int]:
    """
    Computes the canonical cyclic invariant (orbit representative) of a face modulo n.
    
    Input:
        face (tuple[int, int, int]): A triplet of vertex indices representing a 2-simplex.
        n (int): The total number of vertices in the complex (default is 8).
    Output:
        tuple[int, int, int]: A sorted triplet of shortest circular distances between vertices.
    """
    i, j, k = sorted(face)
    
    d1 = (j - i) % n
    d2 = (k - j) % n
    d3 = (i - k) % n
    
    dist1 = min(d1, n - d1)
    dist2 = min(d2, n - d2)
    dist3 = min(d3, n - d3)
    
    return tuple(sorted([dist1, dist2, dist3]))


def get_all_orbits(n: int = 8) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], list[int]], list[tuple[int, int, int]]]:
    """
    Generates all faces of K_n, groups their indices by cyclic orbits, and returns the global face mapping.
    
    Input:
        n (int): Total number of vertices (default is 8).
    Output:
        tuple: (sorted_orbit_keys, orbit_to_face_indices_dict, global_faces_list)
    """
    faces = list(combinations(range(n), 3))
    orbit_groups = {}
    
    for idx, face in enumerate(faces):
        rep = get_orbit_representative(face, n)
        if rep not in orbit_groups:
            orbit_groups[rep] = []
        orbit_groups[rep].append(idx)
        
    return sorted(orbit_groups.keys()), orbit_groups, faces


def filter_unique_orbit_combinations(face_indices: list[int], choose_k: int, all_faces: list, n: int = 8) -> list[tuple[int, ...]]:
    """
    Filters all combinations of choosing k faces from a specific orbit pool,
    retaining only structurally unique (non-isomorphic under Z_n) configurations.
    
    Input:
        face_indices (list[int]): Global face IDs belonging to a specific orbit.
        choose_k (int): Number of faces to select from this orbit.
        all_faces (list): Global mapping of face IDs to vertex triplets.
        n (int): Number of vertices (8).
    Output:
        list[tuple[int, ...]]: A highly reduced list of unique face-index combinations.
    """
    raw_combs = list(combinations(face_indices, choose_k))
    unique_combs = []
    seen_configurations = set()

    for comb in raw_combs:
        actual_triplets = [all_faces[idx] for idx in comb]
        is_duplicate = False
        canonical_variants = []
        
        # Check all 8 cyclic rotations (shifts) of the current face configuration
        for shift in range(n):
            shifted_triplets = []
            for face in actual_triplets:
                sf = tuple(sorted([(v + shift) % n for v in face]))
                shifted_triplets.append(sf)
            shifted_triplets.sort()
            variant_key = tuple(shifted_triplets)
            canonical_variants.append(variant_key)
            if variant_key in seen_configurations:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_combs.append(comb)
            for variant in canonical_variants:
                seen_configurations.add(variant)
                
    return unique_combs

# ─────────────────────────────────────────────────────────────────────────────
# 2. BOUNDARY MATRIX & LINEAR ALGEBRA CORE
# ─────────────────────────────────────────────────────────────────────────────

def create_2d_boundary_matrix(n: int = 8) -> np.ndarray:
    """
    Constructs the dense 2D boundary matrix for linear independence checks.
    """
    edges = list(combinations(range(n), 2))
    faces = list(combinations(range(n), 3))
    edge_idx = {e: i for i, e in enumerate(edges)}

    m = len(edges)
    l = len(faces)
    M = lil_matrix((m, l), dtype=np.float64)

    for b, (i, j, k) in enumerate(faces):
        M[edge_idx[(i, j)], b] = 1.0
        M[edge_idx[(i, k)], b] = -1.0
        M[edge_idx[(j, k)], b] = 1.0

    return np.asarray(M.todense(), dtype=np.float64)


def evaluate_basis_score(M: np.ndarray, basis_indices: list[int], non_basis_indices: list[int]) -> float:
    """
    Computes the average fundamental circuit size for a given basis candidate.
    """
    B = M[:, basis_indices]
    Q, R = qr(B, mode='economic')
    
    if np.any(np.abs(np.diagonal(R)) < 1e-9):
        return -1.0
        
    N = M[:, non_basis_indices]
    QtN = Q.T @ N
    A = solve_triangular(R, QtN, lower=False)
    
    nonzero_counts = np.count_nonzero(np.abs(A) > 1e-9)
    return (nonzero_counts / len(non_basis_indices)) + 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 3. PARALLEL CHUNK WORKER
# ─────────────────────────────────────────────────────────────────────────────

def isomorphism_reduced_worker(args) -> tuple[float, list[list[int]]]:
    """
    Processes a localized pool segment containing pre-filtered non-isomorphic configurations.
    """
    (M, total_faces, fixed_orbit_1_and_2, remaining_filtered_pools) = args
    all_face_indices = set(range(total_faces))
    
    local_max = -1.0
    local_best = []
    
    # Iterate across pre-filtered combinations of orbits 3, 4, and 5
    for remaining_combined in itertools.product(*remaining_filtered_pools):
        basis_candidate = sorted(fixed_orbit_1_and_2 + list(itertools.chain(*remaining_combined)))
        non_basis = sorted(list(all_face_indices - set(basis_candidate)))
        
        score = evaluate_basis_score(M, basis_candidate, non_basis)
        
        if score > 0:
            if score > local_max:
                local_max = score
                local_best = [basis_candidate]
            elif np.abs(score - local_max) < 1e-9:
                local_best.append(basis_candidate)
                
    return local_max, local_best

# ─────────────────────────────────────────────────────────────────────────────
# 4. EXECUTION COORDINATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_isomorphism_reduced_search(n: int, target_profile: list[int], num_processes: int = 10):
    """
    Main controller executing the pre-filtered isomorphic-free parallel brute force.
    """
    t_start = time.time()
    M = create_2d_boundary_matrix(n)
    total_faces = M.shape[1]
    
    sorted_orbits, orbit_groups, all_faces = get_all_orbits(n)
    
    print(f"Executing Pre-filtering Stage over Orbits...")
    print(f"------------------------------------------------------------")
    
    # Pre-filter each orbit individually to keep only structurally unique combinations
    filtered_pools = []
    for i, rep in enumerate(sorted_orbits):
        t_pool = time.time()
        choose_k = target_profile[i]
        face_pool = orbit_groups[rep]
        
        unique_combs = filter_unique_orbit_combinations(face_pool, choose_k, all_faces, n)
        filtered_pools.append(unique_combs)
        
        raw_size = len(list(combinations(face_pool, choose_k)))
        print(f" - Orbit {i+1} {rep}: Reduced combinations from {raw_size:,} to {len(unique_combs):,} (~{raw_size/max(1, len(unique_combs)):.1f}x reduction) in {time.time()-t_pool:.2f}s")
        
    # Calculate the newly collapsed search space size
    reduced_total_space = 1
    for pool in filtered_pools:
        reduced_total_space *= len(pool)
        
    print(f"------------------------------------------------------------")
    print(f"REDUCED COMBINATORIAL SPACE: {reduced_total_space:,} iterations (Down from 1,744,363,008!)")
    print(f"------------------------------------------------------------\n")
    
    # Construct task chunks using a combination of Orbit 1 and Orbit 2 to distribute across 10 processes
    orbit_1_and_2_product = list(itertools.product(filtered_pools[0], filtered_pools[1]))
    remaining_pools = filtered_pools[2:]  # Pools for Orbits 3, 4, 5
    
    worker_tasks = [
        (M, total_faces, list(itertools.chain(*chunk)), remaining_pools)
        for chunk in orbit_1_and_2_product
    ]
    
    print(f"Dispatching task arrays across {num_processes} CPU cores...")
    t_calc = time.time()
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(isomorphism_reduced_worker, worker_tasks)
        
    # Collate outputs
    global_max = -1.0
    global_best_bases = []
    
    for local_max, local_bases in results:
        if local_max > global_max:
            global_max = local_max
            global_best_bases = local_bases
        elif np.abs(local_max - global_max) < 1e-9 and local_max > 0:
            global_best_bases.extend(local_bases)
            
    total_time = time.time() - t_start
    print(f"\n============================================================")
    print(f"EXECUTION COMPLETED")
    print(f"============================================================")
    print(f"Total Filtering + Compute Time : {total_time:.2f} seconds")
    print(f"Calculation Phase Runtime       : {time.time() - t_calc:.2f} seconds")
    print(f"Absolute Max Average Cycle Size : {global_max:.6f}")
    print(f"Total Optimal Bases Discovered : {len(global_best_bases)}")
    print(f"============================================================")
    
    return global_max, global_best_bases


if __name__ == "__main__":
    PROFILE_VECTOR = [4, 4, 5, 3, 5]
    VERTICES = 8
    CORES = 10
    
    max_score, best_bases = run_isomorphism_reduced_search(
        n=VERTICES, 
        target_profile=PROFILE_VECTOR, 
        num_processes=10
    )