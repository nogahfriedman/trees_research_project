import numpy as np
from itertools import combinations
from sympy import Matrix, Rational
import random
import math
import time



def create_2d_simplex(n: int) -> Matrix:
    """
    Create the 2D simplex matrix for n vertices.
    This is the boundary matrix of the complete graph K_n.
    It has n rows (vertices) and C(n, 2) columns (edges).
    Each column has exactly two 1's corresponding to the endpoints of the edge.
    """
    edges = list(combinations(range(n), 2))  # all pairs of vertices
    faces = list(combinations(range(n), 3))  # all triples of vertices (2-simplices)
    m = len(edges)  # number of edges
    l = len(faces)  # number of 2-simplices
    M = np.zeros((m, l), dtype=int)
    for a , (u,v) in enumerate(edges):
        for b ,(i, j, k) in enumerate(faces):
            if (u == i and v == j) or (u == j and v == k):
                M[a, b] = 1
            elif u == i and v == k:
                M[a, b] = -1
    return Matrix(M)


def get_random_basis_with_faces(n: int, required_faces: list, all_faces: list, M_np: np.ndarray, rank: int, n_cols: int, max_attempts: int = 10000) -> list:
    """
    Generate a random basis that includes the specified faces.
    
    Parameters:
    -----------
    n : int
        Number of vertices for the 2D simplex
    required_faces : list
        List of faces (tuples of 3 vertices) that must be included in the basis
    all_faces : list
        Precomputed list of all faces
    M_np : np.ndarray
        Precomputed matrix
    rank : int
        Precomputed rank of the matrix
    n_cols : int
        Precomputed number of columns
    max_attempts : int
        Maximum number of attempts to find a valid basis
    
    Returns:
    --------
    list
        A basis (list of column indices) that includes all required faces
    
    Raises:
    -------
    ValueError
        If the required faces cannot be included in a valid basis or if faces are invalid
    """
    # Map required faces to column indices
    required_indices = set()
    for face in required_faces:
        if face not in all_faces:
            raise ValueError(f"Face {face} is not a valid face for n={n}")
        col_idx = all_faces.index(face)
        required_indices.add(col_idx)
    
    # Check if required faces are too many
    if len(required_indices) > rank:
        raise ValueError(
            f"Cannot include {len(required_indices)} faces in a basis of rank {rank}"
        )
    
    # Get available columns (excluding required ones)
    all_columns = set(range(n_cols))
    available = list(all_columns - required_indices)
    
    # Try to find a valid basis
    for attempt in range(max_attempts):
        # Select additional columns from available columns
        needed = rank - len(required_indices)
        additional = random.sample(available, needed)
        base = sorted(list(required_indices) + additional)
        
        # Check if this forms a valid basis (full rank)
        base_arr = M_np[:, base]
        if np.linalg.matrix_rank(base_arr) == rank:
            return base
    
    raise ValueError(
        f"Could not find a valid basis including faces {required_faces} "
        f"after {max_attempts} attempts"
    )


def stochastic_find_largest_avg_cycle_e_include(n, trials, required_faces):
    """
    Find the largest average cycle size with bases that include the specified faces.
    
    Parameters:
    -----------
    n : int
        Number of vertices
    trials : int
        Number of random bases to test
    required_faces : list
        List of faces (in readable mode: tuples of 3 vertices) to include in each basis
    
    Returns:
    --------
    tuple
        (list of best bases, maximum average cycle size)
    """
    # Precompute matrix and metadata once
    all_faces = list(combinations(range(n), 3))
    print(f"n = {n}, creating matrix, including faces: {required_faces}")
    matrix = create_2d_simplex(n)
    M_np = np.array(matrix.tolist(), dtype=np.float64)
    rank = matrix.rank()
    n_cols = matrix.cols
    n_non_basic = n_cols - rank  # constant

    non_basic_of = lambda base_set: [c for c in range(n_cols) if c not in base_set]

    best_bases = set()
    seen_scores = set()
    max_score = -1
    i = 0
    j = 0
    total_searching_tries = 0

    while i < trials:
        try:
            base = get_random_basis_with_faces(n, required_faces, all_faces, M_np, rank, n_cols, max_attempts=100)
        except ValueError:
            j += 1
            continue
        
        i += 1
        total_searching_tries += j
        j = 0
        
        base_arr = M_np[:, base]

        nb = non_basic_of(set(base))
        G = M_np[:, nb]
        Q, R = np.linalg.qr(base_arr)
        A = np.linalg.solve(R, Q.T @ G)
        score = int(np.sum(np.abs(A) > 1e-9))  # integer, exact

        if score not in seen_scores:
            avg = (score + n_non_basic) / n_non_basic
            print(f"trial {i}: new score {score} → avg {avg:.4f}")
            seen_scores.add(score)

        key = tuple(sorted(base))
        if score > max_score:
            max_score = score
            best_bases = {key}
        elif score == max_score:
            best_bases.add(key)

        if i % 100000 == 0:
            print(f"passed trial {i}")

    max_avg = (max_score + n_non_basic) / n_non_basic
    readable = [[all_faces[c] for c in b] for b in best_bases]
    print(f"\n{len(seen_scores)} unique scores found")
    print(f"best: {len(best_bases)} bases, avg cycle size {max_avg:.4f}")
    for b in readable:
        print(f"Base: {b}")
    print(f"average searching tries per successful trial: {total_searching_tries / i:.2f}")
    return list(best_bases), max_avg


def stochastic_find_largest_avg_cycle_size_e(n, trials):
    faces = list(combinations(range(n), 3))
    print(f"n = {n}, creating matrix")
    matrix = create_2d_simplex(n)
    M_np = np.array(matrix.tolist(), dtype=np.float64)
    rank = matrix.rank()
    n_cols = matrix.cols
    n_non_basic = n_cols - rank  # constant

    remaining = list(range(1, n_cols))
    non_basic_of = lambda base_set: [c for c in range(n_cols) if c not in base_set]

    best_bases = set()
    seen_scores = set()
    max_score = -1
    i = 0
    j = 0
    total_searching_tries = 0

    while i < trials:
        base = [0] + random.sample(remaining, rank - 1)
        base_arr = M_np[:, base]

        if np.linalg.matrix_rank(base_arr) < rank:
            j+= 1
            continue
        i += 1
        total_searching_tries += j
        j = 0
        

        nb = non_basic_of(set(base))
        G = M_np[:, nb]
        Q, R = np.linalg.qr(base_arr)
        A = np.linalg.solve(R, Q.T @ G)
        score = int(np.sum(np.abs(A) > 1e-9))  # integer, exact

        if score not in seen_scores:
            avg = (score + n_non_basic) / n_non_basic
            print(f"trial {i}: new score {score} → avg {avg:.4f}")
            seen_scores.add(score)

        key = tuple(sorted(base))
        if score > max_score:
            max_score = score
            best_bases = {key}
        elif score == max_score:
            best_bases.add(key)

        if i % 100000 == 0:
            print(f"passed trial {i}")

    max_avg = (max_score + n_non_basic) / n_non_basic
    readable = [[faces[c] for c in b] for b in best_bases]
    print(f"\n{len(seen_scores)} unique scores found")
    print(f"best: {len(best_bases)} bases, avg cycle size {max_avg:.4f}")
    for b in readable:
        print(f"Base: {b}")
    print(f"average searching tries per successful trial: {total_searching_tries / i:.2f}")
    return list(best_bases), max_avg


if __name__ == "__main__":
    n = 8
    trials = 100000
    print(f"n = {n}, trials = {trials}")
    start_time = time.time()
    # base = [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (1, 2, 4), (2, 3, 4)]
    # stochastic_find_largest_avg_cycle_e_include(n, trials, base)
    stochastic_find_largest_avg_cycle_size_e(n, trials)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    stochastic_find_largest_avg_cycle_size_e(n, trials)