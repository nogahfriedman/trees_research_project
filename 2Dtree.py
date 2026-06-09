import numpy as np
from itertools import combinations
from sympy import Matrix, Rational
import random
import math
import time
 
 
def find_all_column_bases(A: Matrix) -> list[tuple[int, ...]]:
    """
    Return all column index subsets that form a basis for the column space of A.
 
    Parameters
    ----------
    A : sympy.Matrix  (m x n, entries in Q)
 
    Returns
    -------
    List of tuples, each tuple is a 0-indexed set of column indices
    that form a basis.
    """
    r = A.rank()          # rank over Q (exact)
    n = A.cols
 
    bases = []
    for cols in combinations(range(n), r):
        submatrix = A.extract(list(range(A.rows)), list(cols))
        if submatrix.rank() == r:          # linearly independent  ⟺  full rank
            bases.append(cols)
 
    return bases

def fast_find_all_column_bases(A: Matrix) -> list[tuple[int, ...]]:
    r = A.rank()
    n = A.cols
    
    _, pivot_rows = A.T.rref()
    pivot_rows = list(pivot_rows[:r])
    
    A_np = np.array(A.extract(pivot_rows, list(range(n))).tolist(), dtype=np.int8)
    
    # Precompute: skip columns that are all-zero
    nonzero_cols = [c for c in range(n) if np.any(A_np[:, c])]
    
    bases = []
    for cols in combinations(nonzero_cols, r):  # only iterate nonzero cols
        if np.linalg.matrix_rank(A_np[:, list(cols)]) == r:
            bases.append(cols)
    
    return bases
 
 
def print_bases(A: Matrix) -> None:
    """Pretty-print the matrix, its rank, and every column basis found."""
    print("Matrix A:")
    print(A)
    print()
 
    r = A.rank()
    n = A.cols
    print(f"  rank = {r},  columns = {n}")
    print(f"  Checking C({n}, {r}) = {len(list(combinations(range(n), r)))} subsets...\n")
 
    bases = find_all_column_bases(A)
    print(f"Found {len(bases)} column basis/bases:\n")
 
    for i, cols in enumerate(bases, 1):
        sub = A.extract(list(range(A.rows)), list(cols))
        # 1-indexed column labels for human readability
        col_labels = tuple(c + 1 for c in cols)
        print(f"  Basis {i}: columns {col_labels}")
        print(f"    {sub.tolist()}")
        print()

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


def find_minimal_cycle_size(M, base: list[int], col: int) -> int:
    """
    Given indexes of a base and a column index, formed by adding the column to the base, find the size of the foundemental cycle."""
    # build basis matrix
    base_matrix = M.extract(list(range(M.rows)), list(base))
    #find a such that base_matrix * a = M[:,col] above Q
    a = base_matrix.LUsolve(M[:, col])
    # return number of nonzeroes in a plus one (for the column itself)
    cycle_size = sum(1 for x in a if x != 0) + 1
    for x in a:
        if x != 0 and x != Rational(1) and x != Rational(-1):
            print(f"nonzero entry in a: {x}")
    return cycle_size

def find_average_cycle_size(M, base: list[int])-> float:
    """
    Given indexes of a base, find the average size of the fundamental cycle formed by adding each non-base column to the base."""
    total_cycle_size = 0
    count = 0
    for col in range(M.cols):
        if col not in base:
            cycle_size = find_minimal_cycle_size(M, base, col)
            total_cycle_size += cycle_size
            count += 1
    if count != 0:
        average_cycle_size = total_cycle_size / count 
    else:
        average_cycle_size = 0   
    return average_cycle_size

def efficient_find_average_cycle_size(M, base: list[int]) -> float:
    non_basic = [col for col in range(M.cols) if col not in set(base)]
    if not non_basic:
        return 0.0

    base_matrix = M.extract(list(range(M.rows)), base)
    rhs = M.extract(list(range(M.rows)), non_basic)

    # Single LU factorization, all RHS at once
    A = base_matrix.LUsolve(rhs)

    # Count nonzeros per column using SymPy's applyfunc + columnwise sum
    nonzero_counts = [
        sum(1 for x in A.col(j) if x != 0)
        for j in range(A.cols)
    ]

    total = sum(nonzero_counts) + len(non_basic)  # +1 per non-basic col
    return total / len(non_basic)

def find_largest_avg_cycle_size(M)-> tuple[list[int], float]:
    """
    Find the largest average cycle size among all bases of M."""
    bases = fast_find_all_column_bases(M)
    print(f"searching best basesout of {len(bases)} bases")
    best_bases = []
    max_avg_cycle_size = 0
    for base in bases:
        avg_cycle_size = efficient_find_average_cycle_size(M, base)
        if avg_cycle_size == max_avg_cycle_size:
            best_bases.append(base)
        elif avg_cycle_size > max_avg_cycle_size:
            max_avg_cycle_size = avg_cycle_size
            best_bases = [base]
    return best_bases, max_avg_cycle_size


def is_base(M, cols: list[int]) -> bool:
    """
    Check if the given column indices form a basis for the column space of M."""
    submatrix = M.extract(list(range(M.rows)), cols)
    return submatrix.rank() == len(cols)  # linearly independent  ⟺  full rank


def broutforce_find_largest_avg_cycle_size(n: int)-> tuple[list[int], float]:
    """
    Brute-force search for the largest average cycle size among all bases of M."""
    faces = list(combinations(range(n), 3))
    print("building matrix")
    matrix = create_2d_simplex(n)
    print("finding bases")
    bases = fast_find_all_column_bases(matrix)
    print(f"searching best basesout of {len(bases)} bases")
    paths, avg_cycle_size= find_largest_avg_cycle_size(matrix)
    readable_paths = [[faces[col] for col in base] for base in paths]  # convert to 1-indexed for readability
    print(f"prints best bases {len(paths)} bases, with average cycle size {avg_cycle_size}")
    for base in readable_paths:
        print(f"Base: {base} \n")
    return paths, avg_cycle_size


def stochastic_find_largest_avg_cycle_size(n: int, trials: int) -> tuple[list[int], float]:
    """
    Stochastic search for the largest average cycle size among all bases of M."""
    faces = list(combinations(range(n), 3))
    print("building matrix")
    matrix = create_2d_simplex(n)
    print("finding bases")
    best_bases = []
    list_of_average_cycle_sizes = []
    max_avg_cycle_size = 0
    i= 0
    while i < trials:
        rank = matrix.rank()
        base = random.sample(range(matrix.cols), rank)  # random sample of column indices of size equal to the rank
        if is_base(matrix, base):  
            i += 1  # only count valid bases towards the trial limit 
            avg_cycle_size = efficient_find_average_cycle_size(matrix, base)
            if avg_cycle_size not in list_of_average_cycle_sizes:
                print(f"trial {i}: found new average cycle size {avg_cycle_size} with base {base}")
                list_of_average_cycle_sizes.append(avg_cycle_size)
            if avg_cycle_size == max_avg_cycle_size:
                if base not in best_bases:
                    best_bases.append(base)
            elif avg_cycle_size > max_avg_cycle_size:
                max_avg_cycle_size = avg_cycle_size
                best_bases = [base]
    readable_paths = [[faces[col] for col in base] for base in best_bases]  # convert to 1-indexed for readability
    print(f"prints best bases {len(best_bases)} bases, with average cycle size {max_avg_cycle_size}")
    for base in readable_paths:
        print(f"Base: {base} \n")
    return best_bases, max_avg_cycle_size

# some base for n=6
# base = [0, 3, 4, 7, 9, 11,13,14,17,18]
# base_1 = [0,3,4,7,9,10,12,15,16,19]
# base_2 = [0,3,4,7,9,10,11,15,17,19]
# base_3 = [1,2,4,6,7,11,12,15,17,19]

def efficient_find_path(n:int )-> tuple[list[int], float]:
    faces = list(combinations(range(n), 3))
    print(f"n={n}, building matrix")
    matrix = create_2d_simplex(n)
    print("finding bases")
    bases = fast_find_all_column_bases(matrix) 
    best_bases = [] 
    max_avg_cycle_size = 0
    l = math.comb(n-1, 2)* len(bases)//math.comb(n, 3) # expected average cycle size for a random base, used as a threshold to skip bases with low average cycle size
    print(f"searching best basesout of {l} bases")
    for base in bases[:l]:
        avg_cycle_size = efficient_find_average_cycle_size(matrix, base)
        if avg_cycle_size == max_avg_cycle_size:
            best_bases.append(base)
        elif avg_cycle_size > max_avg_cycle_size:
            max_avg_cycle_size = avg_cycle_size
            readable_base = [faces[col] for col in base]  # convert to 1-indexed for readability
            print(f"found new best average cycle size {max_avg_cycle_size} with base {readable_base}")
            best_bases = [base]
    print(f"prints best bases {len(best_bases)} bases, with average cycle size {max_avg_cycle_size}")
    for base in best_bases:
        readable_base = [faces[col] for col in base]  # convert to 1-indexed for readability
        print(f"Base: {readable_base} \n")  



def stochastic_find_largest_avg_cycle_size_e(n: int, trials: int) -> tuple[list[int], float]:
    faces = list(combinations(range(n), 3))
    
    print("building matrix")
    matrix = create_2d_simplex(n)
    M_np = np.array(matrix.tolist(), dtype=np.float64)
    
    print("computing rank")
    rank = matrix.rank()
    n_cols = matrix.cols
    
    # Always include col 0, sample rest from cols 1..n_cols-1
    remaining = list(range(1, n_cols))
    
    best_bases = []
    seen_avg_sizes = set()
    max_avg_cycle_size = 0.0
    i = 0
    
    print("searching bases")
    while i < trials:
        base = [0] + random.sample(remaining, rank - 1)
        
        # Check rank over numpy (fast) instead of is_base over sympy (slow)
        if np.linalg.matrix_rank(M_np[:, base]) == rank:
            i += 1
            avg_cycle_size = efficient_find_average_cycle_size_np(M_np, base, n_cols)
            
            if avg_cycle_size not in seen_avg_sizes:
                print(f"trial {i}: new avg cycle size {avg_cycle_size:.4f}")
                seen_avg_sizes.add(avg_cycle_size)
            
            if avg_cycle_size > max_avg_cycle_size:
                max_avg_cycle_size = avg_cycle_size
                best_bases = [base]
            elif avg_cycle_size == max_avg_cycle_size:
                if base not in best_bases:
                    best_bases.append(base)
            
            if i % 100000 == 0:
                print(f"passed trial {i}")

    readable_paths = [[faces[col] for col in base] for base in best_bases]
    print(f"{len(seen_avg_sizes)} unique average cycle sizes found during search.")
    print(f"\nbest: {len(best_bases)} bases, avg cycle size {max_avg_cycle_size:.4f}")
    for base in readable_paths:
        print(f"Base: {base}")
    
    return best_bases, max_avg_cycle_size



def efficient_find_average_cycle_size_np(M_np: np.ndarray, base: list[int], n_cols: int) -> float:
    non_basic = [col for col in range(n_cols) if col not in set(base)]
    if not non_basic:
        return 0.0

    base_matrix = M_np[:, base]
    rhs = M_np[:, non_basic]

    A, _, _, _ = np.linalg.lstsq(base_matrix, rhs, rcond=None)
    nonzero_counts = np.sum(np.abs(A) > 1e-9, axis=0)

    return (int(np.sum(nonzero_counts)) + len(non_basic)) / len(non_basic)




def readable_base_to_base(readable_base: list[tuple[int, int, int]], faces: list[tuple[int, int, int]]) -> list[int]:
    return [faces.index(face) for face in readable_base]    
    

if __name__ == "__main__":
    n = 8
    stochastic_find_largest_avg_cycle_size_e(n, 1000000)
    


    
    
