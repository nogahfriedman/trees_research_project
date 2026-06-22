import itertools
from itertools import combinations  
import networkx as nx
from networkx.algorithms import isomorphism
import math
from hopefully_fast import compute_score, create_2d_boundary_sparse
from scipy.linalg import qr, solve_triangular
from scipy.sparse import lil_matrix, csc_matrix
import numpy as np
from collections import defaultdict, Counter

def calculate_degrees(n, base ) -> dict:
    """Calculate the degree of each edge in the base hypertree."""
    edges = combinations(range(n), 2)
    deg_dict = {edge: 0 for edge in edges}
    for face in base:
        for edge in combinations(face, 2):
            deg_dict[edge] += 1
    return deg_dict

def calculate_vertices_degrees(n, base)->dict:
    """calculate degree for each vertex in the base hypertree"""
    deg_dict = {i: 0 for i in range(n)}
    for face in base:
        for i in face:
            deg_dict[i] += 1
    return deg_dict


def calculate_hypergraph_automorphism_order(num_vertices, hyperedges):
    """
    מחשב את גודל חבורת האוטומורפיזמים של היפר-גרף אחיד
    על ידי המרה לגרף דו-צדדי צבוע.
    """
    B = nx.Graph()
    
    # 1. הוספת קודקודים של ההיפר-גרף (נצבע אותם ב-'vertex')
    for v in range(1, num_vertices + 1):
        B.add_node(f"v_{v}", bipartite=0, color="vertex")
        
    # 2. הוספת קודקודים עבור היפר-הקצוות (נצבע אותם ב-'edge')
    for idx, edge in enumerate(hyperedges):
        edge_node = f"e_{idx}"
        B.add_node(edge_node, bipartite=1, color="edge")
        
        # חיבור הקודקודים שמרכיבים את ההיפר-קצה הנוכחי
        for v in edge:
            B.add_edge(f"v_{v}", edge_node)
            
    # 3. הגדרת פונקציית התאמת צבעים (כדי שאוטומורפיזם לא יחליף בין קודקוד לקצה)
    node_match = isomorphism.categorical_node_match('color', 'vertex')
    
    # 4. מציאת כל האוטומורפיזמים של הגרף הדו-צדדי שמשמרים את הצבעים
    GM = isomorphism.GraphMatcher(B, B, node_match=node_match)
    
    # ספירת כמות המיפויים האיזומורפיים לעצמו
    automorphism_count = sum(1 for _ in GM.isomorphisms_iter())
    
    return automorphism_count

def calculate_isomorphism_class_size(num_vertices, hyperedges):
    """
    מחשב את גודל מחלקת האיזומורפיזם (כמות ההיפר-גרפים השונים שאיזומורפיים לו)
    """
    # 1. חישוב |Aut(H)| בעזרת הפונקציה הקודמת
    aut_order = calculate_hypergraph_automorphism_order(num_vertices, hyperedges)
    
    # 2. חישוב !n (סך כל התמורות האפשריות על הקודקודים)
    total_permutations = math.factorial(num_vertices)
    
    # 3. גודל המחלקה הוא !n חלקי |Aut(H)|
    class_size = total_permutations // aut_order
    
    return class_size


def create_cyclic_hypergraph(n, c):
    faces = list(combinations(range(n), 3))
    bases = []
    for i in range(len(faces)):
        if ((c*faces[i][0] + faces[i][1] + faces[i][2]) % n == 0):
            bases.append(i)
        if ((c*faces[i][1] + faces[i][0] + faces[i][2])% n == 0):
            bases.append(i)
        if ((c*faces[i][2] + faces[i][0] + faces[i][1]) % n == 0):
            bases.append(i)
    print(f"c={c} | base size={len(bases)}")
    return bases

def from_readable_to_indices(num_vertices, readable_base):
    faces = list(combinations(range(num_vertices), 3))
    face_to_index = {face: idx for idx, face in enumerate(faces)}
    return [face_to_index[tuple(sorted(face))] for face in readable_base]

def from_indices_to_readable(num_vertices, indices):
    faces = list(combinations(range(num_vertices), 3))
    return [faces[idx] for idx in indices]

def get_orbit_representative(face: tuple[int, int, int], n: int = 8) -> tuple[int, int, int]:
    """
    מחשבת את האינווריאנט הציקלי של משולש מודולו n.
    האינווריאנט הוא שלשת המרחקים הממוינת בין הקודקודים על פני המעגל.
    """
    i, j, k = sorted(face)
    
    # חישוב מרחקים מחזוריים (הפרשים מודולו n)
    d1 = (j - i) % n
    d2 = (k - j) % n
    d3 = (i - k) % n
    
    # לוקחים את המרחק המינימלי בכל כיוון מחזורי
    dist1 = min(d1, n - d1)
    dist2 = min(d2, n - d2)
    dist3 = min(d3, n - d3)
    
    # החזרת השלשה הממוינת שמייצגת את המסלול
    return tuple(sorted([dist1, dist2, dist3]))

def analyze_base_orbits(n: int = 8, basis_indices: list[int] = None):
    """
    ממפה את כל הפאות למסלולים, ומנתחת בסיס נתון.
    """
    # 1. יצירת רשימת הפאות המלאה (זהה ל-create_2d_boundary_sparse בקוד שלך)
    all_faces = list(combinations(range(n), 3))
    
    # 2. חלוקת כל 56 הפאות למסלולים מודולו n
    orbit_groups = defaultdict(list)
    face_to_orbit = {}
    
    for idx, face in enumerate(all_faces):
        rep = get_orbit_representative(face, n)
        orbit_groups[rep].append(idx)
        face_to_orbit[idx] = rep
        
    # מיון המסלולים כדי לתת להם שמות קבועים (מסלול 1 עד 7)
    sorted_orbits = sorted(orbit_groups.keys())
    orbit_to_name = {rep: f"Orbit {i+1} {rep}" for i, rep in enumerate(sorted_orbits)}
    
    print("=" * 50)
    print(f"ORBIT DECOMPOSITION FOR K_{n} (Total {len(all_faces)} faces):")
    print("=" * 50)
    for rep in sorted_orbits:
        name = orbit_to_name[rep]
        size = len(orbit_groups[rep])
        print(f" - {name}: Contains {size} faces")
    print("=" * 50)
    
    # 3. ניתוח הבסיס הספציפי אם הועבר
    if basis_indices is not None:
        print(f"\nANALYSIS OF THE GIVEN BASIS (Size {len(basis_indices)}):")
        print("-" * 50)
        
        basis_orbit_counts = defaultdict(int)
        for face_idx in basis_indices:
            rep = face_to_orbit[face_idx]
            basis_orbit_counts[rep] += 1
            
        for rep in sorted_orbits:
            name = orbit_to_name[rep]
            count_in_basis = basis_orbit_counts[rep]
            total_in_orbit = len(orbit_groups[rep])
            percentage = (count_in_basis / total_in_orbit) * 100
            print(f" * {name:20} -> {count_in_basis}/{total_in_orbit} faces chosen ({percentage:.1f}%)")
        print("-" * 50)


def is_isomorphic(base1: list[tuple[int, int, int]], base2: list[tuple[int, int, int]], n: int) -> bool:
    """
    Determines if two bases of a 2-simplicial complex are structurally isomorphic 
    under any vertex permutation in S_n.
    
    Parameters:
    ----------
    base1, base2 : list of tuple[int, int, int]
        The two bases containing face triplets. Vertices are 0-indexed integers.
    n : int
        The total number of vertices in the complex (K_n).
        
    Returns:
    -------
    bool
        True if an isomorphic vertex mapping exists, False otherwise.
    """
    # 1. Quick Structural Invariant Check: Vertex Degree Distributions
    # Count how many faces each vertex participates in for both bases
    deg1 = Counter(v for face in base1 for v in face)
    deg2 = Counter(v for face in base2 for v in face)
    
    # Pad missing vertices that might have a degree of 0
    deg1_seq = sorted([deg1[v] for v in range(n)])
    deg2_seq = sorted([deg2[v] for v in range(n)])
    
    if deg1_seq != deg2_seq:
        return False

    # 2. Group vertices by degree to prune the permutation search space
    # We only map a vertex in base1 to a vertex in base2 if they have the exact same degree
    v1_by_deg = {}
    v2_by_deg = {}
    for v in range(n):
        d1, d2 = deg1[v], deg2[v]
        v1_by_deg.setdefault(d1, []).append(v)
        v2_by_deg.setdefault(d2, []).append(v)
        
    # Canonicalize and sort base2 for O(log K) lookups via set casting
    set_base2 = {tuple(sorted(face)) for face in base2}
    
    # Extract structural categories sorted by rarest degrees first to fail-fast
    sorted_degrees = sorted(v1_by_deg.keys(), key=lambda d: len(v1_by_deg[d]))
    
    # Generate valid matching permutations per degree partition block
    block_permutations = []
    for deg in sorted_degrees:
        target_v1 = v1_by_deg[deg]
        target_v2 = v2_by_deg[deg]
        # Generate all ways to map target_v1 onto target_v2
        perms = [dict(zip(target_v1, p)) for p in itertools.permutations(target_v2)]
        block_permutations.append(perms)
        
    # 3. Evaluate candidate mappings
    # Product combining the permutations of each degree block into a global mapping dictionary
    for block_comb in itertools.product(*block_permutations):
        # Merge local block dictionaries into a unified global vertex mapping
        mapping = {}
        for d_map in block_comb:
            mapping.update(d_map)
            
        # Transform base1 faces using the current candidate permutation map
        mapped_base1 = set()
        for face in base1:
            mapped_face = tuple(sorted((mapping[face[0]], mapping[face[1]], mapping[face[2]])))
            mapped_base1.add(mapped_face)
            
        # If the transformed set perfectly matches base2, an isomorphism is proven
        if mapped_base1 == set_base2:
            return True
            
    return False

if __name__ == "__main__":
    """
    num_vertices = 8
    base_proposal_1 = [
    # טיפוס א (8 משולשים)
    (0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7), (5, 6, 0), (6, 7, 1), (7, 0, 2),
    # טיפוס ב (8 משולשים)
    (0, 2, 5), (1, 3, 6), (2, 4, 7), (3, 5, 0), (4, 6, 1), (5, 7, 2), (6, 0, 3), (7, 1, 4),
    # משולשי השלמה (5 משולשים)
    (0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 4, 7), (4, 5, 0)
    ]
    M,_,faces = create_2d_boundary_sparse(num_vertices)
    basis_list = []
    for index, face in enumerate(faces):
        if face in base_proposal_1:
            basis_list.append(index)
    basis_set = set(basis_list)
    all_cols = set(range(len(faces)))
    non_basis  = sorted(all_cols - basis_set)

    B = M[:, basis_list].copy()
    if np.linalg.matrix_rank(B) < len(basis_list):
        print("Warning: proposed base is not full rank!")
    else:
        Q, R = qr(B, mode='economic')
        G = M[:, non_basis]
        score = compute_score(Q, R, G)
        avg_score = 1+(score / len(non_basis))
    
        print(f"Average score of non-basis columns: {avg_score:.4f}")

        

        
    
        iso_class_size = calculate_isomorphism_class_size(num_vertices, base_proposal_1)
        print(f"Size of isomorphism class: {iso_class_size}")
        degrees = calculate_degrees(num_vertices, base_proposal_1)
        print("Degrees of edges in the base hypertree:")    
        for edge, degree in degrees.items():
            print(f"Edge {edge}: Degree {degree}")
        """
    """
    base = [(0, 1, 3), (0, 1, 6), (0, 2, 4), (0, 2, 7), (0, 3, 5), (0, 4, 5), (0, 5, 6), (0, 6, 7), (1, 2, 4), (1, 2, 5), (1, 3, 7), (1, 4, 6), (1, 5, 7), (2, 3, 6), (2, 3, 7), (2, 5, 6), (2, 5, 7), (3, 4, 5), (3, 4, 6), (4, 5, 7), (4, 6, 7)]
    indexed_base = from_readable_to_indices(8, base)
    print(f"Indexed base: {indexed_base}")
    """
    """
    n= 8
    indexed_base_2 = [1, 4, 7, 10, 12, 15, 18, 20, 22, 23, 29, 31, 34, 38, 39, 43, 44, 46, 47, 53, 54]
    indexed_base_1 = [1, 2, 8, 10, 12, 14, 16, 20, 22, 23, 28, 32, 34, 35, 38, 39, 41, 46, 48, 52, 55]
    indexed_base_3 = [1, 5, 6, 8, 15, 16, 17, 18, 22, 23, 26, 28, 33, 35, 38, 41, 42, 44, 46, 50, 51]
    iso_1_2 = is_isomorphic(from_indices_to_readable(n, indexed_base_2), from_indices_to_readable(n, indexed_base_1), n)
    print(f"Are the two bases isomorphic? {iso_1_2}")
    iso_2_3 = is_isomorphic(from_indices_to_readable(n, indexed_base_2), from_indices_to_readable(n, indexed_base_3), n)
    print(f"Is the second base isomorphic to itself? {iso_2_3}")
    iso_1_3 = is_isomorphic(from_indices_to_readable(n, indexed_base_1), from_indices_to_readable(n, indexed_base_3), n)
    print(f"Is the first base isomorphic to itself? {iso_1_3}")
    
    readable_base_1 = from_indices_to_readable(n, indexed_base_1)
    print("base 1:")
    iso_class_size_1 = calculate_isomorphism_class_size(n, readable_base_1)
    print(f"Size of isomorphism class: {iso_class_size_1}")
    degrees = calculate_degrees(n, readable_base_1)
    print("Degrees of edges in the base hypertree:")    
    for edge, degree in degrees.items():
        print(f"Edge {edge}: Degree {degree}")
    v_degs = calculate_vertices_degrees(n, readable_base_1)
    print("Degree of vertices in the hypertree")
    for i,degree in v_degs.items():
        print(f"Vertex {i}: Degree {degree}")


    readable_base_2 = from_indices_to_readable(n, indexed_base_2)
    print("base 2:")
    iso_class_size_2 = calculate_isomorphism_class_size(n, readable_base_2)
    print(f"Size of isomorphism class: {iso_class_size_2}")
    degrees = calculate_degrees(n, readable_base_2)
    print("Degrees of edges in the base hypertree:")    
    for edge, degree in degrees.items():
        print(f"Edge {edge}: Degree {degree}")
    v_degs = calculate_vertices_degrees(n, readable_base_2)
    print("Degree of vertices in the hypertree")
    for i,degree in v_degs.items():
        print(f"Vertex {i}: Degree {degree}")

    
    
    readable_base_3 = from_indices_to_readable(n, indexed_base_3)
    print("base 3:")
    iso_class_size_3 = calculate_isomorphism_class_size(n, readable_base_3)
    print(f"Size of isomorphism class: {iso_class_size_3}")
    degrees = calculate_degrees(n, readable_base_3)
    print("Degrees of edges in the base hypertree:")    
    for edge, degree in degrees.items():
        print(f"Edge {edge}: Degree {degree}")
    v_degs = calculate_vertices_degrees(n, readable_base_3)
    print("Degree of vertices in the hypertree")
    for i,degree in v_degs.items():
        print(f"Vertex {i}: Degree {degree}")
    """
    n= 7
    best_base = [(0, 1, 3), (0, 1, 4), (0, 2, 5), (0, 2, 6), (0, 3, 5), (0, 4, 5), (0, 4, 6), (1, 2, 4), (1, 2, 5),
 (1, 2, 6), (1, 3, 6), (2, 3, 4), (2, 3, 5), (3, 4, 6), (3, 5, 6)]
    m = 5
    base =[(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (1, 2, 4), (2, 3, 4)] 
    degrees = calculate_degrees(m, base)
    for edge in degrees:
        print(f"{edge}: {degrees[edge]}")   
    v_deg = calculate_vertices_degrees(m, base)
    for i in range(m):
        print(f"{i} : {v_deg[i]}") 
    


    
