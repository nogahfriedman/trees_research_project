from itertools import combinations  
import networkx as nx
from networkx.algorithms import isomorphism
import math
from hopefully_fast import compute_score, create_2d_boundary_sparse
from scipy.linalg import qr, solve_triangular
from scipy.sparse import lil_matrix, csc_matrix
import numpy as np
from collections import defaultdict

def calculate_degrees(n, base ) -> dict:
    """Calculate the degree of each node in the base hypertree."""
    edges = combinations(range(n), 2)
    deg_dict = {edge: 0 for edge in edges}
    for face in base:
        for edge in combinations(face, 2):
            deg_dict[edge] += 1
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
    n= 8
    indexed_base = [4, 5, 7, 9, 12, 13, 15, 19, 21, 24, 26, 27, 32, 33, 39, 42, 44, 47, 51, 52, 55]
    redable_base = from_indices_to_readable(n, indexed_base)
    iso_class_size = calculate_isomorphism_class_size(n, redable_base)
    print(math.factorial(n))
    print(f"Size of isomorphism class: {iso_class_size}")
    analyze_base_orbits(n, indexed_base)