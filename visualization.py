import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

# רשימת הפאות של הבסיס האופטימלי שמצאת עבור n=7
"""
best_base = [
    (0, 1, 3), (0, 1, 4), (0, 2, 5), (0, 2, 6), (0, 3, 5), 
    (0, 4, 5), (0, 4, 6), (1, 2, 4), (1, 2, 5), (1, 2, 6), 
    (1, 3, 6), (2, 3, 4), (2, 3, 5), (3, 4, 6), (3, 5, 6)
]
"""
best_base = [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5), (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5)]

# בניית הגרף הדואלי
G = nx.Graph()

# הוספת הצמתים (הפאות)
for face in best_base:
    G.add_node(face)

# חיבור קשת אם שתי פאות חולקות בדיוק 2 קודקודים משותפים (צלע)
for f1, f2 in combinations(best_base, 2):
    shared_vertices = set(f1).intersection(set(f2))
    if len(shared_vertices) == 2:
        G.add_edge(f1, f2)

# עיצוב הציור
plt.figure(figsize=(12, 10))
pos = nx.kamada_kawai_layout(G)  # אלגוריתם פיזיקלי שמסדר את הצמתים לפי כוחות מתיחה

# ציור הצמתים והקשתות
nx.draw_networkx_nodes(G, pos, node_size=900, node_color='darkcyan', edgecolors='black')
nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color='gray')

# הוספת תוויות (השמות של השלשות)
labels = {node: f"{node[0]},{node[1]},{node[2]}" for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='white', font_weight='bold')

plt.title("Dual Graph Connectivity of the Optimal Basis (n=7)", fontsize=14, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.show()

# הדפסת מדדי מרכזיות בסיסיים
print("--- מדדי קישוריות של הבסיס ---")
print(f"מספר פאות (צמתים בגרף הדואלי): {G.number_of_nodes()}")
print(f"מספר השקות של צלעות (קשתות בגרף הדואלי): {G.number_of_edges()}")
print(f"דרגה ממוצעת של פאה (כמה שכנים יש לכל משולש): {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")