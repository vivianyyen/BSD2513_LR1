import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from PIL import Image


# ---------- BFS Algorithm ----------
def breadth_first_search(
        graph: Dict[str, List[str]],
        start: str,
        goal: Optional[str] = None,
) -> Tuple[List[str], List[str], Dict[str, Optional[str]], List[Dict[str, Any]]]:
    """
    Breadth-First Search (BFS) - explores level by level.

    Returns:
    - path: list of nodes from start to goal (empty if no goal or not found)
    - expanded_order: nodes visited in order
    - parent: mapping node -> predecessor
    - trace: step-by-step snapshots
    """
    visited = set()
    queue = deque([start])
    parent: Dict[str, Optional[str]] = {start: None}
    expanded_order: List[str] = []
    trace: List[Dict[str, Any]] = []

    visited.add(start)
    goal_found = goal is None

    while queue:
        current = queue.popleft()
        expanded_order.append(current)

        # Snapshot for UI
        trace.append({
            "expanded": current,
            "queue": list(queue),
            "visited": list(visited),
        })

        if goal is not None and current == goal:
            goal_found = True
            break

        # Sort neighbors alphabetically for tie-breaking
        neighbors = sorted(graph.get(current, []))

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    # Reconstruct path
    path: List[str] = []
    if goal is not None and goal_found:
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()

    return path, expanded_order, parent, trace


# ---------- DFS Algorithm ----------
def depth_first_search(
        graph: Dict[str, List[str]],
        start: str,
        goal: Optional[str] = None,
) -> Tuple[List[str], List[str], Dict[str, Optional[str]], List[Dict[str, Any]]]:
    """
    Depth-First Search (DFS) - explores as deep as possible first.

    Returns:
    - path: list of nodes from start to goal (empty if no goal or not found)
    - expanded_order: nodes visited in order
    - parent: mapping node -> predecessor
    - trace: step-by-step snapshots
    """
    visited = set()
    stack = [start]
    parent: Dict[str, Optional[str]] = {start: None}
    expanded_order: List[str] = []
    trace: List[Dict[str, Any]] = []

    goal_found = goal is None

    while stack:
        current = stack.pop()

        if current in visited:
            continue

        visited.add(current)
        expanded_order.append(current)

        # Snapshot for UI
        trace.append({
            "expanded": current,
            "stack": list(stack),
            "visited": list(visited),
        })

        if goal is not None and current == goal:
            goal_found = True
            break

        # Sort neighbors alphabetically in reverse for stack
        neighbors = sorted(graph.get(current, []), reverse=True)

        for neighbor in neighbors:
            if neighbor not in visited:
                if neighbor not in parent:
                    parent[neighbor] = current
                stack.append(neighbor)

    # Reconstruct path
    path: List[str] = []
    if goal is not None and goal_found:
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()

    return path, expanded_order, parent, trace


# ---------- Parsing Helpers ----------
def parse_unweighted_edges(text: str, undirected: bool) -> Dict[str, List[str]]:
    """Parse edges for BFS/DFS (unweighted)."""
    graph: Dict[str, List[str]] = {}

    def ensure_node(node: str):
        if node not in graph:
            graph[node] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        parts = [p for p in line.replace(',', ' ').split() if p]
        if len(parts) != 2:
            raise ValueError(f"Invalid edge line: '{line}'. Expected: src dst")
        u, v = parts[0], parts[1]

        ensure_node(u)
        ensure_node(v)
        graph[u].append(v)
        if undirected:
            graph[v].append(u)

    return graph


def all_nodes(graph: Dict[str, List[str]]) -> List[str]:
    nodes = set(graph.keys())
    for u, nbrs in graph.items():
        nodes.update(nbrs)
    return sorted(nodes)


def to_graphviz(graph: Dict[str, List[str]], path: List[str], directed: bool = True) -> str:
    """Build DOT source for unweighted graph."""
    is_on_path = set()
    for i in range(len(path) - 1):
        is_on_path.add((path[i], path[i + 1]))
        if not directed:
            is_on_path.add((path[i + 1], path[i]))

    gtype = "digraph" if directed else "graph"
    arrow = "->" if directed else "--"

    lines = [
        f"{gtype} G {{",
        "  rankdir=LR;",
        "  node [shape=circle, fontsize=12];",
    ]

    nodes = all_nodes(graph)
    for n in nodes:
        lines.append(f'  "{n}";')

    for u, nbrs in graph.items():
        for v in nbrs:
            on_path = (u, v) in is_on_path
            color = "#d32f2f" if on_path else "#4285f4"
            penwidth = "3" if on_path else "1"
            lines.append(f'  "{u}" {arrow} "{v}" [color="{color}", penwidth={penwidth}];')

    lines.append("}")
    return "\n".join(lines)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Graph Traversal: BFS & DFS", page_icon="🔍", layout="wide")

st.title("🔍 Graph Traversal: BFS & DFS")
st.caption("Compare Breadth-First Search and Depth-First Search with alphabetical tie-breaking")

with st.sidebar:
    st.header("Configuration")

    # Algorithm selection
    algorithm = st.selectbox(
        "Search Algorithm",
        ["BFS (Breadth-First)", "DFS (Depth-First)"],
        index=0
    )

    st.divider()

    # Graph input
    st.subheader("Graph Input")
    input_mode = st.radio("Definition mode", ["Sample", "Custom"], index=0)
    undirected = st.checkbox("Treat edges as undirected", value=False)

    sample_edges = """# source,target
A,B
B,E
B,G
C,A
B,C
A,D
D,C
E,H
G,F
H,G
H,F"""
    help_text = "Format: src,dst (one per line)"

    if input_mode == "Sample":
        edge_text = sample_edges


        image_path = "Screenshot 2025-11-08 232832.png"

        try:
            img = Image.open(image_path)
            st.image(img, caption="Sample Graph", use_container_width=True)
        except:
            st.error("Image file not found. Please check the file path.")

    else:
        edge_text = st.text_area(
            "Edges",
            value=sample_edges,
            height=200,
            help=help_text,
        )

    parse_ok = True
    graph = {}
    nodes = []

    try:
        graph = parse_unweighted_edges(edge_text, undirected=undirected)
        nodes = all_nodes(graph)
    except Exception as e:
        parse_ok = False
        st.error(str(e))

if not parse_ok or not nodes:
    st.stop()

# Main controls
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    start_node = st.selectbox("Start Node", nodes, index=0 if nodes else None)
with col2:
    goal_node = st.selectbox("Goal Node (optional)", ["<none>"] + nodes,
                             index=(nodes.index("F") + 1) if "F" in nodes else 0)
with col3:
    run_btn = st.button(f"▶️ Run {algorithm.split()[0]}", type="primary", use_container_width=True)

st.divider()

# Execute algorithm
if run_btn and start_node:
    goal_value = None if goal_node == "<none>" else goal_node

    try:
        if algorithm == "BFS (Breadth-First)":
            path, expanded, parent, trace = breadth_first_search(graph, start_node, goal_value)

        else:  # DFS
            path, expanded, parent, trace = depth_first_search(graph, start_node, goal_value)

    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Display results
    left, right = st.columns([1, 1])

    with left:
        st.subheader("📊 Results")

        if goal_value is None:
            st.info("Exploring entire graph (no specific goal)")
        elif path:
            st.success(f"**Path:** {' → '.join(path)}")
        else:
            st.warning("No path found to the specified goal.")

        st.write(f"**Expansion order:** {' → '.join(expanded)}")

        # Results table
        rows = [
            {
                "Node": n,
                "Visited": "✓" if n in expanded else "-",
                "Parent": parent.get(n, "-")
            }
            for n in nodes
        ]

        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Step-by-step trace
        with st.expander("🔍 View step-by-step trace"):
            for i, snap in enumerate(trace, start=1):
                if algorithm == "BFS (Breadth-First)":
                    st.write(f"**Step {i}:** Expanded `{snap['expanded']}` | Queue: {snap['queue'][:5]}")
                else:  # DFS
                    st.write(f"**Step {i}:** Expanded `{snap['expanded']}` | Stack: {snap['stack'][:5]}")

    with right:
        st.subheader("🗺️ Graph Visualization")
        gv = to_graphviz(graph, path if path else [], directed=not undirected)
        st.graphviz_chart(gv, use_container_width=True)

else:
    st.info("👆 Select start node, optional goal, and click Run to execute the search algorithm")

    # Algorithm comparison table
    st.subheader("📚 Algorithm Comparison")

    comparison = {
        "Feature": ["Data Structure", "Exploration", "Optimal (Unweighted)", "Complete", "Memory Usage",
                    "Best Use Case"],
        "BFS": ["Queue (FIFO)", "Level by level", "Yes", "Yes", "High", "Shortest path"],
        "DFS": ["Stack (LIFO)", "Depth first", "No", "No (with cycles)", "Low", "Path existence, Topological sort"]
    }

    st.table(comparison)
