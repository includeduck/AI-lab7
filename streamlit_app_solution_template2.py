import math
import heapq

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Search Algorithms", layout="wide")

# -----------------------------------------------------------------------------
# Task 1 dataset
# -----------------------------------------------------------------------------
warehouse_locations = {
    "Receiving_Area": (0, 0),
    "Storage_A": (2, 1),
    "Storage_B": (1, 4),
    "Sorting_Area": (4, 2),
    "Inspection_Area": (5, 5),
    "Packing_Station": (7, 6),
}

warehouse_graph = {
    "Receiving_Area": {"Storage_A": 2.2, "Storage_B": 4.1},
    "Storage_A": {"Sorting_Area": 2.2},
    "Storage_B": {"Inspection_Area": 5.0, "Sorting_Area": 6.0},
    "Sorting_Area": {"Inspection_Area": 3.4, "Packing_Station": 5.6},
    "Inspection_Area": {"Packing_Station": 2.2},
    "Packing_Station": {},
}

# -----------------------------------------------------------------------------
# Task 2 dataset
# -----------------------------------------------------------------------------
airport_locations = {
    "Baggage_Area": (0, 0),
    "Security": (2, 1),
    "Checkpoint": (1, 4),
    "Food_Court": (4, 2),
    "Terminal_Hall": (6, 4),
    "Departure_Gate": (8, 5),
}

airport_graph = {
    "Baggage_Area": {"Security": 2.2, "Checkpoint": 4.1},
    "Security": {"Food_Court": 2.2},
    "Checkpoint": {"Terminal_Hall": 5.0},
    "Food_Court": {"Terminal_Hall": 3.2, "Departure_Gate": 6.0},
    "Terminal_Hall": {"Departure_Gate": 3.2},
    "Departure_Gate": {},
}

# -----------------------------------------------------------------------------
# Task 3 dataset
# -----------------------------------------------------------------------------
hospital_locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6),
}

hospital_graph = {
    "Pharmacy": {"Main_Corridor": 2.2, "Patient_Wing": 4.1},
    "Main_Corridor": {"Nursing_Station": 2.2},
    "Patient_Wing": {"Laboratory": 5.0},
    "Nursing_Station": {"Laboratory": 3.4, "Emergency_Ward": 5.0},
    "Laboratory": {"Emergency_Ward": 2.2},
    "Emergency_Ward": {},
}

# -----------------------------------------------------------------------------
# Task 4 dataset
# -----------------------------------------------------------------------------
drone_locations = {
    "Distribution_Center": (0, 0),
    "Zone_A": (2, 1),
    "Zone_B": (1, 4),
    "Zone_C": (4, 2),
    "Zone_D": (5, 5),
    "Customer_Building": (7, 6),
}

drone_graph = {
    "Distribution_Center": {"Zone_A": 2.2, "Zone_B": 3.0},
    "Zone_A": {"Zone_C": 2.2},
    "Zone_B": {"Zone_D": 5.0},
    "Zone_C": {"Customer_Building": 6.0},
    "Zone_D": {"Customer_Building": 3.2},
    "Customer_Building": {},
}


# -----------------------------------------------------------------------------
# Search helpers
# -----------------------------------------------------------------------------
def heuristic(current, goal, locations):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.hypot(x1 - x2, y1 - y2)


def greedy_best_first_search(start, goal, graph, locations):
    open_heap = [(heuristic(start, goal, locations), start)]
    came_from = {start: None}
    visited = set()
    expansion_order = []

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in visited:
            continue

        visited.add(current)
        expansion_order.append(current)

        if current == goal:
            break

        for neighbor, _ in graph[current].items():
            if neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(open_heap, (heuristic(neighbor, goal, locations), neighbor))

    if goal not in came_from:
        return expansion_order, None, None

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()

    total_cost = 0.0
    for i in range(len(path) - 1):
        total_cost += graph[path[i]][path[i + 1]]

    return expansion_order, path, total_cost


def a_star_search(start, goal, graph, locations):
    open_heap = [(heuristic(start, goal, locations), 0.0, start)]
    came_from = {start: None}
    g_cost = {start: 0.0}
    expansion_order = []
    visited = set()

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)

        if current in visited:
            continue

        visited.add(current)
        expansion_order.append(current)

        if current == goal:
            break

        for neighbor, edge_cost in graph[current].items():
            tentative_g = current_cost + edge_cost
            if tentative_g < g_cost.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g
                f_cost = tentative_g + heuristic(neighbor, goal, locations)
                heapq.heappush(open_heap, (f_cost, tentative_g, neighbor))

    if goal not in came_from:
        return expansion_order, None, None, {}

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()

    total_cost = g_cost[goal]
    return expansion_order, path, total_cost, g_cost


def weighted_a_star(start, goal, graph, locations, weight):
    open_heap = [(weight * heuristic(start, goal, locations), 0.0, start)]
    came_from = {start: None}
    g_cost = {start: 0.0}
    expansion_order = []
    visited = set()

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)

        if current in visited:
            continue

        visited.add(current)
        expansion_order.append(current)

        if current == goal:
            break

        for neighbor, edge_cost in graph[current].items():
            tentative_g = current_cost + edge_cost
            if tentative_g < g_cost.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g
                f_cost = tentative_g + weight * heuristic(neighbor, goal, locations)
                heapq.heappush(open_heap, (f_cost, tentative_g, neighbor))

    if goal not in came_from:
        return expansion_order, None, None

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return expansion_order, path, g_cost[goal]


def plot_search_graph(graph, locations, path=None, title="Search Path"):
    G = nx.DiGraph()
    for node, neighbors in graph.items():
        G.add_node(node, pos=locations[node])
        for neighbor, cost in neighbors.items():
            G.add_edge(node, neighbor, weight=cost)

    pos = locations
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, ax=ax)

    if path and len(path) > 1:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=3, arrows=True, ax=ax)

    ax.set_title(title)
    ax.axis("off")
    return fig


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("AI Search Algorithms Lab")
st.caption("Warehouse robot, airport baggage handling, emergency supply routing, and autonomous drone delivery.")

page = st.sidebar.selectbox(
    "Choose a task",
    [
        "Task 1: Warehouse Heuristic",
        "Task 2: GBFS Airport",
        "Task 3: A* Hospital",
        "Task 4: Weighted A* Drone",
    ],
)

if page == "Task 1: Warehouse Heuristic":
    st.header("Task 1: Designing a Heuristic for a Warehouse Robot")

    heuristic_rows = []
    for location in warehouse_locations:
        heuristic_rows.append({"Node": location, "h(n) to Packing_Station": round(heuristic(location, "Packing_Station", warehouse_locations), 2)})
    st.dataframe(pd.DataFrame(heuristic_rows), use_container_width=True)

    admissibility_rows = []
    for node in warehouse_graph:
        for neighbor, cost in warehouse_graph[node].items():
            h_node = heuristic(node, "Packing_Station", warehouse_locations)
            h_neighbor = heuristic(neighbor, "Packing_Station", warehouse_locations)
            valid = h_node <= cost + h_neighbor
            admissibility_rows.append(
                {
                    "From": node,
                    "To": neighbor,
                    "Cost": round(cost, 2),
                    "h(From)": round(h_node, 2),
                    "h(To)": round(h_neighbor, 2),
                    "Condition": "OK" if valid else "Violation",
                }
            )

    st.subheader("Admissibility Check")
    st.dataframe(pd.DataFrame(admissibility_rows), use_container_width=True)

    st.info(
        "Observation: The Euclidean heuristic is not admissible for this graph because it overestimates the true remaining cost on the edge Inspection_Area → Packing_Station."
    )

    source = "Receiving_Area"
    target = "Packing_Station"
    dijkstra_frontier = [(0.0, source)]
    distances = {source: 0.0}
    came_from = {source: None}

    while dijkstra_frontier:
        current_cost, current = heapq.heappop(dijkstra_frontier)

        if current_cost != distances[current]:
            continue

        if current == target:
            break

        for neighbor, edge_cost in warehouse_graph[current].items():
            new_cost = current_cost + edge_cost
            if new_cost < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(dijkstra_frontier, (new_cost, neighbor))

    path = []
    current = target
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()

    st.subheader("Minimum-Cost Path")
    st.write(" → ".join(path))
    st.write(f"Total cost: {distances[target]:.2f}")
    st.pyplot(plot_search_graph(warehouse_graph, warehouse_locations, path, "Warehouse Robot Graph"))

elif page == "Task 2: GBFS Airport":
    st.header("Task 2: Greedy Best-First Search for Airport Baggage Handling")

    start = st.selectbox("Select start node", list(airport_graph.keys()), index=0)
    goal = st.selectbox("Select goal node", list(airport_graph.keys()), index=5)

    if st.button("Run GBFS"):
        expansion_order, path, total_cost = greedy_best_first_search(start, goal, airport_graph, airport_locations)

        if path is None:
            st.error("No path found.")
        else:
            st.subheader("Search Result")
            st.write(f"**Expansion order:** {' → '.join(expansion_order)}")
            st.write(f"**Solution path:** {' → '.join(path)}")
            st.write(f"**Total path cost:** {total_cost:.2f}")
            st.pyplot(plot_search_graph(airport_graph, airport_locations, path, "Airport Baggage Handling - GBFS"))

elif page == "Task 3: A* Hospital":
    st.header("Task 3: A* Search for Emergency Supply Robot")

    start = st.selectbox("Select start node", list(hospital_graph.keys()), index=0)
    goal = st.selectbox("Select goal node", list(hospital_graph.keys()), index=5)

    if st.button("Run A*"):
        expansion_order, path, total_cost, g_cost = a_star_search(start, goal, hospital_graph, hospital_locations)

        if path is None:
            st.error("No path found.")
        else:
            info_rows = []
            for node in expansion_order:
                g_value = g_cost[node]
                h_value = heuristic(node, goal, hospital_locations)
                info_rows.append(
                    {
                        "Node": node,
                        "g(n)": round(g_value, 2),
                        "h(n)": round(h_value, 2),
                        "f(n)": round(g_value + h_value, 2),
                    }
                )

            st.subheader("Search Result")
            st.write(f"**Expansion order:** {' → '.join(expansion_order)}")
            st.write(f"**Solution path:** {' → '.join(path)}")
            st.write(f"**Total path cost:** {total_cost:.2f}")
            st.dataframe(pd.DataFrame(info_rows), use_container_width=True)
            st.pyplot(plot_search_graph(hospital_graph, hospital_locations, path, "Emergency Supply Robot - A* Search"))

else:
    st.header("Task 4: Weighted A* for Autonomous Delivery Drone")

    start = st.selectbox("Select start node", list(drone_graph.keys()), index=0)
    goal = st.selectbox("Select goal node", list(drone_graph.keys()), index=5)
    weight = st.slider("Select weighted heuristic factor w", 1.0, 3.0, 1.5, 0.5)

    if st.button("Run Weighted A*"):
        expansion_order, path, total_cost = weighted_a_star(start, goal, drone_graph, drone_locations, weight)

        if path is None:
            st.error("No path found.")
        else:
            st.subheader("Search Result")
            st.write(f"**Expansion order:** {' → '.join(expansion_order)}")
            st.write(f"**Solution path:** {' → '.join(path)}")
            st.write(f"**Total path cost:** {total_cost:.2f}")
            st.pyplot(plot_search_graph(drone_graph, drone_locations, path, f"Weighted A* Solution Path | w = {weight}"))

    st.subheader("Comparison for all required weights")
    comparison_rows = []
    for w in [1, 1.5, 2, 3]:
        _, path_for_w, cost_for_w = weighted_a_star(start, goal, drone_graph, drone_locations, w)
        expansion_order_for_w, _, _ = weighted_a_star(start, goal, drone_graph, drone_locations, w)
        comparison_rows.append(
            {
                "Weight": w,
                "Solution Path": " → ".join(path_for_w),
                "Total Cost": round(cost_for_w, 2),
                "Nodes Expanded": len(expansion_order_for_w),
            }
        )
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

    st.markdown(
        """
        Expected main results:
        - w = 1: Distribution_Center → Zone_A → Zone_C → Customer_Building | Cost = 10.4
        - w = 1.5: Distribution_Center → Zone_A → Zone_C → Customer_Building | Cost = 10.4
        - w = 2: Distribution_Center → Zone_B → Zone_D → Customer_Building | Cost = 11.2
        - w = 3: Distribution_Center → Zone_B → Zone_D → Customer_Building | Cost = 11.2
        """
    )