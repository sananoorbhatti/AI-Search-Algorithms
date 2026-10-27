"""Route Finding: BFS, DFS, UCS, A* and GBFS."""
import heapq
import math
import time
import tracemalloc
from collections import deque
import matplotlib.pyplot as plt

START, GOAL = "S", "G"
ALGORITHMS = ["BFS", "DFS", "UCS", "A*", "GBFS"]

# ----------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------
ORIGINAL_COORDS = {
    "S": (0, 0), "A": (2, 1), "B": (4, 1), "C": (6, 1),
    "D": (2, 4), "E": (4, 3), "F": (6, 4), "G": (8, 2),
}
ORIGINAL_ROADS = [
    ("S", "A", 3), ("A", "B", 2), ("B", "C", 3), ("B", "E", 3),
    ("B", "F", 4), ("D", "E", 3), ("E", "F", 3), ("F", "G", 3),
    ("S", "D", 7), ("A", "E", 6), ("E", "G", 6), ("C", "G", 6),
    ("D", "F", 6),
]
# The larger graph adds four cities (H, I, J, K) and nine roads.
LARGER_COORDS = {**ORIGINAL_COORDS,
                 "H": (1, 6), "I": (4, 6), "J": (7, 6), "K": (9, 5)}
LARGER_ROADS = ORIGINAL_ROADS + [
    ("D", "H", 3), ("A", "H", 7), ("H", "I", 4), ("I", "E", 5),
    ("I", "J", 4), ("J", "F", 3), ("J", "K", 3), ("K", "G", 5),
    ("K", "F", 5),
]


def build_graph(roads):
    graph = {}
    for a, b, cost in roads:
        graph.setdefault(a, {})[b] = cost
        graph.setdefault(b, {})[a] = cost
    return graph


ORIGINAL_GRAPH = build_graph(ORIGINAL_ROADS)
LARGER_GRAPH = build_graph(LARGER_ROADS)


def make_heuristic(coords, scale=1.0, goal=GOAL):
    """Euclidean distance to the goal, multiplied by scale. A scale above 1
    can overestimate, so admissibility is checked in admissibility_check()."""
    gx, gy = coords[goal]
    return lambda n: scale * math.hypot(coords[n][0] - gx, coords[n][1] - gy)


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------
def reconstruct(parent, goal):
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def search(graph, start, goal, algorithm, h=None, record=True):
    """Returns the result dictionary. record=False skips the trace rows."""
    h = h or (lambda n: 0)
    parent, g = {start: None}, {start: 0}
    expanded, rows, peak = [], [], 1

    def add_row(node):
        if record:
            rows.append((len(expanded), node, show()))

    # ---------------- BFS and DFS ----------------
    if algorithm in ("BFS", "DFS"):
        frontier = deque([start]) if algorithm == "BFS" else [start]
        show = lambda: ", ".join(frontier if algorithm == "BFS"
                                 else reversed(frontier)) or "Empty"
        if record:
            rows.append((0, "-", show()))
        while frontier:
            node = frontier.popleft() if algorithm == "BFS" else frontier.pop()
            if node == goal:
                if record:
                    rows.append((len(expanded) + 1, f"{node} (goal)",
                                 "Search stops"))
                break
            expanded.append(node)
            nbrs = sorted(graph[node])
            if algorithm == "DFS":
                nbrs.reverse()               # alphabetical city ends on top
            for n in nbrs:
                if n not in parent:          # not expanded, not in frontier
                    parent[n] = node
                    g[n] = g[node] + graph[node][n]
                    frontier.append(n)
            peak = max(peak, len(frontier))
            add_row(node)

    # ---------------- UCS, A* and GBFS ----------------
    else:
        def priority(cost, n):
            return (cost if algorithm == "UCS" else
                    cost + h(n) if algorithm == "A*" else h(n))

        counter = 0                          # ties: first added wins
        in_frontier = {start: priority(0, start)}
        heap = [(in_frontier[start], counter, start)]
        explored = set()
        fmt = (lambda v: f"{v:g}") if algorithm == "UCS" else (lambda v: f"{v:.2f}")

        def show():
            live = sorted((f, c, n) for f, c, n in heap
                          if n in in_frontier and in_frontier[n] == f)
            return ", ".join(f"{n}({fmt(f)})" for f, _, n in live) or "Empty"

        if record:
            rows.append((0, "-", show()))
        while in_frontier:
            f, _, node = heapq.heappop(heap)
            if node not in in_frontier or in_frontier[node] != f:
                continue                     # outdated entry
            del in_frontier[node]
            if node == goal:
                if record:
                    rows.append((len(expanded) + 1, f"{node} (goal)",
                                 "Search stops"))
                break
            explored.add(node)
            expanded.append(node)
            for n in sorted(graph[node]):
                new_g = g[node] + graph[node][n]
                if n in explored:
                    # Only A* reopens an expanded city, and only when a cheaper
                    # route is found. This matters for a non-admissible or
                    # inconsistent heuristic. UCS never needs it.
                    if algorithm == "A*" and new_g < g[n]:
                        explored.discard(n)
                    else:
                        continue
                better = algorithm != "GBFS" and n in g and new_g < g[n]
                if n not in in_frontier or better:
                    g[n], parent[n] = new_g, node
                    counter += 1
                    in_frontier[n] = priority(new_g, n)
                    heapq.heappush(heap, (in_frontier[n], counter, n))
            peak = max(peak, len(in_frontier))
            add_row(node)

    path = reconstruct(parent, goal) if goal in parent else None
    cost = sum(graph[a][b] for a, b in zip(path, path[1:])) if path else math.inf
    return {"algorithm": algorithm, "path": path, "cost": cost,
            "expanded": len(expanded), "expanded_order": expanded,
            "rows": rows, "peak_frontier": peak}


def run(graph, algorithm, h=None, runs=500):
    """Search once for the trace, then measure time and memory separately."""
    result = search(graph, START, GOAL, algorithm, h, record=True)
    t = time.perf_counter()
    for _ in range(runs):
        search(graph, START, GOAL, algorithm, h, record=False)
    result["time_ms"] = (time.perf_counter() - t) / runs * 1000
    tracemalloc.start()
    search(graph, START, GOAL, algorithm, h, record=False)
    result["memory_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    return result


# ----------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------
def path_text(r):
    return " -> ".join(r["path"]) if r["path"] else "No path found"


def guarantee(algorithm, admissible=True):
    if algorithm == "UCS":
        return "Yes (non-negative costs)"
    if algorithm == "A*" and admissible:
        return "Yes (admissible heuristic)"
    if algorithm == "BFS":
        return "Fewest roads only"
    return "No general guarantee"


def guaranteed(algorithm, admissible=True):
    """Theoretical optimality guarantee: Yes or No."""
    return "Yes" if algorithm == "UCS" or (algorithm == "A*" and admissible) else "No"


def true_costs(graph, goal=GOAL):
    """Cheapest cost from every city to the goal (Dijkstra, roads are two-way)."""
    best, heap = {goal: 0}, [(0, goal)]
    while heap:
        c, n = heapq.heappop(heap)
        if c > best[n]:
            continue
        for m, w in graph[n].items():
            if c + w < best.get(m, math.inf):
                best[m] = c + w
                heapq.heappush(heap, (c + w, m))
    return best


def admissibility_check(graph, coords, scale):
    """Prints h*(n), h(n) and the scaled heuristic. Returns True if admissible."""
    true, base, scaled = true_costs(graph), make_heuristic(coords), make_heuristic(coords, scale)
    over = [n for n in sorted(graph) if scaled(n) > true[n] + 1e-9]
    rows = [[n, true[n], f"{base(n):.2f}", f"{scaled(n):.2f}",
             "No" if n in over else "Yes"] for n in sorted(graph)]
    print_table(f"Admissibility check for h'(n) = {scale:g} x h(n)",
                ["City", "True cost to G h*(n)", "h(n)", "h'(n)",
                 "h'(n) <= h*(n)?"], rows)
    if over:
        print(f"h'(n) overestimates at {', '.join(over)}, so it is non-admissible.")
    else:
        print("h'(n) never overestimates, so it is still admissible.")
    return not over


def print_trace(r, title=None):
    print("\n" + "=" * 72 + f"\n{title or r['algorithm']}\n" + "=" * 72)
    print(f"{'Step':<6}{'Node Expanded':<16}Frontier")
    for step, node, frontier in r["rows"]:
        print(f"{step:<6}{node:<16}{frontier}")


def print_summary(r, best_cost, note):
    observed = "Yes" if math.isclose(r["cost"], best_cost) else "No"
    print(f"\nAlgorithm: {r['algorithm']}")
    print(f"Solution path: {path_text(r)}")
    print(f"Total path cost: {r['cost']}")
    print(f"Nodes expanded: {r['expanded']} ({', '.join(r['expanded_order'])})")
    print(f"Execution time (ms): {r['time_ms']:.4f}")
    print(f"Peak traced memory (KB): {r['memory_kb']:.3f}")
    print(f"Optimality guarantee: {note}")
    print(f"Observed least-cost path: {observed} (best cost is {best_cost})")


def print_table(title, headers, rows):
    print("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110)
    rows = [[str(x) for x in row] for row in rows]
    w = [max(len(headers[i]), *(len(r[i]) for r in rows))
         for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(w[i]) for i in range(len(headers))))
    print("-+-".join("-" * x for x in w))
    for r in rows:
        print(" | ".join(r[i].ljust(w[i]) for i in range(len(headers))))


def comparison_table(results, best_cost, title):
    rows = []
    for a in ALGORITHMS:
        r = results[a]
        rows.append([a, path_text(r), r["cost"], r["expanded"],
                     f"{r['time_ms']:.4f}", f"{r['memory_kb']:.3f}",
                     guaranteed(a)])
    print_table(title, ["Algorithm", "Solution Path", "Path Cost",
                        "Nodes Expanded", "Time (ms)", "Memory (KB)",
                        "Guaranteed Optimal?"], rows)


# ----------------------------------------------------------------------
# Pictures
# ----------------------------------------------------------------------
LABEL_POS = {("B", "F"): .33, ("E", "G"): .72, ("D", "F"): .3,
             ("I", "E"): .75, ("A", "H"): .75, ("S", "D"): .25,
             ("K", "F"): .5}


def draw_graph(graph, coords, filename, title="Larger graph"):
    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.set_xlim(-1, 13.8), ax.set_ylim(7.1, -.9), ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", loc="left")
    seen = set()
    for u in graph:
        for v, cost in graph[u].items():
            if (v, u) in seen:
                continue
            seen.add((u, v))
            (x1, y1), (x2, y2) = coords[u], coords[v]
            exp = cost >= 6
            f = LABEL_POS.get((u, v), LABEL_POS.get((v, u), .5))
            ax.plot([x1, x2], [y1, y2], color="#c0282d" if exp else "#222",
                    lw=2.2, ls=(0, (4, 3)) if exp else "-", zorder=1)
            ax.text(x1 + (x2 - x1) * f, y1 + (y2 - y1) * f, str(cost),
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="#c0282d" if exp else "#111", zorder=3,
                    bbox=dict(fc="white", ec="none", pad=1))
    for n, (x, y) in coords.items():
        fc, ec = "#cfe2f7", "#1f4e9c"
        if n == START:
            fc, ec = "#c2ecc9", "#1b7a34"
        if n == GOAL:
            fc, ec = "#f5b8b8", "#a91d22"
        ax.scatter([x], [y], s=3600, fc=fc, ec=ec, lw=2.2, zorder=4)
        ax.text(x, y - .1, n, ha="center", va="center", fontsize=17,
                fontweight="bold", zorder=5)
        ax.text(x, y + .25, f"({x}, {y})", ha="center", va="center",
                fontsize=8, zorder=5)
    # legend written as plain text on the right
    lx = 10.4
    ax.text(lx, 0.0, "Legend", fontsize=14, fontweight="bold", color="#0f2a5a")
    ax.plot([lx, lx + .6], [.7, .7], color="#222", lw=2.2, clip_on=False)
    ax.text(lx + .8, .7, "Cheap road (cost 2 to 5)", va="center", fontsize=10)
    ax.plot([lx, lx + .6], [1.3, 1.3], color="#c0282d", lw=2.2,
            ls=(0, (4, 3)), clip_on=False)
    ax.text(lx + .8, 1.3, "Expensive road (cost 6 or more)", va="center",
            fontsize=10)
    ax.scatter([lx + .3], [1.9], s=130, c="#1b7a34", clip_on=False)
    ax.text(lx + .8, 1.9, "S = Start", va="center", fontsize=10)
    ax.scatter([lx + .3], [2.5], s=130, c="#a91d22", clip_on=False)
    ax.text(lx + .8, 2.5, "G = Goal", va="center", fontsize=10)
    ax.text(lx, 3.3, "Each circle shows the city name\nand its (x, y) coordinates.\n"
            "The number on a road is its cost.", va="top", fontsize=9,
            linespacing=1.5)
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.02)
    fig.savefig(filename, dpi=200, facecolor="white")
    plt.show()
    plt.close(fig)


def plot_metrics(results, tag, suffix=""):
    specs = [("expanded", "Nodes Expanded", "nodes_expanded"),
             ("time_ms", "Execution Time (ms)", "execution_time"),
             ("memory_kb", "Peak Traced Memory (KB)", "memory_usage")]
    for key, label, name in specs:
        plt.figure(figsize=(8, 5))
        plt.bar(ALGORITHMS, [results[a][key] for a in ALGORITHMS])
        plt.title(f"{label}{suffix}")
        plt.xlabel("Search Algorithm")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(f"{tag}_{name}.png", dpi=200)
        plt.show()
        plt.close()


# ----------------------------------------------------------------------
# Running one graph
# ----------------------------------------------------------------------
def run_all(graph, coords, tag, suffix=""):
    h = make_heuristic(coords)
    results = {a: run(graph, a, h) for a in ALGORITHMS}
    best_cost = results["UCS"]["cost"]          # UCS is optimal
    for a in ALGORITHMS:
        print_trace(results[a])
        print_summary(results[a], best_cost, guarantee(a))
    comparison_table(results, best_cost,
                     f"Comparison of BFS, DFS, UCS, A* and GBFS{suffix}")
    plot_metrics(results, tag, suffix)
    return results, h, best_cost


def non_admissible_experiment(original, graph, coords, h, best_cost):
    admissible = admissibility_check(graph, coords, 2)
    h2 = lambda n: 2 * h(n)
    na = {a: run(graph, a, h2) for a in ("A*", "GBFS")}
    for a in ("A*", "GBFS"):
        print_trace(na[a], f"{a} with non-admissible heuristic h'(n) = 2h(n)")
        print_summary(na[a], best_cost,
                      "No general guarantee with h'(n) = 2h(n)")
    rows = []
    for a in ("A*", "GBFS"):
        for label, r, adm in (("Original h(n)", original[a], True),
                              ("Non-admissible h'(n)=2h(n)", na[a], admissible)):
            rows.append([a, label, path_text(r), r["cost"], r["expanded"],
                         f"{r['time_ms']:.4f}", f"{r['memory_kb']:.3f}",
                         guaranteed(a, adm)])
    print_table("Original vs non-admissible heuristic",
                ["Algorithm", "Heuristic", "Solution Path", "Path Cost",
                 "Nodes Expanded", "Time (ms)", "Memory (KB)",
                 "Guaranteed Optimal?"], rows)
    labels = ["A* original", "A* non-admissible",
              "GBFS original", "GBFS non-admissible"]
    paired = [original["A*"], na["A*"], original["GBFS"], na["GBFS"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (key, lab) in zip(axes, [("expanded", "Nodes Expanded"),
                                     ("time_ms", "Execution Time (ms)"),
                                     ("memory_kb", "Peak Traced Memory (KB)")]):
        ax.bar(labels, [r[key] for r in paired])
        ax.set_title(lab)
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Original vs Non-admissible Heuristic")
    fig.tight_layout()
    fig.savefig("non_admissible_heuristic_comparison.png", dpi=200)
    plt.show()
    plt.close(fig)


def main():
    original, h, best_cost = run_all(ORIGINAL_GRAPH, ORIGINAL_COORDS, "original")
    non_admissible_experiment(original, ORIGINAL_GRAPH, ORIGINAL_COORDS, h, best_cost)

    print(f"\nLarger graph: {len(LARGER_GRAPH)} cities, {len(LARGER_ROADS)} roads")
    draw_graph(LARGER_GRAPH, LARGER_COORDS, "larger_graph.png")
    run_all(LARGER_GRAPH, LARGER_COORDS, "larger", " (larger graph)")

    print("\nFinished. All graph pictures were saved in the current folder.")
    print("Time and memory values vary between runs and computers.")


if __name__ == "__main__":
    main()