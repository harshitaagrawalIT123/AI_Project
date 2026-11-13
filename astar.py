"""A* pathfinding - fallback planner and used directly with --planner astar.

Step cost = 1 + a danger penalty for cells close to zombies, so the cheapest
path is both short AND safe. The Manhattan heuristic never overestimates
(every step costs >= 1), so A* stays optimal for this cost model.

Cells inside CAPTURE_RADIUS of a zombie are treated as nearly impassable:
stepping there means capture (Pursuit-Evasion with Obstacles, 2016).
"""
import heapq

import config

DEATH_PENALTY = 1000.0


def danger_penalty(grid, cell, zombies):
    p = 0
    for z in zombies:
        d = grid.dist(z, cell)          # obstacle-aware BFS distance
        if d <= config.CAPTURE_RADIUS:
            p += DEATH_PENALTY
        elif d == 1:
            p += 25
        elif d == 2:
            p += 8
        elif d == 3:
            p += 3
    return p


def in_danger(grid, cell, zombies):
    """True if stepping on `cell` would put the human within capture range."""
    return any(grid.dist(z, cell) <= config.CAPTURE_RADIUS for z in zombies)


def heuristic(a, goals):
    return min(abs(a[0] - g[0]) + abs(a[1] - g[1]) for g in goals)


def astar(grid, start, goals, zombies):
    """Return the path [start, ..., goal] or None if no path exists."""
    zset = set(zombies)
    open_heap = [(heuristic(start, goals), 0, start)]
    came_from = {}
    g_cost = {start: 0}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in goals:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]
        if g > g_cost.get(cur, 1e9):
            continue
        for n in grid.neighbors(cur):
            if n in zset:
                continue
            ng = g + 1 + danger_penalty(grid, n, zombies)
            if ng < g_cost.get(n, 1e9):
                g_cost[n] = ng
                came_from[n] = cur
                heapq.heappush(open_heap, (ng + heuristic(n, goals), ng, n))
    return None


def human_next_move(grid, human, zombies):
    """Pick the human's next cell. Returns (next_cell, planned_path)."""
    path = astar(grid, human, grid.safe_cells, zombies)
    if path and len(path) > 1:
        return path[1], path
    # No safe path: step to the neighbour farthest from the zombies (or stay).
    zset = set(zombies)
    options = [n for n in grid.neighbors(human)
               if n not in zset and not in_danger(grid, n, zombies)] + [human]
    if len(options) == 1 and options[0] == human:
        # every move is deadly - prefer any non-zombie neighbour anyway
        options = [n for n in grid.neighbors(human) if n not in zset] + [human]
    best = max(options, key=lambda c: min(grid.dist(c, z) for z in zombies))
    return best, [human, best]
