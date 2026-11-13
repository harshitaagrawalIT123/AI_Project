import random
from collections import deque

MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]   # up, down, left, right


class Grid:
    def __init__(self, size, wall_density, safe_size, seed=None):
        self.size = size
        self.rng = random.Random(seed)
        self.safe_cells = {(x, y) for x in range(size - safe_size, size)
                           for y in range(size - safe_size, size)}
        self.human_start = (0, 0)
        self._dist_cache = {}
        while True:
            self.walls = set()
            for x in range(size):
                for y in range(size):
                    c = (x, y)
                    if c == self.human_start or c in self.safe_cells:
                        continue
                    if self.rng.random() < wall_density:
                        self.walls.add(c)
            self._dist_cache.clear()
            if min((self.bfs_dist(self.human_start).get(s, 10**9)
                    for s in self.safe_cells)) < 10**9:
                break   # guarantee the map is solvable

    def in_bounds(self, c):
        return 0 <= c[0] < self.size and 0 <= c[1] < self.size

    def walkable(self, c):
        return self.in_bounds(c) and c not in self.walls

    def neighbors(self, c):
        for dx, dy in MOVES:
            n = (c[0] + dx, c[1] + dy)
            if self.walkable(n):
                yield n

    def bfs_dist(self, source):
        """Shortest-path distance from source to every reachable cell (cached)."""
        if source in self._dist_cache:
            return self._dist_cache[source]
        dist = {source: 0}
        q = deque([source])
        while q:
            cur = q.popleft()
            for n in self.neighbors(cur):
                if n not in dist:
                    dist[n] = dist[cur] + 1
                    q.append(n)
        self._dist_cache[source] = dist
        return dist

    def dist(self, a, b):
        return self.bfs_dist(a).get(b, 10**6)

    @property
    def dist_to_safe(self):
        """BFS distance (in steps) from every cell to the nearest safe cell.

        Multi-source BFS from all safe cells, computed once and cached.
        Used as the (consistent) heuristic for the planners.
        """
        if not hasattr(self, "_dist_to_safe"):
            dist = {s: 0 for s in self.safe_cells}
            q = deque(self.safe_cells)
            while q:
                cur = q.popleft()
                for n in self.neighbors(cur):
                    if n not in dist:
                        dist[n] = dist[cur] + 1
                        q.append(n)
            self._dist_to_safe = dist
        return self._dist_to_safe

    def dist_safe(self, cell):
        return self.dist_to_safe.get(cell, 10**6)

    def random_zombie_spots(self, n, human):
        spots = []
        cells = [(x, y) for x in range(self.size) for y in range(self.size)
                 if self.walkable((x, y)) and (x, y) not in self.safe_cells
                 and self.dist(human, (x, y)) >= 6]
        self.rng.shuffle(cells)
        for c in cells:
            if len(spots) == n:
                break
            spots.append(c)
        return spots
