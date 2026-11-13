"""D* Lite incremental path planner (Koenig & Likhachev, 2002).

D* Lite is Lifelong Planning A*: it keeps g/rhs cost-to-go values for every
state between planning episodes and, when edge costs change (here: the risk
field and zombie positions change every turn), it only re-propagates the
states whose costs actually changed instead of searching from scratch like
ordinary A*.

Graph model
------------
States are grid cells plus one virtual GOAL state. Every walkable cell has
edges to its 4 neighbours; every safe-zone cell has an extra zero-cost edge
to GOAL (stepping into the safe zone ends the game, so the cost-to-go of a
safe cell is 0). Edge costs come from the RiskMap: 1 + risk of the cell
being entered, or ~1e6 for zombie-occupied cells.

Algorithm (paper pseudocode, made concrete)
---------------------------------------------
  g(s)    - cost of the best path from s to GOAL found so far
  rhs(s)  - one-step lookahead: min over successors s' of c(s,s') + g(s')
  key(s)  - [min(g(s),rhs(s)) + h(start,s) + km ; min(g(s),rhs(s))]
  km      - correction term so priorities stay valid when the start moves
  h(a,b)  - consistent heuristic = BFS step distance (every edge costs >= 1)

The priority queue is a lazy binary heap: entries carry the key they were
pushed with, and stale entries are discarded on pop by comparing against the
state's currently registered key.
"""
import heapq
import math

GOAL = ("__goal__",)          # virtual goal state (never a real cell)
INF = math.inf
MAX_POP = 200000              # safety cap so a bug can never hang the game


class DStarLite:
    """Incremental shortest-path planner over the risk-weighted grid."""

    def __init__(self, grid, riskmap):
        self.grid = grid
        self.riskmap = riskmap
        self.g = {}
        self.rhs = {}
        self.km = 0.0
        self.start = None        # current start state (a cell)
        self.last_start = None   # previous start state, for the km update
        self.cur_key = {}        # state -> key of its live heap entry
        self.hdist = {}          # BFS step distances from the current start
        self.heap = []
        self._seq = 0
        self.pops = 0            # diagnostics: expansions in last replan

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    def _gv(self, s):
        return self.g.get(s, INF)

    def _rhsv(self, s):
        return self.rhs.get(s, INF)

    def _h(self, s):
        """Heuristic cost from the current start to state s (BFS steps)."""
        if s == GOAL:
            return self.grid.dist_safe(self.start)
        return self.hdist.get(s, INF)

    def _key(self, s):
        m = min(self._gv(s), self._rhsv(s))
        return (m + self._h(s) + self.km, m)

    def _refresh_heuristic(self):
        self.hdist = self.grid.bfs_dist(self.start)

    # ------------------------------------------------------------------
    # graph structure: successors and predecessors
    # ------------------------------------------------------------------
    def succ(self, s):
        """Yield (next_state, edge_cost) pairs for state s."""
        if s == GOAL:
            return
        for n in self.grid.neighbors(s):
            yield n, self.riskmap.edge_cost(s, n)
        if s in self.grid.safe_cells:
            yield GOAL, 0.0

    def pred(self, s):
        """States that have an edge into s (used to propagate rhs changes)."""
        if s == GOAL:
            return self.grid.safe_cells
        return self.grid.neighbors(s)

    # ------------------------------------------------------------------
    # priority queue (lazy deletion; cur_key marks the live entry)
    # ------------------------------------------------------------------
    def _push(self, s):
        self._seq += 1
        k = self._key(s)
        self.cur_key[s] = k
        heapq.heappush(self.heap, (k[0], k[1], self._seq, s))

    def _remove(self, s):
        self.cur_key.pop(s, None)

    def _top_key(self):
        """Smallest live key in the queue, or (inf, inf) if it is empty."""
        while self.heap:
            k1, k2, _, s = self.heap[0]
            if self.cur_key.get(s) == (k1, k2):
                return (k1, k2)
            heapq.heappop(self.heap)
        return (INF, INF)

    # ------------------------------------------------------------------
    # D* Lite core
    # ------------------------------------------------------------------
    def update_vertex(self, u):
        """Recompute rhs(u) from current edge costs and g-values, then fix
        u's place in the priority queue (paper procedure UpdateVertex)."""
        if u != GOAL:
            best = INF
            for s2, c in self.succ(u):
                v = c + self._gv(s2)
                if v < best:
                    best = v
            self.rhs[u] = best
        self._remove(u)
        if self._gv(u) != self._rhsv(u):
            self._push(u)

    def compute_shortest_path(self):
        """Propagate cost-to-go values until the start's rhs == g."""
        self.pops = 0
        while (self._top_key() < self._key(self.start)
               or self._rhsv(self.start) != self._gv(self.start)):
            if self.pops > MAX_POP:
                break
            k1, k2, _, u = heapq.heappop(self.heap)
            self.pops += 1
            if self.cur_key.get(u) != (k1, k2):
                continue                      # stale heap entry
            kold = (k1, k2)
            if kold < self._key(u):           # key outdated -> reinsert
                self._push(u)
                continue
            self.cur_key.pop(u, None)         # u leaves the queue
            if self._gv(u) > self._rhsv(u):   # rhs improved -> adopt it
                self.g[u] = self._rhsv(u)
                for s in self.pred(u):
                    self.update_vertex(s)
            else:                             # g no longer justified
                self.g[u] = INF
                for s in self.pred(u):
                    self.update_vertex(s)
                self.update_vertex(u)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def reset(self, riskmap, start):
        """(Re)initialise for a fresh game."""
        self.riskmap = riskmap
        self.g.clear()
        self.rhs.clear()
        self.cur_key.clear()
        self.heap.clear()
        self.km = 0.0
        self.start = start
        self.last_start = start
        self._prev_riskmap = riskmap
        self._refresh_heuristic()
        # rhs(GOAL) = 0, everything else infinite (paper Initialize)
        self.rhs[GOAL] = 0.0
        self._push(GOAL)
        self.compute_shortest_path()

    def replan(self, riskmap, start):
        """One planning step: absorb cost changes, then re-converge."""
        self.riskmap = riskmap
        if start != self.start:
            # robot moved: keep priorities valid without reordering the queue
            self.km += self.grid.dist(self.last_start, start)
            self.last_start = start
            self.start = start
            self._refresh_heuristic()
        # find destination cells whose enter-cost changed; every edge into
        # such a cell changed, so update each of its predecessors (paper:
        # "for all directed edges (u,v) with changed costs: UpdateVertex(u)")
        changed = riskmap.changed_cells(self._prev_riskmap)
        seen = set()
        for b in changed:
            for a in self.pred(b):
                if a not in seen:
                    seen.add(a)
                    self.update_vertex(a)
        self._prev_riskmap = riskmap
        self.compute_shortest_path()

    def next_move(self):
        """Best successor of the start state, or None if unreachable.

        Returns a cell tuple, GOAL if the start is already a goal state,
        or None when no finite-cost path exists.
        """
        best_s, best_v = None, INF
        for s2, c in self.succ(self.start):
            v = c + self._gv(s2)
            if v < best_v:
                best_s, best_v = s2, v
        if best_s is None or self._gv(best_s) == INF:
            return None
        return best_s

    def path_to_goal(self, max_len=1000):
        """Reconstruct the full cell path start -> ... -> safe zone.

        Follows the best successor (min c + g) at each step, exactly like the
        paper's "move to argmin c(s,s') + g(s')". Returns a list of cells
        starting at the current start; empty if no path exists.
        """
        path = [self.start]
        s = self.start
        for _ in range(max_len):
            if s in self.grid.safe_cells or s == GOAL:
                break
            best_s, best_v = None, INF
            for s2, c in self.succ(s):
                v = c + self._gv(s2)
                if v < best_v:
                    best_s, best_v = s2, v
            if best_s is None or best_s == GOAL or self._gv(best_s) == INF:
                break
            path.append(best_s)
            s = best_s
        return path
