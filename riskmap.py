"""Risk-aware cost field, inspired by D+ (2023).

D+ penalises paths through regions where the outcome is uncertain. Here the
"risk" of a cell comes from the zombies:

  * every zombie emits a radial risk that decays exponentially with the
    *obstacle-aware* BFS distance (Manhattan distance would ignore walls and
    badly misjudge risk around corners - a key point of pursuit-evasion with
    obstacles, 2016),
  * cells a zombie could step into on its next move get an extra penalty, so
    the human does not plan a route that walks into a predicted interception,
  * zombie-occupied cells are treated as impassable (infinite cost) so no
    planner will ever route the human through a zombie.

The field is recomputed every turn and handed to the path planner, which adds
RISK_WEIGHT * risk(cell) to the cost of entering each cell.
"""
import math

import config

INF_COST = 1.0e6        # cost of entering a zombie-occupied cell


class RiskMap:
    """Holds the per-cell risk values for the current turn."""

    def __init__(self, grid, weight=None, tau=None, lookahead=None):
        self.grid = grid
        self.weight = config.RISK_WEIGHT if weight is None else weight
        self.tau = config.RISK_TAU if tau is None else tau
        self.lookahead = (config.RISK_LOOKAHEAD if lookahead is None
                          else lookahead)
        self.risk = {}          # cell -> float risk in [0, ~NUM_ZOMBIES]
        self.blocked = set()    # cells currently occupied by zombies
        self.death = set()      # cells within capture radius of a zombie
        self._enter_cost = {}   # cell -> total cost of entering it

    # ------------------------------------------------------------------
    def update(self, zombies):
        """Recompute the risk field for the given zombie positions."""
        self.blocked = set(zombies)
        self.death = set()         # cells a zombie could capture from
        self.risk = {c: 0.0 for c in self._all_cells()}
        for z in zombies:
            dist = self.grid.bfs_dist(z)      # cached per zombie position
            for c, d in dist.items():
                # exponential decay with BFS distance
                self.risk[c] += math.exp(-d / self.tau)
                # stepping within CAPTURE_RADIUS of a zombie means capture
                # (safe cells are exempt: entering them wins immediately)
                if 0 < d <= config.CAPTURE_RADIUS and c not in self.grid.safe_cells:
                    self.death.add(c)
            # cells reachable in one zombie step are extra risky
            for n in self.grid.neighbors(z):
                self.risk[n] += self.lookahead
        self._enter_cost = {}

    # ------------------------------------------------------------------
    def _all_cells(self):
        g = self.grid
        return [(x, y) for x in range(g.size) for y in range(g.size)
                if g.walkable((x, y))]

    def risk_of(self, cell):
        return self.risk.get(cell, 0.0)

    def enter_cost(self, cell):
        """Total extra cost of stepping onto `cell` this turn."""
        if cell not in self._enter_cost:
            if cell in self.blocked or cell in self.death:
                self._enter_cost[cell] = INF_COST
            else:
                self._enter_cost[cell] = self.weight * self.risk_of(cell)
        return self._enter_cost[cell]

    def edge_cost(self, a, b):
        """Cost of moving from cell `a` to neighbouring cell `b`.

        Every step costs 1 plus the risk of the cell being entered, so the
        planner trades distance against danger.
        """
        if a == b:
            return 0.0
        return 1.0 + self.enter_cost(b)

    def changed_cells(self, old_risk_map):
        """Cells whose enter-cost differs from a previous RiskMap.

        D* Lite only needs to re-examine edges whose cost changed; this tells
        it exactly which destination cells were affected.
        """
        changed = set()
        for c in self._all_cells():
            if ((c in self.blocked) != (c in old_risk_map.blocked)
                    or (c in self.death) != (c in old_risk_map.death)):
                changed.add(c)
            elif abs(self.enter_cost(c)
                     - old_risk_map.enter_cost(c)) > 1e-9:
                changed.add(c)
        return changed
