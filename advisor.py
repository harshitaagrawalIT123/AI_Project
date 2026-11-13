"""AI move advisor for the human player (co-pilot / manual mode).

In manual mode the human chooses every move. Before each choice this module
looks at the position and recommends up to TOP_N *different* moves, each
backed by a different way of thinking:

  PLANNER   - the risk-aware route (D* Lite / A*) the AI itself would follow.
  FASTEST   - the move that gets closest to the safe zone, never stepping
              into capture range if a non-fatal alternative exists.
  SAFEST    - the move that keeps the most distance from the zombies
              (obstacle-aware BFS distance, lowest risk-field value).
  LOOKAHEAD - the move that scores best after the zombies' predicted reply
              (uses the same brain the zombies really use: minimax or Q).

When several strategies pick the same cell they are merged into one
recommendation with several tags ("agreed by 3"). The player may follow any
recommendation or ignore them all and play their own move.
"""
import config
from minimax import captured, move_all_zombies
from qlearn import q_zombie_turn

TOP_N = 3

DIRECTIONS = {(0, -1): "Up", (0, 1): "Down", (-1, 0): "Left",
              (1, 0): "Right", (0, 0): "Stay"}

# outcome codes, best first (used for ranking)
SAFE, OK, CAUGHT_AFTER_REPLY, CAUGHT = "safe", "ok", "caught_after_reply", "caught"
_OUTCOME_RANK = {SAFE: 0, OK: 1, CAUGHT_AFTER_REPLY: 2, CAUGHT: 3}


class Recommendation:
    """One suggested move for the human."""

    def __init__(self, cell, direction, tags, outcome, steps_to_safe,
                 zombie_dist, risk, cost):
        self.cell = cell
        self.direction = direction          # "Up" / "Down" / ... / "Stay"
        self.tags = tags                    # strategies that picked this cell
        self.outcome = outcome
        self.steps_to_safe = steps_to_safe
        self.zombie_dist = zombie_dist      # nearest zombie after their reply
        self.risk = risk
        self.cost = cost
        self.rank = 0                       # 1-based, set by recommend()

    @property
    def risk_level(self):
        if self.outcome in (CAUGHT, CAUGHT_AFTER_REPLY):
            return "DEADLY"
        if self.outcome == SAFE:
            return "LOW"
        if self.zombie_dist <= 2 or self.risk >= 1.2:
            return "HIGH"
        if self.zombie_dist <= 4 or self.risk >= 0.5:
            return "MEDIUM"
        return "LOW"

    @property
    def summary(self):
        if self.outcome == SAFE:
            return "reaches the safe zone - you win"
        if self.outcome == CAUGHT:
            return "walks into a zombie - you lose"
        if self.outcome == CAUGHT_AFTER_REPLY:
            return "zombies can catch you after this"
        return (f"{self.steps_to_safe} steps to safe, "
                f"nearest zombie {self.zombie_dist}")

    def __repr__(self):
        return (f"#{self.rank} {self.direction} {self.cell} "
                f"[{', '.join(self.tags)}] risk={self.risk_level}")


# ----------------------------------------------------------------------
def legal_moves(grid, human, zombies):
    """Cells the human may move to this turn (neighbours + staying put)."""
    zset = set(zombies)
    return [n for n in grid.neighbors(human) if n not in zset] + [human]


def predict_zombies(game, human_cell):
    """Where the zombies would be after replying to `human_cell`.

    Zombies only act every ZOMBIE_MOVE_EVERY turns; on the other turns they
    stay put. The Q-brain is queried greedily (learn=False) so the real
    Q-table is never modified.
    """
    if (game.turn + 1) % config.ZOMBIE_MOVE_EVERY != 0:
        return list(game.zombies)
    if game.zombie_brain == "q":
        new_z, _ = q_zombie_turn(game.grid, human_cell, game.zombies,
                                 game.qlearner, learn=False, epsilon=0.0)
        return new_z
    return move_all_zombies(game.grid, human_cell, list(game.zombies),
                            config.MINIMAX_DEPTH)


def _evaluate(game, cell):
    """Score one candidate move."""
    grid, zombies, rm = game.grid, game.zombies, game.riskmap
    steps = grid.dist_safe(cell)
    risk = rm.risk_of(cell)
    if cell in grid.safe_cells:
        return dict(cell=cell, outcome=SAFE, steps=0, zdist=99, risk=risk,
                    cost=-1e6)
    if captured(grid, cell, zombies):
        return dict(cell=cell, outcome=CAUGHT, steps=steps, zdist=0,
                    risk=risk, cost=1e6)
    new_z = predict_zombies(game, cell)
    zdist = min(grid.dist(z, cell) for z in new_z)
    outcome = CAUGHT_AFTER_REPLY if captured(grid, cell, new_z) else OK
    # cost-to-go estimate: distance to safety + weighted risk (as the planner)
    cost = steps + rm.weight * risk
    if outcome == CAUGHT_AFTER_REPLY:
        cost += 500
    return dict(cell=cell, outcome=outcome, steps=steps, zdist=zdist,
                risk=risk, cost=cost)


def _pick(evals, key):
    """Best candidate under `key`, preferring non-fatal outcomes."""
    return min(evals, key=lambda e: (_OUTCOME_RANK[e["outcome"]], key(e)))


# ----------------------------------------------------------------------
def recommend(game, top_n=TOP_N):
    """Return up to `top_n` Recommendations, best first.

    Expects game.riskmap and game.planner_move to be fresh for this turn
    (Game.begin_turn() takes care of that).
    """
    grid = game.grid
    cands = legal_moves(grid, game.human, game.zombies)
    evals = {c: _evaluate(game, c) for c in cands}
    ev = list(evals.values())

    picks = {}      # cell -> list of strategy tags

    def add(tag, e):
        picks.setdefault(e["cell"], []).append(tag)

    planner_cell = game.planner_move
    if planner_cell in evals:
        add("Planner", evals[planner_cell])
    add("Fastest", _pick(ev, lambda e: (e["steps"], e["risk"])))
    add("Safest", _pick(ev, lambda e: (-e["zdist"], e["risk"], e["steps"])))
    add("Lookahead", _pick(ev, lambda e: (e["cost"], e["steps"])))

    recs = []
    for cell, tags in picks.items():
        e = evals[cell]
        d = (cell[0] - game.human[0], cell[1] - game.human[1])
        recs.append(Recommendation(
            cell, DIRECTIONS.get(d, "Stay"), tags, e["outcome"], e["steps"],
            e["zdist"], e["risk"], e["cost"]))

    # best outcome first, then most strategies agreeing, then lowest cost
    recs.sort(key=lambda r: (_OUTCOME_RANK[r.outcome], -len(r.tags),
                              "Planner" not in r.tags, r.cost))
    recs = recs[:top_n]
    for i, r in enumerate(recs, 1):
        r.rank = i
    return recs


def classify_choice(recs, cell):
    """How did the player's move relate to the AI's advice?

    Returns (kind, rank): ("ai", 1..N) if they followed a recommendation,
    ("own", None) if they played their own move.
    """
    for r in recs:
        if r.cell == cell:
            return "ai", r.rank
    return "own", None
