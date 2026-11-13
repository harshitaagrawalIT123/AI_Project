"""Game state and turn pipeline.

One turn works like this (the "working flow" of the whole project):

  1. RISK FIELD   - RiskMap.update() rebuilds the D+-style risk field from
                    the current zombie positions (BFS-distance decay plus a
                    penalty on cells zombies could reach next turn, plus an
                    impassable death zone inside CAPTURE_RADIUS).
  2. HUMAN MOVE   - the planner replans incrementally. With
                    HUMAN_PLANNER="dstar" the D* Lite search from last turn
                    is reused and only states touched by changed edge costs
                    are re-propagated; "astar" falls back to plain A*.
                    Both minimise  distance + RISK_WEIGHT * risk.
  3. CAPTURE/WIN  - reaching a safe cell wins; being within
                    CAPTURE_RADIUS of a zombie loses.
  4. ZOMBIE MOVE  - every ZOMBIE_MOVE_EVERY turns the zombies act:
                    "minimax" runs alpha-beta search with move ordering,
                    "q" follows the Q-table learned by self-play
                    (Min-Max Q-learning, 2022). With learn=True the
                    transitions are written back into the Q-table.
"""
import config
from grid import Grid
from astar import human_next_move
from minimax import move_all_zombies, captured
from riskmap import RiskMap
from dstar import DStarLite, GOAL
from qlearn import ZombieQLearner, q_zombie_turn, apply_updates
import advisor


class Game:
    def __init__(self, seed=None, zombie_brain=None, qlearner=None,
                 human_planner=None, learn=False, manual=False):
        self.seed = seed
        self.manual = manual           # True = human picks the moves (co-pilot)
        self.zombie_brain = zombie_brain or config.ZOMBIE_BRAIN
        self.human_planner = human_planner or config.HUMAN_PLANNER
        self.qlearner = qlearner or ZombieQLearner()
        self.learn = learn
        self.reset()

    def reset(self):
        self.grid = Grid(config.GRID_SIZE, config.WALL_DENSITY,
                         config.SAFE_ZONE_SIZE, self.seed)
        self.human = self.grid.human_start
        self.zombies = self.grid.random_zombie_spots(config.NUM_ZOMBIES,
                                                     self.human)
        self.turn = 0
        self.status = "playing"        # playing | won | lost
        self.path = []
        self.message = ("Your move - pick a recommendation or play your own!"
                        if self.manual else "Reach the safe zone!")
        self.recommendations = []      # advisor output for the current turn
        self.planner_move = None       # what the AI planner would play now
        self._prepared = False         # begin_turn() already run this turn?
        self.stats = {"ai": 0, "top": 0, "own": 0}   # manual-mode choices
        self.riskmap = RiskMap(self.grid)
        self.riskmap.update(self.zombies)
        if self.human_planner == "dstar":
            self.planner = DStarLite(self.grid, self.riskmap)
            self.planner.reset(self.riskmap, self.human)
        else:
            self.planner = None
        if self.manual:
            self.begin_turn()          # first set of recommendations

    def fresh_riskmap(self):
        """New RiskMap object for this turn.

        A fresh object matters: D* Lite diffs the new field against the
        previous one to know which edge costs changed, so the old snapshot
        must not be mutated.
        """
        rm = RiskMap(self.grid)
        rm.update(self.zombies)
        return rm

    # ------------------------------------------------------------------
    def _plan_move(self):
        """AI planner's choice for the human: returns (next_cell, path)."""
        old = self.human
        if self.planner is not None:
            self.planner.replan(self.riskmap, self.human)
            nxt = self.planner.next_move()
            if nxt is not None and nxt != GOAL:
                return nxt, self.planner.path_to_goal()
        # fallback: plain A* (also used when D* Lite finds no finite path)
        nxt, path = human_next_move(self.grid, self.human, self.zombies)
        if not path:
            path = [old, nxt]
        return nxt, path

    def begin_turn(self):
        """Prepare the human's upcoming move (idempotent within a turn).

        Refreshes the risk field, lets the planner replan (so D* Lite stays
        in sync even when a human is steering) and, in manual mode, asks the
        advisor for ranked move recommendations.
        """
        if self._prepared:
            return
        self.riskmap = self.fresh_riskmap()
        self.planner_move, self.path = self._plan_move()
        self.recommendations = (advisor.recommend(self) if self.manual
                                else [])
        self._prepared = True

    def legal_moves(self):
        return advisor.legal_moves(self.grid, self.human, self.zombies)

    def play_human_move(self, cell):
        """Manual mode: the human plays `cell`. Returns False if illegal."""
        if self.status != "playing":
            return False
        self.begin_turn()
        if cell not in self.legal_moves():
            self.message = "Illegal move - pick an adjacent free cell."
            return False
        kind, rank = advisor.classify_choice(self.recommendations, cell)
        self.step(human_move=cell)
        if kind == "ai":
            self.stats["ai"] += 1
            self.stats["top"] += rank == 1
        else:
            self.stats["own"] += 1
        if self.status == "playing":
            how = (f"AI pick #{rank}" if kind == "ai" else "your own move")
            self.message = f"Played {how}. Your move."
            self.begin_turn()          # recommendations for the next turn
        else:
            self.message += (f"  (AI picks followed: {self.stats['ai']}, "
                             f"own moves: {self.stats['own']})")
        return True

    def set_manual(self, manual):
        """Switch between AI-driven (auto) and human-driven (manual) play."""
        self.manual = manual
        self._prepared = False
        if manual and self.status == "playing":
            self.message = "Your move - pick a recommendation or play your own!"
            self.begin_turn()
        elif not manual:
            self.recommendations = []

    def _zombie_move(self):
        """Move the zombie team; returns Q-transitions when learning."""
        if self.zombie_brain == "q":
            new_z, transitions = q_zombie_turn(
                self.grid, self.human, self.zombies, self.qlearner,
                learn=self.learn)
            self.zombies = new_z
            return transitions
        self.zombies = move_all_zombies(self.grid, self.human, self.zombies,
                                        config.MINIMAX_DEPTH)
        return None

    # ------------------------------------------------------------------
    def step(self, human_move=None):
        """One full turn: risk update -> human -> zombies.

        human_move=None lets the AI planner move the human (auto mode);
        passing a legal cell plays that move instead (manual mode).
        """
        if self.status != "playing":
            return
        # 1+2. refresh the risk field + plan (skipped if already prepared)
        self.begin_turn()
        self.turn += 1
        self.human = self.planner_move if human_move is None else human_move
        self._prepared = False

        # 3. resolve human move
        if self.human in self.grid.safe_cells:
            self.status = "won"
            self.message = f"Human reached the safe zone in {self.turn} turns!"
            return
        if captured(self.grid, self.human, self.zombies):
            self.status = "lost"
            self.message = f"Zombies caught the human on turn {self.turn}."
            return

        # 4. zombies move (every N turns)
        if self.turn % config.ZOMBIE_MOVE_EVERY == 0:
            transitions = self._zombie_move()
            if self.human in self.grid.safe_cells:
                self.status = "won"
                self.message = (f"Human reached the safe zone in "
                                f"{self.turn} turns!")
            elif captured(self.grid, self.human, self.zombies):
                self.status = "lost"
                self.message = (f"Zombies caught the human on turn "
                                f"{self.turn}.")
            elif self.turn >= config.MAX_TURNS:
                self.status = "lost"
                self.message = "Out of time."
            if self.learn and transitions is not None:
                done = self.status != "playing"
                apply_updates(self.qlearner, self.grid, self.human,
                              self.zombies, transitions, done)

    # ------------------------------------------------------------------
    def render_text(self):
        rec_rank = {r.cell: r.rank for r in self.recommendations}
        rows = []
        for y in range(self.grid.size):
            row = ""
            for x in range(self.grid.size):
                c = (x, y)
                if c == self.human:
                    ch = "H"
                elif c in self.zombies:
                    ch = "Z"
                elif c in self.grid.walls:
                    ch = "#"
                elif c in rec_rank:
                    ch = str(rec_rank[c])      # AI recommendation 1/2/3
                elif c in self.grid.safe_cells:
                    ch = "S"
                elif c in self.path:
                    ch = "*"
                else:
                    ch = "."
                row += ch + " "
            rows.append(row)
        return "\n".join(rows)
