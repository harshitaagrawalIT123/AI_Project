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


class Game:
    def __init__(self, seed=None, zombie_brain=None, qlearner=None,
                 human_planner=None, learn=False):
        self.seed = seed
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
        self.message = "Reach the safe zone!"
        self.riskmap = RiskMap(self.grid)
        self.riskmap.update(self.zombies)
        if self.human_planner == "dstar":
            self.planner = DStarLite(self.grid, self.riskmap)
            self.planner.reset(self.riskmap, self.human)
        else:
            self.planner = None

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
    def _human_move(self):
        """Choose the human's next cell and the path used for display."""
        old = self.human
        if self.planner is not None:
            self.planner.replan(self.riskmap, self.human)
            nxt = self.planner.next_move()
            if nxt is not None and nxt != GOAL:
                self.path = self.planner.path_to_goal()
                self.human = nxt
                return
        # fallback: plain A* (also used when D* Lite finds no finite path)
        self.human, self.path = human_next_move(self.grid, self.human,
                                                self.zombies)
        if not self.path:
            self.path = [old, self.human]

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
    def step(self):
        """One full turn: risk update -> human -> zombies."""
        if self.status != "playing":
            return
        self.turn += 1

        # 1+2. refresh the risk field, then move the human
        self.riskmap = self.fresh_riskmap()
        self._human_move()

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
                elif c in self.grid.safe_cells:
                    ch = "S"
                elif c in self.path:
                    ch = "*"
                else:
                    ch = "."
                row += ch + " "
            rows.append(row)
        return "\n".join(rows)
