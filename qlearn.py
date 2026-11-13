"""Min-Max Q-learning for the zombie team (2022 paper).

All zombies share one tabular Q-function - they are symmetric players in the
same pursuit-evasion game, so experience is pooled (multi-player setting).
The team reward is min-max flavoured: every zombie is rewarded for the
*worst-case* outcome of the team (capture = good, human escape = bad) plus
an individual shaping term for closing the distance to the human.

State (discretised, kept small so the table stays readable):
    d_zombie_human : BFS distance zombie -> human, capped at 7
    d_human_safe   : human's BFS distance to the safe zone, capped at 12
    nearby_zombies : how many *other* zombies are within 3 steps of the human

Actions (5): up / down / left / right / stay.

Training is self-play: the human uses the real A*/D* Lite planner, the
zombies explore with epsilon-greedy and update the shared Q-table after
every turn. `python main.py --train N` runs N episodes, then you can play
against the learned policy.
"""
import json
import math
import random

import config
from minimax import captured

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]   # up, down, left, right, stay
STAY = 4

CAPTURE_REWARD = 100.0
ESCAPE_REWARD = -100.0
DIST_WEIGHT = 2.0       # reward per step of closing distance
STEP_PENALTY = 0.1      # small cost per turn (learn to capture quickly)


class ZombieQLearner:
    """Shared tabular Q-learner for all zombies."""

    def __init__(self, alpha=None, gamma=None, epsilon=None):
        self.alpha = config.Q_ALPHA if alpha is None else alpha
        self.gamma = config.Q_GAMMA if gamma is None else gamma
        self.epsilon = config.Q_EPSILON if epsilon is None else epsilon
        self.Q = {}        # state tuple -> list of 5 action values
        self.episodes = 0

    # ------------------------------------------------------------------
    # state representation
    # ------------------------------------------------------------------
    def state(self, grid, z, human, zombies):
        d_zh = min(grid.dist(z, human), 7)
        d_hs = min(grid.dist_safe(human), 12)
        nearby = sum(1 for o in zombies
                     if o != z and grid.dist(o, human) <= 3)
        return (d_zh, d_hs, min(nearby, 3))

    def _qrow(self, s):
        return self.Q.setdefault(s, [0.0] * len(ACTIONS))

    # ------------------------------------------------------------------
    # action selection
    # ------------------------------------------------------------------
    def choose(self, s, epsilon):
        """Epsilon-greedy action index."""
        if random.random() < epsilon:
            return random.randrange(len(ACTIONS))
        row = self._qrow(s)
        return max(range(len(ACTIONS)), key=lambda a: row[a])

    def best_action(self, s):
        row = self._qrow(s)
        return max(range(len(ACTIONS)), key=lambda a: row[a])

    def apply(self, grid, z, a, zombies):
        """Map an action index to a legal new cell for zombie z."""
        dx, dy = ACTIONS[a]
        n = (z[0] + dx, z[1] + dy)
        if (grid.walkable(n) and n not in self._occupied(zombies, z)
                and n not in grid.safe_cells):
            return n
        return z    # illegal move -> stay

    @staticmethod
    def _occupied(zombies, z):
        return {o for o in zombies if o != z}

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def update(self, s, a, reward, s2, done):
        row = self._qrow(s)
        future = 0.0 if done else max(self._qrow(s2))
        row[a] += self.alpha * (reward + self.gamma * future - row[a])

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path):
        data = {json.dumps(list(k)): v for k, v in self.Q.items()}
        with open(path, "w") as f:
            json.dump({"episodes": self.episodes, "Q": data}, f)

    @staticmethod
    def load(path):
        with open(path) as f:
            data = json.load(f)
        q = ZombieQLearner()
        q.Q = {tuple(json.loads(k)): v for k, v in data["Q"].items()}
        q.episodes = data.get("episodes", 0)
        return q


# ----------------------------------------------------------------------
# one zombie phase for the Q-brain; used by Game.step and by training
# ----------------------------------------------------------------------
def q_zombie_turn(grid, human, zombies, learner, learn=False,
                  epsilon=None):
    """Move every zombie one step under the learned policy.

    When learn=True each zombie's transition is recorded so the caller can
    apply rewards once the whole team has moved (rewards depend on the
    joint outcome - that is the min-max team aspect).

    Returns (new_zombies, transitions) where transitions is a list of
    (zombie_index, state, action, distance_before) for learning.
    """
    eps = learner.epsilon if epsilon is None else epsilon
    zombies = list(zombies)
    order = list(range(len(zombies)))
    random.shuffle(order)
    transitions = []
    dist_before = {i: grid.dist(zombies[i], human) for i in order}

    for i in order:
        z = zombies[i]
        s = learner.state(grid, z, human, zombies)
        a = learner.choose(s, eps if learn else 0.0)
        zombies[i] = learner.apply(grid, z, a, zombies)
        transitions.append((i, s, a, dist_before[i]))
    return zombies, transitions


def apply_updates(learner, grid, human, zombies, transitions, done):
    """One Q-update per zombie after a full zombie phase.

    Reward = team outcome (min-max: capture is good for the whole team,
    human escape is bad) + individual shaping for closing distance.
    """
    team = 0.0
    if captured(grid, human, zombies):
        team += CAPTURE_REWARD
    elif human in grid.safe_cells:
        team += ESCAPE_REWARD
    for i, s, a, d_before in transitions:
        z = zombies[i]
        s2 = learner.state(grid, z, human, zombies)
        shaping = (d_before - grid.dist(z, human)) * DIST_WEIGHT - STEP_PENALTY
        learner.update(s, a, team + shaping, s2, done)
