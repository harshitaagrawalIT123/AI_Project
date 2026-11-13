# Zombie Apocalypse Survival

A turn-based grid game. A human tries to reach the safe zone (green,
bottom-right) while zombies hunt it.

## Research basis

| Paper | What it contributes here |
|---|---|
| D+ : A Risk-Aware Platform Agnostic Heterogeneous Path Planner (2023) | `riskmap.py` - per-cell risk field around zombies, decayed by *obstacle-aware* BFS distance, replanned every turn |
| D Lite / D* Lite (Koenig & Likhachev, 2002) | `dstar.py` - incremental replanning: the human's search from last turn is reused and only states touched by changed edge costs are re-propagated |
| Min-Max Q-learning for Multi-player Pursuit-Evasion Games (2022) | `qlearn.py` - all zombies share one tabular Q-function, trained by self-play against the real human planner |
| An Analysis of Alpha-Beta Pruning (1975) + A Survey on Minimax Trees (1995) | `minimax.py` - move ordering (most threatening zombie moves / best human escapes first) so alpha-beta cuts off earlier |
| Pursuit-Evasion Games in the Presence of Obstacles (2016) | BFS-distance evaluation everywhere (walls respected), capture radius, and risk lookahead on cells zombies could reach next turn |
| Catch Me If You Can (2012) | zombie evaluation minimises time-to-capture; `CAPTURE_RADIUS` defines the capture condition |

## Pipeline (one turn)

```
zombie positions
   │
   ▼
RiskMap.update()          risk field = Σ_zombies exp(-BFSdist/τ)
   │                      + lookahead bonus on next-step cells
   │                      + death zone inside CAPTURE_RADIUS (impassable)
   ▼
Human planner             D* Lite replan (incremental) or A* fallback
   │                      cost(cell) = 1 + RISK_WEIGHT * risk(cell)
   ▼
win/lose check            safe cell -> win, within CAPTURE_RADIUS -> loss
   │
   ▼ (every ZOMBIE_MOVE_EVERY turns)
Zombie brain              minimax: alpha-beta, depth MINIMAX_DEPTH
                          q:       greedy action from shared Q-table
   │
   ▼
Q-update (training only)  team reward (capture/escape) + distance shaping
```

## Files

| File | Purpose |
|---|---|
| main.py | Entry point (GUI / console / batch / training) |
| game.py | Game state and the turn pipeline above |
| grid.py | Grid, walls (guaranteed solvable), BFS distances, dist-to-safe heuristic |
| riskmap.py | D+ risk field and edge costs |
| dstar.py | D* Lite incremental planner (verified against Dijkstra) |
| astar.py | A* planner (fallback / `--planner astar`) |
| minimax.py | Alpha-beta zombie search with move ordering |
| qlearn.py | Shared Q-table, self-play training, save/load |
| gui.py | Pygame visualisation (risk field shown as red tint) |
| config.py | All tunables |

## Run

Requires Python 3.8+. Pygame is only needed for the GUI.

    pip install -r requirements.txt          # only for the GUI
    python main.py                           # GUI
    python main.py --console                 # text mode (no pygame needed)
    python main.py --seed 7                  # same map every time
    python main.py --runs 50 --seed 100      # simulate 50 games, print win rate

Train the Q-learning zombies, then play against them:

    python main.py --train 1000 --save q_table.json
    python main.py --zombie q --load q_table.json
    python main.py --runs 50 --zombie q --load q_table.json

Other flags: `--planner astar`, `--zombie minimax|q`, `--verbose`.

GUI keys: SPACE = one turn, A = auto-play, R = restart, ESC = quit.
Legend: blue circle = human, red square = zombie, grey = wall, green = safe
zone, yellow dots = planned path, red cell tint = zombie risk field.
Console: H, Z, #, S, * (path).

## Tweak difficulty (config.py)

`NUM_ZOMBIES`, `ZOMBIE_MOVE_EVERY` (1 = zombies as fast as human),
`MINIMAX_DEPTH`, `WALL_DENSITY`, `GRID_SIZE`, `CAPTURE_RADIUS` (0 = only
same-cell capture, 1 = adjacent also counts), `RISK_WEIGHT`, `RISK_TAU`.
