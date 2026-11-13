"""Game settings. Change these to tweak difficulty."""
GRID_SIZE = 15          # grid is GRID_SIZE x GRID_SIZE
WALL_DENSITY = 0.20     # fraction of cells that are walls
NUM_ZOMBIES = 3
MINIMAX_DEPTH = 3       # plies: zombie -> human -> zombie
MAX_TURNS = 120         # human loses if not safe by then
SAFE_ZONE_SIZE = 2      # safe zone is a SxS block in bottom-right corner
CELL_PX = 40            # pygame cell size
FPS_DELAY_MS = 250      # auto-play delay per turn (GUI)
ZOMBIE_MOVE_EVERY = 2   # zombies move every N turns (1 = same speed as human, 2 = half speed)

# --- Risk-aware planning (D+, 2023) -------------------------------------
RISK_WEIGHT = 6.0       # how strongly the human avoids risky cells
RISK_TAU = 2.0          # decay length (in BFS steps) of the zombie risk field
RISK_LOOKAHEAD = 0.6    # extra risk on cells a zombie could reach next turn

# --- Pursuit / capture (Catch Me If You Can, 2012; Pursuit-Evasion 2016)
CAPTURE_RADIUS = 1      # human is caught when a zombie is within this many steps

# --- Planner / brain selection ------------------------------------------
# HUMAN_PLANNER: "dstar" (D* Lite incremental replanning) or "astar" (plain A*)
HUMAN_PLANNER = "dstar"
# ZOMBIE_BRAIN: "minimax" (alpha-beta search) or "q" (learned Q-policy)
ZOMBIE_BRAIN = "minimax"

# --- Q-learning training (Min-Max Q-learning, 2022) ----------------------
Q_ALPHA = 0.2           # learning rate
Q_GAMMA = 0.9           # discount factor
Q_EPSILON = 0.15        # exploration rate during training
TRAIN_EPISODES = 1000   # default number of training episodes for --train
