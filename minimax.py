"""Minimax with alpha-beta pruning for the zombies.

Based on the classical alpha-beta analysis (1975) and the minimax survey
(1995): the search is only as good as its move ordering, so candidate moves
are sorted before they are evaluated -

  * zombie (MAX) tries the moves that bring it closest to the human first,
    which finds high-value cutoffs earliest,
  * human (MIN) tries its best escape (closest to the safe zone) first.

MAX = zombie team (wants to minimise time-to-capture, Catch Me If You Can
2012), MIN = human (wants to reach the safe zone). Depth 3 means
zombie move -> human reply -> zombie move. Distances are obstacle-aware BFS
distances, so walls are handled correctly (Pursuit-Evasion with Obstacles,
2016).
"""
import config

WIN_SCORE = 1000    # zombie captures the human
LOSE_SCORE = -1000  # human reaches the safe zone


def captured(grid, human, zombies):
    """True when any zombie is within CAPTURE_RADIUS steps of the human."""
    return any(grid.dist(z, human) <= config.CAPTURE_RADIUS for z in zombies)


def evaluate(grid, human, zombies):
    """Score from the zombies' point of view (higher = better for zombies)."""
    if captured(grid, human, zombies):
        return WIN_SCORE
    if human in grid.safe_cells:
        return LOSE_SCORE
    # get close: smaller total distance to the human is better
    chase = -sum(grid.dist(z, human) for z in zombies)
    # and keep the human far from safety: larger distance-to-safe is better
    progress = grid.dist_safe(human)
    return chase + 0.5 * progress


def zombie_options(grid, z, others):
    occupied = set(others)
    opts = [n for n in grid.neighbors(z)
            if n not in occupied and n not in grid.safe_cells]
    opts.append(z)   # may wait
    return opts


def human_options(grid, human, zombies):
    zset = set(zombies)
    opts = [n for n in grid.neighbors(human) if n not in zset]
    opts.append(human)   # may stay still
    return opts


def ordered_zombie_options(grid, z, others, human):
    """Most threatening moves first -> earliest alpha-beta cutoffs."""
    return sorted(zombie_options(grid, z, others),
                  key=lambda m: grid.dist(m, human))


def ordered_human_options(grid, human, zombies):
    """Best escapes first (closest to the safe zone)."""
    return sorted(human_options(grid, human, zombies),
                  key=lambda h: grid.dist_safe(h))


def minimax(grid, human, zombies, idx, depth, zombie_turn, alpha, beta):
    if captured(grid, human, zombies) or human in grid.safe_cells \
            or depth == 0:
        return evaluate(grid, human, zombies)
    z = zombies[idx]
    others = zombies[:idx] + zombies[idx + 1:]
    if zombie_turn:
        best = -10**9
        for m in ordered_zombie_options(grid, z, others, human):
            nz = zombies[:idx] + (m,) + zombies[idx + 1:]
            val = minimax(grid, human, nz, idx, depth - 1, False, alpha, beta)
            if val > best:
                best = val
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    best = 10**9
    for h in ordered_human_options(grid, human, zombies):
        val = minimax(grid, h, zombies, idx, depth - 1, True, alpha, beta)
        if val < best:
            best = val
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def zombie_best_move(grid, human, zombies, idx, depth):
    zombies = tuple(zombies)
    z = zombies[idx]
    others = zombies[:idx] + zombies[idx + 1:]
    best_val, best_move = -10**9, z
    for m in ordered_zombie_options(grid, z, others, human):
        nz = zombies[:idx] + (m,) + zombies[idx + 1:]
        val = minimax(grid, human, nz, idx, depth - 1, False, -10**9, 10**9)
        if val > best_val:
            best_val, best_move = val, m
    return best_move


def move_all_zombies(grid, human, zombies, depth):
    zombies = list(zombies)
    for i in range(len(zombies)):
        zombies[i] = zombie_best_move(grid, human, zombies, i, depth)
        if captured(grid, human, zombies):
            break
    return zombies
