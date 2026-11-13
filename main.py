"""Zombie Apocalypse Survival.

Usage:
    python main.py                    # GUI (needs pygame)
    python main.py --console          # text mode, no extra libraries
    python main.py --seed 7           # reproducible map
    python main.py --runs 20          # simulate 20 games, print win rate
    python main.py --train 500        # train the Q-learning zombies, then play
    python main.py --zombie q --load q_table.json
    python main.py --planner astar    # use plain A* instead of D* Lite
    python main.py --manual           # YOU choose the moves, AI recommends
    python main.py --manual --console # same, in text mode
"""
import argparse
import time

import config
from game import Game
from qlearn import ZombieQLearner


def run_console(seed, delay, **game_kw):
    g = Game(seed, **game_kw)
    print(g.render_text(), "\n")
    while g.status == "playing":
        g.step()
        print(f"Turn {g.turn}")
        print(g.render_text(), "\n")
        time.sleep(delay)
    print(g.status.upper(), "-", g.message)


CONSOLE_KEYS = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0),
                ".": (0, 0)}


def run_console_manual(seed, **game_kw):
    """Text-mode co-pilot: the AI recommends, the human decides."""
    g = Game(seed, manual=True, **game_kw)
    print("MANUAL MODE - you decide every move.\n"
          "  1/2/3 or Enter = take an AI recommendation (Enter = #1)\n"
          "  u/d/l/r        = your own move up/down/left/right\n"
          "  .              = stay still      q = quit\n")
    while g.status == "playing":
        print(f"Turn {g.turn + 1}")
        print(g.render_text(), "\n")
        for r in g.recommendations:
            print(f"  [{r.rank}] {r.direction:<5} by {' + '.join(r.tags)}"
                  f"  | risk {r.risk_level} | {r.summary}")
        choice = input("your move> ").strip().lower()
        if choice == "q":
            return
        cell = None
        if choice == "" and g.recommendations:
            cell = g.recommendations[0].cell
        elif choice.isdigit() and 1 <= int(choice) <= len(g.recommendations):
            cell = g.recommendations[int(choice) - 1].cell
        elif choice in CONSOLE_KEYS:
            dx, dy = CONSOLE_KEYS[choice]
            cell = (g.human[0] + dx, g.human[1] + dy)
        if cell is None or not g.play_human_move(cell):
            print("  -> not a valid move, try again.\n")
            continue
        print(f"  -> {g.message}\n")
    print(g.render_text(), "\n")
    print(g.status.upper(), "-", g.message)


def run_batch(runs, seed, verbose=False, **game_kw):
    wins = 0
    for i in range(runs):
        g = Game(None if seed is None else seed + i, **game_kw)
        while g.status == "playing":
            g.step()
        wins += g.status == "won"
        if verbose:
            print(f"game {i}: {g.status} ({g.turn} turns)")
    print(f"Human won {wins}/{runs} games")
    return wins


def train_episodes(n, seed, learner, planner="dstar"):
    """Self-play training for the shared zombie Q-table (2022 paper)."""
    human_wins = 0
    eps0 = config.Q_EPSILON
    for ep in range(n):
        learner.epsilon = max(0.02, eps0 * (1 - ep / n))   # decay exploration
        g = Game(None if seed is None else seed + ep,
                 zombie_brain="q", qlearner=learner, learn=True,
                 human_planner=planner)
        while g.status == "playing":
            g.step()
        human_wins += g.status == "won"
        learner.episodes += 1
        if (ep + 1) % max(1, n // 10) == 0:
            print(f"  episode {ep + 1}/{n}: human win rate so far "
                  f"{human_wins}/{ep + 1}")
    print(f"Training done: {n} episodes, human won {human_wins} "
          f"({100 * human_wins / n:.0f}%).")
    return learner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--console", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--delay", type=float, default=0.3)
    p.add_argument("--runs", type=int, default=0)
    p.add_argument("--planner", choices=["dstar", "astar"],
                   default=config.HUMAN_PLANNER)
    p.add_argument("--zombie", choices=["minimax", "q"],
                   default=config.ZOMBIE_BRAIN)
    p.add_argument("--train", type=int, default=0,
                   help="train the Q-zombies for N episodes, then play")
    p.add_argument("--save", default=None, help="save Q-table to this file")
    p.add_argument("--load", default=None, help="load Q-table from this file")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--manual", action="store_true",
                   help="human-in-the-loop: AI recommends moves, you choose")
    a = p.parse_args()

    # --- Q-learner setup -------------------------------------------------
    learner = None
    if a.zombie == "q" or a.train:
        learner = ZombieQLearner.load(a.load) if a.load else ZombieQLearner()
        if a.load:
            print(f"Loaded Q-table with {len(learner.Q)} states "
                  f"({learner.episodes} training episodes).")
    if a.zombie == "q" and learner is not None and not a.train \
            and learner.episodes == 0 and not a.load:
        print("note: Q-table is untrained - run --train N first for "
              "meaningful zombie behaviour.\n")

    if a.train:
        train_episodes(a.train, a.seed, learner, planner=a.planner)
        if a.save:
            learner.save(a.save)
            print(f"Q-table saved to {a.save}")

    game_kw = dict(zombie_brain=a.zombie, qlearner=learner,
                   human_planner=a.planner)

    # strip None learner so Game uses a fresh one only when needed
    if learner is None:
        game_kw.pop("qlearner")

    if a.runs:
        run_batch(a.runs, a.seed, verbose=a.verbose, **game_kw)
    elif a.console and a.manual:
        run_console_manual(a.seed, **game_kw)
    elif a.console:
        run_console(a.seed, a.delay, **game_kw)
    else:
        try:
            import gui
        except ModuleNotFoundError:
            print("pygame not installed - falling back to console mode.\n"
                  "Install it with: pip install -r requirements.txt")
            if a.manual:
                run_console_manual(a.seed, **game_kw)
            else:
                run_console(a.seed, a.delay, **game_kw)
        else:
            gui.run(a.seed, zombie_brain=a.zombie, qlearner=learner,
                    human_planner=a.planner, manual=a.manual)


if __name__ == "__main__":
    main()
