import pygame
import config
from game import Game

COL = dict(bg=(25, 25, 30), floor=(45, 48, 55), wall=(90, 90, 100),
           safe=(40, 140, 70), human=(60, 150, 255), zombie=(200, 50, 50),
           path=(255, 220, 80), text=(235, 235, 235), risk=(200, 40, 40))


def _risk_color(base, risk):
    """Blend a floor colour towards red proportional to the cell's risk."""
    t = min(risk / 2.0, 1.0) * 0.55
    return tuple(int(base[i] * (1 - t) + COL["risk"][i] * t)
                 for i in range(3))


def run(seed=None, zombie_brain=None, qlearner=None, human_planner=None):
    pygame.init()
    n, px = config.GRID_SIZE, config.CELL_PX
    screen = pygame.display.set_mode((n * px, n * px + 50))
    brain = zombie_brain or config.ZOMBIE_BRAIN
    planner = human_planner or config.HUMAN_PLANNER
    pygame.display.set_caption(
        f"Zombie Apocalypse - human:{planner} zombies:{brain}")
    font = pygame.font.SysFont(None, 26)
    clock = pygame.time.Clock()
    game = Game(seed, zombie_brain=zombie_brain, qlearner=qlearner,
                human_planner=human_planner)
    auto, last = False, 0

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    game.step()
                elif e.key == pygame.K_a:
                    auto = not auto
                elif e.key == pygame.K_r:
                    game.reset()
                elif e.key == pygame.K_ESCAPE:
                    pygame.quit(); return
        if auto and pygame.time.get_ticks() - last > config.FPS_DELAY_MS:
            game.step(); last = pygame.time.get_ticks()

        screen.fill(COL["bg"])
        for x in range(n):
            for y in range(n):
                c = (x, y)
                if c in game.grid.walls:
                    col = COL["wall"]
                elif c in game.grid.safe_cells:
                    col = COL["safe"]
                else:
                    col = _risk_color(COL["floor"],
                                      game.riskmap.risk_of(c))
                pygame.draw.rect(screen, col, (x*px+1, y*px+1, px-2, px-2))
        for c in game.path[1:]:
            pygame.draw.circle(screen, COL["path"],
                               (c[0]*px+px//2, c[1]*px+px//2), 4)
        for z in game.zombies:
            pygame.draw.rect(screen, COL["zombie"],
                             (z[0]*px+6, z[1]*px+6, px-12, px-12))
        pygame.draw.circle(screen, COL["human"],
                           (game.human[0]*px+px//2,
                            game.human[1]*px+px//2), px//2-5)
        msg = f"Turn {game.turn} | {game.message}"
        screen.blit(font.render(msg, True, COL["text"]), (8, n*px+6))
        hint = ("SPACE=step  A=auto  R=restart  ESC=quit   "
                f"(red tint = zombie risk field)")
        screen.blit(font.render(hint, True, COL["text"]), (8, n*px+28))
        pygame.display.flip()
        clock.tick(60)
