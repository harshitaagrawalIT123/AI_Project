import pygame
import config
from game import Game

COL = dict(bg=(25, 25, 30), floor=(45, 48, 55), wall=(90, 90, 100),
           safe=(40, 140, 70), human=(60, 150, 255), zombie=(200, 50, 50),
           path=(255, 220, 80), text=(235, 235, 235), risk=(200, 40, 40),
           panel=(34, 34, 42), card=(46, 46, 58), dim=(150, 150, 160),
           legal=(190, 200, 225), hover=(255, 255, 255))

# colour of a recommendation depends on how dangerous its move is
RISK_COL = {"LOW": (80, 200, 110), "MEDIUM": (240, 200, 60),
            "HIGH": (240, 130, 50), "DEADLY": (230, 60, 60)}

PANEL_W = 300
MOVE_KEYS = {pygame.K_UP: (0, -1), pygame.K_DOWN: (0, 1),
             pygame.K_LEFT: (-1, 0), pygame.K_RIGHT: (1, 0),
             pygame.K_x: (0, 0)}                      # X = stay in place
REC_KEYS = [pygame.K_1, pygame.K_2, pygame.K_3]


def _risk_color(base, risk):
    """Blend a floor colour towards red proportional to the cell's risk."""
    t = min(risk / 2.0, 1.0) * 0.55
    return tuple(int(base[i] * (1 - t) + COL["risk"][i] * t)
                 for i in range(3))


def _fit(font, text, width):
    """Trim text with '..' so it fits in `width` pixels."""
    while font.size(text)[0] > width and len(text) > 3:
        text = text[:-3] + ".."
    return text


def _draw_panel(screen, fonts, game, x0, height):
    """Right-hand panel: recommendations (manual mode) or help (auto)."""
    small, norm = fonts
    pygame.draw.rect(screen, COL["panel"], (x0, 0, PANEL_W, height))
    x, y, w = x0 + 12, 10, PANEL_W - 24

    def line(text, font=small, color=COL["text"], dy=None):
        nonlocal y
        screen.blit(font.render(text, True, color), (x, y))
        y += dy if dy is not None else font.get_linesize() + 2

    mode = "MANUAL (you decide)" if game.manual else "AUTO (AI plays)"
    line(f"Mode: {mode}", norm,
         RISK_COL["LOW"] if game.manual else COL["dim"], dy=26)

    if not game.manual:
        for t in ("Press M for MANUAL mode:",
                  "the AI recommends moves and",
                  "YOU choose - follow one of",
                  "them or play your own."):
            line(t, small, COL["dim"])
        y += 8
    else:
        line("AI recommendations", norm, dy=24)
        if game.status != "playing":
            line("Game over - press R", small, COL["dim"])
        for r in game.recommendations if game.status == "playing" else []:
            color = RISK_COL[r.risk_level]
            card_h = 66
            pygame.draw.rect(screen, COL["card"], (x - 6, y, w + 12, card_h),
                             border_radius=6)
            pygame.draw.rect(screen, color, (x - 6, y, 4, card_h),
                             border_radius=2)
            cy = y + 6
            title = f"[{r.rank}] {r.direction} -> ({r.cell[0]},{r.cell[1]})"
            screen.blit(norm.render(title, True, COL["text"]), (x + 4, cy))
            tw = norm.size(r.risk_level)[0]
            screen.blit(norm.render(r.risk_level, True, color),
                        (x + w - tw, cy))
            cy += 21
            screen.blit(small.render(
                _fit(small, "by " + " + ".join(r.tags), w - 8), True,
                COL["dim"]), (x + 4, cy))
            cy += 18
            screen.blit(small.render(_fit(small, r.summary, w - 8), True,
                                     COL["text"]), (x + 4, cy))
            y += card_h + 8
        y += 4
        line("Your turn - any of these:", small, COL["dim"])
        for t in ("click a highlighted cell",
                  "arrow keys (X = stay)",
                  "1/2/3 = take that AI pick",
                  "SPACE/ENTER = take AI pick #1"):
            line(t, small, COL["text"])
        y += 6
        line(f"AI picks followed: {game.stats['ai']}", small, COL["dim"])
        line(f"Your own moves: {game.stats['own']}", small, COL["dim"])

    y += 8
    line("Legend", small, COL["dim"])
    for t in ("blue circle = you, red = zombie",
              "green = safe zone, grey = wall",
              "yellow dots = AI planner route",
              "red tint = zombie risk field"):
        line(t, small, COL["dim"])
    if game.manual:
        line("hollow ring = legal move", small, COL["dim"])
        line("numbered badge = AI pick", small, COL["dim"])


def run(seed=None, zombie_brain=None, qlearner=None, human_planner=None,
        manual=False):
    pygame.init()
    n, px = config.GRID_SIZE, config.CELL_PX
    board = n * px
    screen = pygame.display.set_mode((board + PANEL_W, board + 50))
    brain = zombie_brain or config.ZOMBIE_BRAIN
    planner = human_planner or config.HUMAN_PLANNER
    pygame.display.set_caption(
        f"Zombie Apocalypse - human:{planner} zombies:{brain}")
    font = pygame.font.SysFont(None, 26)
    fonts = (pygame.font.SysFont(None, 20), pygame.font.SysFont(None, 24))
    badge_font = pygame.font.SysFont(None, 26)
    clock = pygame.time.Clock()
    game = Game(seed, zombie_brain=zombie_brain, qlearner=qlearner,
                human_planner=human_planner, manual=manual)
    auto, last = False, 0

    def cell_at(pos):
        x, y = pos[0] // px, pos[1] // px
        if pos[0] < board and pos[1] < board and 0 <= x < n and 0 <= y < n:
            return (x, y)
        return None

    def play_cell(cell):
        if game.manual and cell is not None:
            game.play_human_move(cell)

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                play_cell(cell_at(e.pos))
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); return
                elif e.key == pygame.K_r:
                    game.reset()
                elif e.key == pygame.K_m:
                    auto = False
                    game.set_manual(not game.manual)
                elif game.manual:
                    recs = game.recommendations
                    if e.key in MOVE_KEYS:
                        dx, dy = MOVE_KEYS[e.key]
                        play_cell((game.human[0] + dx, game.human[1] + dy))
                    elif e.key in REC_KEYS:
                        i = REC_KEYS.index(e.key)
                        if i < len(recs):
                            play_cell(recs[i].cell)
                    elif e.key in (pygame.K_SPACE, pygame.K_RETURN) and recs:
                        play_cell(recs[0].cell)
                else:
                    if e.key == pygame.K_SPACE:
                        game.step()
                    elif e.key == pygame.K_a:
                        auto = not auto
        if auto and not game.manual \
                and pygame.time.get_ticks() - last > config.FPS_DELAY_MS:
            game.step(); last = pygame.time.get_ticks()

        # ---------------- drawing ----------------
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

        if game.manual and game.status == "playing":
            hover = cell_at(pygame.mouse.get_pos())
            rec_by_cell = {r.cell: r for r in game.recommendations}
            for c in game.legal_moves():
                centre = (c[0]*px + px//2, c[1]*px + px//2)
                if c in rec_by_cell:
                    r = rec_by_cell[c]
                    color = RISK_COL[r.risk_level]
                    pygame.draw.rect(screen, color,
                                     (c[0]*px+2, c[1]*px+2, px-4, px-4), 3,
                                     border_radius=6)
                    pygame.draw.circle(screen, color, centre, 11)
                    label = badge_font.render(str(r.rank), True, (20, 20, 20))
                    screen.blit(label, label.get_rect(center=centre))
                else:
                    pygame.draw.circle(screen, COL["legal"], centre,
                                       px//2 - 8, 2)
                if c == hover:
                    pygame.draw.rect(screen, COL["hover"],
                                     (c[0]*px+1, c[1]*px+1, px-2, px-2), 2,
                                     border_radius=6)

        for z in game.zombies:
            pygame.draw.rect(screen, COL["zombie"],
                             (z[0]*px+6, z[1]*px+6, px-12, px-12))
        pygame.draw.circle(screen, COL["human"],
                           (game.human[0]*px+px//2,
                            game.human[1]*px+px//2), px//2-5)

        _draw_panel(screen, fonts, game, board, board + 50)

        msg = f"Turn {game.turn} | {game.message}"
        screen.blit(font.render(_fit(font, msg, board + PANEL_W - 16), True,
                                COL["text"]), (8, board+6))
        if game.manual:
            hint = "MANUAL: click / arrows / 1-3   M=auto  R=restart  ESC=quit"
        else:
            hint = "SPACE=step  A=auto  M=manual  R=restart  ESC=quit"
        screen.blit(font.render(hint, True, COL["text"]), (8, board+28))
        pygame.display.flip()
        clock.tick(60)
