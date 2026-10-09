
import math
import sys

import pygame
import pymunk


# ============================================================
# Bottle-Dynamics: Interactive Physics Prototype
# ============================================================

WIDTH = 1000
HEIGHT = 650
FPS = 60
SUBSTEPS = 5

GROUND_Y = 75
GRAVITY = 900.0

BOTTLE_WIDTH = 24
BOTTLE_HEIGHT = 80
BOTTLE_MASS = 1.0

# Adjustable launch settings.
# These are initial values, not learned parameters.
DEFAULT_JX = 0.0
DEFAULT_JY = 360.0
DEFAULT_SPIN = 7.85  # radians per second

HORIZONTAL_STEP = 20.0
VERTICAL_STEP = 20.0
SPIN_STEP = 0.5

SUCCESS_ANGLE_DEG = 12.0
SUCCESS_ANGULAR_SPEED = 1.0
SUCCESS_LINEAR_SPEED = 25.0
REQUIRED_STABLE_TIME = 0.5


# ============================================================
# Pygame
# ============================================================

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bottle-Dynamics | Flip Test")

clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 17)


def to_screen(point):
    """Convert Pymunk coordinates to screen coordinates."""
    x, y = point
    return round(x), round(HEIGHT - y)


# ============================================================
# Physics world
# ============================================================

space = pymunk.Space()
space.gravity = (0.0, -GRAVITY)
space.damping = 0.999

ground = pymunk.Segment(
    space.static_body,
    (20, GROUND_Y),
    (WIDTH - 20, GROUND_Y),
    2,
)
ground.friction = 1.0
ground.elasticity = 0.0
ground.collision_type = 1

space.add(ground)

ground_contact = False


def on_ground_begin(arbiter, space, data):
    global ground_contact

    ground_contact = True
    return True


def on_ground_separate(arbiter, space, data):
    global ground_contact

    ground_contact = False


space.on_collision(
    1,
    2,
    begin=on_ground_begin,
    separate=on_ground_separate,
)


# ============================================================
# Bottle creation
# ============================================================

def create_bottle():
    """Create a rectangular rigid-body approximation of a bottle."""

    moment = pymunk.moment_for_box(
        BOTTLE_MASS,
        (BOTTLE_WIDTH, BOTTLE_HEIGHT),
    )

    body = pymunk.Body(BOTTLE_MASS, moment)

    # Start vertically upright, resting on its base.
    body.position = (
        WIDTH * 0.25,
        GROUND_Y + BOTTLE_HEIGHT / 2 + 2,
    )

    shape = pymunk.Poly.create_box(
        body,
        (BOTTLE_WIDTH, BOTTLE_HEIGHT),
    )

    shape.friction = 1.0
    shape.elasticity = 0.0
    shape.collision_type = 2

    space.add(body, shape)

    return body, shape


bottle, bottle_shape = create_bottle()

launch_jx = DEFAULT_JX
launch_jy = DEFAULT_JY
launch_spin = DEFAULT_SPIN

launched = False
attempt_count = 0
stable_frames = 0
result_message = "Ready - adjust settings and press SPACE"


# ============================================================
# Controls and state
# ============================================================

def reset_bottle():
    """Reset the bottle while preserving launch settings."""

    global bottle, bottle_shape
    global launched, ground_contact, stable_frames
    global result_message

    space.remove(bottle_shape, bottle)

    bottle, bottle_shape = create_bottle()

    launched = False
    ground_contact = False
    stable_frames = 0

    result_message = "Ready - adjust settings and press SPACE"


def reset_settings():
    """Restore the default launch parameters."""

    global launch_jx, launch_jy, launch_spin

    launch_jx = DEFAULT_JX
    launch_jy = DEFAULT_JY
    launch_spin = DEFAULT_SPIN

    reset_bottle()


def launch_bottle():
    """Apply a one-time translational and rotational launch."""

    global launched, attempt_count, result_message

    if launched:
        return

    # Apply the linear impulse through the bottle's center.
    # This controls horizontal and vertical launch motion.
    bottle.apply_impulse_at_local_point(
        (launch_jx, launch_jy),
        (0, 0),
    )

    # Represent the short launch flick as an initial angular
    # velocity. The equivalent angular impulse is I * delta_omega.
    bottle.angular_velocity += launch_spin

    launched = True
    attempt_count += 1
    result_message = "In flight"


def wrapped_angle_error():
    """Smallest absolute angular difference from upright."""

    error = (bottle.angle + math.pi) % (2 * math.pi) - math.pi
    return abs(math.degrees(error))


def update_landing_assessment():
    """Require a stable, upright landing rather than brief contact."""

    global stable_frames, result_message

    if not launched:
        return

    if not ground_contact:
        stable_frames = 0
        result_message = "In flight"
        return

    angle_error = wrapped_angle_error()
    angular_speed = abs(bottle.angular_velocity)
    linear_speed = bottle.velocity.length

    is_stable = (
        angle_error <= SUCCESS_ANGLE_DEG
        and angular_speed <= SUCCESS_ANGULAR_SPEED
        and linear_speed <= SUCCESS_LINEAR_SPEED
    )

    if is_stable:
        stable_frames += 1

        required_frames = round(FPS * REQUIRED_STABLE_TIME)

        if stable_frames >= required_frames:
            result_message = "SUCCESS - stable upright landing!"
        else:
            result_message = "Upright contact - checking stability"
    else:
        stable_frames = 0

        if angle_error <= SUCCESS_ANGLE_DEG:
            result_message = "Upright, but still moving"
        else:
            result_message = "Contact - bottle is not upright"


# ============================================================
# Rendering
# ============================================================

def draw_bottle():
    vertices = [
        bottle.local_to_world(vertex)
        for vertex in bottle_shape.get_vertices()
    ]

    points = [to_screen(point) for point in vertices]

    pygame.draw.polygon(screen, (65, 165, 225), points)
    pygame.draw.polygon(screen, (230, 243, 252), points, width=2)

    # Decorative cap. It is not a separate physical component.
    cap_left = bottle.local_to_world((-5, BOTTLE_HEIGHT / 2))
    cap_right = bottle.local_to_world((5, BOTTLE_HEIGHT / 2))

    pygame.draw.line(
        screen,
        (245, 190, 75),
        to_screen(cap_left),
        to_screen(cap_right),
        width=5,
    )

    # Decorative label.
    label_left = bottle.local_to_world((-BOTTLE_WIDTH / 2, 0))
    label_right = bottle.local_to_world((BOTTLE_WIDTH / 2, 0))

    pygame.draw.line(
        screen,
        (225, 245, 255),
        to_screen(label_left),
        to_screen(label_right),
        width=3,
    )


def draw_scene():
    screen.fill((23, 29, 42))

    platform_top = HEIGHT - GROUND_Y - 2

    pygame.draw.rect(
        screen,
        (105, 120, 142),
        (0, platform_top, WIDTH, HEIGHT - platform_top),
    )

    pygame.draw.line(
        screen,
        (205, 215, 230),
        (0, platform_top),
        (WIDTH, platform_top),
        width=3,
    )

    draw_bottle()

    angle_error = wrapped_angle_error()

    lines = [
        "BOTTLE-DYNAMICS | INTERACTIVE FLIP TEST",
        "ARROWS: adjust launch    Q/E: adjust spin",
        "SPACE: launch    R: reset attempt    C: reset settings",
        f"Launch Jx: {launch_jx:.1f}",
        f"Launch Jy: {launch_jy:.1f}",
        f"Initial spin: {launch_spin:.2f} rad/s",
        f"Angle from upright: {angle_error:.1f} deg",
        f"Angular velocity: {bottle.angular_velocity:.2f} rad/s",
        f"Linear velocity: {bottle.velocity.length:.1f}",
        f"Attempts: {attempt_count}",
        f"Status: {result_message}",
    ]

    for index, line in enumerate(lines):
        text = font.render(line, True, (235, 240, 245))
        screen.blit(text, (20, 16 + index * 25))

    pygame.display.flip()


# ============================================================
# Main loop
# ============================================================

def main():
    global launch_jx, launch_jy, launch_spin

    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    launch_bottle()

                elif event.key == pygame.K_r:
                    reset_bottle()

                elif event.key == pygame.K_c:
                    reset_settings()

                elif event.key == pygame.K_LEFT:
                    launch_jx = max(-180.0, launch_jx - HORIZONTAL_STEP)

                elif event.key == pygame.K_RIGHT:
                    launch_jx = min(180.0, launch_jx + HORIZONTAL_STEP)

                elif event.key == pygame.K_UP:
                    launch_jy = min(520.0, launch_jy + VERTICAL_STEP)

                elif event.key == pygame.K_DOWN:
                    launch_jy = max(180.0, launch_jy - VERTICAL_STEP)

                elif event.key == pygame.K_q:
                    launch_spin = max(-14.0, launch_spin - SPIN_STEP)

                elif event.key == pygame.K_e:
                    launch_spin = min(14.0, launch_spin + SPIN_STEP)

        dt = 1.0 / FPS

        for _ in range(SUBSTEPS):
            space.step(dt / SUBSTEPS)

        update_landing_assessment()
        draw_scene()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
