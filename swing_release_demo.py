
import math
import sys

import pygame
import pymunk


# ============================================================
# Bottle-Dynamics: Swing and Release Prototype
# ============================================================

WIDTH = 1000
HEIGHT = 700
FPS = 60
SUBSTEPS = 5

# Rendering scale: pixels per metre.
SCALE = 220.0

WORLD_WIDTH = WIDTH / SCALE
GROUND_Y = 0.22

# Initial prototype geometry.
# Water dynamics have NOT been added yet.
BOTTLE_WIDTH = 0.065
BOTTLE_HEIGHT = 0.260
BOTTLE_MASS = 0.175

GRAVITY = 9.81

PIVOT_POSITION = (1.20, 1.05)

# Bounded torque applied while the bottle is attached.
MAX_TORQUE = 0.12

# Collision types.
GROUND_COLLISION = 1
BOTTLE_COLLISION = 2


# ============================================================
# Pygame setup
# ============================================================

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "Bottle-Dynamics | Swing and Release"
)

clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 16)


def to_screen(point):
    """Convert world metres to screen pixels."""
    return (
        round(point.x * SCALE),
        round(HEIGHT - point.y * SCALE),
    )


# ============================================================
# Physics world
# ============================================================

space = pymunk.Space()
space.gravity = (0.0, -GRAVITY)
space.damping = 1.0
space.iterations = 30

ground = pymunk.Segment(
    space.static_body,
    (0.0, GROUND_Y),
    (WORLD_WIDTH, GROUND_Y),
    0.015,
)

ground.friction = 0.9
ground.elasticity = 0.02
ground.collision_type = GROUND_COLLISION

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
    GROUND_COLLISION,
    BOTTLE_COLLISION,
    begin=on_ground_begin,
    separate=on_ground_separate,
)


# ============================================================
# Bottle and pivot
# ============================================================

def create_bottle():
    """Create a bottle proxy attached at its neck."""

    moment = pymunk.moment_for_box(
        BOTTLE_MASS,
        (BOTTLE_WIDTH, BOTTLE_HEIGHT),
    )

    body = pymunk.Body(BOTTLE_MASS, moment)

    pivot_x, pivot_y = PIVOT_POSITION

    # The bottle's top end coincides with the pivot.
    body.position = (
        pivot_x,
        pivot_y - BOTTLE_HEIGHT / 2,
    )

    body.angle = 0.0

    shape = pymunk.Poly.create_box(
        body,
        (BOTTLE_WIDTH, BOTTLE_HEIGHT),
    )

    shape.friction = 0.9
    shape.elasticity = 0.02
    shape.collision_type = BOTTLE_COLLISION

    # Constrain the bottle's neck to the fixed world pivot.
    joint = pymunk.PivotJoint(
        space.static_body,
        body,
        PIVOT_POSITION,
    )

    joint.collide_bodies = False

    space.add(body, shape, joint)

    return body, shape, joint


bottle, bottle_shape, pivot_joint = create_bottle()

released = False
release_count = 0
applied_torque = 0.0
status = "Attached: hold LEFT or RIGHT to swing"


# ============================================================
# Controls
# ============================================================

def reset_simulation():
    """Reset the bottle and recreate the pivot constraint."""

    global bottle, bottle_shape, pivot_joint
    global released, ground_contact
    global applied_torque, status

    # Remove constraints before the body.
    if pivot_joint is not None:
        space.remove(pivot_joint)

    space.remove(bottle_shape, bottle)

    bottle, bottle_shape, pivot_joint = create_bottle()

    released = False
    ground_contact = False
    applied_torque = 0.0
    status = "Attached: hold LEFT or RIGHT to swing"


def release_bottle():
    """Release the bottle without adding an artificial impulse."""

    global pivot_joint, released, release_count, status

    if released:
        return

    # Removing the joint lets the bottle retain its current
    # linear and angular velocities.
    space.remove(pivot_joint)
    pivot_joint = None

    released = True
    release_count += 1

    status = "Released: bottle is now in free flight"


# ============================================================
# Rendering
# ============================================================

def draw_bottle():
    vertices = [
        bottle.local_to_world(vertex)
        for vertex in bottle_shape.get_vertices()
    ]

    points = [to_screen(vertex) for vertex in vertices]

    pygame.draw.polygon(
        screen,
        (65, 165, 225),
        points,
    )

    pygame.draw.polygon(
        screen,
        (225, 240, 250),
        points,
        width=2,
    )

    # Decorative cap at the bottle's neck.
    cap_left = bottle.local_to_world(
        pymunk.Vec2d(-BOTTLE_WIDTH * 0.28, BOTTLE_HEIGHT / 2)
    )
    cap_right = bottle.local_to_world(
        pymunk.Vec2d(BOTTLE_WIDTH * 0.28, BOTTLE_HEIGHT / 2)
    )

    pygame.draw.line(
        screen,
        (245, 190, 75),
        to_screen(cap_left),
        to_screen(cap_right),
        width=4,
    )

    # Decorative label.
    label_left = bottle.local_to_world(
        pymunk.Vec2d(-BOTTLE_WIDTH / 2, 0)
    )
    label_right = bottle.local_to_world(
        pymunk.Vec2d(BOTTLE_WIDTH / 2, 0)
    )

    pygame.draw.line(
        screen,
        (225, 245, 255),
        to_screen(label_left),
        to_screen(label_right),
        width=3,
    )


def draw_scene():
    screen.fill((23, 29, 42))

    # Ground and platform.
    ground_screen_y = HEIGHT - round(GROUND_Y * SCALE)

    pygame.draw.rect(
        screen,
        (105, 120, 142),
        (
            0,
            ground_screen_y,
            WIDTH,
            HEIGHT - ground_screen_y,
        ),
    )

    pygame.draw.line(
        screen,
        (205, 215, 230),
        (0, ground_screen_y),
        (WIDTH, ground_screen_y),
        width=3,
    )

    # Fixed pivot marker.
    pivot_screen = to_screen(
        pymunk.Vec2d(*PIVOT_POSITION)
    )

    pygame.draw.circle(
        screen,
        (245, 190, 75),
        pivot_screen,
        6,
    )

    if not released:
        # Illustrate the neck's connection to the pivot.
        neck_point = bottle.local_to_world(
            pymunk.Vec2d(0, BOTTLE_HEIGHT / 2)
        )

        pygame.draw.line(
            screen,
            (245, 190, 75),
            pivot_screen,
            to_screen(neck_point),
            width=2,
        )

    draw_bottle()

    angle_degrees = math.degrees(bottle.angle)
    angular_velocity = bottle.angular_velocity
    speed = bottle.velocity.length

    lines = [
        "BOTTLE-DYNAMICS | SWING AND RELEASE",
        "HOLD LEFT/RIGHT: apply torque",
        "SPACE: release    R: reset    ESC: quit",
        f"Mode: {'FREE FLIGHT' if released else 'PIVOT ATTACHED'}",
        f"Applied torque: {applied_torque:.3f} N m",
        f"Angle: {angle_degrees:.1f} deg",
        f"Angular velocity: {angular_velocity:.2f} rad/s",
        f"Linear speed: {speed:.2f} m/s",
        f"Ground contact: {ground_contact}",
        f"Release attempts: {release_count}",
        f"Status: {status}",
    ]

    for index, line in enumerate(lines):
        text = font.render(
            line,
            True,
            (235, 240, 245),
        )
        screen.blit(text, (18, 15 + index * 25))

    pygame.display.flip()


# ============================================================
# Main simulation loop
# ============================================================

def main():
    global applied_torque, status

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
                    release_bottle()

                elif event.key == pygame.K_r:
                    reset_simulation()

        keys = pygame.key.get_pressed()

        applied_torque = 0.0

        if not released:
            if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
                applied_torque = -MAX_TORQUE

            elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
                applied_torque = MAX_TORQUE

        # Advance the physics at a fixed frame duration
        # using multiple smaller integration steps.
        frame_dt = 1.0 / FPS
        substep_dt = frame_dt / SUBSTEPS

        for _ in range(SUBSTEPS):
            if not released:
                # Pymunk force/torque accumulators are reset
                # after a step, so set torque every substep.
                bottle.torque = applied_torque
            else:
                bottle.torque = 0.0

            space.step(substep_dt)

        if released and ground_contact:
            status = "Ground contact: inspect landing stability"
        elif released:
            status = "Released: bottle is in free flight"
        elif abs(applied_torque) > 0:
            status = "Applying torque: build swing momentum"
        else:
            status = "Attached: hold LEFT or RIGHT to swing"

        draw_scene()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
