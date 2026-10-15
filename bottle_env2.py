"""Bottle-Dynamics v2: tabletop bottle flipping with a solid raised platform.

This is an independent environment. It does not modify bottle_env.py or the
BottleDynamics-v0 registration.

Geometry uses metres, kilograms, and seconds. Pymunk/Chipmunk performs rigid-body
polygon collision detection. A small physics timestep and thick platform reduce
collision tunnelling, but no discrete physics engine can guarantee perfect CCD
for arbitrarily high velocities.
"""

import math

import gymnasium as gym
import numpy as np
import pymunk
from gymnasium import spaces
from gymnasium.envs.registration import register, registry


class BottleSwingEnv2(gym.Env):
    """Swing, release, and land a rigid bottle upright on a raised tabletop."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # World dimensions and gravity (SI units).
    WORLD_WIDTH = 1000 / 220
    GRAVITY = 9.81
    LOWER_GROUND_Y = 0.22

    # The bottle hangs vertically with its neck at the pivot.
    PIVOT_X = 1.20
    PIVOT_Y = 2.05
    BOTTLE_WIDTH = 0.075
    BOTTLE_HEIGHT = 0.22
    BOTTLE_MASS = 0.12  # Approximate empty-bottle mass, not a measurement.

    # The raised tabletop top is level with the hanging bottle's base.
    PLATFORM_TOP_Y = PIVOT_Y - BOTTLE_HEIGHT  # 1.83 m
    PLATFORM_X = 2.25
    PLATFORM_HALF_WIDTH = 0.55
    PLATFORM_THICKNESS = 0.12
    TARGET_X_TOLERANCE = 0.40

    # Prefer landings near the centre of the raised tabletop. The score is 1.0
    # at the centre and decays smoothly as the landing moves away from it.
    CENTERING_SCALE = 0.20
    CENTERING_CONTACT_REWARD = 5.0
    CENTERING_SUCCESS_BONUS = 10.0

    # Control and numerical simulation.
    MAX_TORQUE = 0.06
    PHYSICS_DT = 1 / 480
    ACTION_REPEAT = 8  # 8 substeps = 1/60 second per agent action.
    MAX_SWING_TIME = 3.0
    MAX_FLIGHT_TIME = 1.5

    # Minimum release readiness conditions.
    MIN_SWING_TIME = 0.25
    MIN_SWING_ANGLE = math.radians(35)
    MIN_RELEASE_OMEGA = 1.0

    # Landing and rotation criteria.
    SUCCESS_ANGLE = math.radians(12)
    SUCCESS_ANGULAR_SPEED = 1.0
    SUCCESS_LINEAR_SPEED = 0.25
    REQUIRED_STABLE_TIME = 0.35
    MIN_NET_ROTATION = math.radians(300)
    MIN_FLIGHT_ROTATION = math.radians(180)
    LANDING_CHECK_TIME = 0.8

    MAX_OBS_OMEGA = 20.0
    MAX_OBS_SPEED = 8.0
    MAX_OBS_HEIGHT = 3.0

    PLATFORM_COLLISION = 1
    BOTTLE_COLLISION = 2
    LOWER_GROUND_COLLISION = 3

    def __init__(self, render_mode=None):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32)

        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self.window_closed = False
        self._episode_over = True

    # ------------------------------------------------------------------
    # World creation
    # ------------------------------------------------------------------

    def _build_world(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -self.GRAVITY)
        self.space.damping = 1.0
        self.space.iterations = 40
        self.space.collision_slop = 0.003

        self.phase = "swing"
        self.swing_time = 0.0
        self.flight_time = 0.0
        self.net_rotation = 0.0
        self.swing_travel = 0.0
        self.flight_travel = 0.0
        self.max_swing_excursion = 0.0
        self.stable_time = 0.0
        self.landing_elapsed = 0.0
        self.max_flight_height = self.body_start_height()

        self.platform_contact = False
        self.lower_ground_contact = False
        self.swing_platform_hit = False
        self.swing_ground_hit = False
        self.first_contact = False
        self.first_contact_surface = None
        self.first_contact_angle_error = None
        self.first_contact_omega = None
        self.first_contact_speed = None
        self.first_contact_position = None
        self.first_contact_net_rotation = None
        self.first_contact_flight_rotation = None
        self.release_angle = None
        self.release_omega = None
        self.release_velocity = None
        self.outcome = "in_progress"

        # Lower floor is a separate physical failure surface.
        lower_ground = pymunk.Segment(
            self.space.static_body,
            (-2.0, self.LOWER_GROUND_Y),
            (self.WORLD_WIDTH + 2.0, self.LOWER_GROUND_Y),
            0.04,
        )
        lower_ground.friction = 0.9
        lower_ground.elasticity = 0.01
        lower_ground.collision_type = self.LOWER_GROUND_COLLISION
        self.space.add(lower_ground)

        # Solid tabletop: a real static polygon, not just a painted strip.
        platform_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        platform_body.position = (
            self.PLATFORM_X,
            self.PLATFORM_TOP_Y - self.PLATFORM_THICKNESS / 2,
        )
        platform_shape = pymunk.Poly.create_box(
            platform_body,
            (2.0 * self.PLATFORM_HALF_WIDTH, self.PLATFORM_THICKNESS),
        )
        platform_shape.friction = 0.95
        platform_shape.elasticity = 0.01
        platform_shape.collision_type = self.PLATFORM_COLLISION
        self.space.add(platform_body, platform_shape)
        self.platform_body = platform_body
        self.platform_shape = platform_shape

        # Bottle collision callbacks distinguish tabletop from lower floor.
        self.space.on_collision(
            self.PLATFORM_COLLISION,
            self.BOTTLE_COLLISION,
            begin=self._on_platform_begin,
            separate=self._on_platform_separate,
        )
        self.space.on_collision(
            self.LOWER_GROUND_COLLISION,
            self.BOTTLE_COLLISION,
            begin=self._on_lower_ground_begin,
            separate=self._on_lower_ground_separate,
        )

        moment = pymunk.moment_for_box(
            self.BOTTLE_MASS,
            (self.BOTTLE_WIDTH, self.BOTTLE_HEIGHT),
        )
        self.body = pymunk.Body(self.BOTTLE_MASS, moment)
        self.body.position = (
            self.PIVOT_X,
            self.PIVOT_Y - self.BOTTLE_HEIGHT / 2,
        )
        self.body.angle = 0.0
        self.shape = pymunk.Poly.create_box(
            self.body,
            (self.BOTTLE_WIDTH, self.BOTTLE_HEIGHT),
        )
        self.shape.friction = 0.9
        self.shape.elasticity = 0.01
        self.shape.collision_type = self.BOTTLE_COLLISION

        self.pivot_joint = pymunk.PivotJoint(
            self.space.static_body,
            self.body,
            (self.PIVOT_X, self.PIVOT_Y),
        )
        self.pivot_joint.collide_bodies = False
        self.space.add(self.body, self.shape, self.pivot_joint)
        self._last_angle = self.body.angle

    def body_start_height(self):
        return self.PIVOT_Y - self.BOTTLE_HEIGHT / 2

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.window_closed = False
        self._build_world()
        self._episode_over = False
        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------
    # Collision callbacks
    # ------------------------------------------------------------------

    def _remember_first_contact(self, surface):
        if self.first_contact:
            return
        self.first_contact = True
        self.first_contact_surface = surface
        self.first_contact_angle_error = self._upright_error()
        self.first_contact_omega = float(self.body.angular_velocity)
        self.first_contact_speed = float(self.body.velocity.length)
        self.first_contact_position = tuple(map(float, self.body.position))
        self.first_contact_net_rotation = float(self.net_rotation)

    def _on_platform_begin(self, arbiter, space, data):
        self.platform_contact = True
        if self.phase == "swing":
            self.swing_platform_hit = True
        else:
            self._remember_first_contact("platform")
        return True

    def _on_platform_separate(self, arbiter, space, data):
        self.platform_contact = False

    def _on_lower_ground_begin(self, arbiter, space, data):
        self.lower_ground_contact = True
        if self.phase == "swing":
            self.swing_ground_hit = True
        else:
            self._remember_first_contact("lower_ground")
        return True

    def _on_lower_ground_separate(self, arbiter, space, data):
        self.lower_ground_contact = False

    # ------------------------------------------------------------------
    # State, rotation and reward helpers
    # ------------------------------------------------------------------

    def _upright_error(self):
        wrapped = (self.body.angle + math.pi) % (2 * math.pi) - math.pi
        return abs(wrapped)

    @property
    def release_ready(self):
        return (
            self.phase == "swing"
            and self.swing_time >= self.MIN_SWING_TIME
            and self.max_swing_excursion >= self.MIN_SWING_ANGLE
            and abs(self.body.angular_velocity) >= self.MIN_RELEASE_OMEGA
        )

    def _track_rotation(self):
        angle = self.body.angle
        delta = (angle - self._last_angle + math.pi) % (2 * math.pi) - math.pi
        self.net_rotation += delta
        if self.phase == "swing":
            self.swing_travel += abs(delta)
            self.max_swing_excursion = max(
                self.max_swing_excursion, abs(self.net_rotation)
            )
        else:
            self.flight_travel += abs(delta)
        self._last_angle = angle

    def _swing_potential(self):
        excursion = min(self.max_swing_excursion / (math.pi / 2), 1.0)
        rotational_speed = min(abs(self.body.angular_velocity) / 12.0, 1.0)
        return 0.5 * excursion + 0.25 * rotational_speed

    def _flight_potential(self):
        progress = min(self.flight_travel / self.MIN_FLIGHT_ROTATION, 1.0)
        angle_quality = math.exp(-self._upright_error() / math.radians(45))
        position_quality = math.exp(
            -abs(self.body.position.x - self.PLATFORM_X) / self.TARGET_X_TOLERANCE
        )
        speed_quality = math.exp(-abs(self.body.angular_velocity) / 10.0)
        height_quality = min(max(self.body.position.y / self.MAX_OBS_HEIGHT, 0.0), 1.0)
        return progress * (
            0.20 + 0.30 * angle_quality + 0.20 * position_quality
            + 0.10 * speed_quality + 0.10 * height_quality
        )

    def _base_height_error(self):
        """Approximate base-to-table height error for an almost-upright bottle."""
        bottom_center_y = self.body.position.y - (
            self.BOTTLE_HEIGHT / 2 * math.cos(self.body.angle)
        )
        return abs(bottom_center_y - self.PLATFORM_TOP_Y)

    def _center_score(self, x):
        """Smooth centre score: 1.0 at the platform centre, lower farther away."""
        offset = abs(float(x) - self.PLATFORM_X)
        return math.exp(-offset / self.CENTERING_SCALE)

    def _is_stably_upright_on_top(self):
        in_target = (
            abs(self.body.position.x - self.PLATFORM_X)
            <= self.TARGET_X_TOLERANCE
        )
        return (
            self.platform_contact
            and in_target
            and self._base_height_error() <= 0.035
            and self._upright_error() <= self.SUCCESS_ANGLE
            and abs(self.body.angular_velocity) <= self.SUCCESS_ANGULAR_SPEED
            and self.body.velocity.length <= self.SUCCESS_LINEAR_SPEED
        )

    def _get_observation(self):
        x, y = self.body.position
        vx, vy = self.body.velocity
        obs = np.asarray([
            math.sin(self.body.angle),
            math.cos(self.body.angle),
            np.clip(self.body.angular_velocity / self.MAX_OBS_OMEGA, -1, 1),
            np.clip(2.0 * x / self.WORLD_WIDTH - 1.0, -1, 1),
            np.clip(2.0 * y / self.MAX_OBS_HEIGHT - 1.0, -1, 1),
            np.clip(vx / self.MAX_OBS_SPEED, -1, 1),
            np.clip(vy / self.MAX_OBS_SPEED, -1, 1),
            -1.0 if self.phase == "swing" else 1.0,
            np.clip(self.swing_time / self.MAX_SWING_TIME, 0, 1),
            np.clip(self.max_swing_excursion / math.pi, 0, 1),
            np.clip(self.flight_travel / (2 * math.pi), 0, 1),
            np.clip(self.net_rotation / (2 * math.pi), -1, 1),
            float(self.release_ready),
            float(self.platform_contact),
            np.clip(self.flight_time / self.MAX_FLIGHT_TIME, 0, 1),
            np.clip(self.stable_time / self.REQUIRED_STABLE_TIME, 0, 1),
            np.clip((self.PLATFORM_X - x) / self.TARGET_X_TOLERANCE, -1, 1),
        ], dtype=np.float32)
        return obs

    def _get_info(self):
        current_center_offset = abs(float(self.body.position.x) - self.PLATFORM_X)
        first_contact_center_offset = (
            abs(float(self.first_contact_position[0]) - self.PLATFORM_X)
            if self.first_contact_position is not None else None
        )
        return {
            "phase": self.phase,
            "outcome": self.outcome,
            "is_success": self.outcome == "success",
            "release_ready": self.release_ready,
            "platform_contact": self.platform_contact,
            "lower_ground_contact": self.lower_ground_contact,
            "first_contact_surface": self.first_contact_surface,
            "landing_angle_error_deg": math.degrees(self._upright_error()),
            "net_rotation_deg": math.degrees(self.net_rotation),
            "flight_rotation_deg": math.degrees(self.flight_travel),
            "swing_time": self.swing_time,
            "flight_time": self.flight_time,
            "max_flight_height_m": self.max_flight_height,
            "platform_top_y_m": self.PLATFORM_TOP_Y,
            "release_angle_deg": (
                math.degrees(self.release_angle) if self.release_angle is not None else None
            ),
            "release_angular_velocity_rad_s": self.release_omega,
            "release_velocity_m_s": self.release_velocity,
            "first_contact_angle_error_deg": (
                math.degrees(self.first_contact_angle_error)
                if self.first_contact_angle_error is not None else None
            ),
            "first_contact_position_m": self.first_contact_position,
            "platform_center_offset_m": current_center_offset,
            "first_contact_center_offset_m": first_contact_center_offset,
            "platform_center_score": self._center_score(self.body.position.x),
        }

    # ------------------------------------------------------------------
    # Flight: gravity-driven, no agent control after release
    # ------------------------------------------------------------------

    def _finish_flight(self):
        reward = 0.0
        contact_reward_paid = False
        step_count = 0
        render_interval = max(1, round(1 / (self.PHYSICS_DT * self.metadata["render_fps"])))

        while self.flight_time < self.MAX_FLIGHT_TIME:
            self.body.torque = 0.0
            self.space.step(self.PHYSICS_DT)
            self.flight_time += self.PHYSICS_DT
            self._track_rotation()
            self.max_flight_height = max(self.max_flight_height, float(self.body.position.y))
            step_count += 1

            if self.first_contact and self.first_contact_flight_rotation is None:
                self.first_contact_flight_rotation = float(self.flight_travel)
                self.first_contact_net_rotation = float(self.net_rotation)

            if self.first_contact and not contact_reward_paid:
                angle_error = self.first_contact_angle_error or 0.0
                contact_x = self.first_contact_position[0]
                rotation = self.first_contact_flight_rotation or 0.0
                angle_score = math.exp(-angle_error / math.radians(30))
                # Reward accurate landings, with the strongest score at the
                # centre of the tabletop rather than merely anywhere on it.
                center_score = self._center_score(contact_x)
                rotation_score = min(rotation / (2 * math.pi), 1.0)
                reward += (
                    3.0 * angle_score
                    + self.CENTERING_CONTACT_REWARD * center_score
                    + 4.0 * rotation_score
                    - 3.0
                )
                contact_reward_paid = True

            # Hitting the lower floor is always a failure, even if the bottle
            # touched the tabletop edge earlier during the same trajectory.
            if self.lower_ground_contact:
                self.outcome = "hit_lower_ground"
                return reward - 8.0, True, False

            if self.first_contact:
                self.landing_elapsed += self.PHYSICS_DT
                if self._is_stably_upright_on_top():
                    self.stable_time += self.PHYSICS_DT
                else:
                    self.stable_time = 0.0

                if self.stable_time >= self.REQUIRED_STABLE_TIME:
                    enough_rotation = (
                        self.first_contact_net_rotation is not None
                        and self.first_contact_flight_rotation is not None
                        and abs(self.first_contact_net_rotation) >= self.MIN_NET_ROTATION
                        and self.first_contact_flight_rotation >= self.MIN_FLIGHT_ROTATION
                    )
                    if enough_rotation:
                        self.outcome = "success"
                        # Give an additional bonus for finishing close to the
                        # tabletop centre, using the final stable position.
                        center_bonus = (
                            self.CENTERING_SUCCESS_BONUS
                            * self._center_score(self.body.position.x)
                        )
                        return reward + 25.0 + center_bonus, True, False
                    self.outcome = "platform_landing_rotation_incomplete"
                    return reward - 8.0, True, False

                if self.landing_elapsed >= self.LANDING_CHECK_TIME:
                    self.outcome = (
                        "platform_landing_unstable"
                        if self.first_contact_surface == "platform"
                        else "missed_platform"
                    )
                    return reward - 8.0, True, False

            if (
                self.body.position.x < -0.2
                or self.body.position.x > self.WORLD_WIDTH + 0.2
                or self.body.position.y < self.LOWER_GROUND_Y - 0.2
            ):
                self.outcome = "fell_out_of_bounds"
                return reward - 8.0, True, False

            if self.render_mode == "human" and step_count % render_interval == 0:
                self.render()
                if self.window_closed:
                    self.outcome = "window_closed"
                    return reward, True, False

        self.outcome = "flight_timeout"
        if not contact_reward_paid:
            reward -= 5.0
        return reward, False, True

    # ------------------------------------------------------------------
    # Gymnasium step
    # ------------------------------------------------------------------

    def step(self, action):
        if self._episode_over:
            raise RuntimeError("Episode ended; call reset() before step().")

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action.shape}")
        torque_cmd, release_cmd = np.clip(action, -1.0, 1.0)

        reward = -0.002
        terminated = False
        truncated = False

        if self.phase == "swing":
            old_potential = self._swing_potential()
            torque = float(torque_cmd) * self.MAX_TORQUE

            for _ in range(self.ACTION_REPEAT):
                self.body.torque = torque
                self.space.step(self.PHYSICS_DT)
                self.swing_time += self.PHYSICS_DT
                self._track_rotation()
                if self.swing_platform_hit or self.swing_ground_hit:
                    break

            reward += self._swing_potential() - old_potential
            reward -= 0.001 * float(torque_cmd) ** 2

            if self.swing_platform_hit:
                self.outcome = "hit_platform_before_release"
                reward -= 8.0
                terminated = True
            elif self.swing_ground_hit:
                self.outcome = "hit_lower_ground_before_release"
                reward -= 8.0
                terminated = True
            elif float(release_cmd) > 0.0:
                if self.release_ready:
                    self.space.remove(self.pivot_joint)
                    self.pivot_joint = None
                    self.phase = "flight"
                    self.flight_time = 0.0
                    self.release_angle = float(self.body.angle)
                    self.release_omega = float(self.body.angular_velocity)
                    self.release_velocity = tuple(map(float, self.body.velocity))
                    reward += 0.1
                    flight_reward, terminated, truncated = self._finish_flight()
                    reward += flight_reward
                else:
                    reward -= 0.02

            if self.phase == "swing" and not terminated and self.swing_time >= self.MAX_SWING_TIME:
                self.outcome = "failed_to_release"
                reward -= 8.0
                terminated = True

        self._episode_over = terminated or truncated

        if self.render_mode == "human" and not self.window_closed:
            self.render()

        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Pygame visualisation
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode != "human" or self.window_closed:
            return None

        import pygame

        if self._pygame is None:
            pygame.init()
            self._pygame = pygame
            self.render_width, self.render_height = 1000, 700
            self.render_scale = 220.0
            self._screen = pygame.display.set_mode((self.render_width, self.render_height))
            pygame.display.set_caption("Bottle-Dynamics v2 | Raised Tabletop")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("consolas", 16)

        pg = self._pygame
        screen = self._screen
        scale = self.render_scale
        height = self.render_height

        for event in pg.event.get():
            if event.type == pg.QUIT or (
                event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
            ):
                self.window_closed = True
                return None

        def to_screen(point):
            return round(point.x * scale), round(height - point.y * scale)

        screen.fill((23, 29, 42))

        # Lower floor is visibly separate from the tabletop.
        lower_y = round(height - self.LOWER_GROUND_Y * scale)
        pg.draw.rect(screen, (75, 85, 102), (0, lower_y, self.render_width, height - lower_y))
        pg.draw.line(screen, (170, 180, 195), (0, lower_y), (self.render_width, lower_y), 3)

        # Draw the actual raised platform geometry, with a distinct target strip.
        platform_left = round((self.PLATFORM_X - self.PLATFORM_HALF_WIDTH) * scale)
        platform_right = round((self.PLATFORM_X + self.PLATFORM_HALF_WIDTH) * scale)
        platform_top = round(height - self.PLATFORM_TOP_Y * scale)
        platform_bottom = round(
            height - (self.PLATFORM_TOP_Y - self.PLATFORM_THICKNESS) * scale
        )
        pg.draw.rect(
            screen,
            (110, 125, 146),
            (platform_left, platform_top, platform_right - platform_left, platform_bottom - platform_top),
        )
        pg.draw.line(screen, (225, 235, 245), (platform_left, platform_top), (platform_right, platform_top), 3)

        target_left = round((self.PLATFORM_X - self.TARGET_X_TOLERANCE) * scale)
        target_right = round((self.PLATFORM_X + self.TARGET_X_TOLERANCE) * scale)
        pg.draw.line(screen, (70, 230, 155), (target_left, platform_top - 3), (target_right, platform_top - 3), 5)

        if self.phase == "swing":
            pivot = (round(self.PIVOT_X * scale), round(height - self.PIVOT_Y * scale))
            pg.draw.circle(screen, (245, 190, 75), pivot, 7)

        vertices = [self.body.local_to_world(v) for v in self.shape.get_vertices()]
        pg.draw.polygon(screen, (65, 165, 225), [to_screen(v) for v in vertices])
        pg.draw.polygon(screen, (225, 240, 250), [to_screen(v) for v in vertices], 2)

        cap = [
            self.body.local_to_world((-self.BOTTLE_WIDTH * 0.28, self.BOTTLE_HEIGHT / 2)),
            self.body.local_to_world((self.BOTTLE_WIDTH * 0.28, self.BOTTLE_HEIGHT / 2)),
        ]
        base = [
            self.body.local_to_world((-self.BOTTLE_WIDTH / 2, -self.BOTTLE_HEIGHT / 2)),
            self.body.local_to_world((self.BOTTLE_WIDTH / 2, -self.BOTTLE_HEIGHT / 2)),
        ]
        pg.draw.line(screen, (245, 190, 75), *map(to_screen, cap), 5)
        pg.draw.line(screen, (225, 245, 255), *map(to_screen, base), 4)

        angle_deg = math.degrees((self.body.angle + math.pi) % (2 * math.pi) - math.pi)
        lines = [
            "BOTTLE-DYNAMICS v2 | RAISED TABLETOP",
            f"Phase: {self.phase}",
            f"Outcome: {self.outcome}",
            f"Angle: {angle_deg:.1f} deg",
            f"Angular speed: {self.body.angular_velocity:.2f} rad/s",
            f"Linear speed: {self.body.velocity.length:.2f} m/s",
            f"Release ready: {self.release_ready}",
            f"Platform contact: {self.platform_contact}",
            f"Swing time: {self.swing_time:.2f} s",
            f"Flight time: {self.flight_time:.2f} s",
            f"Flight rotation: {math.degrees(self.flight_travel):.1f} deg",
            f"Flight apex: {self.max_flight_height:.2f} m",
            "Green line = target | ESC = close",
        ]
        for i, line in enumerate(lines):
            screen.blit(self._font.render(line, True, (235, 240, 245)), (18, 15 + i * 23))

        pg.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = self._screen = self._clock = self._font = None


# Keep the original BottleDynamics-v0 registration untouched.
if "BottleDynamics2-v0" not in registry:
    register(id="BottleDynamics2-v0", entry_point="bottle_env2:BottleSwingEnv2")
