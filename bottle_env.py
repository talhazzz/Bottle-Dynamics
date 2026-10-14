"""A small Gymnasium environment for learning to flip an empty bottle."""

import math

import gymnasium as gym
import numpy as np
import pymunk
from gymnasium import spaces
from gymnasium.envs.registration import register, registry


class BottleSwingEnv(gym.Env):
    """The agent controls a swinging bottle and chooses when to release it."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # World units are metres, kilograms, and seconds.
    WORLD_WIDTH = 1000 / 220
    GRAVITY = 9.81
    GROUND_Y = 0.22
    PIVOT_X, PIVOT_Y = 1.20, 2.05
    BOTTLE_WIDTH, BOTTLE_HEIGHT = 0.075, 0.22
    BOTTLE_MASS = 0.12  # Assumed mass of an empty reusable plastic bottle.
    MAX_TORQUE = 0.06
    PHYSICS_DT = 1 / 120
    ACTION_REPEAT = 4

    MAX_SWING_TIME = 1.8
    MAX_FLIGHT_TIME = 1.5
    MIN_SWING_TIME = 0.25
    MIN_SWING_ANGLE = math.radians(35)
    MIN_RELEASE_OMEGA = 1.0

    TARGET_X = 2.25
    TARGET_TOLERANCE = 0.40
    SUCCESS_ANGLE = math.radians(12)
    SIDE_ANGLE = math.radians(45)
    SUCCESS_OMEGA = 1.0
    SUCCESS_SPEED = 0.25
    STABLE_TIME_REQUIRED = 0.35
    MIN_NET_ROTATION = math.radians(300)
    MIN_FLIGHT_ROTATION = math.radians(180)

    def __init__(self, render_mode=None):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32)
        self._pygame = self._screen = self._clock = self._font = None
        self.window_closed = False
        self._episode_over = True

    def _build_world(self):
        """Create the bottle, pivot, and one continuous ground surface."""
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -self.GRAVITY)
        self.space.iterations = 30
        self.phase = "swing"
        self.swing_time = self.flight_time = 0.0
        self.net_rotation = self.swing_travel = self.flight_travel = 0.0
        self.max_swing_excursion = self.stable_time = self.landing_elapsed = 0.0
        self.ground_contact = self.first_contact = False
        self.swing_ground_hit = False
        self.first_contact_angle_error = self.first_contact_omega = None
        self.first_contact_speed = self.first_contact_position = None
        self.first_contact_flight_rotation = self.first_contact_net_rotation = None
        self.release_angle = self.release_omega = None
        self.outcome = "in_progress"

        # The target is only a painted zone; the whole floor has collisions.
        ground = pymunk.Segment(
            self.space.static_body, (0, self.GROUND_Y),
            (self.WORLD_WIDTH, self.GROUND_Y), 0.02,
        )
        ground.friction, ground.elasticity = 0.95, 0.01
        ground.collision_type = 1
        self.space.add(ground)
        self.space.on_collision(1, 2, begin=self._on_ground_begin,
                                separate=self._on_ground_separate)

        moment = pymunk.moment_for_box(
            self.BOTTLE_MASS, (self.BOTTLE_WIDTH, self.BOTTLE_HEIGHT)
        )
        self.body = pymunk.Body(self.BOTTLE_MASS, moment)
        self.body.position = (self.PIVOT_X, self.PIVOT_Y - self.BOTTLE_HEIGHT / 2)
        self.body.angle = 0.0
        self.shape = pymunk.Poly.create_box(
            self.body, (self.BOTTLE_WIDTH, self.BOTTLE_HEIGHT)
        )
        self.shape.friction, self.shape.elasticity = 0.9, 0.01
        self.shape.collision_type = 2
        self.pivot = pymunk.PivotJoint(
            self.space.static_body, self.body, (self.PIVOT_X, self.PIVOT_Y)
        )
        self.pivot.collide_bodies = False
        self.space.add(self.body, self.shape, self.pivot)
        self._last_angle = self.body.angle

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._build_world()
        self._episode_over = False
        return self._observation(), self._info()

    def _on_ground_begin(self, arbiter, space, data):
        self.ground_contact = True
        if self.phase == "swing":
            # A collision before release means the swing hit the floor.
            self.swing_ground_hit = True
        elif not self.first_contact:
            # Only flight contact counts as a landing attempt.
            self.first_contact = True
            self.first_contact_angle_error = self._upright_error()
            self.first_contact_omega = float(self.body.angular_velocity)
            self.first_contact_speed = float(self.body.velocity.length)
            self.first_contact_position = tuple(map(float, self.body.position))
        return True

    def _on_ground_separate(self, arbiter, space, data):
        self.ground_contact = False

    def _upright_error(self):
        angle = (self.body.angle + math.pi) % (2 * math.pi) - math.pi
        return abs(angle)

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
            self.max_swing_excursion = max(self.max_swing_excursion, abs(self.net_rotation))
        else:
            self.flight_travel += abs(delta)
        self._last_angle = angle

    def _swing_potential(self):
        swing = min(self.max_swing_excursion / math.radians(90), 1.0)
        speed = min(abs(self.body.angular_velocity) / 12.0, 1.0)
        return 0.5 * swing + 0.25 * speed

    def _observation(self):
        x, y = self.body.position
        vx, vy = self.body.velocity
        return np.asarray([
            math.sin(self.body.angle), math.cos(self.body.angle),
            np.clip(self.body.angular_velocity / 20.0, -1, 1),
            np.clip(2 * x / self.WORLD_WIDTH - 1, -1, 1),
            np.clip(2 * y / 3.0 - 1, -1, 1),
            np.clip(vx / 8.0, -1, 1), np.clip(vy / 8.0, -1, 1),
            -1.0 if self.phase == "swing" else 1.0,
            np.clip(self.swing_time / self.MAX_SWING_TIME, 0, 1),
            np.clip(self.max_swing_excursion / math.pi, 0, 1),
            np.clip(self.flight_travel / (2 * math.pi), 0, 1),
            np.clip(self.net_rotation / (2 * math.pi), -1, 1),
            float(self.release_ready), float(self.ground_contact),
            np.clip(self.flight_time / self.MAX_FLIGHT_TIME, 0, 1),
            np.clip(self.stable_time / self.STABLE_TIME_REQUIRED, 0, 1),
            np.clip((self.TARGET_X - x) / self.TARGET_TOLERANCE, -1, 1),
        ], dtype=np.float32)

    def _info(self):
        return {
            "phase": self.phase,
            "outcome": self.outcome,
            "is_success": self.outcome == "success",
            "ground_contact": self.ground_contact,
            "release_ready": self.release_ready,
            "landing_angle_error_deg": math.degrees(self._upright_error()),
            "first_contact_angle_error_deg": (
                math.degrees(self.first_contact_angle_error)
                if self.first_contact_angle_error is not None else None
            ),
            "first_contact_position_m": self.first_contact_position,
            "first_contact_angular_velocity_rad_s": self.first_contact_omega,
            "first_contact_speed_m_s": self.first_contact_speed,
            "flight_rotation_deg": math.degrees(
                self.first_contact_flight_rotation
                if self.first_contact_flight_rotation is not None
                else self.flight_travel
            ),
            "net_rotation_deg": math.degrees(self.net_rotation),
            "release_angle_deg": (
                math.degrees(self.release_angle) if self.release_angle is not None else None
            ),
            "release_angular_velocity_rad_s": self.release_omega,
            "final_position_m": tuple(map(float, self.body.position)),
        }

    def _finish_flight(self):
        """Simulate the released bottle until it settles or the episode ends."""
        reward = 0.0
        contact_reward_paid = False
        while self.flight_time < self.MAX_FLIGHT_TIME:
            self.body.torque = 0.0
            self.space.step(self.PHYSICS_DT)
            self.flight_time += self.PHYSICS_DT
            self._track_rotation()

            if self.first_contact and self.first_contact_flight_rotation is None:
                self.first_contact_flight_rotation = self.flight_travel
                self.first_contact_net_rotation = self.net_rotation

            if self.first_contact and not contact_reward_paid:
                angle_error = self.first_contact_angle_error
                contact_x = self.first_contact_position[0]
                rotation = self.first_contact_flight_rotation or 0.0
                angle_score = math.exp(-angle_error / math.radians(30))
                target_score = math.exp(-abs(contact_x - self.TARGET_X) / self.TARGET_TOLERANCE)
                rotation_score = min(rotation / (2 * math.pi), 1.0)
                reward += 3.0 * angle_score + 2.0 * target_score + 4.0 * rotation_score - 5.0
                contact_reward_paid = True

            if self.render_mode == "human" and round(self.flight_time / self.PHYSICS_DT) % self.ACTION_REPEAT == 0:
                self.render()
                if self.window_closed:
                    self.outcome = "window_closed"
                    return reward, True, False

            if self.first_contact:
                self.landing_elapsed += self.PHYSICS_DT
                if self.ground_contact:
                    stable = (
                        self._upright_error() <= self.SUCCESS_ANGLE
                        and abs(self.body.angular_velocity) <= self.SUCCESS_OMEGA
                        and self.body.velocity.length <= self.SUCCESS_SPEED
                    )
                    self.stable_time = self.stable_time + self.PHYSICS_DT if stable else 0.0
                    if self.stable_time >= self.STABLE_TIME_REQUIRED:
                        in_target = abs(self.body.position.x - self.TARGET_X) <= self.TARGET_TOLERANCE
                        enough_rotation = (
                            self.first_contact_net_rotation is not None
                            and self.first_contact_flight_rotation is not None
                            and abs(self.first_contact_net_rotation) >= self.MIN_NET_ROTATION
                            and self.first_contact_flight_rotation >= self.MIN_FLIGHT_ROTATION
                        )
                        if in_target and enough_rotation:
                            self.outcome, reward = "success", reward + 25.0
                        elif not in_target:
                            self.outcome = "stable_landing_outside_target"
                        else:
                            self.outcome = "rotation_incomplete"
                        return reward, True, False
                else:
                    self.stable_time = 0.0
                if self.landing_elapsed >= 0.8:
                    self.outcome = (
                        "side_landing"
                        if self.first_contact_angle_error > self.SIDE_ANGLE
                        else "unstable_landing"
                    )
                    return reward, True, False

            if (self.body.position.x < -0.2 or self.body.position.x > self.WORLD_WIDTH + 0.2
                    or self.body.position.y < -0.2):
                self.outcome = "fell_out_of_bounds"
                return reward - 5.0, True, False

        self.outcome = "flight_timeout"
        if not contact_reward_paid:
            reward -= 5.0
        return reward, False, True

    def step(self, action):
        if self._episode_over:
            raise RuntimeError("Episode ended; call reset() before step().")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action.shape}")
        torque_cmd, release_cmd = np.clip(action, -1.0, 1.0)
        old_potential = self._swing_potential()
        for _ in range(self.ACTION_REPEAT):
            self.body.torque = float(torque_cmd) * self.MAX_TORQUE
            self.space.step(self.PHYSICS_DT)
            self.swing_time += self.PHYSICS_DT
            self._track_rotation()
            if self.swing_ground_hit:
                break
        reward = self._swing_potential() - old_potential - 0.002 - 0.001 * float(torque_cmd) ** 2
        terminated = truncated = False

        if self.swing_ground_hit:
            self.outcome = "swing_hit_ground"
            reward -= 5.0
            terminated = True
        elif release_cmd > 0:
            if self.release_ready:
                self.space.remove(self.pivot)
                self.phase = "flight"
                self.release_angle = float(self.body.angle)
                self.release_omega = float(self.body.angular_velocity)
                reward += 0.1
                flight_reward, terminated, truncated = self._finish_flight()
                reward += flight_reward
            else:
                reward -= 0.02
        if self.phase == "swing" and self.swing_time >= self.MAX_SWING_TIME:
            self.outcome, terminated, reward = "failed_to_release", True, reward - 5.0

        if self.render_mode == "human" and not self.window_closed:
            self.render()
        self._episode_over = terminated or truncated
        return self._observation(), float(reward), terminated, truncated, self._info()

    def render(self):
        if self.render_mode != "human" or self.window_closed:
            return
        import pygame
        if self._pygame is None:
            pygame.init()
            self._pygame = pygame
            self.render_width, self.render_height, self.render_scale = 1000, 700, 220
            self._screen = pygame.display.set_mode((self.render_width, self.render_height))
            pygame.display.set_caption("Bottle-Dynamics")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("consolas", 17)
        pg, screen = self._pygame, self._screen
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.window_closed = True
                return

        screen.fill((23, 29, 42))
        ground_y = round(self.render_height - self.GROUND_Y * self.render_scale)
        pg.draw.rect(screen, (95, 108, 125), (0, ground_y, self.render_width, self.render_height - ground_y))
        left = round((self.TARGET_X - self.TARGET_TOLERANCE) * self.render_scale)
        right = round((self.TARGET_X + self.TARGET_TOLERANCE) * self.render_scale)
        pg.draw.rect(screen, (55, 170, 115), (left, ground_y - 7, right - left, 7))
        pg.draw.line(screen, (220, 230, 240), (0, ground_y), (self.render_width, ground_y), 3)

        def screen_point(point):
            return round(point.x * self.render_scale), round(self.render_height - point.y * self.render_scale)

        if self.phase == "swing":
            pivot = (round(self.PIVOT_X * self.render_scale), round(self.render_height - self.PIVOT_Y * self.render_scale))
            pg.draw.circle(screen, (245, 190, 75), pivot, 6)
        points = [screen_point(self.body.local_to_world(v)) for v in self.shape.get_vertices()]
        pg.draw.polygon(screen, (65, 165, 225), points)
        pg.draw.polygon(screen, (225, 240, 250), points, 2)
        cap = [self.body.local_to_world((-self.BOTTLE_WIDTH * .28, self.BOTTLE_HEIGHT / 2)),
               self.body.local_to_world((self.BOTTLE_WIDTH * .28, self.BOTTLE_HEIGHT / 2))]
        base = [self.body.local_to_world((-self.BOTTLE_WIDTH / 2, -self.BOTTLE_HEIGHT / 2)),
                self.body.local_to_world((self.BOTTLE_WIDTH / 2, -self.BOTTLE_HEIGHT / 2))]
        pg.draw.line(screen, (245, 190, 75), *map(screen_point, cap), 5)
        pg.draw.line(screen, (225, 245, 255), *map(screen_point, base), 4)

        angle = math.degrees((self.body.angle + math.pi) % (2 * math.pi) - math.pi)
        lines = ["BOTTLE-DYNAMICS", f"Phase: {self.phase}", f"Outcome: {self.outcome}",
                 f"Angle: {angle:.1f} deg", f"Angular speed: {self.body.angular_velocity:.2f} rad/s",
                 f"Ground contact: {self.ground_contact}", f"Release ready: {self.release_ready}",
                 "Green strip = target zone | ESC = close"]
        for i, line in enumerate(lines):
            screen.blit(self._font.render(line, True, (235, 240, 245)), (18, 15 + i * 24))
        pg.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = self._screen = self._clock = self._font = None


if "BottleDynamics-v0" not in registry:
    register(id="BottleDynamics-v0", entry_point="bottle_env:BottleSwingEnv")
