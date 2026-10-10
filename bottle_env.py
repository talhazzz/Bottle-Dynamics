
import math

import gymnasium as gym
import numpy as np
import pymunk
from gymnasium import spaces


class BottleSwingEnv(gym.Env):
    """
    Baseline reinforcement-learning environment for bottle flipping.

    The bottle first swings around a fixed neck pivot. The agent applies
    torque and requests release. After release, the bottle flies freely
    and must complete a controlled rotation before landing upright.

    This initial version uses a rigid rectangular bottle proxy.
    It does not yet simulate water sloshing.
    """

    metadata = {
    "render_modes": ["human"],
    "render_fps": 30,
}

    # --------------------------------------------------------
    # Physical constants: SI units (metres, kilograms, seconds)
    # --------------------------------------------------------

    WORLD_WIDTH = 1000 / 220
    GRAVITY = 9.81
    GROUND_Y = 0.22

    PIVOT_X = 1.20
    PIVOT_Y = 2.05

    BOTTLE_WIDTH = 0.065
    BOTTLE_HEIGHT = 0.260
    BOTTLE_MASS = 0.175

    MAX_TORQUE = 0.06

    # Physics runs faster than the agent's decision frequency.
    PHYSICS_DT = 1 / 120
    ACTION_REPEAT = 4

    # Episode limits.
    MAX_SWING_TIME = 1.8
    MAX_FLIGHT_TIME = 1.5

    # A release is permitted only after a meaningful swing.
    MIN_SWING_TIME = 0.25
    MIN_SWING_ANGLE = math.radians(35)
    MIN_RELEASE_OMEGA = 1.0

    # Initial landing criteria. These are tunable hypotheses.
    SUCCESS_ANGLE = math.radians(12)
    SIDE_LANDING_ANGLE = math.radians(45)
    SUCCESS_ANGULAR_SPEED = 1.0
    SUCCESS_LINEAR_SPEED = 0.25
    REQUIRED_STABLE_TIME = 0.35

    # Require a genuine rotation, not a simple drop.
    MIN_NET_ROTATION = math.radians(300)
    MIN_FLIGHT_ROTATION = math.radians(180)

    TARGET_X = 2.25
    TARGET_X_TOLERANCE = 0.40
    PLATFORM_HALF_WIDTH=0.55

    MAX_OBS_OMEGA = 20.0
    MAX_OBS_SPEED = 8.0
    MAX_OBS_HEIGHT = 3.0

    GROUND_COLLISION = 1
    BOTTLE_COLLISION = 2

    def __init__(self, render_mode=None):
        super().__init__()

        if render_mode not in (None, "human"):
            raise ValueError(
                f"Unsupported render_mode: {render_mode}"
            )

        self.render_mode = render_mode

# Create the window only when visualization is requested.
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self.window_closed = False

        self.render_width = 1000
        self.render_height = 700
        self.render_scale = 220.0

        # Action[0]: normalized torque command.
        # Action[1]: release request; positive means request release.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # All observations are normalized to [-1, 1].
        # See _get_observation() for the meaning of each element.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(17,),
            dtype=np.float32,
        )

        self._episode_over = True

    # --------------------------------------------------------
    # World creation and reset
    # --------------------------------------------------------

    def _build_world(self):
        """Create a fresh Pymunk world and bottle."""

        self.space = pymunk.Space()
        self.space.gravity = (0.0, -self.GRAVITY)
        self.space.damping = 1.0
        self.space.iterations = 30

        self.phase = "swing"

        self.swing_time = 0.0
        self.flight_time = 0.0

        self.swing_travel = 0.0
        self.flight_travel = 0.0
        self.net_rotation = 0.0
        self.max_swing_excursion = 0.0

        self.ground_contact = False
        self.first_contact = False
        self.first_contact_angle_error = 0.0
        self.first_contact_angular_velocity = 0.0
        self.first_contact_speed = 0.0
        self.first_contact_position = None
        self.contact_angle_checked = False
        self.first_contact_flight_rotation = None
        self.first_contact_net_rotation = None
        self.stable_time = 0.0
        self.landing_elapsed = 0.0

        self.outcome = "in_progress"

        platform_left = (
            self.TARGET_X - self.PLATFORM_HALF_WIDTH
        )
        platform_right = (
            self.TARGET_X + self.PLATFORM_HALF_WIDTH
        )

        ground = pymunk.Segment(
            self.space.static_body,
            (platform_left, self.GROUND_Y),
            (platform_right, self.GROUND_Y),
            0.015,
        )
        ground.friction = 0.9
        ground.elasticity = 0.02
        ground.collision_type = self.GROUND_COLLISION
        self.space.add(ground)

        # Capture initial contact and separation events.
        self.space.on_collision(
            self.GROUND_COLLISION,
            self.BOTTLE_COLLISION,
            begin=self._on_ground_begin,
            separate=self._on_ground_separate,
        )

        moment = pymunk.moment_for_box(
            self.BOTTLE_MASS,
            (self.BOTTLE_WIDTH, self.BOTTLE_HEIGHT),
        )

        self.body = pymunk.Body(self.BOTTLE_MASS, moment)

        # The bottle's neck begins at the pivot.
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
        self.shape.elasticity = 0.02
        self.shape.collision_type = self.BOTTLE_COLLISION

        self.pivot_joint = pymunk.PivotJoint(
            self.space.static_body,
            self.body,
            (self.PIVOT_X, self.PIVOT_Y),
        )

        self.pivot_joint.collide_bodies = False

        self.space.add(self.body, self.shape, self.pivot_joint)

        # Used to measure angular displacement across full rotations.
        self._last_angle = self.body.angle

    def reset(self, *, seed=None, options=None):
        """
        Start a new episode.

        Gymnasium requires reset() to return (observation, info).
        """

        super().reset(seed=seed)

        self._build_world()
        self._episode_over = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    # --------------------------------------------------------
    # Collision callbacks
    # --------------------------------------------------------

    def _on_ground_begin(self, arbiter, space, data):
     self.ground_contact = True

     if not self.first_contact:
         self.first_contact = True

         self.first_contact_angle_error = self._upright_error()
         self.first_contact_angular_velocity = (
             float(self.body.angular_velocity)
         )
         self.first_contact_speed = float(
             self.body.velocity.length
         )
         self.first_contact_position = (
             float(self.body.position.x),
             float(self.body.position.y),
         )

     return True
        

    def _on_ground_separate(self, arbiter, space, data):
        self.ground_contact = False

    # --------------------------------------------------------
    # Physics measurements
    # --------------------------------------------------------

    def _upright_error(self):
        """Smallest angular difference from the upright orientation."""

        angle = self.body.angle
        wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
        return abs(wrapped)

    def _track_rotation(self):
        """Track signed and absolute rotation across physics steps."""

        current_angle = self.body.angle

        delta = (
            current_angle - self._last_angle + math.pi
        ) % (2 * math.pi) - math.pi

        self.net_rotation += delta

        if self.phase == "swing":
            self.swing_travel += abs(delta)
            self.max_swing_excursion = max(
                self.max_swing_excursion,
                abs(self.net_rotation),
            )
        else:
            self.flight_travel += abs(delta)

        self._last_angle = current_angle

    @property
    def release_ready(self):
        """Whether the swing has satisfied the release prerequisites."""

        return (
            self.phase == "swing"
            and self.swing_time >= self.MIN_SWING_TIME
            and self.max_swing_excursion >= self.MIN_SWING_ANGLE
            and abs(self.body.angular_velocity) >= self.MIN_RELEASE_OMEGA
        )

    def _swing_potential(self):
        """
        An intermediate learning signal for building a meaningful swing.

        This is reward shaping, not the definition of success.
        """

        amplitude = min(
            self.max_swing_excursion / (math.pi / 2),
            1.0,
        )

        rotational_speed = min(
            abs(self.body.angular_velocity) / self.MAX_OBS_OMEGA,
            1.0,
        )

        return 0.5 * amplitude + 0.15 * rotational_speed

    def _flight_potential(self):
        """
        Smooth intermediate signal for rotation and landing quality.
        A successful episode still requires the strict terminal criteria.
        """

        progress = min(
            self.flight_travel / self.MIN_FLIGHT_ROTATION,
            1.0,
        )

        angle_quality = math.exp(
            -self._upright_error() / math.radians(45)
        )

        position_quality = math.exp(
            -abs(self.body.position.x - self.TARGET_X)
            / self.TARGET_X_TOLERANCE
        )

        speed_quality = math.exp(
            -abs(self.body.angular_velocity) / 10.0
        )

        return progress * (
            0.25
            + 0.35 * angle_quality
            + 0.20 * position_quality
            + 0.10 * speed_quality
        )


    def _get_observation(self):
        """
        Return a fixed-size numerical state vector.

        Index:
          0  sine of bottle angle
          1  cosine of bottle angle
          2  normalized angular velocity
          3  normalized horizontal position
          4  normalized vertical position
          5  normalized horizontal velocity
          6  normalized vertical velocity
          7  phase: -1 = swing, +1 = flight
          8  normalized swing time
          9  maximum swing excursion
          10 absolute flight rotation
          11 signed net rotation
          12 release readiness
          13 ground contact
          14 normalized flight time
          15 stable-contact progress
          16 normalized horizontal target error
        """

        angle = self.body.angle
        x, y = self.body.position
        vx, vy = self.body.velocity

        observation = np.array(
            [
                math.sin(angle),
                math.cos(angle),

                np.clip(
                    self.body.angular_velocity / self.MAX_OBS_OMEGA,
                    -1.0, 1.0,
                ),

                np.clip(
                    2.0 * x / self.WORLD_WIDTH - 1.0,
                    -1.0, 1.0,
                ),

                np.clip(
                    2.0 * y / self.MAX_OBS_HEIGHT - 1.0,
                    -1.0, 1.0,
                ),

                np.clip(vx / self.MAX_OBS_SPEED, -1.0, 1.0),
                np.clip(vy / self.MAX_OBS_SPEED, -1.0, 1.0),

                -1.0 if self.phase == "swing" else 1.0,

                np.clip(
                    self.swing_time / self.MAX_SWING_TIME,
                    0.0, 1.0,
                ),

                np.clip(
                    self.max_swing_excursion / math.pi,
                    0.0, 1.0,
                ),

                np.clip(
                    self.flight_travel / (2.0 * math.pi),
                    0.0, 1.0,
                ),

                np.clip(
                    self.net_rotation / (2.0 * math.pi),
                    -1.0, 1.0,
                ),

                float(self.release_ready),
                float(self.ground_contact),

                np.clip(
                    self.flight_time / self.MAX_FLIGHT_TIME,
                    0.0, 1.0,
                ),

                np.clip(
                    self.stable_time / self.REQUIRED_STABLE_TIME,
                    0.0, 1.0,
                ),

                np.clip(
                    (self.TARGET_X - x) / self.TARGET_X_TOLERANCE,
                    -1.0, 1.0,
                ),
            ],
            dtype=np.float32,
        )

        return observation

    def _get_info(self):
        return {
            "phase": self.phase,
            "release_ready": self.release_ready,
            "ground_contact": self.ground_contact,
            "landing_angle_error_deg": math.degrees(
                self._upright_error()
            ),
            "net_rotation_deg": math.degrees(self.net_rotation),
            "swing_time": self.swing_time,
            "flight_time": self.flight_time,
            "flight_rotation_deg": (
    math.degrees(self.first_contact_flight_rotation)
    if self.first_contact_flight_rotation is not None
    else math.degrees(self.flight_travel)
),
            "success": self.outcome == "success",
            "outcome": self.outcome,
            "is_success": self.outcome == "success",
            "first_contact_angle_error_deg": (
                math.degrees(self.first_contact_angle_error)
                if self.first_contact else None
            ),
            "first_contact_angular_velocity_rad_s": (
                self.first_contact_angular_velocity
                if self.first_contact else None
            ),
            "first_contact_speed_m_s": (
                self.first_contact_speed
                if self.first_contact else None
            ),
            "first_contact_position_m": self.first_contact_position,
        }

    # --------------------------------------------------------
    # Main Gymnasium step
    # --------------------------------------------------------

    def step(self, action):
        """
        Apply one agent action and advance the physics.

        Required return:
        observation, reward, terminated, truncated, info
        """

        if self._episode_over:
            raise RuntimeError(
                "Episode has ended. Call reset() before step()."
            )

        action = np.asarray(action, dtype=np.float32)

        if action.shape != (2,):
            raise ValueError(
                f"Expected an action with shape (2,), got {action.shape}"
            )

        action = np.clip(action, -1.0, 1.0)

        torque_action = float(action[0])
        release_requested = bool(action[1] > 0.0)

        # Small time cost discourages unnecessarily long episodes.
        reward = -0.002

        terminated = False
        truncated = False
        control_dt = self.PHYSICS_DT * self.ACTION_REPEAT

        # =========================
        # Phase 1: swinging
        # =========================

        if self.phase == "swing":
            old_potential = self._swing_potential()

            applied_torque = torque_action * self.MAX_TORQUE

            for _ in range(self.ACTION_REPEAT):
                # Reapply torque at every physics substep.
                self.body.torque = applied_torque

                self.space.step(self.PHYSICS_DT)

                self.swing_time += self.PHYSICS_DT
                self._track_rotation()
                if (
                    self.first_contact
                    and self.first_contact_flight_rotation is None
                ):
                    self.first_contact_flight_rotation = self.flight_travel
                    self.first_contact_net_rotation = self.net_rotation

            new_potential = self._swing_potential()

            # Reward changes in swing potential, not raw angle alone.
            reward += new_potential - old_potential

            # Small control cost. This does not replace landing reward.
            reward -= 0.001 * torque_action**2

            if release_requested:
                if self.release_ready:
                    # Removing the joint preserves the current motion.
                    self.space.remove(self.pivot_joint)
                    self.pivot_joint = None

                    self.phase = "flight"
                    self.flight_time = 0.0

                    reward += 0.05
                else:
                    # A request cannot bypass the swing prerequisites.
                    reward -= 0.01

            if (
                self.phase == "swing"
                and self.swing_time >= self.MAX_SWING_TIME
            ):
                terminated = True
                reward -= 8.0
                self.outcome = "failed_to_release"

        # =========================
        # Phase 2: free flight
        # =========================

        elif self.phase == "flight":
            old_potential = self._flight_potential()

            for _ in range(self.ACTION_REPEAT):
                # No agent control after release.
                self.body.torque = 0.0

                self.space.step(self.PHYSICS_DT)

                self.flight_time += self.PHYSICS_DT
                self._track_rotation()

            new_potential = self._flight_potential()
            reward += 2.0 * (new_potential - old_potential)

            # Reject a clearly sideways first impact.
            if (
                self.first_contact
                and not self.contact_angle_checked
            ):
                self.contact_angle_checked = True

                if (
                    self.first_contact_angle_error
                    > self.SIDE_LANDING_ANGLE
                ):
                    terminated = True
                    reward -= 8.0
                    self.outcome = "side_landing"

            if not terminated and self.first_contact:
                self.landing_elapsed += control_dt

                if self.ground_contact:
                    stable = (
                        self._upright_error() <= self.SUCCESS_ANGLE
                        and abs(self.body.angular_velocity)
                        <= self.SUCCESS_ANGULAR_SPEED
                        and self.body.velocity.length
                        <= self.SUCCESS_LINEAR_SPEED
                    )

                    if stable:
                        self.stable_time += control_dt
                    else:
                        self.stable_time = 0.0

                    if self.stable_time >= self.REQUIRED_STABLE_TIME:
                        in_target = (
                            abs(
                                self.body.position.x - self.TARGET_X
                            )
                            <= self.TARGET_X_TOLERANCE
                        )

                        sufficient_rotation = (
                           self.first_contact_net_rotation is not None
                           and self.first_contact_flight_rotation is not None
                           and abs(self.first_contact_net_rotation)
                           >= self.MIN_NET_ROTATION
                           and self.first_contact_flight_rotation
                           >= self.MIN_FLIGHT_ROTATION
                       )

                        if in_target and sufficient_rotation:
                            terminated = True
                            reward += 25.0
                            self.outcome = "success"
                        else:
                            terminated = True
                            reward -= 8.0
                            self.outcome = (
                                "stable_landing_but_task_incomplete"
                            )

                else:
                    # A bounce resets the continuous stable-contact time.
                    self.stable_time = 0.0

                if (
                    not terminated
                    and self.landing_elapsed >= 0.8
                ):
                    terminated = True
                    reward -= 8.0
                    self.outcome = "unstable_landing"

            # Falling off the world is a terminal failure.
            if not terminated and (
                self.body.position.x < 0.0
                or self.body.position.x > self.WORLD_WIDTH
                or self.body.position.y < -0.1
            ):
                terminated = True
                reward -= 8.0
                self.outcome = "fell_off_platform"

            # A flight timeout is an episode truncation.
            if (
                not terminated
                and self.flight_time >= self.MAX_FLIGHT_TIME
            ):
                truncated = True
                reward -= 5.0
                self.outcome = "flight_timeout"

        self._episode_over = terminated or truncated

        observation = self._get_observation()
        info = self._get_info()

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    
    def render(self):
        """Render the actual environment state in a Pygame window."""

        if self.render_mode != "human":
            return None

        import pygame

        if self.window_closed:
            return None

        # Initialize Pygame lazily so training remains headless.
        if self._pygame is None:
            pygame.init()

            self._pygame = pygame
            self._screen = pygame.display.set_mode(
                (self.render_width, self.render_height)
            )
            pygame.display.set_caption(
                "Bottle-Dynamics | Trained Agent"
            )
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("consolas", 17)

        pygame = self._pygame
        screen = self._screen
        scale = self.render_scale
        height = self.render_height

        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE
                )
            ):
                self.window_closed = True
                return None

        def to_screen(point):
            return (
                round(point.x * scale),
                round(height - point.y * scale),
            )

        screen.fill((23, 29, 42))

        # Draw the ground.
        
        # Draw a separate landing platform on the right.
        ground_y = height - round(self.GROUND_Y * scale)

        platform_left = round(
            (self.TARGET_X - self.PLATFORM_HALF_WIDTH) * scale
        )
        platform_right = round(
            (self.TARGET_X + self.PLATFORM_HALF_WIDTH) * scale
        )

        platform_width = platform_right - platform_left

        # Platform body.
        pygame.draw.rect(
            screen,
            (105, 120, 142),
            (
                platform_left,
                ground_y,
                platform_width,
                30,
            ),
        )

        # Highlight the permitted landing zone.
        target_left = round(
            (self.TARGET_X - self.TARGET_X_TOLERANCE) * scale
        )
        target_right = round(
            (self.TARGET_X + self.TARGET_X_TOLERANCE) * scale
        )

        target_surface = pygame.Surface(
            (target_right - target_left, 8),
            pygame.SRCALPHA,
        )
        target_surface.fill((70, 210, 145, 180))

        screen.blit(
            target_surface,
            (target_left, ground_y - 8),
        )

        # Top edges of the platform and target.
        pygame.draw.line(
            screen,
            (205, 215, 230),
            (platform_left, ground_y),
            (platform_right, ground_y),
            3,
        )

        pygame.draw.line(
            screen,
            (70, 230, 155),
            (target_left, ground_y - 8),
            (target_right, ground_y - 8),
            2,
        )
        
        # Visual markers for the cap and base.
        # These are rendered from the same body as the physics.

        cap_left = self.body.local_to_world(
            (-self.BOTTLE_WIDTH * 0.28, self.BOTTLE_HEIGHT / 2)
        )
        cap_right = self.body.local_to_world(
            (self.BOTTLE_WIDTH * 0.28, self.BOTTLE_HEIGHT / 2)
        )

        base_left = self.body.local_to_world(
            (-self.BOTTLE_WIDTH / 2, -self.BOTTLE_HEIGHT / 2)
        )
        base_right = self.body.local_to_world(
            (self.BOTTLE_WIDTH / 2, -self.BOTTLE_HEIGHT / 2)
        )

        pygame.draw.line(
            screen,
            (245, 190, 75),
            to_screen(cap_left),
            to_screen(cap_right),
            5,
        )

        pygame.draw.line(
            screen,
            (225, 245, 255),
            to_screen(base_left),
            to_screen(base_right),
            4,
        )



        # Mark the target landing region.
        left = round(
            (self.TARGET_X - self.TARGET_X_TOLERANCE) * scale
        )
        right = round(
            (self.TARGET_X + self.TARGET_X_TOLERANCE) * scale
        )

        target_surface = pygame.Surface(
            (max(1, right - left), 12),
            pygame.SRCALPHA,
        )
        target_surface.fill((70, 210, 145, 100))
        screen.blit(target_surface, (left, ground_y - 12))

        pygame.draw.line(
            screen,
            (70, 210, 145),
            (left, ground_y - 16),
            (right, ground_y - 16),
            2,
        )

        # Draw the pivot while the bottle is attached.
        if self.phase == "swing":
            pivot = pygame.Vector2(
                self.PIVOT_X * scale,
                height - self.PIVOT_Y * scale,
            )

            pygame.draw.circle(
                screen,
                (245, 190, 75),
                (round(pivot.x), round(pivot.y)),
                7,
            )

        # Draw the bottle using its actual physics polygon.
        vertices = [
            self.body.local_to_world(vertex)
            for vertex in self.shape.get_vertices()
        ]

        points = [to_screen(point) for point in vertices]

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
        
        # Identify the bottle's cap and base.
        # These markers follow the bottle's actual rotation.

        cap_left = self.body.local_to_world(
            (-self.BOTTLE_WIDTH * 0.28, self.BOTTLE_HEIGHT / 2)
        )
        cap_right = self.body.local_to_world(
            (self.BOTTLE_WIDTH * 0.28, self.BOTTLE_HEIGHT / 2)
        )

        base_left = self.body.local_to_world(
            (-self.BOTTLE_WIDTH / 2, -self.BOTTLE_HEIGHT / 2)
        )
        base_right = self.body.local_to_world(
            (self.BOTTLE_WIDTH / 2, -self.BOTTLE_HEIGHT / 2)
        )

        # Gold = cap; light blue = base.
        pygame.draw.line(
            screen,
            (245, 190, 75),
            to_screen(cap_left),
            to_screen(cap_right),
            5,
        )

        pygame.draw.line(
            screen,
            (225, 245, 255),
            to_screen(base_left),
            to_screen(base_right),
            4,
        )

        

        angle = (
            math.degrees(self.body.angle) + 180.0
        ) % 360.0 - 180.0

        lines = [
            "BOTTLE-DYNAMICS | POLICY PLAYBACK",
            f"Phase: {self.phase}",
            f"Outcome: {self.outcome}",
            f"Angle: {angle:.1f} deg",
            f"Angular velocity: {self.body.angular_velocity:.2f} rad/s",
            f"Linear speed: {self.body.velocity.length:.2f} m/s",
            f"Swing time: {self.swing_time:.2f} s",
            f"Flight time: {self.flight_time:.2f} s",
            f"Flight rotation: {math.degrees(self.flight_travel):.1f} deg",
            f"Release ready: {self.release_ready}",
            "ESC / close window: stop playback",
        ]

        for index, line in enumerate(lines):
            text = self._font.render(
                line,
                True,
                (235, 240, 245),
            )
            screen.blit(text, (18, 15 + index * 24))

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        """Release rendering resources."""

        if self._pygame is not None:
            self._pygame.quit()

        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None

