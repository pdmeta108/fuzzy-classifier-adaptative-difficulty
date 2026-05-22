__credits__ = ["Andrea PIERRÉ"]

import math
from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
from fuzzy_inference_system import get_lunar_lander_inference_system as infer_system
import gymnasium as gym
from gymnasium import error, spaces
from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle
from gymnasium.utils.play import PlayableGame, MissingKeysToAction
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from gymnasium.wrappers import RecordEpisodeStatistics, RenderCollection


try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
    import pygame
    from pygame import Surface
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e


if TYPE_CHECKING:
    import pygame


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = (
    4  # The Y location of the main engine on the body of the Lander.
)

VIEWPORT_W = 600
VIEWPORT_H = 400

WHITE = (255, 255, 255)

# Teclas de acciones del jugador
keyboard_act = {
    "d": 3,
    "sd": 3,
    "s": 2,
    "a": 1,
    "as": 1,
    "l": 3,
    "kl": 3,
    "k": 2,
    "j": 1,
    "kj": 1,
}

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class LunarLander(gym.Env, EzPickle):
    r"""
    ## Description
    This environment is a classic rocket trajectory optimization problem.
    According to Pontryagin's maximum principle, it is optimal to fire the
    engine at full throttle or turn it off. This is the reason why this
    environment has discrete actions: engine on or off.

    There are two environment versions: discrete or continuous.
    The landing pad is always at coordinates (0,0). The coordinates are the
    first two numbers in the state vector.
    Landing outside of the landing pad is possible. Fuel is infinite, so an agent
    can learn to fly and then land on its first attempt.

    To see a heuristic landing, run:
    ```shell
    python gymnasium/envs/box2d/lunar_lander.py
    ```

    ## Action Space
    There are four discrete actions available:
    - 0: do nothing
    - 1: fire left orientation engine
    - 2: fire main engine
    - 3: fire right orientation engine

    ## Observation Space
    The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
    velocities in `x` & `y`, its angle, its angular velocity, and two booleans
    that represent whether each leg is in contact with the ground or not.

    ## Rewards
    After every step a reward is granted. The total reward of an episode is the
    sum of the rewards for all the steps within that episode.

    For each step, the reward:
    - is increased/decreased the closer/further the lander is to the landing pad.
    - is increased/decreased the slower/faster the lander is moving.
    - is decreased the more the lander is tilted (angle not horizontal).
    - is increased by 10 points for each leg that is in contact with the ground.
    - is decreased by 0.03 points each frame a side engine is firing.
    - is decreased by 0.3 points each frame the main engine is firing.

    The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

    An episode is considered a solution if it scores at least 200 points.

    ## Starting State
    The lander starts at the top center of the viewport with a random initial
    force applied to its center of mass.

    ## Episode Termination
    The episode finishes if:
    1) the lander crashes (the lander body gets in contact with the moon);
    2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
    3) the lander is not awake. From the [Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61),
        a body which is not awake is a body which doesn't move and doesn't
        collide with any other body:
    > When Box2D determines that a body (or group of bodies) has come to rest,
    > the body enters a sleep state which has very little CPU overhead. If a
    > body is awake and collides with a sleeping body, then the sleeping body
    > wakes up. Bodies will also wake up if a joint or contact attached to
    > them is destroyed.

    ## Arguments

    Lunar Lander has a large number of arguments

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
    ...                enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<LunarLander<LunarLander-v3>>>>>

    ```

     * `continuous` determines if discrete or continuous actions (corresponding to the throttle of the engines) will be used with the
     action space being `Discrete(4)` or `Box(-1, +1, (2,), dtype=np.float32)` respectively.
     For continuous actions, the first coordinate of an action determines the throttle of the main engine, while the second
     coordinate specifies the throttle of the lateral boosters. Given an action `np.array([main, lateral])`, the main
     engine will be turned off completely if `main < 0` and the throttle scales affinely from 50% to 100% for
     `0 <= main <= 1` (in particular, the main engine doesn't work  with less than 50% power).
     Similarly, if `-0.5 < lateral < 0.5`, the lateral boosters will not fire at all. If `lateral < -0.5`, the left
     booster will fire, and if `lateral > 0.5`, the right booster will fire. Again, the throttle scales affinely
     from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

    * `gravity` dictates the gravitational constant, this is bounded to be within 0 and -12. Default is -10.0

    * `enable_wind` determines if there will be wind effects applied to the lander. The wind is generated using
     the function `tanh(sin(2 k (t+C)) + sin(pi k (t+C)))` where `k` is set to 0.01 and `C` is sampled randomly between -9999 and 9999.

    * `wind_power` dictates the maximum magnitude of linear wind applied to the craft. The recommended value for
     `wind_power` is between 0.0 and 20.0.

    * `turbulence_power` dictates the maximum magnitude of rotational wind applied to the craft.
     The recommended value for `turbulence_power` is between 0.0 and 2.0.

    ## Version History
    - v3:
        - Reset wind and turbulence offset (`C`) whenever the environment is reset to ensure statistical independence between consecutive episodes (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/954)).
        - Fix non-deterministic behaviour due to not fully destroying the world (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/728)).
        - Changed observation space for `x`, `y`  coordinates from $\pm 1.5$ to $\pm 2.5$, velocities from $\pm 5$ to $\pm 10$ and angles from $\pm \pi$ to $\pm 2\pi$ (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/752)).
    - v2: Count energy spent and in v0.24, added turbulence with wind power and turbulence_power parameters
    - v1: Legs contact with ground added in state vector; contact with ground give +10 reward points, and -10 if then lose contact; reward renormalized to 200; harder initial random push.
    - v0: Initial version

    ## Notes

    There are several unexpected bugs with the implementation of the environment.

    1. The position of the side thrusters on the body of the lander changes, depending on the orientation of the lander.
    This in turn results in an orientation dependent torque being applied to the lander.

    2. The units of the state are not consistent. I.e.
    * The angular velocity is in units of 0.4 radians per second. In order to convert to radians per second, the value needs to be multiplied by a factor of 2.5.

    For the default values of VIEWPORT_W, VIEWPORT_H, SCALE, and FPS, the scale factors equal:
    'x': 10, 'y': 6.666, 'vx': 5, 'vy': 7.5, 'angle': 1, 'angular velocity': 2.5

    After the correction has been made, the units of the state are as follows:
    'x': (units), 'y': (units), 'vx': (units/second), 'vy': (units/second), 'angle': (radians), 'angular velocity': (radians/second)

    <!-- ## References -->

    ## Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        level: int = 1,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            continuous,
            gravity,
            enable_wind,
            wind_power,
            turbulence_power,
        )

        assert (
            -12.0 < gravity and gravity < 0.0
        ), f"gravity (current value: {gravity}) must be between -12 and 0"
        self.gravity = gravity

        if 0.0 > wind_power or wind_power > 20.0:
            gym.logger.warn(
                f"wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})"
            )
        self.wind_power = wind_power

        if 0.0 > turbulence_power or turbulence_power > 2.0:
            gym.logger.warn(
                f"turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})"
            )
        self.turbulence_power = turbulence_power

        self.enable_wind = enable_wind

        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.lander: Box2D.b2Body | None = None
        self.particles = []

        self.prev_reward = None

        self.continuous = continuous
        self.level = level

        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -2.5,  # x coordinate
                -2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                -10.0,
                -10.0,
                -2 * math.pi,
                -10.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                2.5,  # x coordinate
                2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                10.0,
                10.0,
                2 * math.pi,
                10.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(low, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.render_mode = render_mode

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        self._destroy()

        # Bug's workaround for: https://github.com/Farama-Foundation/Gymnasium/issues/728
        # Not sure why the self._destroy() is not enough to clean(reset) the total world environment elements, need more investigation on the root cause,
        # we must create a totally new world for self.reset(), or the bug#728 will happen
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.level += 1

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create Terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # Create Lander body
        initial_y = VIEWPORT_H / SCALE
        initial_x = VIEWPORT_W / SCALE / 2
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # Apply the initial random impulse to the lander
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        if self.enable_wind:  # Initialize wind pattern based on index
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)

        # Create Lander Legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = (
                    +0.9 - 0.5
                )  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particle):
        while self.particles and (all_particle or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        assert self.lander is not None

        # Update wind and apply to the lander
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + (math.sin(math.pi * 0.01 * self.torque_idx))
                )
                * self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(
                torque_mag,
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Apply Engine Impulses

        # Tip is the (X and Y) components of the rotation of the lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            # The components of the impulse to be applied by the main engine.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )

            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (
                        ox * MAIN_ENGINE_POWER * m_power,
                        oy * MAIN_ENGINE_POWER * m_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation/Side engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                # action = 1 is left, action = 3 is right
                direction = action - 2
                s_power = 1.0

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This causes the position of the thrust on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation dependent torque being applied to the lander.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= (
            m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
        if not self.lander.awake:
            terminated = True
            reward = +100

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # pygame.font.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        pygame.font.init()
        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        # 2. Create a Font object
        # Use the default system font, or specify a font file path (e.g., 'font.ttf')
        # You can use pygame.font.SysFont to access system fonts
        font = pygame.font.SysFont("Arial", 36) # Using 'Arial' font with size 36
        font_gui = pygame.font.SysFont("Arial", 18)

        # 3. Render the text to a Surface
        # Arguments: text string, antialias (True/False), text color, optional background color
        text_surface = font.render("Hello, Pygame!", True, WHITE)
        episode_surface = font_gui.render("Episodio: " + str(self.unwrapped.level + 1), True, WHITE)
        gravity_surface = font_gui.render("Gravedad: " + str(self.unwrapped.unwrapped.gravity), True, WHITE)
        wind_surface = font_gui.render("Viento poder: " + str(self.unwrapped.unwrapped.wind_power), True, WHITE)

        # Get a rectangle for positioning
        text_rect = text_surface.get_rect()
        episode_rect = episode_surface.get_rect()
        gravity_rect = gravity_surface.get_rect()
        wind_rect = wind_surface.get_rect()
        # Center the text on the screen
        text_rect.center = (VIEWPORT_W // 2, 50)
        episode_rect.center = (60, 25)
        gravity_rect.center = (70, 45)
        wind_rect.center = (80, 65)

        # Center the text on the screen
        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )

                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )

                for x in [self.helipad_x1, self.helipad_x2]:
                    x = x * SCALE
                    flagy1 = self.helipad_y * SCALE
                    flagy2 = flagy1 + 50
                    pygame.draw.line(
                        self.surf,
                        color=(255, 255, 255),
                        start_pos=(x, flagy1),
                        end_pos=(x, flagy2),
                        width=1,
                    )
                    pygame.draw.polygon(
                        self.surf,
                        color=(204, 204, 0),
                        points=[
                            (x, flagy2),
                            (x, flagy2 - 10),
                            (x + 25, flagy2 - 5),
                        ],
                    )
                    gfxdraw.aapolygon(
                        self.surf,
                        [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                        (204, 204, 0),
                    )

         # TODO Draw text
        # pygame.draw.line(self.surf, pygame.Color("Yellow"), (VIEWPORT_W / 4, 0), (VIEWPORT_W / 4, VIEWPORT_H), 3)
        # pygame.draw.line(self.surf, pygame.Color("Red"), (3 * VIEWPORT_W / 4, 0), (3 * VIEWPORT_W / 4, VIEWPORT_H), 3)

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            # 4. Blit the text surface onto the screen
            self.screen.blit(text_surface, text_rect)
            self.screen.blit(episode_surface, episode_rect)
            self.screen.blit(gravity_surface, gravity_rect)
            self.screen.blit(wind_surface, wind_rect)
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed
            s[6] 1 if first leg has contact, else 0
            s[7] 1 if second leg has contact, else 0

    Returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.unwrapped.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def demo_heuristic_lander(env, seed=None, render=False):
    total_reward = 0
    steps = 0
    episodes = 0
    s, info = env.reset(seed=seed)
    while True:
        while episodes < 3:
            a = heuristic(env, s)
            s, r, terminated, truncated, info = step_api_compatibility(env.step(a), True)
            total_reward += r

            if render:
                still_open = env.render()
                if still_open is False:
                    break

            if steps % 20 == 0 or terminated or truncated:
                print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated:
                episodes += 1
                print(f"Episode {episodes} finished with reward {total_reward:+0.2f}")

                s, info = env.reset(seed=seed)
        # Calculo de metricas (promedios de puntaje, duración del juego, desviación estándar y tasa de éxito)
        avg_reward = np.sum(env.return_queue)
        avg_length = np.sum(env.length_queue)
        std_reward = np.std(env.return_queue)
        return_queue_array = [f"{x:.2f}" for x in env.return_queue]
        avg_points = str(return_queue_array).replace("[","").replace("]","")
        avg_actions = str(list(env.length_queue)).replace("[","").replace("]","")
        time_game = str(list(env.time_queue)).replace("[","").replace("]","")
        success_percentage = sum(1 for r in env.return_queue if r > 0) / len(env.return_queue)
        break
    if render:
        env.close()
    return avg_reward, avg_length, std_reward, avg_points, avg_actions, time_game, success_percentage


class LunarLanderContinuous:
    def __init__(self):
        raise error.Error(
            "Error initializing LunarLanderContinuous Environment.\n"
            "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
            "To use this environment, instead create it by specifying the continuous keyword in gym.make, i.e.\n"
            'gym.make("LunarLander-v3", continuous=True)'
        )

def display_arr(
    screen: Surface, arr: np.ndarray, video_size: tuple[int, int], transpose: bool, ep_count, gravity, wind_power
):
    """Displays a numpy array on screen.

    Args:
        screen: The screen to show the array on
        arr: The array to show
        video_size: The video size of the screen
        transpose: If to transpose the array on the screen
    """
    assert isinstance(arr, np.ndarray) and arr.dtype == np.uint8
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    # We might have to add black bars if surface_size is larger than video_size
    surface_size = screen.get_size()

    # 2. Create a Font object
    # Use the default system font, or specify a font file path (e.g., 'font.ttf')
    # You can use pygame.font.SysFont to access system fonts
    font_gui = pygame.font.SysFont("Arial", 18)

    # 3. Render the text to a Surface
    # Arguments: text string, antialias (True/False), text color, optional background color
    episode_surface = font_gui.render("Episodio: " + str(ep_count + 1), True, WHITE)
    gravity_surface = font_gui.render("Gravedad: " + str(gravity), True, WHITE)
    wind_surface = font_gui.render("Viento poder: " + str(wind_power), True, WHITE)

    # Get a rectangle for positioning
    episode_rect = episode_surface.get_rect()
    gravity_rect = gravity_surface.get_rect()
    wind_rect = wind_surface.get_rect()
    # Center the text on the screen
    episode_rect.center = (60, 25)
    gravity_rect.center = (70, 45)
    wind_rect.center = (80, 65)

    # Center the text on the screen
    pygame.transform.scale(screen, (SCALE, SCALE))
    pygame.draw.rect(screen, (255, 255, 255), screen.get_rect())

    width_offset = (surface_size[0] - video_size[0]) / 2
    height_offset = (surface_size[1] - video_size[1]) / 2
    screen.fill((0, 0, 0))
    screen.blit(pyg_img, (width_offset, height_offset))
    # 4. Blit the text surface onto the screen
    screen.blit(episode_surface, episode_rect)
    screen.blit(gravity_surface, gravity_rect)
    screen.blit(wind_surface, wind_rect)

def play(
    env: Env,
    transpose: bool | None = True,
    fps: int | None = None,
    zoom: float | None = None,
    callback: Callable | None = None,
    keys_to_action: dict[tuple[str | int, ...] | str | int, ActType] | None = None,
    seed: int | None = None,
    noop: ActType = 0,
    wait_on_player: bool = False,
):
    """Allows the user to play the environment using a keyboard.

    If playing in a turn-based environment, set wait_on_player to True.

    Args:
        env: Environment to use for playing.
        transpose: If this is ``True``, the output of observation is transposed. Defaults to ``True``.
        fps: Maximum number of steps of the environment executed every second. If ``None`` (the default),
            ``env.metadata["render_fps""]`` (or 30, if the environment does not specify "render_fps") is used.
        zoom: Zoom the observation in, ``zoom`` amount, should be positive float
        callback: If a callback is provided, it will be executed after every step. It takes the following input:

            * obs_t: observation before performing action
            * obs_tp1: observation after performing action
            * action: action that was executed
            * rew: reward that was received
            * terminated: whether the environment is terminated or not
            * truncated: whether the environment is truncated or not
            * info: debug info
        keys_to_action:  Mapping from keys pressed to action performed.
            Different formats are supported: Key combinations can either be expressed as a tuple of unicode code
            points of the keys, as a tuple of characters, or as a string where each character of the string represents
            one key.
            For example if pressing 'w' and space at the same time is supposed
            to trigger action number 2 then ``key_to_action`` dict could look like this:

                >>> key_to_action = {
                ...    # ...
                ...    (ord('w'), ord(' ')): 2
                ...    # ...
                ... }

            or like this:

                >>> key_to_action = {
                ...    # ...
                ...    ("w", " "): 2
                ...    # ...
                ... }

            or like this:

                >>> key_to_action = {
                ...    # ...
                ...    "w ": 2
                ...    # ...
                ... }

            If ``None``, default ``key_to_action`` mapping for that environment is used, if provided.
        seed: Random seed used when resetting the environment. If None, no seed is used.
        noop: The action used when no key input has been entered, or the entered key combination is unknown.
        wait_on_player: Play should wait for a user action

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> from gymnasium.utils.play import play
        >>> play(gym.make("CarRacing-v3", render_mode="rgb_array"),  # doctest: +SKIP
        ...     keys_to_action={
        ...         "w": np.array([0, 0.7, 0], dtype=np.float32),
        ...         "a": np.array([-1, 0, 0], dtype=np.float32),
        ...         "s": np.array([0, 0, 1], dtype=np.float32),
        ...         "d": np.array([1, 0, 0], dtype=np.float32),
        ...         "wa": np.array([-1, 0.7, 0], dtype=np.float32),
        ...         "dw": np.array([1, 0.7, 0], dtype=np.float32),
        ...         "ds": np.array([1, 0, 1], dtype=np.float32),
        ...         "as": np.array([-1, 0, 1], dtype=np.float32),
        ...     },
        ...     noop=np.array([0, 0, 0], dtype=np.float32)
        ... )

        Above code works also if the environment is wrapped, so it's particularly useful in
        verifying that the frame-level preprocessing does not render the game
        unplayable.

        If you wish to plot real time statistics as you play, you can use
        :class:`PlayPlot`. Here's a sample code for plotting the reward
        for last 150 steps.

        >>> from gymnasium.utils.play import PlayPlot, play
        >>> def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        ...        return [rew,]
        >>> plotter = PlayPlot(callback, 150, ["reward"])             # doctest: +SKIP
        >>> play(gym.make("CartPole-v1"), callback=plotter.callback)  # doctest: +SKIP
    """
    env.reset(seed=seed)

    if keys_to_action is None:
        if env.has_wrapper_attr("get_keys_to_action"):
            keys_to_action = env.get_wrapper_attr("get_keys_to_action")()
        else:
            assert env.spec is not None
            raise MissingKeysToAction(
                f"{env.spec.id} does not have explicit key to action mapping, "
                "please specify one manually"
            )

    assert keys_to_action is not None

    # validate the `keys_to_action` set provided
    assert isinstance(keys_to_action, dict)
    for key, action in keys_to_action.items():
        if isinstance(key, tuple):
            assert len(key) > 0
            assert all(isinstance(k, (str, int)) for k in key)
        else:
            assert isinstance(key, (str, int))

        assert action in env.action_space

    key_code_to_action = {}
    for key_combination, action in keys_to_action.items():
        if isinstance(key_combination, int):
            key_combination = (key_combination,)
        key_code = tuple(
            sorted(ord(key) if isinstance(key, str) else key for key in key_combination)
        )
        key_code_to_action[key_code] = action

    game = PlayableGame(env, key_code_to_action, zoom)

    if fps is None:
        fps = env.metadata.get("render_fps", 30)

    done, obs = True, None
    clock = pygame.time.Clock()

    while game.running:
        if done:
            done = False
            obs = env.reset(seed=seed)
        elif wait_on_player is False or len(game.pressed_keys) > 0:
            action = key_code_to_action.get(tuple(sorted(game.pressed_keys)), noop)
            prev_obs = obs
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if callback is not None:
                callback(prev_obs, obs, action, rew, terminated, truncated, info)
        if obs is not None:
            rendered = env.render()
            if isinstance(rendered, list):
                rendered = rendered[-1]
            assert rendered is not None and isinstance(rendered, np.ndarray)
            display_arr(
                game.screen, rendered, transpose=transpose, video_size=game.video_size, ep_count=5, gravity=env.unwrapped.gravity, wind_power=env.unwrapped.wind_power
            )

        # process pygame events
        for event in pygame.event.get():
            game.process_event(event)

        pygame.display.flip()
        clock.tick(fps)
    # Calculo de metricas (promedios de puntaje, duración del juego, desviación estándar y tasa de éxito)
    avg_reward = np.sum(env.return_queue)
    avg_length = np.sum(env.length_queue)
    std_reward = np.std(env.return_queue)
    return_queue_array = [f"{x:.2f}" for x in env.return_queue]
    avg_points = str(return_queue_array).replace("[","").replace("]","")
    avg_actions = str(list(env.length_queue)).replace("[","").replace("]","")
    time_game = str(list(env.time_queue)).replace("[","").replace("]","")
    success_percentage = sum(1 for r in env.return_queue if r > 0) / len(env.return_queue)
    pygame.quit()
    return avg_reward, avg_length, std_reward, avg_points, avg_actions, time_game, success_percentage

if __name__ == "__main__":

    lunar_difficulty_gravity = -5
    lunar_difficulty_wind = 0.4
    level_seed = 67
    min_time = 5
    max_time = 300
    min_points = 0
    max_points = 250
    # Register the environment so we can create it with gym.make()
    gym.register(
        id="lunar_fuzzy/LunarLanderFuzzy-v0",
        entry_point=LunarLander,
        max_episode_steps=1000,  # Prevent infinite episodes
    )
    # env = gym.make(
    #     "lunar_fuzzy/LunarLanderFuzzy-v0",
    #     gravity=lunar_difficulty_gravity,
    #     level=-1,
    #     render_mode="human")

    # env = RenderCollection(env, pop_frames=False, reset_clean=False)
    # env = RecordEpisodeStatistics(env, buffer_length=10)

    # # Obtener resultado de ambiente
    # avg_reward, avg_length, std_reward, avg_points, avg_actions, time_game, success_percentage = demo_heuristic_lander(env, seed=level_seed, render=True)
    # print(f"Total Reward: {avg_reward:.2f}")
    # print(f"Standard Deviation of Reward: {std_reward:.2f}")
    # print(f"Total Length: {avg_length:.2f}")
    # print(f"Rewards per episode: {avg_points}")
    # print(f"Actions per episode: {avg_actions}")
    # print(f"Time per episode: {time_game}")
    # print(f"Success Percentage: {success_percentage:.2%}")

    # # Transformar array en numpy
    # float_array = [round(float(x), 3) for x in time_game.split(", ")]
    # float_array_points = [round(float(x.replace("'", "")), 3) for x in avg_points.split(", ")]
    # norm_time = np.asarray(float_array, dtype=np.float32)
    # norm_points = np.asarray(float_array_points, dtype=np.float32)
    # norm_win = success_percentage / 100

    # # Normalizar los datos
    # # res_time = (norm_time - min_time) / (max_time - min_time)
    # # res_points = (norm_points - min_points) / (max_points - min_points)
    # # res_time = np.where(res_time > 1, 1, res_time)
    # # res_points = np.where(res_points > 1, 1, res_points)

    mamdani_gravity = infer_system()
    mamdani_wind = infer_system("wind")
    print(mamdani_gravity)

    # # Realizar la inferencia utilizando el sistema difuso
    # mamdani_result_gravity: float = mamdani_gravity.infer(
    #     time=norm_time.mean(),
    #     reward=norm_points.mean(),
    #     win=norm_win,
    # )["gravity"]

    # mamdani_result_wind: float = mamdani_wind.infer(
    #     time=norm_time.mean(),
    #     reward=norm_points.mean(),
    #     win=norm_win,
    # )["wind"]

    # print(f"Gravity: {mamdani_result_gravity:.2f}")
    # print(f"Wind power: {mamdani_result_wind:.2f}")
    # if (mamdani_result_gravity < 50):
    #     print("Aumentar el nivel de dificultad.")
    #     lunar_difficulty_gravity -= 2
    #     if lunar_difficulty_gravity < -11:
    #         lunar_difficulty_gravity = -11.5
    # else:
    #     print("Disminuir el nivel de dificultad.")
    #     lunar_difficulty_gravity += 2
    #     if lunar_difficulty_gravity >= 0:
    #         lunar_difficulty_gravity = -0.5
    play_eps = 0
    while play_eps < 3:
        # Correr juego nuevo con otra dificultad
        env = gym.make(
            "lunar_fuzzy/LunarLanderFuzzy-v0",
            gravity=round(lunar_difficulty_gravity, 2),
            enable_wind=False,
            wind_power=round(lunar_difficulty_wind, 2),
            render_mode="rgb_array")

        env = RenderCollection(env, pop_frames=False, reset_clean=False)
        env = RecordEpisodeStatistics(env, buffer_length=10)

        avg_reward, avg_length, std_reward, avg_points, avg_actions, time_game, success_percentage = play(env, keys_to_action=keyboard_act, fps=30)

        # Obtener resultado de ambiente
        # avg_reward, avg_length, std_reward, avg_points, avg_actions, time_game, success_percentage = demo_heuristic_lander(env, seed=level_seed, render=True)
        print("resultado para jugador")
        print(f"Total Reward: {avg_reward:.2f}")
        print(f"Standard Deviation of Reward: {std_reward:.2f}")
        print(f"Total Length: {avg_length:.2f}")
        print(f"Rewards per episode: {avg_points}")
        print(f"Actions per episode: {avg_actions}")
        print(f"Time per episode: {time_game}")
        print(f"Success Percentage: {success_percentage:.2%}")

        # Transformar array en numpy
        float_array = [round(float(x), 3) for x in time_game.split(", ")]
        float_array_points = [round(float(x.replace("'", "")), 3) for x in avg_points.split(", ")]
        norm_time = np.asarray(float_array, dtype=np.float32)
        norm_points = np.asarray(float_array_points, dtype=np.float32)
        norm_win = success_percentage / 100

        # Realizar la inferencia utilizando el sistema difuso
        mamdani_result_gravity: float = mamdani_gravity.infer(
            time=norm_time.mean(),
            reward=norm_points.mean(),
            win=norm_win,
        )["gravity"]

        mamdani_result_wind: float = mamdani_wind.infer(
            time=norm_time.mean(),
            reward=norm_points.mean(),
            win=norm_win,
        )["wind"]

        print(f"Gravity: {mamdani_result_gravity:.2f}")
        print(f"Wind power: {mamdani_result_wind:.2f}")
        # Modificar la dificultad de gravedad
        print(success_percentage)
        if (success_percentage > 0.9):
            print("Aumentar el nivel de dificultad.")
            lunar_difficulty_gravity -= 2
        else:
            if (mamdani_result_gravity > -6):
                print("Aumentar el nivel de dificultad.")
                lunar_difficulty_gravity -= 2
                if lunar_difficulty_gravity < -11:
                    lunar_difficulty_gravity = -11.5
            else:
                print("Disminuir el nivel de dificultad.")
                lunar_difficulty_gravity += 2
                if lunar_difficulty_gravity >= 0:
                    lunar_difficulty_gravity = -0.5
        # Modificar la dificultad de viento
        if (success_percentage > 0.9):
            lunar_difficulty_wind += 3
        else:
            if (mamdani_result_wind < 2):
                # print("Aumentar el nivel de dificultad.")
                lunar_difficulty_wind += 5
                if lunar_difficulty_wind < 0:
                    lunar_difficulty_wind = 0.5
            else:
                # print("Disminuir el nivel de dificultad.")
                mamdani_result_wind -= 5
                if lunar_difficulty_wind >= 15:
                    lunar_difficulty_wind = 14.5