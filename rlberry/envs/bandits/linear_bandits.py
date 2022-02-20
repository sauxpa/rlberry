import numpy as np
from scipy import stats
import logging

from rlberry.envs.interface import Model
import rlberry.spaces as spaces


logger = logging.getLogger(__name__)


class LinearBandit(Model):
    """
    Base class for a Linear Contextual Bandit.
    At time time t, the learner observes a decision set D_t and receives a
    random reward Y_t for selecting action X_t in D_t:
    Y_t = <theta, X_t> + noise.

    Only finite decision sets are supported for now.

    Parameters
    ----------

    theta: np.ndarray of shape (d,)
        Linear parameter of the reward.

    decision_set_func: function of t returning a list of d-dimensional vectors.
        Set of possible actions at time t.

    noise_law: law of additive noise.
        Can either be a frozen scipy law or any class that
        has a method .rvs().

    **kwargs: keywords arguments
        additional arguments sent to :class:`~rlberry.envs.interface.Model`

    """

    name = ""

    def __init__(self, theta=None, decision_set_func=None, noise_law=None, **kwargs):
        Model.__init__(self, **kwargs)
        self.theta = theta
        self.decision_set_func = decision_set_func
        self.noise_law = noise_law
        self.reset()

    def forward_time(self, t=None):
        if t is None:
            self.t += 1
        else:
            self.t = t
        self.contexts = self.decision_set_func(self.t)
        self.action_space = spaces.Discrete(len(self.contexts))

    def step(self, action):
        """
        Sample the reward associated to the action.
        """
        # test that the action exists
        assert action < self.action_space.n

        reward = self.theta @ self.contexts[action] + self.noise_law.rvs(
            random_state=self.rng
        )
        done = True
        self.forward_time()

        return 0, reward, done, {}

    def reset(self):
        """
        Reset the environment to a default state.
        """
        self.forward_time(0)
        return 0


class NormalNoiseLinearBandit(LinearBandit):
    """
    Class for Linear Contextual Bandits with Normal noise.

    Parameters
    ----------

    theta: np.ndarray of shape (d,)
        Linear parameter of the reward.

    decision_set_func: function of t returning a list of d-dimensional vectors.
        Set of possible actions at time t.

    std: float
        Standard deviation of the Normal noise.

    """

    def __init__(self, theta=None, decision_set_func=None, std=1.0):
        noise_law = self.make_noise_law(std)
        LinearBandit.__init__(
            self, theta=theta, decision_set_func=decision_set_func, noise_law=noise_law
        )

    def make_noise_law(self, std):
        self.std = std
        return stats.norm(loc=0.0, scale=self.std)


class UniformUnitBallNormalNoiseLinearBandit(NormalNoiseLinearBandit):
    """
    Class for Linear Contextual Bandits with Normal noise and fixed number
    of contexts drawn uniformly from the unit ball.

    Parameters
    ----------

    A: int
        Number of arms.

    theta: np.ndarray of shape (d,)
        Linear parameter of the reward.

    std: float
        Standard deviation of the Normal noise.

    """

    def __init__(self, theta=None, A=1, std=1.0):
        d = theta.shape[0]
        decision_set_func = self.make_decision_set_func(A, d)
        NormalNoiseLinearBandit.__init__(
            self, theta=theta, decision_set_func=decision_set_func, std=std
        )

    def make_decision_set_func(self, A, d):
        def decision_set_func(t):
            # Generate A d-dimensional standard Gaussian vectors
            X = stats.multivariate_normal(np.zeros(d)).rvs(A, random_state=self.rng)
            # Generate A uniform random radii
            R = stats.uniform().rvs(A, random_state=self.rng)
            # Project onto the unit ball
            X /= np.repeat(
                (R ** (1 / d) / np.sqrt(np.sum(X**2, axis=1))).reshape(-1, 1),
                d,
                axis=1,
            )
            return X

        return decision_set_func
