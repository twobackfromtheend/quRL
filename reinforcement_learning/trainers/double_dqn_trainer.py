import logging

import numpy as np

from reinforcement_learning.trainers.base_classes.hyperparameters import QLearningHyperparameters, ExplorationOptions, \
    ExplorationMethod
from reinforcement_learning.trainers.dqn_options import DQNTrainerOptions
from reinforcement_learning.trainers.dqn_trainer import DQNTrainer

logger = logging.getLogger(__name__)


class DoubleDQNTrainer(DQNTrainer):
    """
    Differs from DQN in that target calculation is based on the policy's chosen action, not argmax.
    """

    def get_targets(self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        Calculates targets based on done. Chooses the next action with the policy network, and evalautes
        the Q value of that chosen action with the target network.
        If done,
            target = reward
        If not done,
            target = r + gamma * target_network(best action from policy_network(next_state))
        :param rewards:
        :param next_states:
        :param dones:
        :return: Targets - 1D np.array
        """
        # Targets initialised w/ done == True steps
        targets = rewards.copy()

        # Targets for done == False steps calculated with target network
        done_false_indices = dones == False
        gamma = self.hyperparameters.discount_rate(self.episode_number)

        done_false_next_states = next_states[done_false_indices]

        # Ask policy network to choose next actions
        target_actions = np.argmax(self.model.predict(done_false_next_states), axis=1)

        # Evaluate Q values of the policy-network-chosen actions with the target network.
        done_false_target_q_values = self.target_model.predict(done_false_next_states)
        done_false_targets = targets[done_false_indices]
        done_false_rewards = rewards[done_false_indices]
        for i, action in enumerate(target_actions):
            done_false_targets[i] = done_false_rewards[i] + gamma * done_false_target_q_values[i, action]
        targets[done_false_indices] = done_false_targets
        return targets


if __name__ == '__main__':
    from reinforcement_learning.models.dense_model import DenseModel

    logging.basicConfig(level=logging.INFO)

    from reinforcement_learning.time_sensitive_envs.cartpole_env import CartPoleTSEnv
    time_sensitive = False
    env = CartPoleTSEnv(time_sensitive=time_sensitive)
    inputs = 5 if time_sensitive else 4
    model = DenseModel(inputs=inputs, outputs=2, layer_nodes=(48, 48), learning_rate=3e-3,
                       inner_activation='relu', output_activation='linear')

    EPISODES = 20000

    trainer = DoubleDQNTrainer(
        model, env,
        hyperparameters=QLearningHyperparameters(
            0.95,
            ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=0.5, epsilon_decay=0.999,
                               limiting_value=0.1)
            # ExplorationOptions(method=ExplorationMethod.SOFTMAX, starting_value=0.5, softmax_total_episodes=EPISODES)
        ),
        options=DQNTrainerOptions(render=True)
    )
    trainer.train(episodes=EPISODES)
    logger.info(f"max reward total: {max(trainer.reward_totals)}")
    logger.info(f"last evaluation reward: {trainer.evaluation_rewards[-1]}")
