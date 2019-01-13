import logging

import numpy as np

from reinforcement_learning.trainers.base_classes.hyperparameters import QLearningHyperparameters, ExplorationOptions, \
    ExplorationMethod
from reinforcement_learning.trainers.drqn_batched_trainer import DRQNBatchedTrainer
from reinforcement_learning.trainers.drqn_options import DRQNTrainerOptions
from quantum_evolution.envs.multi_qubit import MultiQEnv

logger = logging.getLogger(__name__)


class DoubleDRQNBatchedTrainer(DRQNBatchedTrainer):
    """
    Differs from DRQN in that target calculation is based on the policy's chosen action, not argmax.
    """

    def get_targets(self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray) -> np.ndarray:
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
    from reinforcement_learning.models.lstm_non_stateful_model import LSTMNonStatefulModel

    logging.basicConfig(level=logging.INFO)

    for t in np.arange(0.5, 0.6, 0.1):
        time_sensitive = False
        t_t = t
        t_N = int(round(t_t / 0.05))
        env = MultiQEnv(t_t, t_N)
        inputs = 2
        rnn_steps = 10
        model = LSTMNonStatefulModel(inputs=inputs, outputs=2, rnn_steps=rnn_steps, learning_rate=3e-3,
                                     inner_activation='relu', output_activation='linear')

        # if t < 0.6:
        #     EPISODES = 1000
        #
        # else:
        #     EPISODES = 50000
        EPISODES = 20000
        trainer = DoubleDRQNBatchedTrainer(
            model, env,
            hyperparameters=QLearningHyperparameters(
                0.95,
                # ExplorationOptions(method=ExplorationMethod.EPSILON, starting_value=0.5, epsilon_decay=0.999,
                #                    limiting_value=0.1)
                ExplorationOptions(method=ExplorationMethod.SOFTMAX, starting_value=100,
                                   softmax_total_episodes=EPISODES, limiting_value=100000)
            ),
            options=DRQNTrainerOptions(save_every=10000, rnn_steps=rnn_steps, update_target_soft=True, render=False)
        )
        trainer.train(episodes=EPISODES)
        logger.info(f"max reward total: {max(trainer.reward_totals)}")
        logger.info(f"last evaluation reward: {trainer.evaluation_rewards[-1]}")
