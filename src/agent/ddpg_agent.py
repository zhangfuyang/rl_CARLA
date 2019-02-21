import tensorflow as tf
import gym
from tqdm import tqdm
import numpy as np
from src.agent.agent import BaseAgent
import os
import random

class DDPGAgent(BaseAgent):
    def __init__(self, sess, action_type, actor, critic, gamma, env, replay_buffer=None, noise=None,
                 exploration_episodes=10000, max_episodes=10000, max_steps_episode=10000,
                 warmup_steps=5000, mini_batch=32, eval_episodes=10, eval_periods=100, env_render=False,
                 summary_dir=None, model_dir=None, detail=True, model_store_periods=1000, render_interval=50,
                 Inference=False, Inference_net_dir=None):
        """
        Deep Deterministic Policy Gradient Agent.
        Args:
            actor: actor network.
            critic: critic network.
            gamma: discount factor.
        """
        super(DDPGAgent, self).__init__(sess, env, replay_buffer, noise=noise,
                                        exploration_episodes=exploration_episodes, max_episodes=max_episodes,
                                        max_steps_episode=max_steps_episode, warmup_steps=warmup_steps,
                                        mini_batch=mini_batch, eval_episodes=eval_episodes, eval_periods=eval_periods,
                                        env_render=env_render, summary_dir=summary_dir)

        self.action_type = action_type
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.cur_episode = 0
        if Inference is False:
            self.env.Image_agent.load_model()
        else:
            self.Restore(Inference_net_dir)
        self.detail = detail
        self.model_dir = model_dir
        self.model_store_periods = model_store_periods
        self.train_t = 0
        self.render_interval = render_interval

    def Restore(self, net_dir):
        saver = tf.train.Saver()
        if not os.path.exists(net_dir):
            raise RuntimeError('failed to find the models path')
        ckpt = tf.train.get_checkpoint_state(net_dir)
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        print('Restoring from ', net_dir)

    def get_episode(self):
        return self.cur_episode

    def train(self):
        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()
        reward_flag = False #used to set whether store too much positive sample

        for cur_episode in tqdm(range(self.max_episodes)):
            if cur_episode < self.cur_episode:
                continue
            self.cur_episode = cur_episode
            # evaluate here. 
            if cur_episode % self.eval_periods == 0 and cur_episode != 0:
                ave_reward = self.evaluate(cur_episode)
                if ave_reward > 150:
                    reward_flag = True
                else:
                    reward_flag = False

            state = self.env.reset()

            episode_reward = 0
            episode_ave_max_q = 0

            for cur_step in range(self.max_steps_episode):

                if self.env_render and cur_episode % self.render_interval == 0:
                    self.env.render(str(cur_episode))

                # Add exploratory noise according to Ornstein-Uhlenbeck process to action
                if self.replay_buffer.size() < self.warmup_steps:
                    action = self.env.action_space.sample()
                else: 
                    if self.action_type == 'Continuous':
                        if cur_episode < self.exploration_episodes and self.noise is not None:
                            action = np.clip(self.actor.predict(np.expand_dims(state, 0))[0] + self.noise.generate(cur_episode), -1, 1) 
                        else: 
                            action = self.actor.predict(np.expand_dims(state, 0))[0] 
                    else:
                        action = self.noise.generate(self.actor.predict(np.expand_dims(state, 0))[0, 0], cur_episode)

                if action[1] >= 0:
                    step_ = {'steer': action[0], 'acc': action[1], 'brake': 0.0}
                else:
                    step_ = {'steer': action[0], 'acc': 0.0, 'brake': action[1]}

                next_state, reward, terminal, info = self.env.step(step_)

                if self.detail and cur_episode % 5 == 0:
                    [grad_, action_check] = self.actor.check_(np.expand_dims(state, 0))
                    print("true_action: ", action,"action: ", action_check, "\treward: ", reward, "\tspeed: ", next_state[-3] * 10.0, "\toffroad: ", next_state[-1] + next_state[-2])
                if reward >= -5 and reward_flag:
                    if random.random() > 0.8:
                        self.replay_buffer.add(state, action, reward_flag, terminal, next_state)
                else:
                    self.replay_buffer.add(state, action, reward, terminal, next_state)

                # Keep adding experience to the memory until there are at least minibatch size samples
                if self.replay_buffer.size() > self.warmup_steps:
                    state_batch, action_batch, reward_batch, terminal_batch, next_state_batch = \
                        self.replay_buffer.sample_batch(self.mini_batch)

                    # Calculate targets
                    target_q = self.critic.predict_target(next_state_batch, self.actor.predict_target(next_state_batch))

                    y_i = np.reshape(reward_batch, (self.mini_batch, 1)) + (1
                            - np.reshape(terminal_batch, (self.mini_batch, 1)).astype(float))\
                            * self.gamma * np.reshape(target_q, (self.mini_batch, 1))

                    # Update the critic given the targets
                    if self.action_type == 'Discrete':
                        action_batch = np.reshape(action_batch, [self.mini_batch, 1])
                    predicted_q_value, _ = self.critic.train(state_batch, action_batch, y_i)

                    episode_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    if self.action_type == 'Continuous':
                        a_outs = self.actor.predict(state_batch)
                        a_grads = self.critic.action_gradients(state_batch, a_outs)
                        self.actor.train(state_batch, a_grads[0])
                    else:
                        a_outs = self.actor.predict(state_batch)
                        a_grads = self.critic.action_gradients(state_batch, a_outs)
                        self.actor.train(state_batch, a_grads[0])

                    self.train_t += 1
                    # store model here
                    if self.train_t % self.model_store_periods == 0:
                        self.saver.save(self.sess, os.path.join(self.model_dir, 'saveNet_' + str(cur_episode) + '.ckpt'), global_step=self.train_t)

                    # Update target networks
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                state = next_state
                episode_reward += reward

                if terminal or cur_step == self.max_steps_episode-1:
                    train_episode_summary = tf.Summary() 
                    train_episode_summary.value.add(simple_value=episode_reward, tag="train/episode_reward")
                    train_episode_summary.value.add(simple_value=episode_ave_max_q/float(cur_step), tag="train/episode_ave_max_q")
                    self.writer.add_summary(train_episode_summary, cur_episode)
                    self.writer.flush()

                    print ('Reward: %.2i' % int(episode_reward), ' | Episode', cur_episode,
                          '| Qmax: %.4f' % (episode_ave_max_q / float(cur_step)), ' | Total train: ', self.train_t)

                    break

    def evaluate(self, cur_episode):
        # evaluate here.
        print("==================evaluation====================")
        total_episode_reward = 0 
        for eval_i in range(self.eval_episodes):
            state = self.env.reset() 
            terminal = False
            while not terminal:
                if self.action_type == 'Continuous':
                    action = self.actor.predict(np.expand_dims(state, 0))[0]
                else:
                    action = self.actor.predict(np.expand_dims(state, 0))[0, 0]
                if action[1] >= 0:
                    step_ = {'steer': action[0], 'acc': action[1], 'brake': 0.0}
                else:
                    step_ = {'steer': action[0], 'acc': 0.0, 'brake': action[1]}
                state, reward, terminal, info = self.env.step(step_)
                total_episode_reward += reward
        ave_episode_reward = total_episode_reward / float(self.eval_episodes)
        print("\nAverage reward {}\n".format(ave_episode_reward))
        # Add ave reward to Tensorboard
        eval_episode_summary = tf.Summary()
        eval_episode_summary.value.add(simple_value=ave_episode_reward, tag="eval/reward")
        self.writer.add_summary(eval_episode_summary, cur_episode)
        print("===============evaluation finish=================")
        return ave_episode_reward

