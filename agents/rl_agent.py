from soccerpy.agent import Agent as BaseAgent
from soccerpy.world_model import WorldModel

import collections
import numpy as np
import tensorflow as tf
import statistics 
import tqdm
# from tensorflow.keras import layers

layers = tf.keras.layers
seed = 42
tf.random.set_seed(seed)

from typing import Any, List, Sequence, Tuple

# reinforcement learning agent
env = None

class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)

class Agent():

  def __init__(self): # DO WE WANT TO HAVE PARAMETERS IN INIT TO CHANGE THIS?
    # for example
    # self.eps = eps

    num_actions = env.action_space.n
    num_hidden_units = 128
    self.model = ActorCritic(num_actions, num_hidden_units)

    self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    self.eps = np.finfo(np.float32).eps.item()
    self.learning_rate = 0.01



    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

  def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""

    # state, reward, done, truncated, info = env.step(action)
    state, reward, done, truncated = env.step(action)

    return (state.astype(np.float32), 
            np.array(reward, np.int32), 
            np.array(done, np.int32))

  def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(self.env_step, [action], 
                            [tf.float32, tf.int32, tf.int32])

  def run_episode(
      self,
      initial_state: tf.Tensor,  
      model: tf.keras.Model, 
      max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
      # Convert state into a batched tensor (batch size = 1)

      prev = state

      state = tf.expand_dims(state, 0) # [x,y] => [[x,y]]
    
      # Run the model and to get action probabilities and critic value
      action_logits_t, value = model(state) # model: actor critic class defined above, returns the actor and critic values for current step
    
      # Sample next action from the action probability distribution
      action = tf.random.categorical(action_logits_t, 1)[0, 0]
      action_probs_t = tf.nn.softmax(action_logits_t) # converts action values to probabilities based off given values

      # Store critic values
      values = values.write(t, tf.squeeze(value)) # squeeze: removes any size 1 dim in the shape of t (in this context, just pops the value out of array)

      # Store log probability of the action chosen
      action_probs = action_probs.write(t, action_probs_t[0, action]) # stores the log prob of the chosen occuring along with the time step it occurred in
    
      # Apply action to the environment to get next state and reward
      state, reward, done = self.tf_env_step(action)
      state.set_shape(initial_state_shape) # why do we need to change the shape back to the initial shape? (what is the point of changing the shape in the first place?)
    
      # Store reward
      rewards = rewards.write(t, reward)

      if tf.cast(done, tf.bool): # if done = true, but why do we have to cast it to a new tensor?
        break
      
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    
    return action_probs, values, rewards

  def get_expected_return(
      self,
      rewards: tf.Tensor, 
      gamma: float, 
      standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
      reward = rewards[i]
      discounted_sum = reward + gamma * discounted_sum
      discounted_sum.set_shape(discounted_sum_shape)
      returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
      returns = ((returns - tf.math.reduce_mean(returns)) / 
                (tf.math.reduce_std(returns) + self.eps))

    return returns


  def compute_loss(
      self, 
      action_probs: tf.Tensor,  
      values: tf.Tensor,  
      returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined Actor-Critic loss."""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = self.huber_loss(values, returns)

    return actor_loss + critic_loss

  @tf.function
  def train_step(
      self,
      initial_state: tf.Tensor, 
      model: tf.keras.Model, 
      optimizer: tf.keras.optimizers.Optimizer, 
      gamma: float, 
      max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""
    ''' runs an episode then handles calculation (expected return, loss, gradients, episode reward) and updating of training parameters and the episode return'''

    with tf.GradientTape() as tape: # GradientTape allows for automated differentiation

      # Run the model for one episode to collect training data

      # print('TRAIN_STEP')

      action_probs, values, rewards = self.run_episode(
          initial_state, model, max_steps_per_episode) # action_probs, values, rewards are arrays of the values of each time step in one episode

      # Calculate the expected returns
      returns = self.get_expected_return(rewards, gamma) # array of expected return for each time step (with discounts of future time steps), float32

      # print(f'RETURNS = {returns}')

      # Convert training data to appropriate TF tensor shapes

      action_probs, values, returns = [
          tf.expand_dims(x, 1) for x in [action_probs, values, returns] # puts all values into their individual 'rows'
          ]

      # Calculate the loss values to update our network
      loss = self.compute_loss(action_probs, values, returns) # the loss of network, float32

      # print(f'LOSS = {loss}')

    # Compute the gradients from the loss 
    grads = tape.gradient(loss, model.trainable_variables) # array of gradients to train parameters, float32

    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards) #sums rewards, int32

    return episode_reward

  def train(self):
    # %%time

    tf.config.run_functions_eagerly(True)

    min_episodes_criterion = 100
    max_episodes = 10000
    max_steps_per_episode = 500

    # `CartPole-v1` is considered solved if average reward is >= 475 over 500 
    # consecutive trials
    reward_threshold = 475
    running_reward = 0

    # The discount factor for future rewards
    gamma = 0.99

    # Keep the last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    t = tqdm.trange(max_episodes)

    for i in t:
        ''' sets up initial episode state and culminates all episode rewards'''
        # print('iter', i)
        # print(env.reset())
        # initial_state, info = env.reset()
        
        initial_state = env.reset() # state: cart position, cart-velocity, pole angle and pole velocity

        initial_state = tf.constant(initial_state, dtype=tf.float32)


        episode_reward = int(self.train_step(
            initial_state, self.model, self.optimizer, gamma, max_steps_per_episode)) # runs an episode via train step function and gets reward of the episode
        

        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)

        # print(f'EPISODE RESULTS = initial state: {initial_state}; reward for episode {i}: {episode_reward}')

        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)
      
        # Show the average episode reward every 10 episodes
        if i % 10 == 0:
          pass # print(f'Episode {i}: average reward: {avg_reward}')
      
        if running_reward > reward_threshold and i >= min_episodes_criterion:  
            break
        
        # input(f'episode {i}')


    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')