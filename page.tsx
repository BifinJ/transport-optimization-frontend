import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- Data Classes for Environment Objects ---

@dataclass
class Package:
    """Represents a package with its properties."""
    id: int
    pickup_location: tuple[int, int]
    delivery_location: tuple[int, int]
    weight: float
    # Status: 0 = waiting, 1 = assigned/in-transit, 2 = delivered
    status: int = 0

@dataclass
class Vehicle:
    """Represents a vehicle with its properties."""
    id: int
    capacity: float
    current_location: tuple[int, int]
    speed: float
    cost_per_km: float
    # Time when the vehicle becomes available for its next task
    available_at_time: int = 0

# --- The Stabilized Logistics Environment ---

class LogisticsEnvironment:
    """Simulates the logistics environment for the RL agent with reduced variance."""
    def __init__(self, grid_size=15, n_packages=8, n_vehicles=3):  # Reduced complexity
        self.grid_size = grid_size
        self.n_packages = n_packages
        self.n_vehicles = n_vehicles
        self.max_time = 300  # Reduced further for more predictable episodes
        self.min_distance = 3  # Minimum distance between pickup and delivery

        self.packages = []
        self.vehicles = []
        self.current_time = 0
        self.packages_delivered = 0

        # Include "do nothing" action
        self.action_space_size = self.n_packages + 1
        self.state_size = (self.n_vehicles * 4) + (self.n_packages * 7) + 3  # Enhanced state

    def _generate_packages(self):
        """Generates packages with controlled variance."""
        packages = []
        for i in range(self.n_packages):
            pickup = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            
            # Ensure reasonable delivery distance (not too short or too long)
            max_attempts = 50
            attempts = 0
            while attempts < max_attempts:
                delivery = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                distance = abs(pickup[0] - delivery[0]) + abs(pickup[1] - delivery[1])
                if self.min_distance <= distance <= self.grid_size:  # Reasonable distance
                    break
                attempts += 1
            
            if attempts == max_attempts:  # Fallback
                delivery = (min(self.grid_size-1, pickup[0] + self.min_distance), pickup[1])
            
            packages.append(Package(
                id=i,
                pickup_location=pickup,
                delivery_location=delivery,
                weight=np.random.uniform(2, 6)  # More controlled weight range
            ))
        return packages

    def _generate_vehicles(self):
        """Generates vehicles with consistent capacities."""
        return [
            Vehicle(
                id=i,
                capacity=12.0,  # Fixed capacity for consistency
                current_location=(np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)),
                speed=1.0,
                cost_per_km=0.3  # Reduced cost variance
            ) for i in range(self.n_vehicles)
        ]

    def reset(self):
        """Resets the environment to an initial state."""
        self.packages = self._generate_packages()
        self.vehicles = self._generate_vehicles()
        self.current_time = 0
        self.packages_delivered = 0
        return self._get_state()

    def _get_state(self):
        """Constructs the normalized state vector with better features."""
        state = []
        
        # Vehicle states
        for vehicle in self.vehicles:
            state.extend([
                vehicle.current_location[0] / self.grid_size,
                vehicle.current_location[1] / self.grid_size,
                min(vehicle.available_at_time / self.max_time, 1.0),
                1.0  # Normalized capacity (fixed now)
            ])
        
        # Package states with additional features
        for package in self.packages:
            distance = self._calculate_distance(package.pickup_location, package.delivery_location)
            # Find closest vehicle to this package
            min_vehicle_dist = min([
                self._calculate_distance(v.current_location, package.pickup_location) 
                for v in self.vehicles
            ])
            
            state.extend([
                package.status / 2.0,
                package.pickup_location[0] / self.grid_size,
                package.pickup_location[1] / self.grid_size,
                package.delivery_location[0] / self.grid_size,
                package.delivery_location[1] / self.grid_size,
                distance / (2 * self.grid_size),  # Delivery distance
                min_vehicle_dist / (2 * self.grid_size)  # Distance to closest vehicle
            ])
        
        # Global state
        state.extend([
            self.current_time / self.max_time,
            self.packages_delivered / self.n_packages,
            len([p for p in self.packages if p.status == 0]) / self.n_packages  # Available packages ratio
        ])
        
        return np.array(state, dtype=np.float32)

    def _calculate_distance(self, loc1, loc2):
        """Calculates Manhattan distance."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def step(self, action):
        """Executes one time step with more stable rewards."""
        # Find the vehicle that becomes available earliest
        available_vehicle = min(self.vehicles, key=lambda v: v.available_at_time)
        self.current_time = max(self.current_time, available_vehicle.available_at_time)
        
        # Handle "wait" action (last action index)
        if action == len(self.packages):
            self.current_time += 1
            available_vehicle.available_at_time = self.current_time
            reward = -0.5  # Very small penalty for waiting
            done = self.packages_delivered == self.n_packages or self.current_time >= self.max_time
            return self._get_state(), reward, done, {}

        # Validate action
        if not (0 <= action < len(self.packages)):
            reward = -5  # Small penalty for invalid action
            done = self.packages_delivered == self.n_packages or self.current_time >= self.max_time
            return self._get_state(), reward, done, {}

        package_to_assign = self.packages[action]

        # Check if package is available and can fit
        if package_to_assign.status != 0:
            reward = -2  # Small penalty for already assigned package
            done = self.packages_delivered == self.n_packages or self.current_time >= self.max_time
            return self._get_state(), reward, done, {}
        
        if package_to_assign.weight > available_vehicle.capacity:
            reward = -3  # Small penalty for capacity violation
            done = self.packages_delivered == self.n_packages or self.current_time >= self.max_time
            return self._get_state(), reward, done, {}
        
        # Execute the delivery
        package_to_assign.status = 1
        dist_to_pickup = self._calculate_distance(available_vehicle.current_location, package_to_assign.pickup_location)
        dist_to_delivery = self._calculate_distance(package_to_assign.pickup_location, package_to_assign.delivery_location)
        total_dist = dist_to_pickup + dist_to_delivery
        travel_time = total_dist / available_vehicle.speed
        travel_cost = total_dist * available_vehicle.cost_per_km

        delivery_completion_time = self.current_time + travel_time
        available_vehicle.current_location = package_to_assign.delivery_location
        available_vehicle.available_at_time = delivery_completion_time
        package_to_assign.status = 2
        self.packages_delivered += 1

        # Simplified, more stable reward structure
        base_reward = 10  # Base reward for successful delivery
        distance_penalty = total_dist * 0.1  # Small distance penalty
        time_factor = min(delivery_completion_time / self.max_time, 1.0)
        time_penalty = time_factor * 2  # Small time penalty
        
        reward = base_reward - distance_penalty - time_penalty
        
        # Check if episode is done
        done = False
        if self.packages_delivered == self.n_packages:
            done = True
            # Modest completion bonus
            remaining_time_bonus = max(0, (self.max_time - self.current_time) / self.max_time) * 20
            reward += 30 + remaining_time_bonus
        elif self.current_time >= self.max_time:
            done = True
            # Small penalty for timeout
            reward -= 10

        return self._get_state(), reward, done, {}

# --- Simplified Experience Replay Buffer ---

class SimpleReplayBuffer:
    """Simple experience replay buffer for more stable training."""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# --- Stabilized DQN Agent ---

class StableDQNAgent:
    """A more stable DQN agent with reduced variance."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = SimpleReplayBuffer(100000)
        self.gamma = 0.99
        self.epsilon = 0.9  # Start with lower epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001  # Lower learning rate for stability
        self.batch_size = 32
        self.update_target_freq = 200  # Less frequent updates
        self.train_freq = 4  # Train every 4 steps
        
        # Build networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
        # Training tracking
        self.training_step = 0
        self.loss_history = []
        self.step_count = 0

    def _build_model(self):
        """Builds a simpler, more stable network."""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        # Use a more stable optimizer configuration
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=1.0,  # Gradient clipping
            epsilon=1e-8
        )
        
        model.compile(optimizer=optimizer, loss='huber')  # Huber loss is more stable
        return model

    def update_target_network(self):
        """Soft update of target network for stability."""
        tau = 0.005  # Soft update parameter
        target_weights = self.target_network.get_weights()
        main_weights = self.q_network.get_weights()
        
        for i in range(len(target_weights)):
            target_weights[i] = tau * main_weights[i] + (1 - tau) * target_weights[i]
        
        self.target_network.set_weights(target_weights)

    def remember(self, state, action, reward, next_state, done):
        # Clip rewards for stability
        clipped_reward = np.clip(reward, -50, 50)
        self.memory.add((state, action, clipped_reward, next_state, done))

    def act(self, state, valid_actions=None):
        """Choose action with epsilon-greedy policy."""
        self.step_count += 1
        
        # Epsilon-greedy with valid action masking
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)[0]
        
        if valid_actions is not None and len(valid_actions) > 0:
            # Mask invalid actions
            masked_q_values = np.full(self.action_size, -np.inf)
            for action in valid_actions:
                if 0 <= action < self.action_size:
                    masked_q_values[action] = q_values[action]
            return np.argmax(masked_q_values)
        
        return np.argmax(q_values)

    def replay(self):
        """Train the agent with experience replay."""
        if len(self.memory) < self.batch_size * 4:
            return
            
        if self.step_count % self.train_freq != 0:
            return

        minibatch = self.memory.sample(self.batch_size)
        if not minibatch:
            return
            
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])

        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Calculate targets
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train the network
        history = self.q_network.fit(
            states, targets,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0
        )
        
        if 'loss' in history.history:
            self.loss_history.append(history.history['loss'][0])
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.update_target_freq == 0:
            self.update_target_network()

        # Decay epsilon more gradually
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_path):
        print(f"Saving trained model weights to {file_path}")
        self.q_network.save_weights(file_path)

    def load_model(self, file_path):
        if os.path.exists(file_path):
            print(f"Loading model weights from {file_path}")
            self.q_network.load_weights(file_path)
            self.update_target_network()
        else:
            print(f"Model file not found at {file_path}. Starting with a new model.")

# --- Stabilized Training Function ---

def train_stable_optimizer(episodes=3000, save_interval=500, early_stopping_patience=500):
    env = LogisticsEnvironment()
    agent = StableDQNAgent(env.state_size, env.action_space_size)
    training_history = []
    best_avg_reward = -float('inf')
    patience_counter = 0
    
    print(f"Starting stable training for {episodes} episodes...")
    print(f"State size: {env.state_size}, Action size: {env.action_space_size}")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_rewards = []
        
        while True:
            # Get valid actions
            available_packages = [i for i, pkg in enumerate(env.packages) if pkg.status == 0]
            valid_actions = available_packages + [env.action_space_size - 1]  # Add wait action
            
            action = agent.act(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            episode_rewards.append(reward)
            steps += 1
            
            if done:
                break
        
        # Train the agent
        agent.replay()
        
        # Record training data with additional stability metrics
        reward_std = np.std(episode_rewards) if len(episode_rewards) > 1 else 0
        training_history.append({
            'episode': episode,
            'total_reward': total_reward,
            'delivered': env.packages_delivered,
            'epsilon': agent.epsilon,
            'steps': steps,
            'completion_rate': env.packages_delivered / env.n_packages,
            'reward_std': reward_std,
            'avg_step_reward': total_reward / steps if steps > 0 else 0
        })

        # Progress reporting with stability metrics
        if episode % 100 == 0 and episode > 0:
            recent_history = training_history[-100:]
            avg_reward = np.mean([h['total_reward'] for h in recent_history])
            reward_variance = np.var([h['total_reward'] for h in recent_history])
            avg_delivered = np.mean([h['delivered'] for h in recent_history])
            avg_completion = np.mean([h['completion_rate'] for h in recent_history])
            avg_steps = np.mean([h['steps'] for h in recent_history])
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} ± {np.sqrt(reward_variance):5.2f} | "
                  f"Avg Delivered: {avg_delivered:4.1f} | "
                  f"Completion: {avg_completion:5.1%} | "
                  f"Avg Steps: {avg_steps:4.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
            
            # Early stopping based on reward stability
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                patience_counter = 0
                agent.save_model("best_stable_logistics_model.h5")
            else:
                patience_counter += 100
                
            if patience_counter >= early_stopping_patience and episode > 1000:
                print(f"Early stopping at episode {episode} due to no improvement")
                break
        
        # Periodic saves
        if episode % save_interval == 0 and episode > 0:
            agent.save_model(f"stable_logistics_model_ep_{episode}.h5")

    print("Training completed!")
    agent.save_model("final_stable_logistics_model.h5")
    
    # Enhanced visualization with variance analysis
    episodes_list = [h['episode'] for h in training_history]
    rewards_list = [h['total_reward'] for h in training_history]
    delivered_list = [h['delivered'] for h in training_history]
    completion_rates = [h['completion_rate'] for h in training_history]
    reward_stds = [h['reward_std'] for h in training_history]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Rewards plot with confidence intervals
    ax1.plot(episodes_list, rewards_list, color='b', alpha=0.2, label='Episode Reward')
    if len(rewards_list) >= 100:
        running_avg = np.convolve(rewards_list, np.ones(100)/100, mode='valid')
        running_std = np.array([np.std(rewards_list[max(0, i-49):i+50]) for i in range(49, len(rewards_list))])
        ax1.plot(episodes_list[99:], running_avg, color='r', linewidth=2, label='100-ep Moving Avg')
        ax1.fill_between(episodes_list[99:], 
                        running_avg - running_std, 
                        running_avg + running_std, 
                        color='r', alpha=0.2, label='±1 Std Dev')
    ax1.set_title('Training Rewards over Time (with Variance)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Completion rate
    ax2.plot(episodes_list, completion_rates, color='g', alpha=0.3)
    if len(completion_rates) >= 100:
        running_avg_completion = np.convolve(completion_rates, np.ones(100)/100, mode='valid')
        ax2.plot(episodes_list[99:], running_avg_completion, color='darkgreen', linewidth=2)
    ax2.set_title('Completion Rate over Time')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Completion Rate')
    ax2.set_ylim([0, 1.1])
    ax2.grid(True)
    
    # Reward standard deviation (within episode)
    ax3.plot(episodes_list, reward_stds, color='orange', alpha=0.5)
    if len(reward_stds) >= 100:
        running_avg_std = np.convolve(reward_stds, np.ones(100)/100, mode='valid')
        ax3.plot(episodes_list[99:], running_avg_std, color='red', linewidth=2)
    ax3.set_title('Reward Variance within Episodes')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward Std Dev')
    ax3.grid(True)
    
    # Training loss
    if agent.loss_history:
        steps_per_loss = len(episodes_list) / len(agent.loss_history)
        loss_episodes = [i * steps_per_loss for i in range(len(agent.loss_history))]
        ax4.plot(loss_episodes, agent.loss_history, color='purple', alpha=0.3)
        if len(agent.loss_history) >= 50:
            loss_smooth = np.convolve(agent.loss_history, np.ones(50)/50, mode='valid')
            ax4.plot(loss_episodes[49:], loss_smooth, color='red', linewidth=2)
        ax4.set_title('Training Loss over Time')
        ax4.set_xlabel('Episode (approx)')
        ax4.set_ylabel('Loss')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('stable_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return training_history

def evaluate_stable_model(model_path, episodes=20):
    """Evaluate the trained stable model."""
    env = LogisticsEnvironment()
    agent = StableDQNAgent(env.state_size, env.action_space_size)
    agent.load_model(model_path)
    agent.epsilon = 0  # No exploration during evaluation
    
    results = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_rewards = []
        
        while True:
            available_packages = [i for i, pkg in enumerate(env.packages) if pkg.status == 0]
            valid_actions = available_packages + [env.action_space_size - 1]
            
            action = agent.act(state, valid_actions)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            episode_rewards.append(reward)
            steps += 1
            
            if done:
                break
        
        reward_std = np.std(episode_rewards) if len(episode_rewards) > 1 else 0
        results.append({
            'episode': episode,
            'reward': total_reward,
            'delivered': env.packages_delivered,
            'completion_rate': env.packages_delivered / env.n_packages,
            'steps': steps,
            'reward_std': reward_std
        })
        
        print(f"Eval Episode {episode:2d}: Reward={total_reward:6.2f}, "
              f"Delivered={env.packages_delivered}/{env.n_packages}, "
              f"Steps={steps:3d}, Reward_Std={reward_std:.2f}")
    
    avg_reward = np.mean([r['reward'] for r in results])
    reward_variance = np.var([r['reward'] for r in results])
    avg_completion = np.mean([r['completion_rate'] for r in results])
    avg_reward_std = np.mean([r['reward_std'] for r in results])
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {np.sqrt(reward_variance):.2f}")
    print(f"Average Completion Rate: {avg_completion:.2%}")
    print(f"Average Within-Episode Reward Std: {avg_reward_std:.2f}")
    print(f"Reward Coefficient of Variation: {np.sqrt(reward_variance)/abs(avg_reward):.3f}")
    
    return results

if __name__ == "__main__":
    # Train the stable model
    history = train_stable_optimizer(episodes=3000)
    
    # Evaluate the best model
    print("\n" + "="*60)
    print("Evaluating best stable model...")
    evaluate_stable_model("best_stable_logistics_model.h5", episodes=20)