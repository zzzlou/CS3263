import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------
# Environment Definition
# --------------------------
class PreferenceEnv:
    def __init__(self):
        # Domains for each variable
        self.domains = {
            'CS3264': [0, 1, 2],
            'CS3263': [0, 1, 2, 3],
            'CS4248': [0, 1, 2, 3, 4, 5, 6]
        }
        self.state = None
        self.reset()

    def reset(self):
        """Randomly initialize the state."""
        self.state = (
            random.choice(self.domains['CS3264']),
            random.choice(self.domains['CS3263']),
            random.choice(self.domains['CS4248'])
        )
        return self.state

    def step(self, action):
        """
        Action space (total 7 actions):
           0: increase CS3264,
           1: decrease CS3264,
           2: increase CS3263,
           3: decrease CS3263,
           4: increase CS4248,
           5: decrease CS4248,
           6: no-op.
        """
        s = list(self.state)
        if action == 0 and s[0] < max(self.domains['CS3264']):
            s[0] += 1
        elif action == 1 and s[0] > min(self.domains['CS3264']):
            s[0] -= 1
        elif action == 2 and s[1] < max(self.domains['CS3263']):
            s[1] += 1
        elif action == 3 and s[1] > min(self.domains['CS3263']):
            s[1] -= 1
        elif action == 4 and s[2] < max(self.domains['CS4248']):
            s[2] += 1
        elif action == 5 and s[2] > min(self.domains['CS4248']):
            s[2] -= 1
        # If action == 6: do nothing

        self.state = tuple(s)
        # Calculate ground–truth reward (hidden to the agent)
        reward = self.ground_truth_reward(self.state)
        done = False  # Termination can be defined by episode length.
        return self.state, reward, done

    def ground_truth_reward(self, state):
        """
        Hidden reward: for illustration we define it as the negative squared Euclidean
        distance to a goal state (here the goal is set to the maximum in each domain).
        """
        goal = (
            max(self.domains['CS3264']),
            max(self.domains['CS3263']),
            max(self.domains['CS4248'])
        )
        distance_sq = sum((s - g) ** 2 for s, g in zip(state, goal))
        return -distance_sq

# --------------------------
# Reward Model (Neural Network)
# --------------------------
class RewardModel(nn.Module):
    def __init__(self, input_dim=3):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)  # output is a scalar reward

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --------------------------
# Trajectory Generation & Simulated Preference Feedback
# --------------------------
def generate_trajectory(env, policy, traj_len=5, start_state=None):
    """
    Generate a trajectory (a list of (state, action, next_state)) starting from
    start_state if provided; otherwise, reset the environment.
    """
    trajectory = []
    # If a starting state is provided, force the environment to begin there.
    if start_state is not None:
        env.state = start_state
    else:
        start_state = env.reset()
    
    state = env.state
    for _ in range(traj_len):
        action = policy(state)
        next_state, _, _ = env.step(action)
        trajectory.append((state, action, next_state))
        state = next_state
    return trajectory

def trajectory_reward(traj, env):
    """
    For simulation purposes, compute the cumulative reward of a trajectory
    using the ground–truth reward (this is hidden from the agent).
    """
    total = 0
    # You might use state or next_state; here we sum over next_states.
    for (_, _, next_state) in traj:
        total += env.ground_truth_reward(next_state)
    return total

def prefer(traj_A, traj_B, env):
    """
    Simulated human preference: compares two trajectories.
    Returns 1 if traj_A is preferred (has a higher cumulative reward)
    and 0 otherwise.
    """
    if trajectory_reward(traj_A, env) > trajectory_reward(traj_B, env):
        return 1
    else:
        return 0

# --------------------------
# Utility Functions for Reward Model Training
# --------------------------
def compute_traj_predicted_reward(traj, reward_model):
    """
    Compute cumulative predicted reward for a trajectory (summing over states).
    Here we use the state (not next_state) for prediction.
    """
    total = 0
    for (state, _, _) in traj:
        state_tensor = torch.FloatTensor(state)
        total += reward_model(state_tensor)
    return total

def train_reward_model(reward_model, optimizer, preference_pairs, num_epochs=10):
    """
    Train the reward model using preference pairs.
    Each pair is a tuple: (traj_A, traj_B, preference)
    where preference is 1 if A is preferred over B, else 0.
    We use a binary cross-entropy loss on the difference between cumulative rewards.
    """
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        total_loss = 0
        for traj_A, traj_B, pref in preference_pairs:
            r_A = compute_traj_predicted_reward(traj_A, reward_model)
            r_B = compute_traj_predicted_reward(traj_B, reward_model)
            # The logit difference
            diff = r_A - r_B  
            target = torch.tensor([[float(pref)]])
            loss = criterion(diff.unsqueeze(0), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Reward Model Train - Epoch {epoch+1}, Loss: {total_loss:.4f}")

# --------------------------
# Tabular Q-Learning Setup
# --------------------------
def initialize_Q(domains):
    """Initialize Q–values for every state in the Cartesian product."""
    Q = {}
    for s1 in domains['CS3264']:
        for s2 in domains['CS3263']:
            for s3 in domains['CS4248']:
                state = (s1, s2, s3)
                Q[state] = [0.0] * 7  # 7 possible actions
    return Q

def q_learning_update(Q, state, action, reward, next_state, alpha, gamma):
    """A simple Q–learning update on the Q–table."""
    current = Q[state][action]
    next_max = max(Q[next_state])
    Q[state][action] = current + alpha * (reward + gamma * next_max - current)

def q_policy(state, Q, epsilon):
    """Epsilon–greedy policy based on Q–table."""
    if random.random() < epsilon:
        return random.choice(range(7))
    else:
        return int(np.argmax(Q[state]))

def find_best_state(Q):
    best_state = None
    best_value = -float('inf')
    for state, q_values in Q.items():
        state_value = max(q_values)  # Value of the state is the best Q–value from that state.
        if state_value > best_value:
            best_value = state_value
            best_state = state
    return best_state, best_value

def simulated_user_preference(traj_A, traj_B, env, noise_std=0.5):
    # Compute the true cumulative rewards using the ground–truth function.
    true_reward_A = trajectory_reward(traj_A, env)
    true_reward_B = trajectory_reward(traj_B, env)
    
    # Add Gaussian noise to simulate uncertainty or variability in human judgment.
    noisy_reward_A = true_reward_A + np.random.normal(0, noise_std)
    noisy_reward_B = true_reward_B + np.random.normal(0, noise_std)
    
    # Return 1 if trajectory A is (noisily) better, 0 otherwise.
    return 1 if noisy_reward_A > noisy_reward_B else 0

def interactive_user_preference(traj_A, traj_B, env):
    # Convert trajectory to a readable string, showing the sequence of states.
    def traj_to_str(traj):
        return " -> ".join([str(state) for (state, _, _) in traj])
    
    print("\nTrajectory A:")
    print(traj_to_str(traj_A))
    
    print("\nTrajectory B:")
    print(traj_to_str(traj_B))
    
    # Prompt the user for their preference.
    while True:
        choice = input("Which trajectory do you prefer? (Enter 'A' or 'B'): ").strip().upper()
        if choice in ["A", "B"]:
            break
        else:
            print("Invalid input. Please enter 'A' or 'B'.")
    
    # Return 1 if trajectory A is preferred, and 0 if trajectory B is preferred.
    return 1 if choice == "A" else 0

# After training, call the function:


# --------------------------
# Main Preference-based RL Loop
# --------------------------
if __name__ == "__main__":
    # Initialize environment, Q–table, and reward model
    env = PreferenceEnv()
    Q = initialize_Q(env.domains)
    reward_model = RewardModel(input_dim=3)
    optimizer = optim.Adam(reward_model.parameters(), lr=0.01)
    
    # Hyperparameters for training
    num_episodes = 10
    max_steps = 5
    alpha = 1          # learning rate for Q–learning
    gamma = 0.99       # discount factor
    epsilon = 0.6      # exploration rate
    update_reward_model_every = 5  # update every 10 episodes
    
    # Buffers to store trajectories for preference comparisons
    traj_buffer = []
    preference_pairs = []  # List of tuples: (traj_A, traj_B, preference)
    
    # Main loop: alternate between Q–learning updates and reward model training
    for episode in range(1, num_episodes + 1):
        # Store the starting state for this episode for later trajectory comparison.
        init_state = env.reset()
        state = init_state
        trajectory = []
        
        # Run the Q–learning episode from init_state.
        for t in range(max_steps):
            action = q_policy(state, Q, epsilon)
            next_state, _, _ = env.step(action)
            
            # Get predicted reward from the reward model.
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                pred_reward = reward_model(state_tensor).item()
            
            # Q–learning update using the predicted reward.
            q_learning_update(Q, state, action, pred_reward, next_state, alpha, gamma)
            
            trajectory.append((state, action, next_state))
            state = next_state
            
        # Save the trajectory from the Q–learning episode.
        traj_buffer.append(trajectory)
        
        # Generate two trajectories for interactive preference feedback.
        # Both trajectories are generated from the same fixed starting state (init_state).
        env.state = init_state  
        traj_A = generate_trajectory(env, lambda s: q_policy(s, Q, epsilon), traj_len=max_steps, start_state=init_state)
        
        env.state = init_state  
        traj_B = generate_trajectory(env, lambda s: q_policy(s, Q, epsilon), traj_len=max_steps, start_state=init_state)
        
        # Use the interactive function for user to compare the two trajectories.
        pref = interactive_user_preference(traj_A, traj_B, env)
        preference_pairs.append((traj_A, traj_B, pref))
        
        # Periodically update the reward model using the collected preference pairs.
        if episode % update_reward_model_every == 0 and preference_pairs:
            print(f"\n=== Updating Reward Model at episode {episode} ===")
            train_reward_model(reward_model, optimizer, preference_pairs, num_epochs=5)
            # Optionally clear the preference pairs after updating.
            preference_pairs = []
        
        if episode % 10 == 0:
            print(f"Episode {episode} complete.")


    # After training, you might test the learned policy.
    # Here we simply print the Q–values for a sample state.
    sample_state = (0, 0, 0)
    print("\nLearned Q–values for state", sample_state, ":", Q[sample_state])
    best_state, best_value = find_best_state(Q)
    print("Best state found:", best_state, "with value:", best_value)