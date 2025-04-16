import gym
import numpy as np
import random
import torch.nn as nn
import torch

# ===============================
# 简化版一天内课表环境，状态表示 s = (earliest_start, total_attended, gpa, fatigue, day)
# ===============================
import gym
import numpy as np
import random
import torch
import torch.nn as nn

class SimpleCourseScheduleEnv(gym.Env):
    """
    A simplified daily timetable environment where the state is defined as:
      s = (gpa_level, fatigue_level, day, current_required, current_start_level)
    
    Here:
      - gpa_level: Discrete GPA level (1: GPA < 3, 2: 3 ≤ GPA < 4, 3: 4 ≤ GPA < 5)
      - fatigue_level: Discrete fatigue level (1, 2, or 3) computed from a continuous fatigue value
      - day: Day of the week (e.g., 0 for Monday)
      - current_required: Binary indicator for the upcoming course (1 if required, 0 otherwise)
      - current_start_level: Discrete representation of the current course's start time:
            1 for courses starting before 10 AM,
            2 for courses starting between 10 AM and 14 PM,
            3 for courses starting at or after 14 PM.
    
    The environment uses the given day_schedule (a list of tuples, e.g.,
       [("ES2660", (9, 12), True), ("CS3263", (14, 15), True), ("CS2105", (17, 18), True)])
    to simulate daily attendance decisions.
    """
    def __init__(self, day_schedule, day=0, reward_model=None):
        super().__init__()
        self.day_schedule = day_schedule
        self.num_courses = len(day_schedule)
        self.day = day
        self.action_space = gym.spaces.Discrete(2)  # 0: skip class, 1: attend class
        self.reward_model = reward_model
        self.reset()
        
    def reset(self):
        # Initialize state variables
        # Initial GPA is 4.0; initial fatigue is 0.0; no course has been decided yet.
        self.gpa = 4.0
        self.fatigue = 0.0
        self.current_course = 0
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self):
        # Discretize fatigue into 3 levels
        if self.fatigue < 1.5:
            fatigue_level = 1
        elif self.fatigue < 2.5:
            fatigue_level = 2
        else:
            fatigue_level = 3
        
        # Discretize GPA into three levels:
        if self.gpa < 3.0:
            gpa_level = 1
        elif self.gpa < 4.0:
            gpa_level = 2
        else:
            gpa_level = 3

        # For the upcoming course, if any:
        if self.current_course < self.num_courses:
            course = self.day_schedule[self.current_course]
            # course is of the form (course_code, (start, end), required)
            current_required = 1 if course[2] else 0
            start = course[1][0]  # course start time (in hours)
            # Discretize current_start_time: assume range 8-18 hours, then subtract 8 and divide by 10,
            # then assign bins: 1 if < (10-8)/10, 2 if < (14-8)/10, 3 otherwise.
            if start < 10:
                current_start_level = 1
            elif start < 14:
                current_start_level = 2
            else:
                current_start_level = 3
        else:
            current_required = 0
            current_start_level = 0

        return (gpa_level, fatigue_level, self.day, current_required, current_start_level)

    def step(self, action):
        """
        Full step function:
          1. Call transition(action) to obtain next_state, done, info;
          2. Compute immediate reward using compute_reward(next_state);
          3. Accumulate total_reward and return (next_state, reward, done, info).
        """
        next_state, done, info = self.transition(action)
        reward = self.compute_reward(next_state)
        self.total_reward += reward
        return next_state, reward, done, info
    
    def transition(self, action):
        """
        Update state based on action without computing reward.
        Update rules:
          - If action == 1 (attend):
                Increase fatigue by 0.5 * duration (or more if the current class starts before 10 AM).
                If the current course starts before 10 AM (i.e., current_start_level == 1),
                increase fatigue by 0.7 * duration instead.
                With a probability of 30%, increase GPA by a random value between 0.3 and 0.6 (capped at 5.0).
          - If action == 0 (skip):
                Decrease fatigue by a random value between 0.5 and 1.
                If the course is required, then with a 70% probability, decrease GPA by 1.
          Increment current_course. When all courses are processed, set done = True.
        Return the new state, done flag, and an info dictionary.
        """
        done = False
        info = {}
        if self.current_course >= self.num_courses:
            done = True
            return self._get_state(), done, info

        course = self.day_schedule[self.current_course]
        course_code, time_range, required = course
        start, end = time_range
        duration = end - start

        if action == 1:  # Attend class
            # Increase fatigue: if the current class starts before 10, use a higher multiplier.
            if start < 10:
                self.fatigue = min(4, self.fatigue + 0.7 * duration)
            else:
                self.fatigue = min(4, self.fatigue + 0.5 * duration)
            if random.random() < 0.3:
                self.gpa = min(5.0, self.gpa + random.uniform(0.3, 0.6))
        else:  # Skip class
            self.fatigue = max(0.0, self.fatigue - random.uniform(0.5, 1))
            if required:
                if random.random() < 0.7:
                    self.gpa = max(0.0, self.gpa - 1)

        self.current_course += 1
        if self.current_course >= self.num_courses:
            done = True

        next_state = self._get_state()
        return next_state, done, info

    def compute_reward(self, state):
        """
        Compute the reward using the trained reward model if provided.
        The state format is: (gpa_level, fatigue_level, day, current_required, current_start_level)
        The state must match the format used during reward model training.
        If no reward model is provided, a fallback rule-based reward is used.
        """
        if self.reward_model is not None:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            self.reward_model.eval()
            with torch.no_grad():
                reward_tensor = self.reward_model(state_tensor)
            return reward_tensor.item()
        else:
            # Fallback rule-based reward:
            # Here we want to reward higher GPA, lower fatigue, and penalize very early start times.
            gpa_level, fatigue_level, day, current_required, current_start_level = state
            return 1.0 * gpa_level - 3.0 * fatigue_level - 10.0 * current_start_level

    def render(self, mode='human'):
        print("State (gpa_level, fatigue_level, day, current_required, current_start_level):", self._get_state())
        print("Total Reward:", self.total_reward)
        print("-" * 40)

# ===============================
# Q-Learning Agent (Tabular)
# ===============================

class RewardModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim1=32, hidden_dim2=16, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x  # Output shape: [batch_size, 1]
def discretize_state(state):

    return state

def q_learning_agent(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.2):

    Q = {}
    
    def get_Q(state, action):
        return Q.get((state, action), 0.0)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            state_key = discretize_state(state)
            # ε-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_vals = [get_Q(state_key, a) for a in range(env.action_space.n)]
                action = int(np.argmax(q_vals))
            next_state, reward, done, info = env.step(action)
            next_state_key = discretize_state(next_state)
            # Q-learning update
            future_q = max([get_Q(next_state_key, a) for a in range(env.action_space.n)]) if not done else 0.0
            current_q = get_Q(state_key, action)
            new_q = current_q + alpha * (reward + gamma * future_q - current_q)
            Q[(state_key, action)] = new_q
            state = next_state
    return Q

def q_policy(state, env, Q):
    state_key = discretize_state(state)
    q_vals = [Q.get((state_key, a), 0.0) for a in range(env.action_space.n)]
    return int(np.argmax(q_vals))

# ===============================
# Other agent policies
# ===============================
def random_agent_policy(state, env):
    # Returns a random action from the environment's action space.
    return env.action_space.sample()

def hardworking_agent_policy(state, env):
    """
    A hardworking agent always attends class.
    This policy is simplistic: it returns 1 for every decision.
    """
    return 1

def lazy_agent_policy(state, env):
    """
    A lazy agent prefers skipping classes, with some exceptions.

    Given the state: 
      (gpa_level, fatigue_level, day, current_required, current_start_level)
    
    The policy behaves as follows:
      - If it's Friday (day == 5), the agent always skips (returns 0).
      - For required courses (current_required == 1):  
          * If gpa_level is low (i.e., 1), attend (return 1) to help maintain GPA.
          * Otherwise, skip with a high probability (e.g., 80% skip, 20% attend).
      - For non-required courses:
          * If the course starts early (current_start_level == 1), skip (return 0).
          * Otherwise, skip with a high probability (e.g., 90% skip, 10% attend).
    """
    gpa_level, fatigue_level, day, current_required, current_start_level = state
    
    # Always skip on Friday.
    if day == 5:
        return 0

    if current_required == 1:
        if gpa_level == 1:  # Very low GPA: might decide to attend to prevent a drop.
            return 1
        else:
            # For required classes with moderate/high GPA, mostly skip (80% skip).
            return 0 if random.random() < 0.8 else 1
    else:
        # For non-required classes, be extra lazy.
        if current_start_level == 1:  # Early classes are heavily avoided.
            return 0
        else:
            return 0 if random.random() < 0.9 else 1


# ===============================
# 策略评估
# ===============================
def evaluate_policy(env, policy_func, num_episodes=10, Q=None, render=False):
    total_rewards = []
    trajs = []
    for episode in range(num_episodes):
        traj = []
        state = env.reset()
        done = False
        ep_reward = 0
        print(f"Episode {episode+1} start:")
        while not done:
            if Q is not None:
                action = q_policy(state, env, Q)
            else:
                action = policy_func(state, env)
            state, reward, done, info = env.step(action)
            traj.append((action,state))
            ep_reward += reward
            if render:
                env.render()
        total_rewards.append(ep_reward)
        # print(f"Episode {episode+1} total reward: {ep_reward:.2f}\n")
        
        trajs.append(traj)
    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    return trajs

def generate_schedule_trajectory(day_schedule, actions):
    """
    根据完整课表 day_schedule 和决策序列 actions（列表，其中元素为0或1），
    产生表面形式上的课表演化轨迹。
    
    参数:
      day_schedule: 完整课表列表，每个元素为 (course_code, (start_time, end_time), required)
      actions: 决策序列，列表中的每个元素 0 表示“不去上课”，1 表示“去上课”
    
    返回:
      trajectory: 轨迹列表，每个元素是一个列表，表示到当前步骤为止代理所选（上）的课程。
                  初始状态为空列表。
    """
    trajectory = []
    current_schedule = []
    # 初始状态：还没有决策，显示空的课表
    trajectory.append(current_schedule.copy())
    
    # 对于 day_schedule 中的每一门课程，对应一个决策
    for i, action in enumerate(actions):
        if i >= len(day_schedule):
            break  # 如果决策数量超过课程数量则停止
        if action == 1:
            current_schedule.append(day_schedule[i])
        trajectory.append(current_schedule.copy())
    
    return trajectory

# day_schedules = [
#     [("CSE1895", (8, 9), True), ("CSE4037", (9, 10), True), ("CSE4016", (11, 12), True), ("CSE3418", (16, 17), True)],
#     [("CSE1814", (8, 11), False), ("CSE4108", (12, 13), False), ("CSE2138", (14, 16), True), ("CSE1902", (16, 18), True)],
#     [("CSE2556", (9, 10), True), ("CSE2083", (13, 16), False), ("CSE3364", (17, 19), True)],
#     [("CSE1933", (8, 11), False), ("CSE3603", (12, 14), False)],
#     [("CSE3600", (9, 12), True), ("CSE1858", (13, 15), True)],
#     [("CSE4366", (8, 9), True), ("CSE2893", (11, 12), True), ("CSE2096", (13, 15), True), ("CSE3804", (16, 17), True)],
#     [("CSE3087", (11, 12), True), ("CSE3684", (13, 14), True), ("CSE2084", (15, 16), True), ("CSE2076", (16, 19), False)],
#     [("CSE3570", (9, 10), True), ("CSE1260", (14, 17), True)],
#     [("CSE3952", (8, 11), True), ("CSE2531", (11, 12), False), ("CSE2202", (13, 14), True), ("CSE2029", (15, 18), False)],
#     [("CSE4805", (8, 9), True), ("CSE1237", (12, 13), True), ("CSE3453", (16, 17), True), ("CSE1350", (17, 18), False)],
#     [("CSE4573", (12, 15), True), ("CSE1525", (16, 17), True)],
#     [("CSE3751", (12, 14), True), ("CSE3825", (16, 19), True)],
#     [("CSE1086", (9, 11), True), ("CSE1902", (11, 14), True), ("CSE1495", (16, 18), True)],
#     [("CSE1276", (8, 9), False), ("CSE3105", (13, 14), True), ("CSE3208", (15, 16), True), ("CSE2936", (17, 20), True)],
#     [("CSE2649", (9, 10), False), ("CSE1386", (14, 15), True), ("CSE1221", (15, 18), True)],
#     [("CSE1779", (11, 12), True), ("CSE4581", (15, 16), False)],
#     [("CSE1681", (9, 10), True), ("CSE4541", (15, 16), True), ("CSE1207", (16, 17), True)],
#     [("CSE1008", (10, 12), False), ("CSE2168", (12, 14), True)],
#     [("CSE4935", (10, 11), True), ("CSE1891", (11, 13), False), ("CSE1205", (13, 14), True), ("CSE3942", (16, 19), True)],
#     [("CSE1278", (9, 12), True), ("CSE3536", (17, 18), True)],
#     [("CSE5001", (8, 10), True), ("CSE5002", (10, 11), False)],
#     [("CSE5101", (9, 10), True), ("CSE5102", (11, 12), True), ("CSE5103", (13, 14), True)],
#     [("CSE5201", (8, 9), True), ("CSE5202", (9, 10), False), ("CSE5203", (15, 16), True)],
#     [("CSE5301", (10, 11), True), ("CSE5302", (13, 15), True)],
#     [("CSE5401", (9, 12), False)],
#     [("CSE5501", (8, 9), True), ("CSE5502", (12, 13), True)],
#     [("CSE5601", (8, 10), False), ("CSE5602", (14, 15), True)],
#     [("CSE5701", (9, 10), True), ("CSE5702", (11, 12), True), ("CSE5703", (16, 17), True)],
#     [("CSE5801", (8, 9), False), ("CSE5802", (9, 11), True)],
#     [("CSE5901", (10, 12), True), ("CSE5902", (14, 15), False)]
# ]

# days = [1,5,5,3,3,2,3,2,3,4,2,3,4,3,1,1,4,5,5,1,1,2,3,4,5,1,2,3,4,5]

# import json

# if __name__ == '__main__':

#     all_results = [] 
#     for day, day_schedule in zip(days, day_schedules):

#         env = SimpleCourseScheduleEnv(day_schedule, day)
    
#         # Evaluate different agents:
#         print("\nEvaluating Random Agent:")
#         traj1 = evaluate_policy(env, random_agent_policy, num_episodes=5, render=False)
#         traj2 = evaluate_policy(env, lazy_agent_policy, num_episodes=2, render=False)
#         traj3 = evaluate_policy(env, hardworking_agent_policy, num_episodes=2, render=False)
#         trajs = traj1 + traj2 + traj3
#         for traj in trajs:
#             actions = []
#             states = []
#             for action_state in traj:
#                 action, state = action_state
#                 actions.append(action)
#                 states.append(state)
#             final_state = states[-1]
#             # In the new state: index 0 is gpa_level, index 2 is day.
#             gpa_level = final_state[0]
#             day_value = final_state[2]
#             schedule_traj = generate_schedule_trajectory(day_schedule, actions)
            
#             info = f"\nday: {day_value}, gpa_level: {gpa_level}"
#             result = "schedule trajectory: " + str(schedule_traj) + info
#             print(result)
#             print(states)
#             all_results.append((result, states))
#     print(len(all_results))
    
#     # Write all results to result_2.json file.
#     wrapped = [{"result": r, "states": states} for r, states in all_results]

#     with open("result_2.json", "w", encoding="utf-8") as f:
#         json.dump(wrapped, f, indent=2, ensure_ascii=False)


# import json

if __name__ == '__main__':

    # Load the trained reward model
    model_path = '/Users/zzlou/Documents/Code/cs3263/CS3263/best_reward_model_2.pt'
    reward_model = RewardModel(input_dim=5, hidden_dim1=32, hidden_dim2=16, dropout_prob=0.2)
    reward_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    reward_model.eval()

    # Define a sample day_schedule and set the day.
    # (In this example, we use a day_schedule with three courses.)
    day_schedule = [
        ("PC1101", (10, 12), False),
        ("CS2105", (14,16), False)
    ]
    day = 5

    # Create the environment with the loaded reward model.
    env = SimpleCourseScheduleEnv(day_schedule, day=day, reward_model=reward_model)
    
    # Train the Q-learning agent using the environment that uses the trained reward model.
    print("Training Q-Learning agent...")
    Q = q_learning_agent(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.2)
    print("Training finished.\n")
    
    # Evaluate the learned Q-learning policy using the q_policy.
    # The evaluate_policy function will use the provided Q table.
    print("Evaluating Q-Learning policy:")
    trajectories = evaluate_policy(env, None, num_episodes=1, Q=Q, render=True)
    actions = []
    traj = trajectories[0]
    for action_state in traj:
        action, state = action_state
        actions.append(action)
    print(actions)
    print("Evaluation finished.")

#     # all_results = [] 
#     # for day, day_schedule in zip(days, day_schedules):

#     #     env = SimpleCourseScheduleEnv(day_schedule, day)
    
#     #     # 评估 Random Agent
#     #     print("\nEvaluating Random Agent:")
#     #     trajs = evaluate_policy(env, random_agent_policy, num_episodes=5, render=False)
#     #     for traj in trajs:
#     #         actions = []
#     #         states = []
#     #         for action_state in traj:
#     #             action, state = action_state
#     #             actions.append(action)
#     #             states.append(state)
#     #         final_state = states[-1]
#     #         gpa = final_state[2]
#     #         day_value = final_state[4]
#     #         schedule_traj = generate_schedule_trajectory(day_schedule, actions)
            
#     #         info = f"\nday: {day}, gpa: {gpa}"
#     #         result = "schedule trajectory: " + str(schedule_traj) + info
#     #         print(result)
#     #         print(states)
#     #         all_results.append((result,states))
#     # print(len(all_results))
#     # # 将所有结果写入 result.json 文件
    
#     # wrapped = [{"result": r, "states": states} for r,states in all_results]

#     # with open("result.json", "w", encoding="utf-8") as f:
#     #     json.dump(wrapped, f, indent=2, ensure_ascii=False)

    # print("\nEvaluating All-Attend Agent:")
    # evaluate_policy(env, all_attend_agent_policy, num_episodes=5, render=False)
# actions = [1, 1, 1]  # 例如 random agent 的决策

# traj = generate_schedule_trajectory(day_schedule, actions)
# for i, state in enumerate(traj):
#     print(f"State {i}: {state}")