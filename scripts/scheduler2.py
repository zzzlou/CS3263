import gym
import numpy as np
import random
import torch.nn as nn
import torch

# ===============================
# 简化版一天内课表环境，状态表示 s = (earliest_start, total_attended, gpa, fatigue, day)
# ===============================
class SimpleCourseScheduleEnv(gym.Env):
    """
    针对一天的课表环境，状态表示为：
      s = (earliest_start, total_attended, gpa, fatigue, day)
    其中：
      - earliest_start: 当前已上课中最早的开始时间归一化到 [0,1]（如果未上课则为 1.0）
      - total_attended: 累计上课时长（小时）
      - gpa: 当前绩点
      - fatigue: 当前疲劳值
      - day: 当天标记（例如 0 表示星期一）
      
    环境依据传入的 day_schedule 来决策。day_schedule 为列表，例如：
      [("ES2660", (9, 12), True), ("CS3263", (14, 15), True), ("CS2105", (17, 18), True)]
    决策规则示例：
      - 若选择上课 (action == 1)：
            total_attended 增加持续时间；
            如果目前没有上过课，earliest_start 更新为课程开始时间归一化值；
            fatigue 增加 0.2*duration.
      - 若选择不上课 (action == 0)：
            fatigue 降低 0.1；
            如果课程必修，有 70% 机率使 gpa 下降 0.1.
    """
    def __init__(self, day_schedule, day=0, reward_model = None):
        super().__init__()
        self.day_schedule = day_schedule
        self.num_courses = len(day_schedule)
        self.day = day
        self.action_space = gym.spaces.Discrete(2)  # 0: 不上课, 1: 上课
        self.reward_model = reward_model
        self.reset()
        
    def reset(self):
        # 初始化状态
        self.earliest_start = 1.0      # 没有上过课：设为1.0（代表24:00归一化）
        self.total_attended = 0.0
        self.gpa = 4.0
        self.fatigue = 0.0
        self.current_course = 0        # 待决策的课程索引
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self):
        # 状态以元组表示，取整或四舍五入处理连续值以减小状态空间
        return (round(self.earliest_start, 2),
                round(self.total_attended, 2),
                round(self.gpa, 2),
                round(self.fatigue, 2),
                self.day)

    def step(self, action):
        """
        完整的 step 函数：
          1. 调用 transition(action) 得到 next_state, done, info；
          2. 通过 compute_reward(next_state) 计算即时 reward；
          3. 累加 total_reward，然后返回 (next_state, reward, done, info)
        """
        next_state, done, info = self.transition(action)
        reward = self.compute_reward(next_state)
        self.total_reward += reward
        return next_state, reward, done, info
    def transition(self, action):
        """
        根据动作更新状态，仅进行状态转移，不计算 reward。
        更新规则如下：
          - 如果 action==1 (上课)：
                total_attended 加上课程持续时长；
                如果 earliest_start==1.0，则设置为当前课程的归一化开始时间，否则取二者最小；
                fatigue 增加 0.2 * duration；
                并有20%概率（随机）使得 GPA 增加 [0.05, 0.1]（上限 5.0）。
          - 如果 action==0 (不上课)：
                fatigue 降低一个在[0.05, 0.15]之间的随机值（最低0）；
                如果该课为必修，则有70%的概率使 GPA 下降 0.1。
          然后 current_course 加 1，当所有课程决策完毕时，done = True。
        返回更新后的下一状态、done 标志和 info 字典。
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
        norm_start = start / 24.0

        if action == 1:  # 上课
            self.total_attended += duration
            if self.earliest_start == 1.0:
                self.earliest_start = norm_start
            else:
                self.earliest_start = min(self.earliest_start, norm_start)
            self.fatigue += 0.2 * duration
            if random.random() < 0.2:
                self.gpa = min(5.0, self.gpa + random.uniform(0.05, 0.1))
        else:  # 不上课
            self.fatigue = max(0.0, self.fatigue - random.uniform(0.05, 0.15))
            if required:
                if random.random() < 0.7:
                    self.gpa = max(0.0, self.gpa - 0.1)

        self.current_course += 1
        if self.current_course >= self.num_courses:
            done = True

        next_state = self._get_state()
        return next_state, done, info
    
    def compute_reward(self, state):
        """
        使用训练好的 reward model 计算奖励。
        state 格式：(earliest_start, total_attended, gpa, fatigue, day)
        注意：输入状态需要与训练 reward model 时保持一致。
        """
        if self.reward_model is not None:
            # 转换为 tensor，并增加 batch 维度：形状 [1, 5]
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # 确保模型处于 eval 模式，并使用 no_grad 避免梯度计算
            self.reward_model.eval()
            with torch.no_grad():
                reward_tensor = self.reward_model(state_tensor)
            # reward_tensor 的形状一般为 [1, 1]，返回标量
            return reward_tensor.item()
        else:
            # 若没有提供 reward model，则退回到 rule-based 的方式
            earliest_start, total_attended, gpa, fatigue, day = state
            return 0.9 * total_attended - 3.0 * fatigue + 1.0 * gpa - 10.0 * earliest_start

    def render(self, mode='human'):
        print("State (earliest_start, total_attended, gpa, fatigue, day):", self._get_state())
        print("Total Reward:", self.total_reward)
        print("-"*40)

# ===============================
# Q-Learning Agent (Tabular)
# ===============================

class RewardModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出单个标量

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # 输出形状 [batch_size, 1]
def discretize_state(state):
    """
    state 是一个 5 元组 (earliest_start, total_attended, gpa, fatigue, day)。
    已经经过四舍五入处理，但这里可直接用作 key。
    """
    return state

def q_learning_agent(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    # Q 表以 (state, action) 为 key。
    Q = {}
    
    def get_Q(state, action):
        return Q.get((state, action), 0.0)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            state_key = discretize_state(state)
            # ε-greedy 策略
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
# 定义其他 Agent 策略
# ===============================
def random_agent_policy(state, env):
    return env.action_space.sample()

def all_attend_agent_policy(state, env):
    return 1
def convert_day_schedule_to_state(day_schedule, day=0, initial_gpa=4.0):
    """
    将实际去上了的课表（day_schedule）转换为环境内部状态表示 s = (earliest_start, total_attended, gpa, fatigue, day)
    
    参数:
      day_schedule: list of tuples, 每个元组 (course_code, (start, end), required)
                    表示一天中一门课程的信息。假设每门课均为已上（action==1）。
      day: 整数，表示当天，默认 0（例如星期一）
      initial_gpa: 初始绩点，默认 4.0
      
    返回:
      state: tuple, (earliest_start, total_attended, gpa, fatigue, day)
             其中 earliest_start 为所有课程中最早的上课开始时间（归一化到 [0,1]），
             total_attended 为累计上课时长（小时），
             fatigue 为累计疲劳（0.2 * 课程时长之和），
             gpa 和 day 分别为给定的初始 gpa 和 day。
    """
    if not day_schedule:
        # 如果课表为空，则返回默认状态
        return (1.0, 0.0, initial_gpa, 0.0, day)
    
    # 计算累计上课时长
    total_attended = 0.0
    fatigue = 0.0
    # earliest_start 取所有课程中的最小开始时间
    earliest_start_raw = float('inf')
    for course in day_schedule:
        _, (start, end), _ = course
        duration = end - start
        total_attended += duration
        fatigue += 0.2 * duration  # 每门课上课增加疲劳 0.2 * duration
        if start < earliest_start_raw:
            earliest_start_raw = start
    # 将 earliest_start 归一化（假设一天以 24 小时为界）
    earliest_start = earliest_start_raw / 24.0

    # 进行四舍五入处理，使得状态离散化一些
    earliest_start = round(earliest_start, 2)
    total_attended = round(total_attended, 2)
    fatigue = round(fatigue, 2)
    gpa = round(initial_gpa, 2)
    
    state = (earliest_start, total_attended, gpa, fatigue, day)
    return state
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
# ===============================
# 主函数：训练和评估
# ===============================

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
#     [("CSE1278", (9, 12), True), ("CSE3536", (17, 18), True)]
# ]

# days = [5,5,5,3,3,2,3,2,3,4,2,3,4,3,1,1,4,5,5,1]


import json

if __name__ == '__main__':
    model_path = '/Users/zzlou/Documents/Code/cs3263/CS3263/best_reward_model.pt'
    reward_model = RewardModel(input_dim=5, hidden_dim=16)
    # 加载模型参数（如果在 GPU 上训练，CPU 上加载时需要 map_location）
    reward_model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # 2. 创建环境时传入 reward_model
    day_schedule = [("ES1103", (9, 12), True), ("CS3263", (14, 15), True), ("CS2105", (17, 18), True)]
    env = SimpleCourseScheduleEnv(day_schedule, day=1, reward_model=reward_model)
    
    # 测试环境：运行一个 episode 并渲染每步状态和总奖励
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # 随机动作演示
        state, reward, done, info = env.step(action)
        env.render()

    print("Episode finished. Total reward (by reward_model):", env.total_reward)
    
    # all_results = [] 
    # for day, day_schedule in zip(days, day_schedules):

    #     env = SimpleCourseScheduleEnv(day_schedule, day)
    
    #     # 评估 Random Agent
    #     print("\nEvaluating Random Agent:")
    #     trajs = evaluate_policy(env, random_agent_policy, num_episodes=5, render=False)
    #     for traj in trajs:
    #         actions = []
    #         states = []
    #         for action_state in traj:
    #             action, state = action_state
    #             actions.append(action)
    #             states.append(state)
    #         final_state = states[-1]
    #         gpa = final_state[2]
    #         day_value = final_state[4]
    #         schedule_traj = generate_schedule_trajectory(day_schedule, actions)
            
    #         info = f"\nday: {day}, gpa: {gpa}"
    #         result = "schedule trajectory: " + str(schedule_traj) + info
    #         print(result)
    #         print(states)
    #         all_results.append((result,states))
    # print(len(all_results))
    # # 将所有结果写入 result.json 文件
    
    # wrapped = [{"result": r, "states": states} for r,states in all_results]

    # with open("result.json", "w", encoding="utf-8") as f:
    #     json.dump(wrapped, f, indent=2, ensure_ascii=False)

    # print("\nEvaluating All-Attend Agent:")
    # evaluate_policy(env, all_attend_agent_policy, num_episodes=5, render=False)
# actions = [1, 1, 1]  # 例如 random agent 的决策

# traj = generate_schedule_trajectory(day_schedule, actions)
# for i, state in enumerate(traj):
#     print(f"State {i}: {state}")