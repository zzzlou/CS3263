import gym
from gym import spaces
import numpy as np
import random

class CourseScheduleEnv(gym.Env):
    def __init__(self, schedule, max_attend_per_day=3):
        """
        :param schedule: 一个列表，每天是一个列表，每个元素是一个元组，
                         格式为 (course_code, (start_time, end_time), attendance_flag)
        :param max_attend_per_day: 一天中理想上课时段数量阈值
        """
        super(CourseScheduleEnv, self).__init__()
        self.schedule = schedule  
        self.num_days = len(schedule)
        self.max_events = max(len(day) for day in schedule)
        self.max_attend_per_day = max_attend_per_day

        # 动作空间：0 不去上课，1 去上课
        self.action_space = spaces.Discrete(2)
        # 状态表示：(current_day, current_event_index, attended_today, fatigue, gpa)
        # 这里 fatigue 和 gpa 以标量的形式返回（Python 的 float）
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.num_days + 1),
            spaces.Discrete(self.max_events + 1),
            spaces.Discrete(self.max_events + 1),
            spaces.Box(low=0.0, high=10.0, shape=(), dtype=np.float32),  # fatigue
            spaces.Box(low=0.0, high=4.0, shape=(), dtype=np.float32)    # gpa
        ))
        self.reset()

    def reset(self):
        # 重置所有状态变量
        self.current_day = 0
        self.current_event_index = 0
        self.attended_today = 0
        self.fatigue = 0.0
        self.gpa = 4.0
        self.total_reward = 0.0
        return (self.current_day, self.current_event_index, self.attended_today,
                self.fatigue, self.gpa)

    def transition(self, action):
        """
        仅负责状态更新，不直接计算 reward。
        根据 action 更新 fatigue、gpa、attended_today 等状态内部变量，
        并管理课程进度（天数、当前事件索引等），返回新的 state、done 标志和 info。
        """
        done = False
        info = {}
        # 获取当前天的课程表
        day_schedule = self.schedule[self.current_day]

        # 如果当前事件索引超出当天课表，则结束当天，执行“休息”更新
        if self.current_event_index >= len(day_schedule):
            # 休息时降低疲劳
            if self.attended_today == 0:
                self.fatigue = max(0, self.fatigue - 0.5)
                info['rest_bonus'] = True
            self.current_day += 1
            self.current_event_index = 0
            self.attended_today = 0
            if self.current_day >= self.num_days:
                done = True
            return (self.current_day, self.current_event_index, self.attended_today,
                    self.fatigue, self.gpa), done, info

        # 处理当前课程事件
        course_code, (start_time, end_time), attendance_flag = day_schedule[self.current_event_index]
        duration = end_time - start_time

        if attendance_flag:  # 课程要求出勤
            if action == 1:
                self.attended_today += 1
                # 上课增加疲劳
                self.fatigue += 0.2 * duration
                # 若上课超过理想数量，额外增加疲劳
                if self.attended_today > self.max_attend_per_day:
                    self.fatigue += 0.1
                    info['over_attend_penalty'] = True
            else:
                # 不去上必修课，70% 概率降低 GPA
                if random.random() < 0.7:
                    self.gpa = max(0.0, self.gpa - 0.1)
                    info['gpa_penalty'] = True
        else:  # 非必修课
            if action == 1:
                self.attended_today += 1
                self.fatigue += 0.1 * duration
                
            else:
                # 不上非必修课，可能休息降低疲劳
                self.fatigue = max(0, self.fatigue - 0.1)

        self.current_event_index += 1

        # 当天的课程结束处理：补充休息效果
        if self.current_event_index >= len(day_schedule):
            if self.attended_today == 0:
                self.fatigue = max(0, self.fatigue - 0.5)
                info['endday_rest'] = True
            self.current_day += 1
            self.current_event_index = 0
            self.attended_today = 0

        if self.current_day >= self.num_days:
            done = True

        state = (self.current_day, self.current_event_index, self.attended_today,
                 self.fatigue, self.gpa)
        return state, done, info

    def compute_reward(self, state):
        """
        定义一个势能函数（潜在奖励函数），用来衡量状态的“好坏”。
        这里只是举例，解释如下：
            - attended_today 对奖励有正贡献（每上一节课奖励 1 分）
            - fatigue 会带来负贡献（每单位疲劳扣 0.5 分）
            - gpa 有正贡献（每一点 GPA 奖励 2 分）
        """
        # state: (day, event_index, attended_today, fatigue, gpa)
        return 1.1 * state[2] - 1 * state[3] + 3 * state[4]

    

    def step(self, action):
        """
        step() 先记录当前 state，然后调用 transition 更新状态，
        最后根据 state 变化调用 reward_model 得到 reward。
        """
        # 保存当前 state 作为计算奖励的参照
        prev_state = (self.current_day, self.current_event_index, self.attended_today,
                      self.fatigue, self.gpa)
        new_state, done, info = self.transition(action)
        reward = self.compute_reward(new_state)
        # reward = self.compute_reward(prev_state, new_state, action, info)
        # 累积 total_reward（如果需要观察总收益）
        self.total_reward += reward
        return new_state, reward, done, info

    def render(self, mode='human'):
        print(f"Day: {self.current_day+1}, Event Index: {self.current_event_index}, "
              f"Attended Today: {self.attended_today}, Fatigue: {self.fatigue:.2f}, GPA: {self.gpa:.2f}, "
              f"Total Reward: {self.total_reward:.2f}")


# -----------------------------
# 以下为示例 Q-learning 算法和策略评估函数
# -----------------------------
def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = {}

    def get_Q(state, action):
        return Q.get((tuple(state), action), 0.0)

    def update_Q(state, action, value):
        Q[(tuple(state), action)] = value

    def epsilon_greedy(state, epsilon):
        if random.random() < epsilon:
            return env.action_space.sample()
        else:
            qs = [get_Q(state, a) for a in range(env.action_space.n)]
            max_actions = [a for a, q in enumerate(qs) if q == max(qs)]
            return random.choice(max_actions)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = epsilon_greedy(state, epsilon)
            next_state, reward, done, info = env.step(action)
            q_current = get_Q(state, action)
            qs_next = [get_Q(next_state, a2) for a2 in range(env.action_space.n)]
            q_next_max = max(qs_next) if qs_next else 0.0
            q_new = q_current + alpha * (reward + gamma * q_next_max - q_current)
            update_Q(state, action, q_new)
            state = next_state
            
    return Q

def baseline_policy(state, env):
    """
    简单规则：如果当前课程要求出勤，则选择上课，否则不去上课。
    """
    current_day = state[0]
    current_event_index = state[1]
    day_schedule = env.schedule[current_day]
    if current_event_index >= len(day_schedule):
        return 0
    course_code, (start_time, end_time), attendance_flag = day_schedule[current_day][current_event_index]
    return 1 if attendance_flag else 0

def evaluate_policy_with_course_logging(env, policy_func, num_episodes=10, render=False):
    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            # 解包所有状态变量
            current_day, current_event_index, attended_today, fatigue, gpa = state
            if current_day < len(env.schedule) and current_event_index < len(env.schedule[current_day]):
                course_code, time_range, attendance_flag = env.schedule[current_day][current_event_index]
                print(f"Episode {episode+1}, Day {current_day+1}, Course {course_code} ({time_range}), Required: {attendance_flag}")
            else:
                print(f"Episode {episode+1}, Day {current_day+1}, No more courses.")
            
            action = policy_func(state, env)
            action_str = "上课" if action == 1 else "不去上课"
            print(f"--> 状态: {state} -> 动作选择: {action_str}")
            
            state, reward, done, info = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        total_rewards.append(ep_reward)
        print(f"Episode {episode+1} 总奖励: {ep_reward:.2f}\n")
    average_reward = np.mean(total_rewards)
    print(f"平均奖励: {average_reward:.2f}")
    return average_reward

# -----------------------------
# 示例课表（区间表示）
# -----------------------------
example_schedule = [
    [("ES1103", (9, 12), True), ("CS3263", (14, 15), True), ("CS2105", (17, 18), True)],  # 第一天
    [("PC1101", (10, 12), False), ("QF1100", (14, 16), False)],                           # 第二天
    [("CS4248", (8, 10), True)],                                                          # 第三天
    [("CS3263", (10, 12), False), ("QF1100", (14, 16), False)],                           # 第四天
    [("QF1100", (9, 10), True), ("PC1101", (10, 12), False), ("CS2105", (14, 16), False)]    # 第五天
]

# -----------------------------
# 创建环境并训练 Q-learning agent
# -----------------------------
env_rl = CourseScheduleEnv(schedule=example_schedule, max_attend_per_day=3)
print("Training Q-learning agent...")
Q = q_learning(env_rl, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1)
print("Training finished.")

def q_policy(state, env, Q):
    qs = [Q.get((tuple(state), a), 0.0) for a in range(env.action_space.n)]
    return int(np.argmax(qs))

def all_go_policy(state, env):
    return 1

def all_not_go_policy(state, env):
    return 0

def rl_agent_policy(state, env, Q):
    return q_policy(state, env, Q)

def rl_policy_wrapper(state, env):
    return rl_agent_policy(state, env, Q)

avg_reward_rl = evaluate_policy_with_course_logging(env_rl, rl_policy_wrapper, num_episodes=10, render=False)
print(f"Average Reward of RL agent: {avg_reward_rl:.2f}")

# avg_reward_allgo = evaluate_policy_with_course_logging(env_rl, all_go_policy, num_episodes=10)
# avg_reward_all_not_go = evaluate_policy_with_course_logging(env_rl, all_not_go_policy, num_episodes=10)
# print(f"Average Reward of All-Go policy: {avg_reward_allgo:.2f}")

# print(f"Average Reward of All-Not-Go policy: {avg_reward_all_not_go:.2f}")


print(f"Average Reward of RL agent: {avg_reward_rl:.2f}")
