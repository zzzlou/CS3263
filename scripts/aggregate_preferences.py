import pickle
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# 映射星期字符串到数字
DAY_TO_INDEX = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def extract_schedule(final_state, permutation_dict, course_order):
    schedule = []
    for i, course_code in enumerate(course_order):
        perm_index = final_state[i]
        sessions = permutation_dict[course_code][perm_index]
        for s in sessions:
            # 处理字符串星期
            if isinstance(s.day, str):
                day = DAY_TO_INDEX.get(s.day, -1)
            else:
                day = s.day
            if day == -1:
                continue  # 忽略无效 weekday

            # 强制转换时间为整数
            try:
                start = int(s.start_time)
                end = int(s.end_time)
                schedule.append((day, start, end))
            except ValueError:
                continue  # 有些start_time可能是非法字符串
    return schedule

def aggregate_schedule_data(directory):
    day_hour_counter = Counter()

    for fname in os.listdir(directory):
        if fname.endswith("_final_state.pkl"):
            student_id = fname.split("_final_state.pkl")[0]
            final_state_path = os.path.join(directory, f"{student_id}_final_state.pkl")
            perm_dict_path = os.path.join(directory, f"{student_id}_permutation_dict.pkl")
            code_order_path = os.path.join(directory, f"{student_id}_code_order.pkl")

            final_state = load_pickle(final_state_path)
            permutation_dict = load_pickle(perm_dict_path)
            course_order = load_pickle(code_order_path)

            schedule = extract_schedule(final_state, permutation_dict, course_order)

            for (day, start, end) in schedule:
                for hour in range(start, end):
                    day_hour_counter[(day, hour)] += 1

    return day_hour_counter

def visualize_heatmap(counter):
    """
    counter: (day, hour) -> count
    """
    heatmap = [[0]*24 for _ in range(7)]
    for (day, hour), count in counter.items():
        if 0 <= hour < 24 and 0 <= day <= 6:
            heatmap[day][hour] = count

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap, cmap="YlGnBu", xticklabels=range(24), yticklabels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.title("Aggregate Preferred Class Times")
    plt.tight_layout()
    plt.show()

def main():
    directory = "student_outputs"  # 修改为你保存.pkl文件的文件夹路径
    counter = aggregate_schedule_data(directory)
    visualize_heatmap(counter)

if __name__ == "__main__":
    main()
