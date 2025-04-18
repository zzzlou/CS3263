import pickle
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def extract_schedule(final_state, permutation_dict, course_order):
    schedule = []
    for i, course_code in enumerate(course_order):
        perm_index = final_state[i]
        sessions = permutation_dict[course_code][perm_index]
        for s in sessions:
            schedule.append((s.day, s.start_time, s.end_time))  # e.g., (1, 10, 12)
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
        heatmap[day][hour] = count

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap, cmap="YlGnBu", xticklabels=range(24), yticklabels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.title("Aggregate Preferred Class Times")
    plt.tight_layout()
    plt.show()

def main():
    directory = "pkl_outputs"  
    counter = aggregate_schedule_data(directory)
    visualize_heatmap(counter)

if __name__ == "__main__":
    main()
