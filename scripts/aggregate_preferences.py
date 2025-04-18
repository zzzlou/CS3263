import pickle
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

DAY_TO_INDEX = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
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
            print(f"Session: day={s.day}, start={s.start_time}, end={s.end_time}")
            day = DAY_TO_INDEX.get(s.day.strip(), -1) if isinstance(s.day, str) else s.day
            if not isinstance(day, int) or not (0 <= day <= 6):
                continue
            try:
                start = int(float(s.start_time)) // 100
                end = int(float(s.end_time)) // 100
                if 0 <= start < end <= 24:
                    schedule.append((day, start, end))
            except Exception as e:
                print(f"Error converting times: {e}")
    return schedule


def aggregate_schedule_data(directory):
    counter = Counter()
    for fname in os.listdir(directory):
        if fname.endswith("_final_state.pkl"):
            student_id = fname.replace("_final_state.pkl", "")
            final_state = load_pickle(os.path.join(directory, f"{student_id}_final_state.pkl"))
            perm_dict = load_pickle(os.path.join(directory, f"{student_id}_permutation_dict.pkl"))
            code_order = load_pickle(os.path.join(directory, f"{student_id}_code_order.pkl"))
            schedule = extract_schedule(final_state, perm_dict, code_order)
            for (day, start, end) in schedule:
                for hour in range(start, end):
                    counter[(day, hour)] += 1
    return counter

def visualize_heatmap(counter):
    heatmap = [[0]*24 for _ in range(5)] 

    for (day, hour), count in counter.items():
        if 0 <= day < 5 and 0 <= hour < 24:
            heatmap[4 - day][23 - hour] = count

    plt.figure(figsize=(12, 4))
    ax = sns.heatmap(
        heatmap,
        cmap="YlGnBu",
        xticklabels=False,  
        yticklabels=["Mon", "Tue", "Wed", "Thu", "Fri"],
    )

    ax.set_xticks([i for i in range(24)])
    ax.set_xticklabels([str(i) for i in range(24)], rotation=0)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_title("Aggregate Preferred Class Times")
    plt.tight_layout()
    plt.show()


def main():
    directory = "student_outputs"
    counter = aggregate_schedule_data(directory)
    visualize_heatmap(counter)

if __name__ == "__main__":
    main()
