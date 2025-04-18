from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


def extract_time_blocks(permutation_dict, code_order, final_state):
    """
    Given one student's final state, extract all (day, hour) blocks they attend.
    Returns a list of (day, hour) tuples.
    """
    time_blocks = []
    for i, course_code in enumerate(code_order):
        selected_index = final_state[i]
        sessions = permutation_dict[course_code][selected_index]
        for session in sessions:
            start = int(session.start_time)
            end = int(session.end_time)
            day = session.day  # e.g., "Monday"
            for hour in range(start, end, 100):  # in 100-unit steps (e.g., 0900, 1000, etc.)
                time_blocks.append((day, hour))
    return time_blocks

def aggregate_all_preferences(final_states, permutation_dict, code_order):
    """
    Aggregate time preferences across many students.
    final_states: List of tuples, each representing one student's final schedule.
    Returns a dictionary of (day, hour) -> count.
    """
    counter = defaultdict(int)
    for final_state in final_states:
        blocks = extract_time_blocks(permutation_dict, code_order, final_state)
        for block in blocks:
            counter[block] += 1
    return counter

def visualize_heatmap(counter):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    hours = list(range(800, 2000, 100))  # 8am to 7pm

    matrix = np.zeros((len(days), len(hours)))
    for i, day in enumerate(days):
        for j, hour in enumerate(hours):
            matrix[i, j] = counter.get((day, hour), 0)

    plt.figure(figsize=(12, 5))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlGnBu",
                xticklabels=[f"{h//100}:00" for h in hours], yticklabels=days)
    plt.title("Aggregated Student Class Time Preferences")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.show()

def save_pickle_outputs(final_state, permutation_dict, code_order, path_prefix="output"):
    """
    Save the key components needed for preference aggregation into pickle files.
    final_state can be a single one (tuple) or a list of states.
    """
    if not isinstance(final_state, list):
        final_state = [final_state]  # make it a list for consistency

    with open(f"{path_prefix}_final_states.pkl", "wb") as f:
        pickle.dump(final_state, f)

    with open(f"{path_prefix}_permutation_dict.pkl", "wb") as f:
        pickle.dump(permutation_dict, f)

    with open(f"{path_prefix}_code_order.pkl", "wb") as f:
        pickle.dump(code_order, f)

    print("Saved pickle files:")
    print(f"- {path_prefix}_final_states.pkl")
    print(f"- {path_prefix}_permutation_dict.pkl")
    print(f"- {path_prefix}_code_order.pkl")



def main():
    # Example placeholder: load final_states, permutation_dict, code_order from file
    with open("final_states.pkl", "rb") as f:
        final_states = pickle.load(f)
    with open("permutation_dict.pkl", "rb") as f:
        permutation_dict = pickle.load(f)
    with open("code_order.pkl", "rb") as f:
        code_order = pickle.load(f)

    counter = aggregate_all_preferences(final_states, permutation_dict, code_order)
    visualize_heatmap(counter)


if __name__ == "__main__":
    main()