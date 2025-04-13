import evaluate
from collections import defaultdict
import random
import math

def is_valid(state, permutation_dict, keys):
    def check_overlap(schedule):
        day_to_intervals = defaultdict(list)
        for day, start, end in schedule:
            day_to_intervals[day].append((start, end))
        for day, intervals in day_to_intervals.items():
            intervals.sort(key=lambda x: x[0])
            for i in range(1, len(intervals)):
                previous_interval = intervals[i-1]
                current_interval = intervals[i]
                if current_interval[0] < previous_interval[1]:
                    return True
        return False

    event_list = []
    for i in range(len(state)):
        number = state[i]
        code = keys[i]
        events = permutation_dict[code][number]
        event_list = event_list + events
    time = list(map(lambda x: (x.day, int(x.start_time), int(x.end_time)), event_list))
    return not check_overlap(time)


def get_neighbors(state, state_space, keys):
    neighbors = []
    for i, key in enumerate(keys):
        possible_values = state_space[key]
        current_value = state[i]
        index = possible_values.index(current_value)
        
        if index > 0:
            new_state = list(state)
            new_state[i] = possible_values[index - 1]
            neighbors.append(tuple(new_state))
        
        if index < len(possible_values) - 1:
            new_state = list(state)
            new_state[i] = possible_values[index + 1]
            neighbors.append(tuple(new_state))
    
    return neighbors


def search(initial_state, state_space, permutation_dict, code_tuple, person_type,
            beam_width=5, max_iterations=10000,
            initial_temperature=100.0, cooling_rate=0.95):
    beam = [(initial_state, evaluate.f(initial_state, permutation_dict, code_tuple, person_type))]
    temperature = initial_temperature
    for iteration in range(max_iterations):
        candidate_states = []
        for state, parent_score in beam:
            neighbors = get_neighbors(state, state_space, code_tuple)
            for neighbor in neighbors:
                if is_valid(neighbor, permutation_dict, code_tuple):
                    neighbor_score = evaluate.f(neighbor, permutation_dict, code_tuple, person_type)
                    delta = neighbor_score - parent_score
                    if delta >= 0:
                        candidate_states.append((neighbor, neighbor_score))
                    else:
                        acceptance_probability = math.exp(delta / temperature) if temperature > 0 else 0
                        if random.random() < acceptance_probability:
                            candidate_states.append((neighbor, neighbor_score))
        all_states = beam + candidate_states
        all_states.sort(key=lambda x: x[1], reverse=True)
        beam = all_states[:beam_width]
        temperature *= cooling_rate
        if temperature < 1e-8:
            break
    return beam