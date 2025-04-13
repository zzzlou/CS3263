from collections import defaultdict
import random
import itertools
import sys
from AttendanceBehavior import *


def convert_domain_to_state_space(domain):
    course_list = list(domain.keys())
    state_space = list(itertools.product(*(domain[course] for course in course_list)))
    return state_space



def calculate_reward(permutation_dict, courses, state, person: AttendanceBehavior):
    schedule = []
    
    for i, course_code in enumerate(courses):
        selected_permutation = permutation_dict[course_code][state[i]]
        for session in selected_permutation:
            schedule.append((session, session.start_time))
    
    schedule.sort(key=lambda x: x[1])
    return person.compute_attendance(schedule)


def q_learning(domain, init_state, permutation_dict, EPISODES=1000, MAX_STEPS=50, alpha=0.1, gamma=0.9, epsilon=0.2,theta=0.01, verbose=False):
    
    # Initialize Q-table as a dictionary with keys as (state, action) pairs
    Q = defaultdict(float)
    course_options = list(domain.values())
    course_code_list = list(domain.keys())
    state_space = convert_domain_to_state_space(domain)
    visited_history = []

    
    if verbose:
        print("Starting Q-learning...")
        print(f"Total state space size: {len(state_space)}")
        
    for episode in range(EPISODES):
        # If no initial state is provided, choose a random valid state from the state_space
        state = tuple(init_state) if init_state is not None else random.choice(state_space)
        max_delta = 0 
        
        if verbose:
            print(f"\nEpisode {episode+1} starts with state: {state}")
        visited_history = []
        visited_history.append(state)
        for step in range(MAX_STEPS):
            actions = possible_actions(state, course_options)
            if not actions:
                if verbose:
                    print("No valid actions available, ending episode.")
                break  # End episode if no valid actions are available

            # Choose action using epsilon-greedy strategy
            if random.random() < epsilon:
                action = random.choice(actions)
                if verbose:
                    print(f"Step {step+1}: Randomly chosen action {action}")
            else:
                action = best_q_action(Q, state, actions)
                if verbose:
                    print(f"Step {step+1}: Best Q action chosen {action}")
            
            # Execute the action to obtain the next state
            next_state = apply_action(state, action)
            # Calculate the reward using the provided function
            reward = calculate_reward(permutation_dict,course_code_list,next_state,LazyPerson())

            if verbose:
                print(f"Step {step+1}: Transition from state {state} to {next_state} with reward {reward:.4f}")
            
            # Get the maximum Q-value for the next state over all possible actions
            next_actions = possible_actions(next_state, course_options)
            if next_actions:
                best_next_q = max(Q[(next_state, a)] for a in next_actions)
            else:
                best_next_q = 0

            # Update Q-value using the Q-learning update rule
            update = alpha * (reward + gamma * best_next_q - Q[(state, action)])
            max_delta = max(max_delta, abs(update))
            Q[(state, action)] += update
            if verbose:
                print(f"Step {step+1}: Q[{state}, {action}] updated by {update:.6f} to {Q[(state, action)]:.6f}")
            
            # Transition to the next state
            state = next_state
            visited_history.append(state)


        if verbose:
            print(f"Episode {episode+1} ended. Maximum Q-update delta: {max_delta:.6f}")
            print(f"Visited States: {visited_history}")

        if max_delta < theta:
            print(f"Converged after {episode+1} episodes with max update delta {max_delta:.6f}.")
            break

    if verbose:
        print("Training completed.")
    return Q

def possible_actions(state,course_options):
    """
    Returns all possible actions in the given state.

    Action format:
    - (course_index, new_perm_index)
    where 'state' is a tuple with each element representing the current selection for a course,
    and permutation_dict holds the available options for each course.
    """
    actions = []
    for i, course_permutation in enumerate(state):
        # course_options[i] holds allowed values for the i-th course in the fixed order.
        for option in course_options[i]:
            if option != course_permutation:
                actions.append((i, option))
    return actions


    
    

def best_q_action(Q, state, actions):
    """
    Returns the action with the highest Q-value from the candidate actions in the given state.
    If all Q-values are equal, a random action is selected.
    """
    best_action = None
    best_value = float('-inf')
    for action in actions:
        value = Q[(state, action)]
        if value > best_value:
            best_value = value
            best_action = action
    # if best_action is None:
    #     best_action = random.choice(actions)
    return best_action

def apply_action(state, action):
    """
    Applies the given action to the current state and returns the new state.

    Action format: (course_index, new_perm_index)
    """
    state_list = list(state)
    course_idx, new_perm_index = action
    state_list[course_idx] = new_perm_index
    return tuple(state_list)

def get_final_state(Q, init_state, domain, max_steps=50):
    """
    Roll out the greedy policy using the learned Q function starting from init_state.
    
    Parameters:
      - Q: The learned Q-table (a dictionary with keys as (state, action) pairs).
      - init_state: The initial state (tuple).
      - course_options: A list of lists containing allowed permutation indices for each course (in fixed order).
      - max_steps: Maximum number of steps for the rollout.
    
    Returns:
      - state: The final state reached after following the greedy policy.
    """
    state = tuple(init_state)
    course_options = list(domain.values())
    for step in range(max_steps):
        actions = possible_actions(state, course_options)
        if not actions:
            break
        # Choose the best action according to Q
        best_action = best_q_action(Q, state, actions)
        if best_action is None:
            break
        next_state = apply_action(state, best_action)
        # Terminate if the state does not change (or you might compare rewards)
        if next_state == state:
            break
        state = next_state
    return state
