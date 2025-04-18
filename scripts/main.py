import CSP
from collections import defaultdict
from Course_API import process_course
import beam_search
from AttendanceBehavior import *
import sys
from Q_learning import q_learning, get_final_state
import random
import pickle
import os
import seaborn


BEAM_WIDTH = 10

def main():
    # Construct dictionary of permutations of states
    # course_codes = input("Enter courses you would like to enrol in, separated by commas: ").split(",")
    course_codes = ["CS1101S"] # , "IS1108", "MA1521", "MA1522", "CS1231S"]
    courses = []
    for code in course_codes:
        code = code.upper()
        course = process_course(code)
        courses.append(course)
    permutation_dict = {}
    for course in courses:
        dict = {}
        lec_list = course.lectures
        grouped_lecs = defaultdict(list)
        for lec in lec_list:
            grouped_lecs[lec.id].append(lec)
        new_lec_list = list(grouped_lecs.values())
        tut_list = course.tutorials
        permutations = []
        for lec in new_lec_list:
            for tut in tut_list:
                permutations.append(lec + [tut])
        for i in range(len(permutations)):
            dict[i] = permutations[i]
        permutation_dict[course.code] = dict
    code_order = tuple(permutation_dict.keys())
    
    # AC3 algorithm 
    variables = list(code_order)
    domain = {}
    for var in variables:
        domain[var] = list(permutation_dict[var].keys())
    neighbors = {var: [v for v in variables if v != var] for var in variables}
    csp = CSP.CSP(variables, domain, neighbors, permutation_dict)
    result = CSP.ac3(csp)

    if result:
        variables = csp.variables
        domain = csp.domains
        neighbors = csp.neighbors
        # print(domain)
        # print(permutation_dict)
    else:
        print("False")

    # Ultra Search (beam search, simulated annealing, 10 restarts)
    initial_states = [] 
    for i in range(10):
        state = tuple(random.choice(domain[code]) for code in code_order)
        while not beam_search.is_valid(state, permutation_dict, code_order):
            state = tuple(random.choice(domain[code]) for code in code_order)
        initial_states.append(state)
    res = list(map(lambda x: beam_search.search(x, domain, permutation_dict, code_order, person_type="Lazy",
                                                beam_width=10, max_iterations=1000, 
                                                initial_temperature=200.0, cooling_rate=0.95)[0], initial_states))
    res.sort(key=lambda x: x[1], reverse=True)
    print(res[0])


    Q = q_learning(domain,res[0][0],permutation_dict,verbose=True)
 
    final_state = get_final_state(Q,res[0][0],domain,500)
    print(f"final state according to learned q function: {final_state}")
    print(Q)


    # calculate_attendance

    # print(calculate_attendance(permutation_dict, courses, res, LazyPerson()))
    # print(calculate_attendance(permutation_dict, courses, res, NormalPerson()))
    # print(calculate_attendance(permutation_dict, courses, res, Grinder()))

    # save to pickle outputs
    output_dir = "student_outputs"
    os.makedirs(output_dir, exist_ok=True)

    student_id = "A0287922R"  
    with open(f"{output_dir}/{student_id}_final_state.pkl", "wb") as f:
        pickle.dump(final_state, f)

    with open(f"{output_dir}/{student_id}_permutation_dict.pkl", "wb") as f:
        pickle.dump(permutation_dict, f)

    with open(f"{output_dir}/{student_id}_code_order.pkl", "wb") as f:
        pickle.dump(code_order, f)


if __name__ == "__main__":
    main()
