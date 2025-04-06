import CSP
from collections import defaultdict
from Course_API import process_course
import beam_search
from AttendanceBehavior import *
import sys
from Q_learning import q_learning, get_final_state


BEAM_WIDTH = 10

def main():
    # Construct dictionary of permutations of states
    # course_codes = input("Enter courses you would like to enrol in, separated by commas: ").split(",")
    course_codes = ["CS2030","CS2040","CS2100"]
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
    
    # beam search
    res = beam_search.search(list(domain.values()), BEAM_WIDTH, permutation_dict, code_order)
    print(res)

    Q = q_learning(domain,res,permutation_dict,verbose=True)

    final_state = get_final_state(Q,res,domain,500)
    print(f"final state according to learned q function: {final_state}")
    # print(Q)


    # calculate_attendance

    # print(calculate_attendance(permutation_dict, courses, res, LazyPerson()))
    # print(calculate_attendance(permutation_dict, courses, res, NormalPerson()))
    # print(calculate_attendance(permutation_dict, courses, res, Grinder()))


if __name__ == "__main__":
    main()
