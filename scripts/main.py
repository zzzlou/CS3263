import CSP
from collections import defaultdict
from Course_API import process_course
import beam_search

BEAM_WIDTH = 10

def main():
    # Construct dictionary of permutations of states
    course_codes = input("Enter courses you would like to enrol in, separated by commas: ").split(",")
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
        print(domain)
    else:
        print("False")
    
    # beam search
    res = beam_search.search(list(domain.values()), BEAM_WIDTH, permutation_dict, code_order)
    print(res)

    #calculate_attendance
    def calculate_attendance(permutation_dict, courses, state, person: AttendanceBehavior):
        schedule = []
        
        for i, course in enumerate(courses):
            course_code = course.code
            selected_permutation = permutation_dict[course_code][state[i]]
            for session in selected_permutation:
                schedule.append((session, session.start_time))
        
        schedule.sort(key=lambda x: x[1])
        return person.compute_attendance(schedule)

    print(calculate_attendance(permutation_dict, courses, res, NormalPerson()))

if __name__ == "__main__":
    main()
