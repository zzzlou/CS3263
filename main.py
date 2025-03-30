from basic_info.Lecture import Lecture
from basic_info.Tutorial import Tutorial
from basic_info.Course import Course
import requests
from collections import defaultdict, deque
import itertools

def get_all_possible_courses(academic_year="2024-2025"):
    url = f"https://api.nusmods.com/v2/{academic_year}/moduleList.json"
    response = requests.get(url).json()
    course_code_list = []
    for course in response:
        course_code_list.append(course['moduleCode'])
    return course_code_list


def get_course_info(academic_year="2024-2025", code="CS3263",sem=1):
    url = f"https://api.nusmods.com/v2/{academic_year}/modules/{code}.json"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Request {code} failed(Status code:{response.status_code})")
        return None
    response = response.json()
    data = response["semesterData"]
    
    #find index of target semester
    
    idx = -1
    for i in range(len(data)):
        sem_data = data[i]
        if sem_data["semester"] == sem:
            idx = i
            break
    if idx == -1:
        print(f"Data for {code} does not exist in semester {sem}")
        return None
    
    exam_date = data[i]["examDate"]
    exam_duration = data[i]["examDuration"]
    timetable = data[i]["timetable"]
    
    return exam_date, exam_duration, timetable

def parse_course_info(code, exam_date, exam_duaration, timetable):
    lectures = []
    tutorials = []
    for lesson in timetable:
        class_no = lesson["classNo"]
        start_time = lesson["startTime"]
        end_time = lesson["endTime"]
        day = lesson["day"]
        weeks = lesson["weeks"]
        venue = lesson["venue"]
        lesson_type = lesson["lessonType"]
        if lesson_type == "Lecture":
            lecture = Lecture(id=class_no, start_time=start_time,end_time=end_time, day = day, weeks = weeks, venue = venue)
            lectures.append(lecture)
        elif lesson_type == "Tutorial":
            tutorial = Tutorial(id=class_no, start_time=start_time,end_time=end_time, day = day, weeks = weeks, venue = venue)
            tutorials.append(tutorial)
        else:
            print("Error, unknown lesson type")
            return None
    course = Course(code, exam_date, exam_duaration, lectures, tutorials)
    return course
    
def process_course(code):
    exam_date, exam_duration, timetable = get_course_info(code=code)
    course = parse_course_info(code,exam_date,exam_duration,timetable)
    return course

def ac3(domains, neighbors, constraint):
    queue = deque([(xi, xj) for xi in domains for xj in neighbors.get(xi, [])])
    while queue:
        xi, xj = queue.popleft()
        if remove_inconsistent_values(xi, xj, domains, constraint):
            if not domains[xi]:
                return False
            for xk in neighbors.get(xi, []):
                if xk != xj:
                    queue.append((xk, xi))
    return True

def remove_inconsistent_values(xi, xj, domains, constraint):
    removed = False
    # Iterate over a copy of the domain list for safe removal.
    for x in domains[xi][:]:
        if not any(constraint(xi, x, xj, y) for y in domains[xj]):
            domains[xi].remove(x)
            removed = True
    return removed

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
    
    # print initial state space
    ls = list(map(lambda x: list(permutation_dict[x].keys()), list(code_order)))
    state_space = list(itertools.product(*ls))
    print(state_space)

    # Remove time-overlapped states
    state_space = tuple(filter(lambda x: not is_overlap(x), state_space))
    
        


if __name__ == "__main__":
    main()
