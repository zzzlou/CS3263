from basic_info.Lecture import Lecture
from basic_info.Tutorial import Tutorial
from basic_info.Course import Course
import requests

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
    try:
        exam_date = data[i]["examDate"]
        exam_duration = data[i]["examDuration"]
    except Exception as e:
        exam_date = None
        exam_duration = None
        print("No exam info found: ", e)
    try:
        timetable = data[i]["timetable"]
    except Exception as e:
        timetable = None
        print("No timetable info found: ", e)
    
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
        else:
            tutorial = Tutorial(id=class_no, start_time=start_time,end_time=end_time, day = day, weeks = weeks, venue = venue)
            tutorials.append(tutorial)
    course = Course(code, exam_date, exam_duaration, lectures, tutorials)
    return course
    
def process_course(code):
    try:
        exam_date, exam_duration, timetable = get_course_info(code=code)
        course = parse_course_info(code,exam_date,exam_duration,timetable)
    except:
        course = None
    return course