import random

def calculate_attendance(permutation_dict, courses, state, mode):
    if mode not in ["lazy", "grinder"]:
        raise ValueError("Mode must be 'lazy' or 'grinder'")
    
    total_sessions = 0
    attended_sessions = 0
    schedule = []

    for i, course in enumerate(courses):
        course_code = course.code
        selected_permutation = permutation_dict[course_code][state[i]]

        for session in selected_permutation:
            total_sessions += 1
            schedule.append((session, session.start_time))
        
    schedule.sort(key=lambda x:x[1])

    if mode == "lazy":
        prev_time = None
        for session, start_time in schedule:
            if start_time == "0800": 
                continue
            elif prev_time is not None and int(start_time) - int(prev_time) > 300:
                attended_sessions += 0.5
            else:
                attended_sessions += 0.9
            prev_time = start_time
    
    elif mode == "grinder":
        attended_sessions = total_sessions * 0.95
    
    attendance_rate = attended_sessions / total_sessions if total_sessions > 0 else 0
    return attendance_rate