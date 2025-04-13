def f(state, permutation_dict, code_tuple, person_type): # person_type accepts "Lazy", "??", "???", ...
    def count_lunch_days(schedule):
        lunch_start, lunch_end = 1000, 1400
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_intervals = {}
        for day, start, end in schedule:
            if day not in weekdays:
                continue
            if end <= lunch_start or start >= lunch_end:
                continue
            clipped = (max(start, lunch_start), min(end, lunch_end))
            day_intervals.setdefault(day, []).append(clipped)
        count = 0
        for day in weekdays:
            if day not in day_intervals:
                count += 1
                continue
            intervals = sorted(day_intervals[day])
            if intervals[0][0] > lunch_start or intervals[-1][1] < lunch_end:
                count += 1
                continue
            gap_found = any(intervals[i][0] > intervals[i-1][1] 
                            for i in range(1, len(intervals)))
            if gap_found:
                count += 1
        return count
    
    def count_no_events_days(schedule):
        weekdays = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
        event_days = {day for day, start, end in schedule if day in weekdays}
        no_events_days = weekdays - event_days
        return len(no_events_days)
    
    def count_early_class_days(schedule):
        early_class_days = set()
        for day, start, _ in schedule:
            if start <= 900:
                early_class_days.add(day)
        return len(early_class_days)
    
    def count_exhaustive_days(schedule):
        day_totals = {}      
        for day, start, end in schedule:
            duration = (end - start) / 100.0
            if day in day_totals:
                day_totals[day] += duration
            else:
                day_totals[day] = duration             
        count = 0
        for total in day_totals.values():
            if total >= 6:
                count += 1   
        return count

    event_list = []
    for i in range(len(state)):
        number = state[i]
        code = code_tuple[i]
        events = permutation_dict[code][number]
        event_list += events
    time = list(map(lambda x: (x.day, int(x.start_time), int(x.end_time)), event_list))
 
    lunch_days = count_lunch_days(time) # How many days do I have lunch time? (has blank between 10am and 2pm)
    no_events_days = count_no_events_days(time) # How many days do I not having any lectures or tutorials?
    early_class_days = count_early_class_days(time) # How many days do I have class before (including) 9am?
    exhaustive_days = count_exhaustive_days(time) # How many days do I work for more than 6 hours (inclusive)?

    '''YOUR CODE HERE'''
    if person_type == "Lazy":
        weights = [0.5, 0.3, -0.1, -0.1]
    elif person_type == "Hardworking":
        weights = [0.1, -0.1, 0.2, 0.4]
    elif person_type == "Balanced":
        weights = [0.3, 0.2, -0.1, -0.1]
    elif person_type == "EarlyBird":
        weights = [0.2, 0.1, 0.5, -0.3]
    elif person_type == "Chill":
        weights = [0.6, 0.5, -0.4, -0.3]
    else:
        weights = [0.4, 0.3, -0.2, -0.2]
#
    weights = [0.5, 0.3, -0.1, -0.1]  # Example weights for the Lazy person type
    return weights[0] * lunch_days + weights[1] * no_events_days + weights[2] * early_class_days + weights[3] * exhaustive_days