from abc import ABC, abstractmethod

class AttendanceBehavior(ABC):
    @abstractmethod
    def compute_attendance(self, schedule):
        pass

class LazyPerson(AttendanceBehavior):
    def compute_attendance(self, schedule):
        total = len(schedule)
        attended = 0
        prev_time = None

        for session, start_time in schedule:
            if start_time == "0800":
                continue
            elif prev_time is not None and int(start_time) - int(prev_time) > 300:
                attended += 0.5 
            else:
                attended += 0.8
            prev_time = start_time
        return attended / total if total > 0 else 0
    
class Grinder(AttendanceBehavior):
    def compute_attendance(self, schedule):
        total = len(schedule)
        return 0.95 if total > 0 else 0


class NormalPerson(AttendanceBehavior):
    def compute_attendance(self, schedule):
        total = len(schedule)
        attended = 0
        for session, start_time in schedule:
            hour = int(start_time[:2])
            if hour <= 10:
                attended += 0.8
            elif hour >= 18:
                attended += 0.8
            else:
                attended += 0.85
        return attended / total if total > 0 else 0
    
class MealCare(AttendanceBehavior):
    def compute_attendance(self, schedule):
        total = len(schedule)
        attended = 0
        for session, start_time in schedule:
            hour = int(start_time[:2])
            if hour >= 12 and hour <=13:
                attended += 0.6
            elif hour == 8:
                attended += 0.7
            elif hour == 17 or 18 or 19:
                attended += 0.7
            else:
                attended += 0.9
        return attended / total if total > 0 else 0

class ExtremeLazyPerson(AttendanceBehavior):
    def compute_attendance(self, schedule):
        total = len(schedule)
        attended = 0
        for session, start_time in schedule:
            if start_time == "0800" or "0900" or "1000":
                attended += 0.5
            elif start_time == "1800" or "1900":
                attended += 0.5
            else:
                attended += 0.7
        return attended / total if total > 0 else 0