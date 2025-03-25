class Lecture:
    id = 1
    def __init__(self, time, location):
        self.lecture_id = Lecture.id
        Lecture.id += 1
        self.time = time
        self.location = location
    
    def show(self):
        return f"Lecture({self.lecture_id}, {self.time}, {self.location})"