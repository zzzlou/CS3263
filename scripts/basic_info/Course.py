class Course:
    def __init__(self, code, exam_date, exam_dur, lectures = [], tutorials = []):
        self.code = code
        self.lectures = lectures  # lecture choices
        self.tutorials = tutorials  # tut choices
        self.exam_date = exam_date
        self.exam_dur = exam_dur
    
    def add_lecture(self, lecture):
        self.lectures.append(lecture)
    
    def add_tutorial(self, tutorial):
        self.tutorials.append(tutorial)
    
    def __str__(self):
        return f"[{self.code}] Exam: {self.exam_date} ({self.exam_dur} mins)\nLectures:\n" + \
               "\n".join(["  " + str(l) for l in self.lectures]) + \
               "\nTutorials:\n" + \
               "\n".join(["  " + str(t) for t in self.tutorials])

    def __repr__(self):
        return self.__str__()