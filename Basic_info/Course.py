class Course:
    def __init__(self, name):
        self.name = name
        self.lectures = []  # lecture choices
        self.tutorials = []  # tut choices
    
    def add_lecture(self, lecture):
        self.lectures.append(lecture)
    
    def add_tutorial(self, tutorial):
        self.tutorials.append(tutorial)
    
    def show(self):
        return f"Course({self.name}, Lectures: {self.lectures}, Tutorials: {self.tutorials})"

