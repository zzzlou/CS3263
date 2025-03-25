class Tutorial:
    def __init__(self, id, time, location):
        self.id = id
        self.time = time
        self.location = location
    
    def show(self):
        return f"Tutorial({self.id}, {self.time}, {self.location})"