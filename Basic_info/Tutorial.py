class Tutorial:
    def __init__(self, id, start_time, end_time, day, weeks, venue):
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
        self.day = day
        self.weeks = weeks
        self.venue = venue
    def __str__(self):
        return f"Tutorial {self.id}: {self.day} {self.start_time}-{self.end_time} at {self.venue}, Weeks: {self.weeks}"

    