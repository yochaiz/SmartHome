class Results:
    def __init__(self):
        self.loss = []
        self.score = []

    def toJSON(self):
        return dict(vars(self))  # make dict copy
