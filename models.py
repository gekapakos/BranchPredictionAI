# 1-bit classifier (1-bit branch predictor)

class OneBitPredictor:
    def __init__(self):
        self.state = 0  # 0: not taken, 1: taken
    def predict(self):
        return self.state
    def update(self, taken):
        if taken:
            self.state = 1
        else:
            self.state = 0

# 2-bit classifier (2-bit branch predictor)

class TwoBitPredictor:
    def __init__(self):
        self.state = 0  # 00: strongly not taken, 01: weakly not taken, 10: weakly taken, 11: strongly taken

    def predict(self):
        return self.state >= 2

    def update(self, taken):
        if taken:
            if self.state < 3:
                self.state += 1
        else:
            if self.state > 0:
                self.state -= 1