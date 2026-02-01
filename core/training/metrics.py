class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.count = 0

    def update(self, loss_value, batch_size):
        self.total_loss += loss_value * batch_size
        self.count += batch_size

    def compute(self):
        return {
            "loss": self.total_loss / max(1, self.count)
        }
