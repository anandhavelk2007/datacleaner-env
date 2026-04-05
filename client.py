from env.environment import DataCleanerEnv
from env.models import Observation, Action, Reward

class DataCleanerClient:
    def __init__(self):
        self.env = DataCleanerEnv()
    def reset(self, task_id: str = "easy") -> Observation:
        return self.env.reset(task_id)
    def step(self, action: Action):
        return self.env.step(action)
    def state(self):
        return self.env.state()