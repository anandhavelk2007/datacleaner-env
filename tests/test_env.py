import unittest
from env.environment import DataCleanerEnv
from env.models import Action

class TestDataCleanerEnv(unittest.TestCase):
    def test_reset(self):
        env = DataCleanerEnv()
        obs = env.reset("easy")
        self.assertEqual(obs.task_id, "easy")
        self.assertIsNotNone(obs.dataset_summary)
    def test_step_impute(self):
        env = DataCleanerEnv()
        env.reset("easy")
        action = Action(type="impute", column="age", method="median")
        obs, reward, done, info = env.step(action)
        self.assertGreaterEqual(reward.value, -1.0)
        self.assertLessEqual(reward.value, 2.0)
        self.assertIn("score", info)
    def test_reward_penalty(self):
        env = DataCleanerEnv()
        env.reset("easy")
        action1 = Action(type="impute", column="age", method="median")
        env.step(action1)
        action2 = Action(type="impute", column="age", method="median")
        obs, reward, done, info = env.step(action2)
        self.assertLess(reward.value, 0)

if __name__ == "__main__":
    unittest.main()