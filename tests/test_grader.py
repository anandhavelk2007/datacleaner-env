import unittest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.tasks import grader_easy, grader_medium, grader_hard, NORMALIZATION_MAP

class TestGraders(unittest.TestCase):
    def test_grader_easy(self):
        df = pd.DataFrame({'age': [25,30,35], 'salary': [50000,60000,70000]})
        self.assertAlmostEqual(grader_easy(df), 1.0)
        df2 = pd.DataFrame({'age': [25,None,35], 'salary': [50000,None,70000]})
        self.assertAlmostEqual(grader_easy(df2), 1.0 - (2/6))
    def test_grader_medium(self):
        df = pd.DataFrame({'date': ['2023-01-15', '15/02/2023', 'invalid']})
        score = grader_medium(df)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    def test_grader_hard(self):
        df = pd.DataFrame({'country': ['USA', 'U.S.A', 'United States', 'Canada']})
        self.assertAlmostEqual(grader_hard(df), 0.75)
    def test_normalization_map(self):
        self.assertIn('usa', NORMALIZATION_MAP)
        self.assertEqual(NORMALIZATION_MAP['united states'], 'USA')

if __name__ == "__main__":
    unittest.main()