from unittest import TestCase
from chatBot import LearningAlgorithm
import data


class TestLearningAlgorithm(TestCase):
    def setUp(self):
        self.sut = LearningAlgorithm()
        self.sut.optimize(data)

    def test_blocked(self):
        self.assertTrue(self.sut.blocked(data.CrossValidationData[0]))
        self.assertTrue(self.sut.blocked(data.CrossValidationData[1]))

    def test_not_blocked(self):
        self.assertFalse(self.sut.blocked(data.CrossValidationData[2]))
        self.assertFalse(self.sut.blocked(data.CrossValidationData[3]))
