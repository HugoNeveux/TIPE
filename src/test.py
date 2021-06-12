import unittest
import random
from perceptron import *

class TestHiddenLayer(unittest.TestCase):
    def test_random_weights_generation(self):
        n, p = random.randint(1, 20), random.randint(1, 20)
        layer = Hidden(n, p, lambda x: x)
        self.assertEqual(layer.weights.shape, (n, p))
        for i in range(n):
            for j in range(p):
                self.assertTrue(0 <= layer.weights[i][j] <= 1)
    
    
if __name__ == "__main__":
    unittest.main()
