import unittest
import sys
sys.path.insert(1, '..')
from test_utils import *

class MyTestCase(unittest.TestCase):

    def assertNear(self, a, b, tol=1E-6):
        self.assertTrue(near(a, b, tol))

    def assertNotNear(self, a, b, tol=1E-6):
        self.assertTrue(not near(a, b, tol))

    def assertApproxZero(self, a, tol=1E-6):
        self.assertTrue(near(a, 0, tol))

    def assertNotApproxZero(self, a, tol=1E-6):
        self.assertTrue(not near(a, 0, tol))
