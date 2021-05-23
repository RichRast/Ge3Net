import unittest
import doctest
import pytest
from src.main import evaluation
from src.utils import dataUtil

suite = unittest.TestSuite()
suite.addTest(doctest.DocTestSuite(evaluation))
suite.addTest(doctest.DocTestSuite(dataUtil))
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)






