#!/usr/bin/env python

import unittest
from mlModel.experiments import basicExperiment

class testMlNetwork(unittest.TestCase):

    def test_contructor(self):

        jsonFile = 'testMl/auxFiles/test_paramFile.json'
        self.myExperiment = basicExperiment.basicExperiment(jsonFile)