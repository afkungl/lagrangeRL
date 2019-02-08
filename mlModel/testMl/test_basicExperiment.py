#!/usr/bin/env python

import unittest
from mlModel.experiments import basicExperiment
import shutil
import os

class testMlNetwork(unittest.TestCase):

    def setUp(self):

        jsonFile = 'testMl/auxFiles/test_paramFile.json'
        self.myExperiment = basicExperiment.basicExperiment(jsonFile)
        self.myExperiment.initializeExperiment()

    def test_constructor(self):

        jsonFile = 'testMl/auxFiles/test_paramFile.json'

        if os.path.exists('output'):
            shutil.rmtree('output')
        self.myExperiment = basicExperiment.basicExperiment(jsonFile)

    def test_experimentInitialization(self):

        self.myExperiment.initializeExperiment()

    def test_experimentSingleIteration(self):

        self.myExperiment.singleIteration()

    def test_runFullExperiment(self):

        self.myExperiment.runFullExperiment()

    def tearDown(self):

        if os.path.exists('output'):
            shutil.rmtree('output')

