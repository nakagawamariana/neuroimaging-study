# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:29:45 2020

@author: nakag
"""

import os
import unittest
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import numpy as np

#
# Correlacion
#

class Correlation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Correlation" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["Mariana Nakagawa, María Pardo, Gema Pérez"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
    This is an example of scripted loadable module bundled in an extension.
    """
    self.parent.acknowledgementText = """
    This file was developed for academical purposes
""" # replace with organization, grant and thanks.

#
# CorrelationWidget
#

class CorrelationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ( ("vtkMRMLScalarVolumeNode"), "" )
    self.inputSelector.addAttribute( "vtkMRMLScalarVolumeNode", "LabelMap", 0 )
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    #
    # input2 volume selector
    #
    self.input2Selector = slicer.qMRMLNodeComboBox()
    self.input2Selector.nodeTypes = ( ("vtkMRMLScalarVolumeNode"), "" )
    self.input2Selector.addAttribute( "vtkMRMLScalarVolumeNode", "LabelMap", 0 )
    self.input2Selector.selectNodeUponCreation = True
    self.input2Selector.addEnabled = True
    self.input2Selector.removeEnabled = True
    self.input2Selector.noneEnabled = False
    self.input2Selector.showHidden = False
    self.input2Selector.showChildNodeTypes = False
    self.input2Selector.renameEnabled = True
    self.input2Selector.setMRMLScene( slicer.mrmlScene )
    self.input2Selector.setToolTip( "Pick the input2 to the algorithm." )
    parametersFormLayout.addRow("input2 Volume: ", self.input2Selector)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply Correlation")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.input2Selector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and self.input2Selector.currentNode()

  def onApplyButton(self):
    logic = CorrelationLogic()

    inputVolume = self.inputSelector.currentNode()
    input2Volume = self.input2Selector.currentNode()
    if not (inputVolume and input2Volume):
      qt.QMessageBox.critical(slicer.util.mainWindow(), 'Correlation', 'Input and input2 volumes are required for Correlation')
      return

    print("Run the algorithm")
    #logic.run(inputVolume, input2Volume)

    result = logic.run(inputVolume, input2Volume)
    qt.QMessageBox.information(slicer.util.mainWindow(),'The correlation value is:',round(result,4))
#
# CorrelationLogic
#

class CorrelationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is a dummy logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      print('no volume node')
      return False
    if volumeNode.GetImageData() == None:
      print('no image data')
      return False
    return True

  def run(self,inputVolume,input2Volume):
    """
    Run the actual algorithm
    """

    self.delayDisplay('Running the algorithm')
    image1 = slicer.util.arrayFromVolume(inputVolume)
    image2 = slicer.util.arrayFromVolume(input2Volume)
    a = np.mean(image1)
    b = np.mean(image2)
    den = np.std(image1)*np.std(image2)
    
    sus1 = image1 - a
    sus2 = image2 - b
    

    matrix = (sus1*sus2)/den


    
    corr_norm = np.sum(matrix)/(matrix.shape[0]*matrix.shape[1]*matrix.shape[2])
    print('La correlacion es: ', round(corr_norm,4))
    




    return corr_norm


class CorrelationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_Correlation1()

  def test_Correlation1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests sould exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        print('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        print('Loading %s...\n' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading\n')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = CorrelationLogic()
    self.assertTrue( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')