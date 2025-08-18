from pathlib import Path
import json
import math
import os
import argparse
import csv

# Need to be installed on a fresh system
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
from plotly.subplots import make_subplots

pio.kaleido.scope.mathjax = None


class ExperimentSettings:
    pass

# NOTE: Needs to add the "sceneObject" key where is needed
def getProcessingSettings(mode, fileContents):
    try:
        experimentID = fileContents["experiment"]["index"]
    except:
        experimentID = 'None'
    
    try:
        experimentName = fileContents["configuration"]["experimentsToRun"][experimentID]["name"]
    except:
        print("Computing execution time chart...")
        experimentName = 'synthetic'
    settings = ExperimentSettings()
    settings.keepPreviousChartFile = False
    settings.minSamplesPerBin = 25
    settings.binCount = 75
    settings.xAxisTitleAdjustment = 0
    settings.heatmapRankLimit = 0
    settings.xAxisOutOfRangeMode = 'discard'
    settings.experimentName = experimentName
    settings.PRCEnabled = "enableComparisonToPRC" in fileContents["configuration"] and fileContents["configuration"]["enableComparisonToPRC"]
    try:
        settings.PRCSupportRadius = fileContents["computedConfiguration"][settings.methodName]["supportRadius"]
    except:
        settings.PRCSupportRadius = False
    try:
        settings.methodName = fileContents["method"]['name']
        settings.methodName = "Spin Image" if settings.methodName == "SI" else settings.methodName
    except:
        settings.methodName = 'tmp'
    settings.title = settings.methodName
    settings.enable3D = False
    sharedYAxisTitle = "Proportion of DDI"

    if experimentName == "normal-noise-only":
        settings.chartShortName = "Deviating<br>normal vector"
        settings.xAxisTitle = "Normal vector rotation (°)"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisMin = 0
        settings.xAxisMax = fileContents['configuration']['filterSettings']['normalVectorNoise'][
            'maxAngleDeviationDegrees']
        settings.xTick = 5
        settings.xAxisTitleAdjustment = 3
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["filterOutput"]["normal-noise-deviationAngle"]
        return settings
    elif experimentName == "subtractive-noise-only":
        settings.chartShortName = "Occlusion"
        settings.xAxisTitle = "Occlusion"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisMin = 0
        settings.xAxisMax = 1
        settings.xTick = 0.2
        settings.enable2D = False
        settings.reverse = True
        settings.readValueX = lambda x: x["fractionSurfacePartiality"]
        return settings
    elif experimentName == "additive-noise-only":
        settings.chartShortName = "Clutter"
        settings.xAxisTitle = "Clutter"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisMin = 0
        settings.xAxisMax = 10
        settings.xTick = 1
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["fractionAddedNoise"]
        return settings
    elif experimentName == "support-radius-deviation-only":
        settings.chartShortName = "Deviating<br>support radius"
        settings.xAxisTitle = "Support radius scale factor"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisMin = 1 - fileContents['configuration']['filterSettings']['supportRadiusDeviation'][
            'maxRadiusDeviation']
        settings.xAxisMax = 1 + fileContents['configuration']['filterSettings']['supportRadiusDeviation'][
            'maxRadiusDeviation']
        settings.xTick = 0.125
        settings.xAxisTitleAdjustment = 2
        settings.enable2D = False
        settings.reverse = True # scale factor used is stored as-is, but the relative change to the support radius is the inverse
        settings.readValueX = lambda x: x["filterOutput"]["support-radius-scale-factor"]
        return settings
    elif experimentName == "repeated-capture-only":
        settings.chartShortName = "Alternate<br>triangulation"
        settings.xAxisTitle = "Vertex displacement distance"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisMin = 0
        settings.xAxisMax = 0.15
        settings.xAxisTitleAdjustment = 5
        settings.xTick = 0.03
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["filterOutput"]["triangle-shift-average-edge-length"]
        return settings
    elif experimentName == "gaussian-noise-only":
        settings.chartShortName = "Gaussian<br>noise"
        settings.xAxisTitle = "Standard Deviation"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisMin = 0
        settings.xAxisMax = 0.01
        settings.xTick = 0.005
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["filterOutput"]["gaussian-noise-max-deviation"]
        return settings
    elif experimentName == "depth-camera-capture-only":
        settings.chartShortName = "Alternate<br>mesh resolution"
        settings.xAxisTitle = "Object distance from camera"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisTitleAdjustment = 6
        settings.xAxisMin = 2
        settings.xAxisMax = 10
        settings.xTick = 1
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["filterOutput"]["depth-camera-capture-distance-from-camera"]  # (float(x["filterOutput"]["depth-camera-capture-initial-vertex-count"])
        # / float(x["filterOutput"]["depth-camera-capture-filtered-vertex-count"]))
        return settings
    elif experimentName == "additive-and-gaussian-noise":
        settings.chartShortName = "Clutter and Gaussian noise"
        settings.xAxisTitle = "Clutter"
        settings.yAxisTitle = "Standard Deviation"
        settings.xAxisBounds = [0, 25]
        settings.yAxisBounds = [0, 0.01]
        settings.xTick = 5
        settings.yTick = 0.002
        settings.binCount = 50
        settings.enable2D = True
        settings.reverseX = False
        settings.reverseY = False
        settings.readValueX = lambda x: x["fractionAddedNoise"]
        settings.readValueY = lambda x: x["filterOutput"]["gaussian-noise-max-deviation"]
        return settings
    elif experimentName == "additive-and-subtractive-noise":
        settings.chartShortName = "Clutter and occlusion"
        settings.xAxisTitle = "Occlusion"
        settings.yAxisTitle = "Clutter"
        settings.xAxisTitleAdjustment = 0
        settings.xAxisBounds = [0, 1]
        settings.yAxisBounds = [0, 25]
        settings.xTick = 0.2
        settings.yTick = 5
        settings.binCount = 50
        settings.enable2D = True
        settings.reverseX = True # Used for occlusion because the definition in the paper is the inverse of what is recorded by the benchmark
        settings.reverseY = False
        settings.readValueX = lambda x: x["fractionSurfacePartiality"]
        settings.readValueY = lambda x: x["fractionAddedNoise"]
        return settings
    elif experimentName == "subtractive-and-gaussian-noise":
        settings.chartShortName = "Occlusion and Gaussian noise"
        settings.xAxisTitle = "Occlusion"
        settings.yAxisTitle = "Standard Deviation"
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisTitleAdjustment = 0
        settings.xAxisBounds = [0, 1]
        settings.yAxisBounds = [0, 0.01]
        settings.xTick = 0.2
        settings.yTick = 0.002
        settings.binCount = 35 #25 50 35
        settings.enable2D = True
        settings.reverseX = True
        settings.reverseY = False
        settings.readValueX = lambda x: x["fractionSurfacePartiality"]
        settings.readValueY = lambda x: x["filterOutput"]["gaussian-noise-max-deviation"]
        return settings
    elif experimentName == "_multi-filter-subtractive-gaussian": #To bypass a createDDI2DChart in createDDIChart, after testing remove _
        settings.chartShortName = "Occlusion and Gaussian noise"
        settings.xAxisTitle = "Scene: Occlusion"
        settings.yAxisTitle = "Model: Gaussian Noise"
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisTitleAdjustment = 0
        settings.xAxisBounds = [0, 1]
        settings.yAxisBounds = [0, 0.01]
        settings.yTick = 0.002
        settings.xTick = 0.2
        settings.binCount = 35
        settings.enable2D = True
        settings.reverseX = True
        settings.reverseY = False
        settings.readValueX = lambda x: x["sceneObject"]["fractionSurfacePartiality"]
        settings.readValueY = lambda x: x["modelObject"]["filterOutput"]["gaussian-noise-max-deviation"]
        return settings
    elif experimentName == "multi-filter-subtractive-gaussian":
        settings.chartShortName = "Occlusion and Gaussian noise on object and scene"
        settings.xAxisTitle = "Scene: Occlusion"
        settings.yAxisTitle = sharedYAxisTitle
        settings.zAxisTitle = "Model: Gaussian Noise"
        settings.wAxisTitle = "Model: Normal Noise"
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisTitleAdjustment = 0
        settings.xAxisBounds = [0, 1]
        settings.zAxisBounds = [0, 0.01]
        settings.wAxisBounds = [0, fileContents["configuration"]["filterSettings"]["normalVectorNoise"]["maxAngleDeviationDegrees"]]
        settings.wTick = (settings.wAxisBounds[1] - settings.wAxisBounds[0])/5
        settings.zTick = 0.002
        settings.xTick = 0.2
        settings.binCount = 35
        settings.enable2D = False
        settings.enable3D = True
        settings.reverseX = True
        settings.reverseZ = False
        settings.reverseW = False
        settings.readValueX = lambda x: x["sceneObject"]["fractionSurfacePartiality"]
        settings.readValueZ = lambda x: x["modelObject"]["filterOutput"]["gaussian-noise-max-deviation"]
        settings.readValueW = lambda x: x["modelObject"]["filterOutput"]["normal-noise-deviationAngle"]
        return settings
    elif experimentName == "synthetic":
        print('Correctly in synthetic...')
        settings.chartShortName = "Execution Time Measurament"
        settings.xAxisTitle = "Workload Size"
        settings.yAxisTitle = "Execution Time"
        settings.reverse = False
        #settings.workloadType = fileContents["workloadType"] #Fill from the JSON results file, tmp value
        settings.xAxisMin = 0
        settings.xAxisMax = 0
        settings.binCount = 10
        settings.xTick = 10
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisTitleAdjustment = 0
        return settings
    
    elif experimentName == "experiment1-level1-occlusion-only":
        settings.chartShortName = "Occlusion"
        settings.xAxisTitle = "Occlusion"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisOutOfRangeMode = 'discard'
        settings.xAxisMin = 0
        settings.xAxisMax = 1
        settings.xAxisBounds = [0, 1]
        settings.xTick = 0.2
        settings.enable2D = False
        settings.reverse = True
        settings.readValueX = lambda x: x["sceneObject"]["fractionSurfacePartiality"]
        return settings
    elif experimentName == "experiment2-level1-clutter-only":
        settings.chartShortName = "Clutter"
        settings.xAxisTitle = "Clutter"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisMin = 0
        settings.xAxisMax = 10
        settings.xTick = 1
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["sceneObject"]["fractionAddedNoise"]
        return settings
    elif experimentName == "experiment3-level1-gaussian-noise-only":
        settings.chartShortName = "Gaussian<br>noise"
        settings.xAxisTitle = "Standard Deviation"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisMin = 0
        settings.xAxisMax = 0.01
        settings.xTick = 0.005
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["sceneObject"]["filterOutput"]["gaussian-noise-max-deviation"]
        return settings
    elif experimentName == "experiment4-level2-occlusion-and-gaussian-noise":
        settings.chartShortName = "Occlusion and Gaussian noise"
        settings.xAxisTitle = "Occlusion"
        settings.yAxisTitle = "Standard Deviation"
        settings.xAxisOutOfRangeMode = 'discard'
        settings.xAxisTitleAdjustment = 0
        settings.xAxisBounds = [0, 1]
        settings.yAxisBounds = [0, 0.01]
        settings.xTick = 0.2
        settings.yTick = 0.002
        settings.binCount = 35 #25 50 35
        settings.enable2D = True
        settings.reverseX = True
        settings.reverseY = False
        settings.readValueX = lambda x: x["sceneObject"]["fractionSurfacePartiality"]
        settings.readValueY = lambda x: x["sceneObject"]["filterOutput"]["gaussian-noise-max-deviation"]
        return settings
    elif experimentName == "experiment5-level2-occlusion-both":
        settings.chartShortName = "Occlusion on both"
        settings.xAxisTitle = "Common Area"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisMin = 0
        settings.xAxisMax = 1
        settings.xTick = 0.2
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["sceneObject"]["filterOutput"]["multi-view-occlusion-common-area"]
        
        '''settings.chartShortName = "Occlusion on both"
        settings.xAxisTitle = "Common Area"
        settings.yAxisTitle = "Occlusion"
        settings.xAxisOutOfRangeMode = 'discard'
        settings.xAxisTitleAdjustment = 0
        settings.xAxisBounds = [0, 1]
        settings.yAxisBounds = [0, 1]
        settings.xTick = 0.2
        settings.yTick = 0.2
        settings.binCount = 35 #25 50 35
        settings.enable2D = True
        settings.reverseX = False
        settings.reverseY = True
        settings.readValueX = lambda x: x["sceneObject"]["filterOutput"]["multi-view-occlusion-common-area"]
        settings.readValueY = lambda x: x["modelObject"]["fractionSurfacePartiality"]'''
        return settings
    elif experimentName == "experiment6-level2-occlusion-fixed-gaussian-both":
        settings.chartShortName = "Occlusion on both with gaussian noise"
        settings.xAxisTitle = "Common Area"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisMin = 0
        settings.xAxisMax = 1
        settings.xTick = 0.2
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["sceneObject"]["filterOutput"]["multi-view-occlusion-common-area"]
        
        '''settings.chartShortName = "Occlusion on both and fixed gaussian"
        settings.xAxisTitle = "Common Area"
        settings.yAxisTitle = "Occlusion"
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisTitleAdjustment = 0
        settings.xAxisBounds = [0, 1]
        settings.yAxisBounds = [0, 1]
        settings.xTick = 0.2
        settings.yTick = 0.2
        settings.binCount = 35 #25 50 35
        settings.enable2D = True
        settings.reverseX = False
        settings.reverseY = True
        settings.readValueX = lambda x: x["sceneObject"]["filterOutput"]["multi-view-occlusion-common-area"]
        settings.readValueY = lambda x: x["modelObject"]["fractionSurfacePartiality"]'''
        return settings
    elif experimentName == "experiment7-level2-occlusion-both-clutter":
        settings.chartShortName = "Occlusion on both and clutter"
        settings.xAxisTitle = "Common Area"
        settings.yAxisTitle = "Clutter"
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisTitleAdjustment = 0
        settings.xAxisBounds = [0, 1]
        settings.yAxisBounds = [0, 10]
        settings.xTick = 0.2
        settings.yTick = 2
        settings.binCount = 35 #25 50 35
        settings.enable2D = True
        settings.reverseX = False
        settings.reverseY = False
        settings.readValueX = lambda x: x["sceneObject"]["filterOutput"]["multi-view-occlusion-common-area"]
        settings.readValueY = lambda x: x["modelObject"]["fractionAddedNoise"]
        return settings
    elif experimentName == "experiment8-level2-occlusion-noise-both-clutter":
        settings.chartShortName = "Occlusion on both and clutter with fixed gaussian noise"
        settings.xAxisTitle = "Common Area"
        settings.yAxisTitle = "Clutter"
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisTitleAdjustment = 0
        settings.xAxisBounds = [0, 1]
        settings.yAxisBounds = [0, 10]
        settings.xTick = 0.2
        settings.yTick = 2
        settings.binCount = 35 #25 50 35
        settings.enable2D = True
        settings.reverseX = False
        settings.reverseY = False
        settings.readValueX = lambda x: x["sceneObject"]["filterOutput"]["multi-view-occlusion-common-area"]
        settings.readValueY = lambda x: x["modelObject"]["fractionAddedNoise"]
        return settings
    elif experimentName == "experiment9-level3-ultimate-test":
        settings.chartShortName = "Ultimate test"
        settings.xAxisTitle = "Occlusion"
        settings.yAxisTitle = sharedYAxisTitle
        settings.zAxisTitle = "Vertex displacement" #Vertex displacement distance
        settings.wAxisTitle = "Clutter"
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisTitleAdjustment = 0
        settings.xAxisBounds = [0, 1]
        settings.zAxisBounds = [0, 0.15]
        settings.wAxisBounds = [0, 2]
        settings.xTick = 0.2
        settings.zTick = (settings.zAxisBounds[1] - settings.zAxisBounds[0])/5
        settings.wTick = 1
        settings.binCount = 35
        settings.enable2D = False
        settings.enable3D = True
        settings.reverseX = True
        settings.reverseZ = False
        settings.reverseW = False
        settings.readValueX = lambda x: x["sceneObject"]["fractionSurfacePartiality"]
        settings.readValueZ = lambda x: x["sceneObject"]["filterOutput"]["triangle-shift-average-edge-length"]
        settings.readValueW = lambda x: x["sceneObject"]["fractionAddedNoise"]
        return settings
    else:
        raise Exception("Failed to determine chart settings: Unknown experiment name: " + experimentName)


def dot(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


def processSingleFile(jsonContent, settings):
    chartDataSequence = {}
    chartDataSequence["name"] = settings.methodName
    chartDataSequence["x"] = []
    chartDataSequence["y"] = []
    if settings.enable2D:
        chartDataSequence["ranks"] = []
    if settings.PRCEnabled:
        chartDataSequence["PRC"] = []

    rawResults = []

    for result in jsonContent["results"]:
        if settings.enable2D:
            rawResult = [settings.readValueX(result), settings.readValueY(result), result['filteredDescriptorRank']]
        elif settings.enable3D:
            rawResult = [settings.readValueX(result), settings.readValueZ(result), settings.readValueW(result), result['filteredDescriptorRank']]
        else:
            rawResult = [settings.readValueX(result), result['filteredDescriptorRank']]
        if settings.PRCEnabled:
            tao = 0 if result["PRC"]["distanceToSecondNearestNeighbour"] == 0 else result["PRC"]["distanceToNearestNeighbour"] / result["PRC"]["distanceToSecondNearestNeighbour"]
            delta = [result["PRC"]["nearestNeighbourVertexModel"][i] - result["PRC"]["nearestNeighbourVertexScene"][i] for i in range(0, 3)]
            distanceBetweenVertices = math.sqrt(dot(delta, delta))
            meshIDsEquivalent = result["PRC"]["modelPointMeshID"] == result["PRC"]["scenePointMeshID"]

            rawResult.append((tao, distanceBetweenVertices, meshIDsEquivalent))

        rawResults.append(rawResult)
        chartDataSequence["x"].append(rawResult[0])
        chartDataSequence["y"].append(rawResult[1])
        if settings.enable2D:
            chartDataSequence["ranks"].append(rawResult[2])

    return chartDataSequence, rawResults


def computeStackedHistogram(rawResults, config, settings):
    histogramMax = settings.xAxisMax
    histogramMin = settings.xAxisMin

    delta = (histogramMax - histogramMin) / settings.binCount

    representativeSetSize = config['commonExperimentSettings']['representativeSetSize']
    stepsPerBin = int(math.log10(representativeSetSize)) + 1

    histogram = []
    prcInfo = [[] for _ in range(0, settings.binCount + 1)]
    labels = []
    for i in range(stepsPerBin):
        if i != stepsPerBin: # Shouldent this always be satisfyed
            histogram.append([0] * (settings.binCount + 1))
            if i == 0:
                labels.append('0')
            elif i == 1:
                labels.append('1 - 10')
            else:
                labels.append(str(int(10 ** (i - 1)) + 1) + " - " + str(int(10 ** i)))

    xValues = [((float(x + 1) * delta) + histogramMin) for x in range(settings.binCount)]

    removedCount = 0
    for rawResult in rawResults:
        if rawResult[0] is None:
            rawResult[0] = 0
        if rawResult[0] < histogramMin or rawResult[0] > histogramMax:
            if settings.xAxisOutOfRangeMode == 'discard':
                removedCount += 1
                continue
            elif settings.xAxisOutOfRangeMode == 'clamp':
                rawResult[0] = max(histogramMin, min(histogramMax, rawResult[0]))

        if settings.reverse:
            rawResult[0] = (histogramMax + histogramMin) - rawResult[0]

        binIndexX = int((rawResult[0] - histogramMin) / delta)
        binIndexY = int(0 if rawResult[1] == 0 else (math.log10(rawResult[1]) + 1))
        if rawResult[1] == representativeSetSize:
            binIndexY -= 1
        if settings.PRCEnabled:
            tao, distanceBetweenVertices, meshIDsEquivalent = rawResult[-1]
            criterion1_isWithinRange = distanceBetweenVertices <= settings.PRCSupportRadius / 2
            criterion2_meshIDIsEquivalent = meshIDsEquivalent
            isValidMatch = criterion1_isWithinRange and criterion2_meshIDIsEquivalent
            prcInfo[binIndexX].append((tao, isValidMatch))
        if binIndexY >= len(histogram):
            continue
        histogram[binIndexY][binIndexX] += 1



    if settings.PRCEnabled:
        taoStep = 0.01
        taoStepCount = int(1 / taoStep)
        taoValues = [x * taoStep for x in range(0, taoStepCount)]

        areaUnderCurves = []

        #countsFigure = go.Figure()
        for binIndex, prcBin in enumerate(prcInfo):
            # compute a PRC curve for each of these

            prcCurvePoints = []
            for taoValue in taoValues:
                correctMatchCount = 0
                matchCount = 0
                for computedTao, satisfiesMatchCriteria in prcBin:
                    isMatch = computedTao <= taoValue
                    if isMatch:
                        matchCount += 1
                    if isMatch and satisfiesMatchCriteria:
                        correctMatchCount += 1

                precision = 0 if matchCount == 0 else correctMatchCount / matchCount
                recall = 0 if len(prcBin) == 0 else correctMatchCount / len(prcBin)
                #print(matchCount, correctMatchCount, len(prcBin), precision, recall)
                prcCurvePoints.append((recall, precision))
            sortedPRCPoints = sorted(prcCurvePoints, key=lambda tup: tup[0])
            # continue area to the left of curve
            prcArea = 0 # sortedPRCPoints[0][0] * sortedPRCPoints[0][1]
            for i in range(0, len(sortedPRCPoints) - 1):
                point1 = sortedPRCPoints[i]
                point2 = sortedPRCPoints[i + 1]
                deltaX = (point2[0] - point1[0])
                averageY = (point2[1] + point1[1]) / 2
                prcArea += averageY * deltaX
            areaUnderCurves.append(prcArea)
            #countsFigure.add_trace(go.Scatter(x=[x[0] for x in prcCurvePoints], y=[x[1] for x in prcCurvePoints], mode='lines', name="PRC_" + str(binIndex)))
        #countsFigure.add_trace(go.Scatter(x=xValues, y=areaUnderCurves, mode='lines', name="PRC"))
        #countsFigure.show()



    counts = [sum(x) for x in zip(*histogram)]

    print('Excluded', removedCount, 'entries')

    # Time to normalise

    for i in range(settings.binCount):
        stepSum = 0
        for j in range(stepsPerBin):
            stepSum += histogram[j][i]

        if stepSum > settings.minSamplesPerBin:

            for j in range(stepsPerBin):
                histogram[j][i] = float(histogram[j][i]) / float(stepSum)
        else:
            for j in range(stepsPerBin):
                histogram[j][i] = None

    if settings.PRCEnabled:
        return xValues, histogram, labels, counts, areaUnderCurves
    else:
        return xValues, histogram, labels, counts, []


def generateSupportRadiusChart(results_directory, output_directory):
    csvFilePaths = [x.name for x in os.scandir(results_directory) if x.name.endswith(".txt")]
    csvFilePaths.sort()

    if len(csvFilePaths) == 0:
        print("No json files were found in this directory. Aborting.")
        return

    for index, csvFileName in enumerate(csvFilePaths):
        isLastFile = index + 1 == len(csvFilePaths)
        csvFilePath = os.path.join(results_directory, csvFileName)
        with open(csvFilePath) as csvFile:
            print('Processing file:', csvFileName)
            reader = csv.DictReader(csvFile)
            sequenceDict = {}
            for row in reader:
                for key in row:
                    key_stripped = key.strip()
                    if key_stripped not in sequenceDict:
                        sequenceDict[key_stripped] = []
                    sequenceDict[key_stripped].append(float(row[key]))
            xValues = sequenceDict["radius"]
            yValues_Sequence1 = sequenceDict["Min mean"]
            yValues_Sequence2 = sequenceDict["Mean"]
            yValues_Sequence3 = sequenceDict["Max mean"]

            countsFigure = go.Figure()

            lineLocation = xValues[yValues_Sequence2.index(max(yValues_Sequence2))]

            methodName = csvFileName.split("_")[3]
            yAxisRange = [0, max(yValues_Sequence3)]
            useLog = False
            if methodName == 'SI':
                # The Spin image method uses Pearson Correlation
                # Because as opposed to all other distance metrics, a higher number is better for Pearson, we computed 1 - pearsonDistance
                # such that any distance would be between 0 and 2, where lower is better like all other methods
                # We therefore need to do an inverse of that here to show correct distances in the chart
                # (that make sense to the reader anyway)
                yValues_Sequence1 = [1 - x for x in yValues_Sequence1]
                yValues_Sequence2 = [1 - x for x in yValues_Sequence2]
                yValues_Sequence3 = [1 - x for x in yValues_Sequence3]
                yAxisRange = [-0.5, 0.5]
            elif methodName == "RICI":
                useLog = True
                # Make line visible
                yValues_Sequence1 = [max(x, 1) for x in yValues_Sequence1]
                yAxisRange = [0, math.log10(max(yValues_Sequence3))]

            yAxisTickMode = 'linear' if not useLog else 'log'

            chartWidth = 285
            if isLastFile:
                #countsFigure.update_layout(legend=dict(y=0, orientation="h", yanchor="bottom", yref="container", xref="paper", xanchor="left"))
                chartWidth = 350
            else:
                countsFigure.update_layout(showlegend=False)

            pio.kaleido.scope.default_width = chartWidth
            pio.kaleido.scope.default_height = 300

            countsFigure.add_trace(go.Scatter(x=xValues, y=yValues_Sequence1, mode='lines', name="Min"))
            countsFigure.add_trace(go.Scatter(x=xValues, y=yValues_Sequence2, mode='lines', name="Mean"))
            countsFigure.add_trace(go.Scatter(x=xValues, y=yValues_Sequence3, mode='lines', name="Max"))
            countsFigure.add_vline(x=lineLocation)
            countsFigure.update_yaxes(range=yAxisRange, type=yAxisTickMode)
            countsFigure.update_layout(xaxis_title="Radius", yaxis_title='Distance',
                                       title_x=0.5, margin={'t': 2, 'l': 0, 'b': 0, 'r': 10}, width=chartWidth, height=270,
                                       font=dict(size=18), xaxis=dict(tickmode='linear', dtick=0.5, range=(0, max(xValues))))

            outputFile = os.path.join(output_directory, "support-radius-" + methodName + ".pdf")

            pio.write_image(countsFigure, outputFile, format='pdf', validate=True)
    print('Done.')


def create2DChart(rawResults, configuration, settings, output_directory, jsonFilePath, jsonFilePaths):
    histogramAccepted = [[0] * settings.binCount for i in range(settings.binCount)]
    histogramTotal = [[0] * settings.binCount for i in range(settings.binCount)]

    removedCount = 0

    deltaX = (settings.xAxisBounds[1] - settings.xAxisBounds[0]) / settings.binCount
    deltaY = (settings.yAxisBounds[1] - settings.yAxisBounds[0]) / settings.binCount

    for rawResult in rawResults:
        # Ignore the PRC information for this chart type
        resultX, resultY, rank, _ = rawResult
        if resultX is None:
            resultX = 0
        if resultY is None:
            resultY = 0
        if settings.reverseX:
            resultX = settings.xAxisBounds[1] - resultX
        if settings.reverseY:
            resultY = settings.yAxisBounds[1] - resultY

        if resultX < settings.xAxisBounds[0] or resultX > settings.xAxisBounds[1]:
            if settings.xAxisOutOfRangeMode == 'discard':
                removedCount += 1
                continue
            elif settings.xAxisOutOfRangeMode == 'clamp':
                rawResult[0] = max(settings.xAxisBounds[0], min(settings.xAxisBounds[1], rawResult[0]))
        if resultY < settings.yAxisBounds[0] or resultY > settings.yAxisBounds[1]:
            removedCount += 1
            continue

        binIndexX = max(0, min(settings.binCount - 1, int((resultX - settings.xAxisBounds[0]) / deltaX)))
        binIndexY = max(0, min(settings.binCount - 1, int((resultY - settings.yAxisBounds[0]) / deltaY)))
        if rank <= settings.heatmapRankLimit:
            histogramAccepted[binIndexY][binIndexX] += 1
        histogramTotal[binIndexY][binIndexX] += 1

    dataRangeX = []
    dataRangeY = []
    dataRangeZ = []
    print("Removed", removedCount, "samples")

    for row in range(0, settings.binCount):
        for col in range(0, settings.binCount):
            dataRangeX.append(col * deltaX + 0.5 * deltaX)
            dataRangeY.append(row * deltaY + 0.5 * deltaY)
            if histogramTotal[row][col] < 10:
                dataRangeZ.append(None)
            else:
                dataRangeZ.append(float(histogramAccepted[row][col]) / histogramTotal[row][col])

    stackFigure = go.Figure(go.Heatmap(x=dataRangeX, y=dataRangeY, z=dataRangeZ,
                                       zmin=0, zmax=1, colorscale=
        [
            [0, 'rgb(200, 200, 200)'],
            [0.25, 'rgb(220, 50, 47)'],
            [0.5, 'rgb(203, 75, 22)'],
            [0.75, 'rgb(181, 137, 000)'],
            [1.0, 'rgb(0, 128, 64)']
        ]))

    xAxisTitle = settings.xAxisTitle
    if jsonFilePath is not jsonFilePaths[-1]:
        stackFigure.update_coloraxes(showscale=False)
        stackFigure.update_traces(showscale=False)
        pio.kaleido.scope.default_width = 300
        pio.kaleido.scope.default_height = 300
        if settings.xAxisTitleAdjustment > 0:
            xAxisTitle += ' ' * settings.xAxisTitleAdjustment
            xAxisTitle += 't'
    else:
        pio.kaleido.scope.default_width = 368
        pio.kaleido.scope.default_height = 300

    stackFigure.update_layout(xaxis_title=xAxisTitle, yaxis_title=settings.yAxisTitle,
                              margin={'t': 0, 'l': 0, 'b': 45, 'r': 15}, font=dict(size=18),
                              xaxis=dict(autorange=False, automargin=True, dtick=settings.xTick, range=settings.xAxisBounds),
                              yaxis=dict(autorange=False, automargin=True, dtick=settings.yTick, range=settings.yAxisBounds))
    #stackFigure.show()

    outputFile = os.path.join(output_directory, settings.experimentName + "-" + settings.methodName + ".pdf")
    pio.write_image(stackFigure, outputFile, format='pdf', validate=True)




def createChart(results_directory, output_directory, mode):
    pio.templates[pio.templates.default].layout.colorway = [
        '#%02x%02x%02x' % (0, 128, 64),
        '#%02x%02x%02x' % (181, 137, 000),
        '#%02x%02x%02x' % (203, 75, 22),
        '#%02x%02x%02x' % (220, 50, 47),
        '#%02x%02x%02x' % (211, 54, 130),
        '#%02x%02x%02x' % (108, 113, 196),
        '#%02x%02x%02x' % (38, 139, 210),
        '#%02x%02x%02x' % (42, 161, 152),
        '#%02x%02x%02x' % (133, 153, 000),
    ]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    if mode == "support-radius":
        generateSupportRadiusChart(results_directory, output_directory)
        return None

    # Find all JSON files in results directory
    jsonFilePaths = [x.name for x in os.scandir(results_directory) if x.name.endswith(".json")]
    jsonFilePaths.sort()

    if len(jsonFilePaths) == 0:
        print("No json files were found in this directory. Aborting.")
        return

    totalAreas = []
    allCounts = []
    countXValues = []
    countLabels = []
    lastSettings = ""
    chartAreas = {}

    print("Found {} json files".format(len(jsonFilePaths)))
    for jsonFilePath in jsonFilePaths:
        with open(os.path.join(results_directory, jsonFilePath)) as inFile:
            print('Loading file: {}'.format(jsonFilePath))
            jsonContents = json.load(inFile)
            settings = getProcessingSettings(mode, jsonContents)
            fileNameParts = jsonFilePath.split('/')[-1].split('.json')[0].split('-')
            settings.fileDateTime = fileNameParts[5] + '-' + fileNameParts[6]
            print('Creating chart for method ' + settings.methodName + "..")
            lastSettings = settings
            dataSequence, rawResults = processSingleFile(jsonContents, settings)

            if settings.enable2D:
                create2DChart(rawResults, jsonContents["configuration"], settings, output_directory, jsonFilePath, jsonFilePaths)
                createDDI2DChart(rawResults, settings, output_directory, jsonFilePath, jsonFilePaths)
                continue
            elif settings.enable3D:
                #allAreas = {}
                #allAreas[settings.methodName] = createDDI3DChart(rawResults, settings, output_directory)
                continue
            else:
                stackedXValues, stackedYValues, stackedLabels, counts, areaUnderCurves = computeStackedHistogram(rawResults, jsonContents["configuration"], settings)
            
            allCounts.append(counts)

            areaUnderDDIZeroCurve = 0
            previousX = stackedXValues[0]
            previousY = stackedYValues[0][0]
            for currentX, currentY in zip(stackedXValues, stackedYValues[0]):
                if currentY is None:
                    previousX = currentX
                    previousY = 0
                    continue
                deltaX = currentX - previousX
                averageY = (currentY + previousY) / 2.0
                areaUnderDDIZeroCurve += deltaX * averageY
                previousY = currentY
                previousX = currentX

            print('Total area under zero curve:', areaUnderDDIZeroCurve)
            maxDDIArea = (settings.xAxisMax - settings.xAxisMin) * 1
            normalisedAreaUnderDDICurve = areaUnderDDIZeroCurve / maxDDIArea
            chartAreas[settings.methodName] = normalisedAreaUnderDDICurve
            countLabels.append(settings.methodName)
            countXValues = stackedXValues

            # stackFigure
            stackFigure = go.Figure()

            for index, yValueStack in enumerate(stackedYValues):
                stackFigure.add_trace(
                    go.Scatter(x=stackedXValues, y=yValueStack, name=stackedLabels[index], stackgroup="main"))
            if settings.PRCEnabled:
                stackFigure.add_trace(go.Scatter(x=stackedXValues, y=areaUnderCurves, name="AUC", line_color=pio.templates[pio.templates.default].layout.colorway[7]))

            xAxisTitle = settings.xAxisTitle
            if jsonFilePath is not jsonFilePaths[-1]:
                if settings.xAxisTitleAdjustment > 0:
                    xAxisTitle += ' ' * settings.xAxisTitleAdjustment
                    xAxisTitle += 't'
                stackFigure.update_layout(showlegend=False)
                titleX = 0.5
                pio.kaleido.scope.default_width = 300
                pio.kaleido.scope.default_height = 300
            else:
                pio.kaleido.scope.default_width = 475
                pio.kaleido.scope.default_height = 300
                titleX = (float(200) / float(500)) * 0.5

            stackFigure.update_yaxes(range=[0, 1])
            stackFigure.update_xaxes(range=[settings.xAxisMin, settings.xAxisMax])

            stackFigure.update_layout(xaxis_title=xAxisTitle, yaxis_title=settings.yAxisTitle, title_x=titleX,
                                      margin={'t': 0, 'l': 0, 'b': 45, 'r': 15}, font=dict(size=18), xaxis=dict(tickmode='linear', dtick=settings.xTick))

            outputFile = os.path.join(output_directory, settings.experimentName + "-" + settings.methodName + '-' + settings.fileDateTime + ".pdf")
            if settings.keepPreviousChartFile:
                outputFileIndex = 0
                while os.path.exists(outputFile):
                    outputFileIndex += 1
                    outputFile = os.path.join(output_directory,
                                              settings.experimentName + "-" + settings.methodName + '-' + str(outputFileIndex) + ".pdf")
            pio.write_image(stackFigure, outputFile, format='pdf', validate=True)
            
            # single DDI Chart
            DDIChart = go.Figure()
            DDIChart.add_trace(
                go.Scatter(x=stackedXValues, y=stackedYValues[0], name=stackedLabels[0], mode="lines")
            )
            
            if jsonFilePath is not jsonFilePaths[-1]:
                if settings.xAxisTitleAdjustment > 0:
                    xAxisTitle += ' ' * settings.xAxisTitleAdjustment
                    xAxisTitle += 't'
                DDIChart.update_layout(showlegend=False)
                titleX = 0.5
                pio.kaleido.scope.default_width = 300
                pio.kaleido.scope.default_height = 300
            else:
                pio.kaleido.scope.default_width = 475
                pio.kaleido.scope.default_height = 300
                titleX = (float(200) / float(500)) * 0.5
            
            DDIChart.update_yaxes(range=[0, 1])
            DDIChart.update_xaxes(range=[settings.xAxisMin, settings.xAxisMax])

            DDIChart.update_layout(xaxis_title=xAxisTitle,
                                   yaxis_title=settings.yAxisTitle,
                                   title_x=titleX,
                                   margin={'t': 0, 'l': 0, 'b': 45, 'r': 15},
                                   font=dict(size=18),
                                   xaxis=dict(tickmode='linear', dtick=settings.xTick))

            outputFile = os.path.join(output_directory, settings.experimentName + "-" + settings.methodName + '-' + settings.fileDateTime + "-DDI0Chart.pdf")

            if settings.keepPreviousChartFile:
                outputFileIndex = 0
                while os.path.exists(outputFile):
                    outputFileIndex += 1
                    outputFile = os.path.join(output_directory,
                                              settings.experimentName + "-" + settings.methodName + '-' + str(outputFileIndex) + ".pdf")
            pio.write_image(DDIChart, outputFile, format='pdf', validate=True)

    if not settings.enable2D and not settings.enable3D:
        print('Writing counts chart..')
        countsFigure = go.Figure()
        for index, countSet in enumerate(allCounts):
            countSet = [x if x >= settings.minSamplesPerBin else None for x in countSet]
            countsFigure.add_trace(go.Scatter(x=countXValues, y=countSet, mode='lines', name=countLabels[index]))
        countsFigure.update_yaxes(range=[0, math.log10(max([max(x) for x in allCounts]))], type="log")
        countsFigure.update_xaxes(range=[lastSettings.xAxisMin, lastSettings.xAxisMax], dtick=settings.xTick)
        countsFigure.update_layout(xaxis_title=settings.xAxisTitle, yaxis_title='Sample Count',
                                   title_x=0.5, margin={'t': 0, 'l': 0, 'b': 0, 'r': 0},
                                   font=dict(size=18), xaxis=dict(
                tickmode='linear',
                dtick=settings.xTick,
                range=(lastSettings.xAxisMin, lastSettings.xAxisMax)

            ))
        #countsFigure.update_layout(
        #    legend=dict(y=0, orientation="h", yanchor="bottom", yref="container", xref="paper", xanchor="left"))
        outputFile = os.path.join(output_directory, lastSettings.experimentName + "-counts.pdf")
        pio.kaleido.scope.default_width = 435
        pio.kaleido.scope.default_height = 300
        pio.write_image(countsFigure, outputFile, format='pdf', validate=True)
        return (chartAreas, settings.chartShortName)

    print('Done.')
    return None

def writeOverviewChart(contents, outputFile):
    # have: chart name -> method name -> value
    # need: method name -> chart name -> value
    resultsByMethod = {}
    for chartName in contents:
        for methodName in contents[chartName]:
            if methodName not in resultsByMethod:
                resultsByMethod[methodName] = {}
            resultsByMethod[methodName][chartName] = contents[chartName][methodName]

    countsFigure = go.Figure()
    for methodName in ['QUICCI', 'RICI', 'Spin Image', 'RoPS', 'SHOT', 'USC']:
        chartTitles = [x for x in resultsByMethod[methodName]]
        areas = [resultsByMethod[methodName][x] for x in chartTitles]
        countsFigure.add_trace(go.Bar(x=chartTitles, y=areas, name=methodName))
    countsFigure.update_xaxes(categoryorder='array',
                              categoryarray=['Clutter', 'Occlusion', 'Alternate<br>triangulation', 'Deviating<br>normal vector', 'Deviating<br>support radius', 'Gaussian<br>noise', 'Alternate<br>mesh resolution'])
    countsFigure.update_yaxes(range=[0, 1], dtick=0.1)
    countsFigure.update_layout(margin={'t': 0, 'l': 0, 'b': 0, 'r': 0}, font=dict(size=18), yaxis_title='Normalised DDI AUC')
    pio.kaleido.scope.default_width = 1400
    pio.kaleido.scope.default_height = 300
    pio.write_image(countsFigure, outputFile, format='pdf', validate=True)

#DDI Charts for multiple filters and not 

def createDDI2DChart(rawResults, settings, output_directory, jsonFilePath, jsonFilePaths):

    #initialize histogram 
    # NOTE: x-axis -> one filter, y-axis -> proportion of DDI, z-axis -> other filter
    xBinCount = settings.binCount
    deltaX = (settings.xAxisBounds[1] - settings.xAxisBounds[0]) / xBinCount
    deltaY = settings.yTick
    yBinCount = int((settings.yAxisBounds[1] - settings.yAxisBounds[0]) / deltaY)
     
    histogramTotal = []
    histogramAccepted = []
    labels = [settings.yAxisTitle]
    for j in range(yBinCount):
        histogramTotal.append([0] * (xBinCount + 1))
        histogramAccepted.append([0] * (xBinCount + 1))
        labels.append(f'{(settings.yAxisBounds[0] + j * settings.yTick):.3f} - {(settings.yAxisBounds[0] + (j + 1) * settings.yTick):.3f}')
    
    removedCount = 0 

    for rawResult in rawResults:
        # Ignore the PRC information for this chart type
        resultX, resultY, rank, _ = rawResult
        if resultX is None:
            resultX = 0
        if resultY is None:
            resultY = 0
        if settings.reverseX:
            resultX = settings.xAxisBounds[1] - resultX
        if settings.reverseY:
            resultY = settings.yAxisBounds[1] - resultY

        if resultX < settings.xAxisBounds[0] or resultX > settings.xAxisBounds[1]:
            if settings.xAxisOutOfRangeMode == 'discard':
                removedCount += 1
                continue
            elif settings.xAxisOutOfRangeMode == 'clamp':
                resultX = max(settings.xAxisBounds[0], min(settings.xAxisBounds[1], rawResult[0]))
        if resultY < settings.yAxisBounds[0] or resultY > settings.yAxisBounds[1]:
            removedCount += 1
            continue
        
        binIndexX = min(xBinCount - 1, int((resultX - settings.xAxisBounds[0]) / deltaX))
        binIndexY = min(yBinCount - 1, int((resultY - settings.yAxisBounds[0]) / deltaY))
        if rank == 0:
            histogramAccepted[binIndexY][binIndexX] += 1
        histogramTotal[binIndexY][binIndexX] += 1

    for j in range(yBinCount):
        for i in range(xBinCount):
            if histogramTotal[j][i] < 10:
                histogramAccepted[j][i] = None
            else:
                histogramAccepted[j][i] = float(histogramAccepted[j][i]) / histogramTotal[j][i]
    
    print("Removed", removedCount, "samples")

    xValues = [((float(x + 1) * deltaX) + settings.xAxisBounds[0]) for x in range(settings.binCount)]

    stackFigure = go.Figure()
    for index, yValueStack in enumerate(histogramAccepted):
                stackFigure.add_trace(
                    go.Scatter(x=xValues, y=yValueStack, name=labels[index + 1], mode="lines")# , mode="lines" stackgroup="main"
                    )
    
    xAxisTitle = settings.xAxisTitle
    if jsonFilePath is not jsonFilePaths[-1]:
        stackFigure.update_coloraxes(showscale=False)
        stackFigure.update_traces(showlegend=False)
        pio.kaleido.scope.default_width = 300
        pio.kaleido.scope.default_height = 300
        if settings.xAxisTitleAdjustment > 0:
            xAxisTitle += ' ' * settings.xAxisTitleAdjustment
            xAxisTitle += 't'
    else:
        pio.kaleido.scope.default_width = 500
        pio.kaleido.scope.default_height = 300
    
    if settings.xAxisTitleAdjustment > 0:
        xAxisTitle += ' ' * settings.xAxisTitleAdjustment
        xAxisTitle += 't'

    stackFigure.update_layout(xaxis_title=xAxisTitle, yaxis_title="Proportion of DDI",
                              margin={'t': 0, 'l': 0, 'b': 45, 'r': 15}, font=dict(size=18), #originally 18
                              xaxis=dict(autorange=False, automargin=True, dtick=settings.xTick, range=settings.xAxisBounds),
                              yaxis=dict(autorange=False, automargin=True, dtick=0.2, range=[0,1]),
                              legend=dict(title=labels[0], x=1))
    #stackFigure.show()

    stackFigure.update_yaxes(range=[0, 1])
    stackFigure.update_xaxes(range=settings.xAxisBounds)

    outputFile = os.path.join(output_directory, settings.experimentName + "-" + settings.methodName + "-2DChart.pdf")
    pio.write_image(stackFigure, outputFile, format='pdf', validate=True)
 
def createDDI3DChart_old(results_directory, output_directory, mode):
    jsonFilePaths = [x.name for x in os.scandir(results_directory) if x.name.endswith(".json")]
    jsonFilePaths.sort()
    
    allAreas = {}

    for jsonFilePath in jsonFilePaths:
        with open(os.path.join(results_directory, jsonFilePath)) as inFile:
            print('Loading file: {}'.format(jsonFilePath))
            jsonContents = json.load(inFile)
            settings = getProcessingSettings(mode, jsonContents)
            print('Creating chart for method ' + settings.methodName + "..")
            lastSettings = settings
            dataSequence, rawResults = processSingleFile(jsonContents, settings)
            #initialize histogram 
            # NOTE: x-axis -> one filter, y-axis -> proportion of DDI, z-axis -> other filter
            xBinCount = settings.binCount
            deltaX = (settings.xAxisBounds[1] - settings.xAxisBounds[0]) / xBinCount
            deltaZ = settings.zTick
            zBinCount = int((settings.zAxisBounds[1] - settings.zAxisBounds[0]) / deltaZ)
            deltaW = settings.wTick
            wBinCount = int((settings.wAxisBounds[1] - settings.wAxisBounds[0]) / deltaW)
            
            histogramsTotal = []
            histogramsAccepted = []
            labels = [settings.zAxisTitle]
            titles = []
            for k in range(wBinCount):
                histogramTotal = []
                histogramAccepted = []
                for j in range(zBinCount):
                    histogramTotal.append([0] * (xBinCount + 1))
                    histogramAccepted.append([0] * (xBinCount + 1))
                    labels.append(f'{settings.zAxisBounds[0] + j * settings.zTick} - {settings.zAxisBounds[0] + (j + 1) * settings.zTick}')
                histogramsTotal.append(histogramTotal)
                histogramsAccepted.append(histogramAccepted)
                titles.append(f'{settings.wAxisTitle} -- {settings.wAxisBounds[0] + k * settings.wTick} - {settings.wAxisBounds[0] + (k + 1) * settings.wTick}')
            
            removedCount = 0 

            for rawResult in rawResults:
                # Ignore the PRC information for this chart type
                resultX, resultZ, resultW, rank, _ = rawResult
                if resultX is None:
                    resultX = 0
                if resultZ is None:
                    resultZ = 0
                if resultW is None:
                    resultW = 0
                if settings.reverseX:
                    resultX = settings.xAxisBounds[1] - resultX
                if settings.reverseZ:
                    resultZ = settings.zAxisBounds[1] - resultZ
                if settings.reverseW:
                    resultW = settings.wAxisBounds[1] - resultW

                if resultX < settings.xAxisBounds[0] or resultX > settings.xAxisBounds[1]:
                    if settings.xAxisOutOfRangeMode == 'discard':
                        removedCount += 1
                        continue
                    elif settings.xAxisOutOfRangeMode == 'clamp':
                        resultX = max(settings.xAxisBounds[0], min(settings.xAxisBounds[1], rawResult[0]))
                #Maybe to be removed or maybe implemented like for the xAxis one 
                if resultZ < settings.zAxisBounds[0] or resultZ > settings.zAxisBounds[1]:
                    removedCount += 1
                    continue
                #Maybe to be removed
                if resultW < settings.wAxisBounds[0] or resultW > settings.wAxisBounds[1]:
                    removedCount += 1
                    continue

                binIndexX = min(xBinCount - 1, int((resultX - settings.xAxisBounds[0]) / deltaX)) #NOTE: There must be a prittier way to do this
                binIndexZ = min(zBinCount - 1, int((resultZ - settings.zAxisBounds[0]) / deltaZ))
                binIndexW = min(wBinCount - 1, int((resultW - settings.wAxisBounds[0]) / deltaW))
                #binIndexY = int(0 if rank == 0 else (math.log10(rank)+1))
                if rank == 0:
                    histogramsAccepted[binIndexW][binIndexZ][binIndexX] += 1
                histogramsTotal[binIndexW][binIndexZ][binIndexX] += 1

            for k in range(wBinCount):
                for j in range(zBinCount):
                    for i in range(xBinCount):
                        if histogramsTotal[k][j][i] < 10:
                            histogramsAccepted[k][j][i] = 0
                        else:
                            histogramsAccepted[k][j][i] = float(histogramsAccepted[k][j][i]) / histogramsTotal[k][j][i]

            print("Removed", removedCount, "samples")

            xValues = [((float(x + 1) * deltaX) + settings.xAxisBounds[0]) for x in range(settings.binCount)]
            #print(f'{xValues}')

            # Computing the areas
            areasUnderTheCurve = []
            for _ in range(wBinCount):
                tmp = [[0, 0] for _ in range(2)]
                areasUnderTheCurve.append(tmp)

            # defining thresholds
            thresholdOcclusion = 0.3
            thresholdClutter = 1
            thresholdVertex = 0.07

            tOcclusionBin = min(xBinCount - 1, int((thresholdOcclusion - settings.xAxisBounds[0]) / deltaX))
            tClutterBin = min(wBinCount - 1, int((thresholdClutter - settings.wAxisBounds[0]) / deltaW))
            tVertexBin = min(zBinCount - 1, int((thresholdVertex - settings.zAxisBounds[0]) / deltaZ))

            #Computing the area under the curve 
            for binW in range(wBinCount):
                #print(f'Clutter index: {binW = }')
                finalAUCLow = 0
                finalAUCHigh = 0
                for binZ in range(zBinCount):
                    #print(f'\tVErtex index: {binZ = }')
                    points1 = histogramsAccepted[binW][binZ][:-2]
                    points2 = histogramsAccepted[binW][binZ][2:]
                    assert len(points1) == len(points2), f'{len(points1) = }\t{len(points2) = }'

                    areaLow = np.sum([(p2+p1)/2*deltaX for p1, p2 in zip(points1[:tOcclusionBin+1], points2[:tOcclusionBin+1])])
                    areaHigh = np.sum([(p2+p1)/2*deltaX for p1, p2 in zip(points1[tOcclusionBin:], points2[tOcclusionBin:])])
                    
                    if binZ <= tVertexBin:
                        areasUnderTheCurve[binW][0][0] += areaLow 
                        areasUnderTheCurve[binW][0][1] += areaHigh
                    else:
                        areasUnderTheCurve[binW][1][0] += areaLow
                        areasUnderTheCurve[binW][1][1] += areaHigh

                    area  =  0
                    for p1, p2 in zip(points1, points2):
                        average = (p2+p1)/2
                        area += average * deltaX
                    
                    #print(f'\t\t{area}')
                    #print(f'\t\t\t{areaLow = }\n\t\t\t{areaHigh = }')
                    #print(f'\t\t\t{finalAUCLow = }\n\t\t\t{finalAUCHigh = }')
                    #print(f'\t\t\t{areaLow+areaHigh = }')

            allAreas[settings.methodName] = areasUnderTheCurve
        '''
        print(f'{allAreas}')
        print(areasUnderTheCurve)
        print(f'{xBinCount = }')
        print(f'{wBinCount = }')
        print(f'{zBinCount = }')
        print(f'{tOcclusionBin = }')
        print(f'{tClutterBin = }')
        print(f'{tVertexBin = }')
        '''
    yAxisTitle = 'Area under DDI curve'

    barChart = go.Figure()
    methods = list(allAreas.keys())

    
    for w in range(2):
        clutter = '≤ 1' if w == 0 else '> 1'
        for z in range(2):
            vertex = '≤ 0.07' if z == 0 else '> 0.07'
            for x in range(2):
                occlusion = '≤ 0.3' if x == 0 else '> 0.3'
                name = f'Clutter {clutter};\tVertex Displacement {vertex};\tOcclusion {occlusion}'
                barChart.add_trace(go.Bar(name=name, x=methods, y=[allAreas[methodName][w][z][x] for methodName in methods]))
    
    barChart.update_coloraxes(showscale=False)
    pio.kaleido.scope.default_width = 600
    pio.kaleido.scope.default_height = 900

    barChart.update_layout(yaxis_title=yAxisTitle,
                           margin={'t':30, 'l':10, 'b':100, 'r':5},
                           font=dict(size=18),
                           legend = dict(yanchor='top',
                                         y=-0.2,
                                         xanchor='center',
                                         x=0.5,))
                                         #bgcolor="rgba(212, 203, 203, 0.5)",
                                         #title=dict(text='Clutter; Vertex Displacement; Occlusion'))


                           #legend = 

    outputFile = os.path.join(output_directory, f"exxperiment-9-barPlot.pdf")
    pio.write_image(barChart, outputFile, engine='kaleido', validate=True)
    '''
    for cIndex, chart in enumerate(histogramsAccepted):
        stackFigure = go.Figure()
        for index, yValueStack in enumerate(chart):
                    stackFigure.add_trace(
                        go.Scatter(x=xValues, y=yValueStack, name=labels[index + 1], mode="lines")#stackgroup="main",
                        )

        xAxisTitle = settings.xAxisTitle
        if chart is not histogramsAccepted[-1]:
            stackFigure.update_coloraxes(showscale=False)
            stackFigure.update_traces(showlegend=False)
            pio.kaleido.scope.default_width = 250
            pio.kaleido.scope.default_height = 300
            if settings.xAxisTitleAdjustment > 0:
                xAxisTitle += ' ' * settings.xAxisTitleAdjustment
                xAxisTitle += 't'
        else:
            pio.kaleido.scope.default_width = 500
            pio.kaleido.scope.default_height = 300
        
        if settings.xAxisTitleAdjustment > 0:
            xAxisTitle += ' ' * settings.xAxisTitleAdjustment
            xAxisTitle += 't'
        
        stackFigure.update_layout(xaxis_title=xAxisTitle, yaxis_title=settings.yAxisTitle,
                                margin={'t': 45, 'l': 0, 'b': 45, 'r': 15}, font=dict(size=18),
                                xaxis=dict(autorange=False, automargin=True, dtick=settings.xTick, range=settings.xAxisBounds),
                                yaxis=dict(autorange=False, automargin=True, dtick=0.2, range=[0,1]),
                                legend=dict(title=labels[0], x=1),
                                title={'text': titles[cIndex],
                                       'x': 0.5,
                                       'xanchor': 'center'}
                                )
        #stackFigure.show()
        
        stackFigure.update_yaxes(range=[0, 1])

        outputFile = os.path.join(output_directory, settings.experimentName + "-" + settings.methodName + "-3D" + str(cIndex) + "-Chart.pdf")
        pio.write_image(stackFigure, outputFile, format='pdf', validate=True)
        '''

def createDDI3DChart(results_directory, output_directory, mode):
    jsonFilePaths = [x.name for x in os.scandir(results_directory) if x.name.endswith(".json")]
    jsonFilePaths.sort()

    allAreas = {}

    for jsonFilePath in jsonFilePaths:
        with open(os.path.join(results_directory, jsonFilePath)) as inFile:
            print('Loading file: {}'.format(jsonFilePath))
            jsonContents = json.load(inFile)
            settings = getProcessingSettings(mode, jsonContents)
            print('Creating chart for method ' + settings.methodName + "..")
            lastSettings = settings
            dataSequence, rawResults = processSingleFile(jsonContents, settings)
            #initialize histogram
            # NOTE: x-axis -> one filter, y-axis -> proportion of DDI, z-axis -> other filter
            xBinCount = settings.binCount
            deltaX = (settings.xAxisBounds[1] - settings.xAxisBounds[0]) / xBinCount
            deltaZ = settings.zTick
            zBinCount = int((settings.zAxisBounds[1] - settings.zAxisBounds[0]) / deltaZ)
            deltaW = settings.wTick
            wBinCount = int((settings.wAxisBounds[1] - settings.wAxisBounds[0]) / deltaW)

            histogramsTotal = []
            histogramsAccepted = []
            labels = [settings.zAxisTitle]
            titles = []
            for k in range(wBinCount):
                histogramTotal = []
                histogramAccepted = []
                for j in range(zBinCount):
                    histogramTotal.append([0] * (xBinCount + 1))
                    histogramAccepted.append([0] * (xBinCount + 1))
                    labels.append(f'{settings.zAxisBounds[0] + j * settings.zTick} - {settings.zAxisBounds[0] + (j + 1) * settings.zTick}')
                histogramsTotal.append(histogramTotal)
                histogramsAccepted.append(histogramAccepted)
                titles.append(f'{settings.wAxisTitle} -- {settings.wAxisBounds[0] + k * settings.wTick} - {settings.wAxisBounds[0] + (k + 1) * settings.wTick}')

            removedCount = 0

            for rawResult in rawResults:
                # Ignore the PRC information for this chart type
                resultX, resultZ, resultW, rank, _ = rawResult
                if resultX is None:
                    resultX = 0
                if resultZ is None:
                    resultZ = 0
                if resultW is None:
                    resultW = 0
                if settings.reverseX:
                    resultX = settings.xAxisBounds[1] - resultX
                if settings.reverseZ:
                    resultZ = settings.zAxisBounds[1] - resultZ
                if settings.reverseW:
                    resultW = settings.wAxisBounds[1] - resultW

                if resultX < settings.xAxisBounds[0] or resultX > settings.xAxisBounds[1]:
                    if settings.xAxisOutOfRangeMode == 'discard':
                        removedCount += 1
                        continue
                    elif settings.xAxisOutOfRangeMode == 'clamp':
                        resultX = max(settings.xAxisBounds[0], min(settings.xAxisBounds[1], rawResult[0]))
                #Maybe to be removed or maybe implemented like for the xAxis one
                if resultZ < settings.zAxisBounds[0] or resultZ > settings.zAxisBounds[1]:
                    removedCount += 1
                    continue
                #Maybe to be removed
                if resultW < settings.wAxisBounds[0] or resultW > settings.wAxisBounds[1]:
                    removedCount += 1
                    continue

                binIndexX = min(xBinCount - 1, int((resultX - settings.xAxisBounds[0]) / deltaX)) #NOTE: There must be a prittier way to do this
                binIndexZ = min(zBinCount - 1, int((resultZ - settings.zAxisBounds[0]) / deltaZ))
                binIndexW = min(wBinCount - 1, int((resultW - settings.wAxisBounds[0]) / deltaW))
                #binIndexY = int(0 if rank == 0 else (math.log10(rank)+1))
                if rank == 0:
                    histogramsAccepted[binIndexW][binIndexZ][binIndexX] += 1
                histogramsTotal[binIndexW][binIndexZ][binIndexX] += 1

            for k in range(wBinCount):
                for j in range(zBinCount):
                    for i in range(xBinCount):
                        if histogramsTotal[k][j][i] < 10:
                            histogramsAccepted[k][j][i] = 0
                        else:
                            histogramsAccepted[k][j][i] = float(histogramsAccepted[k][j][i]) / histogramsTotal[k][j][i]

            print("Removed", removedCount, "samples")

            xValues = [((float(x + 1) * deltaX) + settings.xAxisBounds[0]) for x in range(settings.binCount)]
            #print(f'{xValues}')

            # Computing the areas
            areasUnderTheCurve = []
            for _ in range(wBinCount):
                tmp = [[0, 0] for _ in range(2)]
                areasUnderTheCurve.append(tmp)

            # defining thresholds
            thresholdOcclusion = 0.3
            thresholdClutter = 1
            thresholdVertex = 0.07

            tOcclusionBin = min(xBinCount - 1, int((thresholdOcclusion - settings.xAxisBounds[0]) / deltaX))
            tClutterBin = min(wBinCount - 1, int((thresholdClutter - settings.wAxisBounds[0]) / deltaW))
            tVertexBin = min(zBinCount - 1, int((thresholdVertex - settings.zAxisBounds[0]) / deltaZ))

            #Computing the area under the curve
            for binW in range(wBinCount):
                finalAUCLow = 0
                finalAUCHigh = 0
                for binZ in range(zBinCount):
                    points1 = histogramsAccepted[binW][binZ][:-2]
                    points2 = histogramsAccepted[binW][binZ][2:]
                    assert len(points1) == len(points2), f'{len(points1) = }\t{len(points2) = }'

                    areaLow = np.sum([(p2+p1)/2*deltaX for p1, p2 in zip(points1[:tOcclusionBin+1], points2[:tOcclusionBin+1])])
                    areaHigh = np.sum([(p2+p1)/2*deltaX for p1, p2 in zip(points1[tOcclusionBin:], points2[tOcclusionBin:])])

                    if binZ <= tVertexBin:
                        areasUnderTheCurve[binW][0][0] += areaLow
                        areasUnderTheCurve[binW][0][1] += areaHigh
                    else:
                        areasUnderTheCurve[binW][1][0] += areaLow
                        areasUnderTheCurve[binW][1][1] += areaHigh

                    area = 0
                    for p1, p2 in zip(points1, points2):
                        average = (p2+p1)/2
                        area += average * deltaX

            allAreas[settings.methodName] = areasUnderTheCurve

    yAxisTitle = 'Area under DDI curve'
    methods_raw = list(allAreas.keys())
    methods_raw.sort()

    configuration_combinations = []
    configuration_labels = []
    clutter_statuses = []
    vertex_statuses = []
    occlusion_statuses = []

    for w_idx, clutter_status in enumerate(['Low', 'High']):
        for z_idx, vertex_status in enumerate(['Low', 'High']):
            for x_idx, occlusion_status in enumerate(['Low', 'High']):
                configuration_combinations.append((w_idx, z_idx, x_idx))
                configuration_labels.append(f'C:{clutter_status} V:{vertex_status} O:{occlusion_status}')
                clutter_statuses.append(clutter_status)
                vertex_statuses.append(vertex_status)
                occlusion_statuses.append(occlusion_status)

    methods_with_auc = []
    for method_name in methods_raw:
        # Assuming you want to sort by the first AUC value for now
        auc_value = allAreas[method_name][0][0][0]
        methods_with_auc.append((auc_value, method_name))

    methods_with_auc.sort()

    methods = [method_name for auc_value, method_name in methods_with_auc]

    fig = make_subplots(
        rows=2, cols=2,
        row_titles=['', ''],
        vertical_spacing=0.02,
        specs=[
            [{"type": "pie"}, {"type": "xy"}],
            [{"type": "table", 'colspan': 2}, None]
        ],
        row_heights=[0.85, 0.15], # Initial height distribution
        column_widths=[0.01, 0.9]
    )

    fig.add_trace(go.Pie(values=[]), row=1, col=1)

    for methodName in methods:
        y_values_for_method = []
        for w, z, x in configuration_combinations:
            y_values_for_method.append(allAreas[methodName][w][z][x])

        fig.add_trace(
            go.Bar(name=methodName, x=configuration_labels, y=y_values_for_method),
            row=1, col=2
        )

    num_config_columns = len(configuration_labels)
    table_label_column_ratio = 0.11

    relative_column_widths = [table_label_column_ratio] + [(1 - table_label_column_ratio) / num_config_columns] * num_config_columns
    relative_column_widths = [w / sum(relative_column_widths) for w in relative_column_widths]

    fig.update_layout(
        barmode='group',
        yaxis_title=yAxisTitle,
        xaxis_title="",
        margin={'t': 50, 'l': 0, 'b': 0, 'r': 20}, # Reduced left margin
        font=dict(size=12),
        xaxis=dict(
            showticklabels=False,
            categoryorder='array',
            categoryarray=configuration_labels,
            range=[-0.5, len(configuration_labels) - 0.5],
            tickmode='array',
            tickvals=list(range(len(configuration_labels))),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.0,
            xanchor="center",
            x=0.5
        )
    )

    fill_colors = []
    for i, status_type in enumerate([clutter_statuses, vertex_statuses, occlusion_statuses]):
        row_colors = ['lavender']
        for status in status_type:
            if status == 'Low':
                row_colors.append('rgba(200, 255, 200, 0.7)')
            else:
                row_colors.append('rgba(255, 200, 200, 0.7)')
        fill_colors.append(row_colors)

    table_cells_values = []
    table_cells_values.append(["Clutter"] + clutter_statuses)
    table_cells_values.append(["Vertex Disp."] + vertex_statuses)
    table_cells_values.append(["Occlusion"] + occlusion_statuses)

    fig.add_trace(
        go.Table(
            cells=dict(
                values=np.array(table_cells_values).T,
                fill_color=np.array(fill_colors).T,
                align=['left'] + ['center'] * num_config_columns,
                font=dict(size=10),
                height=20
            ),
            header=dict(
                values=[""] + [""] * num_config_columns,
                fill_color='white',
                align='center',
                line_color='white',
                height=0
            ),
            columnwidth=relative_column_widths
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=550,
        width=1200,
    )

    pio.kaleido.scope.default_width = fig.layout.width
    pio.kaleido.scope.default_height = fig.layout.height

    outputFile = os.path.join(output_directory, f"experiment-9-combined_chart_and_table_final_v1.pdf")
    pio.write_image(fig, outputFile, engine='kaleido', validate=True)
    print(f"Experiment 9 chart saved to: {outputFile}")

#   NOTE: Integrate this function inside the createChart one
def executionTimeChart(results_directory, output_directory, mode):
    jsonFilePaths = [x.name for x in os.scandir(results_directory) if x.name.endswith(".json")]
    jsonFilePaths.sort()

    diagram = {}
    allMethods = {
        'point': {'xAxis': []},
        'triangle': {'xAxis': []}
    }
    barChart = {}

    for jsonFile in jsonFilePaths:
        with open(os.path.join(results_directory, jsonFile)) as inFile:
            print('Loading file: {}'.format(jsonFile))
            jsonContents = json.load(inFile)
            workType = jsonContents['workloadType']
            settings = getProcessingSettings(mode, jsonContents)
            methodName = (Path(results_directory)/jsonFile).stem.split('_')[-2]
            experimentName = 'synthetic'

            compare = (jsonContents['results']['comparison']['comparisonsPerSecond'])
            diagram[methodName] = compare
            #print(f'Method: {methodName}')
            #print(f'{compare = }')

            # collect the four experiments
            barChart[methodName] = []
            allExpHisto = []

            experiments = jsonContents['results']['generation']['syntheticScene']

            for idx, experiment in enumerate(experiments):
                results = []
                xMin, xMax = 1e6, 0

                # loading data for one experiment
                for result in experiment:
                    # respectively the x value and the y value for each chart
                    workloadPerDescriptor = result['workloadPerDescriptor']
                    if workloadPerDescriptor < xMin:
                        xMin = workloadPerDescriptor

                    if workloadPerDescriptor > xMax:
                        xMax = workloadPerDescriptor

                    results.append([workloadPerDescriptor, result['workloadItemsProcessed']/np.mean(np.array(result['executionTimes']))])

                settings.xAxisMin, settings.xAxisMax = xMin, xMax
                #print(settings.xAxisMin)
                #print(settings.xAxisMax)
                deltaX = (settings.xAxisMax - settings.xAxisMin)/settings.binCount
                allMethods[workType]['xAxis'].append((settings.xAxisMin, settings.xAxisMax, deltaX))

                # this section of the code to generate the bar chart
                values = [result[1] for result in results]
                barChart[methodName].append({
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'workType': workType
                })
                # end of section

                histogram = []
                for _ in range(settings.binCount + 1):
                    histogram.append([])

                removedCount = 0 # Initialize removedCount
                for result in results:
                    if result[0] is None:
                        result[0] = 0

                    if result[0] < settings.xAxisMin or result[0] > settings.xAxisMax:
                        if settings.xAxisOutOfRangeMode == 'discard':
                            removedCount += 1
                            continue
                        elif settings.xAxisOutOfRangeMode == 'clamp':
                            result[0] = max(settings.xAxisMin, min(settings.xAxisMax, result[0]))

                    if settings.reverse:
                        result[0] = (settings.xAxisMax + settings.xAxisMin) - result[0]

                    binIndexX = int((result[0] - settings.xAxisMin) / deltaX)
                    histogram[binIndexX].append(result[1])

                allExpHisto.append(histogram)

            allMethods[workType][methodName] = allExpHisto
            print('Done')

    for work in allMethods:
        for i in range(4):
            xValues = [allMethods[work]['xAxis'][i][0] + float(k)*allMethods[work]['xAxis'][i][2] for k in range(settings.binCount)]
            print(xValues)
            yAxisMax = 0
            chart = go.Figure()

            for MName, data in allMethods[work].items():
                if MName == 'xAxis':
                    continue

                histo = data[i]
                yValues = [np.mean(bin) for bin in histo if len(bin) > 0]

                if len(yValues) > 0 and np.max(yValues) > yAxisMax:
                    yAxisMax = np.max(yValues)

                chart.add_trace(go.Scatter(x=xValues, y=yValues, mode='lines', name=MName))

            print(f'{yAxisMax = }')
            yTick = yAxisMax//10 if yAxisMax > 0 else 1

            xAxisTitle = 'WorkloadSIze'
            chart.update_coloraxes(showscale=False)
            pio.kaleido.scope.default_width = 600
            pio.kaleido.scope.default_height = 600

            if settings.xAxisTitleAdjustment > 0:
                xAxisTitle += ' ' * settings.xAxisTitleAdjustment
                xAxisTitle += ''

            chart.update_layout(xaxis_title=xAxisTitle, yaxis_title=f"throughput ({work}/s)",
                                margin={'t': 15, 'l': 0, 'b': 45, 'r': 15}, font=dict(size=18),
                                xaxis=dict(range=[xValues[0], xValues[-1]] if xValues else [0,1]),
                                yaxis=dict(autorange=False, automargin=True, dtick=yTick, range=[0, yAxisMax]))

            outputFile = os.path.join(output_directory, experimentName + "-experiment-" + str(i+1) + '-' + work + ".pdf")
            pio.write_image(chart, outputFile, format='pdf', validate=True)


    # Generating the bar chart
    methods = [
        'COPS',
        'GEDI',
        'MICCI-PointCloud',
        'SHOT',
        'SI',
        'MICCI-Triangle',
        'QUICCI',
        'RICI'
    ]
    yAxisTitle = 'Points/Triangles processed per second'
    yAxisCutoff = 120e6

    method_categories = {
        'GEDI': 'Python method',
        'COPS': 'Python method',
        'MICCI-PointCloud': 'Point based method',
        'SHOT': 'Point based method',
        'SI': 'Point based method',
        'MICCI-Triangle': 'Triangle based method',
        'QUICCI': 'Triangle based method',
        'RICI': 'Triangle based method',
    }

    method_category_colors = {
        'Python method': 'rgba(173, 216, 230, 0.5)',  # Light blue, 40% opaque
        'Point based method': 'rgba(255, 223, 186, 0.5)',  # Light orange, 40% opaque
        'Triangle based method': 'rgba(221, 160, 221, 0.5)'  # Plum, 40% opaque
    }


    for i in range(4):
        showlegend = False if i < 2 else True

        bar_traces = [
            go.Bar(name='Minimum', x=methods, y=[barChart[methodName][i]['min'] for methodName in methods], showlegend=showlegend, legendgroup='metrics'),
            go.Bar(name='Average', x=methods, y=[barChart[methodName][i]['mean'] for methodName in methods], showlegend=showlegend, legendgroup='metrics'),
            go.Bar(name='Maximum', x=methods, y=[barChart[methodName][i]['max'] for methodName in methods], showlegend=showlegend, legendgroup='metrics'),
        ]

        dummy_legend_traces = []
        category_order = ['Python method', 'Point based method', 'Triangle based method']
        for category_name in category_order:
            if category_name in method_category_colors:
                bg_color = method_category_colors[category_name]
                dummy_legend_traces.append(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode='markers',
                        marker=dict(size=10, color=bg_color, symbol='square'),
                        name=category_name,
                        showlegend=showlegend,
                        hoverinfo='none',
                        legendgroup='background_categories'
                    )
                )
        
        bar = go.Figure(data=bar_traces + dummy_legend_traces)


        bar.update_coloraxes(showscale=False)

        pio.kaleido.scope.default_width = 700 if i > 1 else 500
        pio.kaleido.scope.default_height = 600

        bar.update_layout(
            barmode='group',
            yaxis = dict(range=[0,yAxisCutoff], title_text=yAxisTitle),
            margin={'t':100, 'l':0, 'b':45, 'r':15},
            font=dict(size=18),
            xaxis=dict(tickangle=-45, showgrid=False),
        )

        # Inserting annotations
        annotations = []
        xOffsetMap = {
            'Minimum':-0.3,
            'Maximum':0.0,
            'Average':0.3
        }
        method_to_x_index = {method: i for i, method in enumerate(methods)}
        annotation_angle = -45

        metric_bar_colors = {
            'Minimum': '#636EFA',
            'Maximum': '#EF553B',
            'Average': '#00CC96'
        }

        for trace in bar.data:
            if trace.name in ['Minimum', 'Maximum', 'Average']:
                for k, value in enumerate(trace.y):
                    if value > yAxisCutoff:
                        method_name = trace.x[k]
                        xAnnotation = method_to_x_index[method_name] + xOffsetMap[trace.name]
                        yAnnotation = yAxisCutoff * 0.99
                        annotatedValueFormatted = f'{value/1_000_000:.0f}M'

                        annotation_color = metric_bar_colors.get(trace.name, "black")

                        annotations.append(
                            go.layout.Annotation(
                                x=xAnnotation,
                                y=yAnnotation,
                                text=f'<b>{annotatedValueFormatted}</b>',
                                showarrow=True,
                                arrowhead=1,
                                arrowsize=1,
                                arrowwidth=1,
                                arrowcolor=annotation_color,
                                ax=0,
                                ay=-20,
                                font=dict(color=annotation_color, size=10, family="Arial, sans-serif"),
                                #bgcolor="rgba(255, 255, 255, 0.7)",
                                #bordercolor="gray",
                                #borderwidth=0.5,
                                #borderpad=2,
                                #opacity=0.9,
                                textangle=annotation_angle
                            )
                        )

        # Adding shapes
        shapes = []
        for k, method in enumerate(methods):
            x0_coord = k - 0.5
            x1_coord = k + 0.5
            category = method_categories.get(method, 'Unknown category')
            bg_color = method_category_colors.get(category, 'rgba(240, 240, 240, 0.4)') # Ensure 0.4 opacity here too

            shapes.append(
                go.layout.Shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=x0_coord,
                    y1=1,
                    x1=x1_coord,
                    y0=0,
                    fillcolor=bg_color,
                    layer="below",
                    line_width=0,
                )
            )

        bar.update_layout(annotations=annotations, shapes=shapes)
        outputFile = os.path.join(output_directory, f"executionTime-exp-{i}.pdf")
        pio.write_image(bar, outputFile, engine='kaleido', validate=True)


    diag = go.Figure()
    diag.add_trace(go.Bar(x=methods, y=[diagram[method] for method in methods]))
    diag.update_coloraxes(showscale=False)
    pio.kaleido.scope.default_width = 600
    pio.kaleido.scope.default_height = 600
    if settings.xAxisTitleAdjustment > 0:
        xAxisTitle += ' ' * settings.xAxisTitleAdjustment
        xAxisTitle += 't'

    diag.update_layout(yaxis_title=f"Descriptor comparisons per seconds",
                        margin={'t': 35, 'l': 0, 'b': 45, 'r': 15}, font=dict(size=18),)
    outputFile = os.path.join(output_directory, "Comparison" + ".pdf")
    pio.write_image(diag, outputFile, format='pdf', validate=True)            
                
                
                
                
                
                
                
                


def main():
    parser = argparse.ArgumentParser(description="Generates charts for the experiment results")
    parser.add_argument("--results-directory", help="Results directory specified in the configuration JSON file",
                        required=True)
    parser.add_argument("--output-dir", help="Where to write the chart images", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.results_directory):
        print(f"The specified directory '{args.results_directory}' does not exist.")
        return
    if not os.path.isdir(args.results_directory):
        print(f"The specified directory '{args.results_directory}' is not a directory. You need to specify the "
              f"directory rather than individual JSON files.")
        return

    directoriesToProcess = os.listdir(args.results_directory)
    overviewChartContents = {}
    for directoryToProcess in directoriesToProcess:
        print('Entering directory:', directoryToProcess)
        if directoryToProcess == 'charts':
            continue
        elif not os.path.isdir(os.path.join(args.results_directory, directoryToProcess)):
            continue
        elif directoryToProcess == 'execution_times':
            executionTimeChart(os.path.join(args.results_directory, directoryToProcess), args.output_dir, '')
        elif directoryToProcess == 'support_radius_estimation':
            createChart(args.results_directory, args.output_dir, 'support-radius')
        elif directoryToProcess == 'experiment9-level3-ultimate-test':
            createDDI3DChart(os.path.join(args.results_directory, directoryToProcess), args.output_dir, 'auto')
        else:
            #continue
            createChart(os.path.join(args.results_directory, directoryToProcess), args.output_dir, 'auto')

            #singleDDIChart(os.path.join(args.results_directory, directoryToProcess), args.output_dir, 'auto')
            #raise #CHECHPOINT
            #print(overallTableEntry)

    #print(overviewChartContents)
    '''overviewChartContents = {'Clutter': {'QUICCI': 0.27537532256709957, 'RICI': 0.4817495333351075, 'RoPS': 0.0009022093801893217, 'SHOT': 0.0010631403592512914, 'USC': 9.909914802135964e-06},
                             'Alternate<br>mesh resolution': {'QUICCI': 0.0962596790459941, 'RICI': 0.03974363574185445, 'RoPS': 0.03168536649079715, 'SHOT': 0.1104174845859617, 'Spin Image': 0.3428006584188961, 'USC': 0.09709257533197806},
                             'Gaussian<br>noise': {'QUICCI': 0.39707256910400024, 'RICI': 0.4074890509257194, 'RoPS': 0.45957462383613135, 'SHOT': 0.7562214479671908, 'Spin Image': 0.8620130648078511},
                             'Deviating<br>normal vector': {'QUICCI': 0.1218925743771634, 'RICI': 0.12186936063433425, 'RoPS': 0.9618210014817647, 'SHOT': 0.4891446012780454},
                             'Alternate<br>triangulation': {'QUICCI': 0.19443432036867744, 'RICI': 0.17943185429462727, 'RoPS': 0.3629179457231485, 'SHOT': 0.35798841271997395},
                             'Occlusion': {'QUICCI': 0.6869910717319178, 'RICI': 0.4852920579620921, 'RoPS': 0.06538493124967049, 'SHOT': 0.3394370957042049, 'Spin Image': 0.5394741221349988, 'USC': 0.2310877690226886},
                             'Deviating<br>support radius': {'QUICCI': 0.17763820035060865, 'RICI': 0.1788041819189112, 'RoPS': 0.2504897186655886, 'SHOT': 0.9161040264480456, 'Spin Image': 0.5040284113714978}}
'''
    #writeOverviewChart(overviewChartContents, os.path.join(args.output_dir, 'overview.pdf')) It raises a key error






if __name__ == "__main__":
    main()
