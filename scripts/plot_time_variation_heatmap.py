
import csv
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
import plotly.io as pio

pio.kaleido.scope.mathjax = None

# Hardcoded CSV filename
file_path = '../input/time_variation.csv'

# Read columns 2 and 3 using the built-in csv module
bincount = 70
max_x = 1.5
max_y = 6.5
values = []
for i in range(0, bincount):
	values.append([0] * bincount)
values_x = []
values_y = []
values_z = []
xAxisTitle = "Support radius"
yAxisTitle = "Execution time (s)"
xAxisBounds = [0, 1.5]
yAxisBounds = [0, 6.5]
xTick = 0.5
yTick = 1

deltaX = (xAxisBounds[1] - xAxisBounds[0]) / bincount
deltaY = (yAxisBounds[1] - yAxisBounds[0]) / bincount

with open(file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header if present
    for row in reader:
        if len(row) >= 3:
            xValue = float(row[1]) / max_x
            yValue = float(row[2]) / max_y
            if xValue > 1:
                continue
            if yValue > 1:
                continue
            xIndex = int(xValue * bincount)
            yIndex = int(yValue * bincount)
            print(row[1], row[2], xIndex, yIndex)
            values[yIndex][xIndex] += 1


dataRangeX = [(x + 0.5) * deltaX for x in range(0, bincount)]
dataRangeY = [(y + 0.5) * deltaY for y in range(0, bincount)]
dataRangeZ = values


stackFigure = go.Figure(go.Heatmap(x=dataRangeX, y=dataRangeY, z=dataRangeZ,
                                   zmin=0, zmax=50, colorscale=
    [
        [0, 'rgb(200, 200, 200)'],
        [0.25, 'rgb(220, 50, 47)'],
        [0.5, 'rgb(203, 75, 22)'],
        [0.75, 'rgb(181, 137, 000)'],
        [1.0, 'rgb(0, 128, 64)']
    ]))

stackFigure.update_coloraxes(showscale=False)
stackFigure.update_traces(showscale=True)
pio.kaleido.scope.default_width = 370
pio.kaleido.scope.default_height = 310



stackFigure.update_layout(xaxis_title=xAxisTitle, yaxis_title=yAxisTitle,
                          margin={'t': 0, 'l': 0, 'b': 45, 'r': 15}, font=dict(size=18),
                          xaxis=dict(autorange=False, automargin=True, dtick=xTick, range=xAxisBounds),
                          yaxis=dict(autorange=False, automargin=True, dtick=yTick, range=yAxisBounds))
pio.write_image(stackFigure, '../output/execution_time_variation.pdf', engine="kaleido", validate=True)

