from pathlib import Path
import os
import json

import plotly.express as px
import plotly.graph_objs as go


def JSON_Create(diction: dict, FileDirectory: str, FileName: str) -> None:
    """
    Function for creating JSON log-file with dictionary.

    :param diction: Dictionary for writing
    :param FileDirectory: Path to logging file
    :param FileName: Name of logging file. Should be ended with ".txt"
    """
    filename = Path(FileDirectory) / FileName  # Full file-path with file-name
    os.makedirs(FileDirectory, exist_ok=True)  # Creating / checking existing of file-path
    with open(filename, 'w') as f:
        json.dump(diction, f, indent=4)  # Writing file


def JSON_Read(FileDirectory: str, FileName: str) -> dict:
    """
    Function for loading dictionary from log-file.

    :param FileDirectory: Path to logging file
    :param FileName: Name of logging file
    """
    filename = Path(FileDirectory) / FileName  # Full file-path with file-name
    with open(filename) as f:
        return json.load(f)  # Loading dictionary


def plotly_multi_scatter(mult_x_y,
                         names = None,
                         main_title="",
                         x_title="",
                         y_title=""):
    """Draws plotly scatter of (x,y).
    mult_x_y - [(x1, y1), (x2, y2), ...] to plot.
    """

    fig = go.Figure()
    fig.update_layout(title=main_title,
                      xaxis_title=x_title,
                      yaxis_title=y_title)
    
    if names==None:
        names = list(range(1, len(mult_x_y)+1))
    
    # Iterating through (x, y) pairs
    for i, (x, y) in enumerate(mult_x_y):
        #fig = px.scatter(x=x, y=y)
        print(x[:5], y[:5])
        fig.add_trace(go.Scatter(x = x, y = y,
                                 name = names[i]))

    fig.show()