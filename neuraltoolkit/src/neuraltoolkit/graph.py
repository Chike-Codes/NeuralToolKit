import numpy as np
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, x_label="", y_label="", title=""):
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.lines = []
        self.plots = {}

        self.width = 1
        self.height = 1
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.grid(True)
        self.ax.autoscale(enable=True, axis='both', tight=None) 
        plt.ion()
        plt.show()

    def set_xlabel(self, label:str):
        self.x_label = label

    def set_ylabel(self, label:str):
        self.y_label = label

    def set_title(self, label:str):
        self.title = label

    def close(self, event=None):
        plt.close(self.fig)

    def add_plot(self, name:str, color="blue"):
        self.plots[name] = {
            "plot": self.ax.plot([], [], color),
            "x": [],
            "y": []
        }

    def append(self, name:str, x, y):
        self.plots[name]["x"].append(x)
        self.plots[name]["y"].append(y)

        self.plots[name]["plot"][0].set_data(
            self.plots[name]["x"],
            self.plots[name]["y"]
        )

    def update(self):
        #for plot in self.plots:
        #    if self.plots[plot]["x"][-1] > self.width:
        #        self.width = self.plots[plot]["x"][-1]
        #    if self.plots[plot]["y"][-1] > self.height:
        #        self.height = self.plots[plot]["y"][-1] + 0.1
        #    
        #
        #self.ax.set_xlim(0, self.width)
        #self.ax.set_ylim(0, self.height)

        self.fig.canvas.mpl_connect('close_event', self.close)
        self.ax.relim()
        #self.ax.autoscale_view()
        self.ax.autoscale(enable=True, axis='both', tight=None) 
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
