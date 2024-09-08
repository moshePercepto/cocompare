from tkinter import Tk


class Root(Tk):
    def __init__(self, title, width, height):
        super().__init__()
        self.title(title)
        self.geometry_centered(width, height)


    def geometry_centered(self, width, height):
        x_coordinate = int((self.winfo_screenwidth() / 2) - (width / 2))
        y_coordinate = int((self.winfo_screenheight() / 3) - (height / 3))
        self.geometry(f'{width}x{height}+{x_coordinate}+{y_coordinate}')

