import tkinter as tk
from PIL import Image, ImageDraw, ImageOps


class ImageGenerator:
    def __init__(self, parent, posx, posy, *kwargs):
        self.is_new_image = False
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 200
        self.sizey = 200
        self.b1 = "up"
        self.xold = None
        self.yold = None
        self.drawing_area = tk.Canvas(self.parent, width=self.sizex, height=self.sizey, highlightthickness=1, highlightbackground="black")
        self.drawing_area.place(x=self.posx, y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button = tk.Button(self.parent, text="Done!", width=10, bg='white', command=self.save)
        self.button.place(x=self.sizex / 7, y=self.sizey + 20)
        self.button1 = tk.Button(self.parent, text="Clear!", width=10, bg='white', command=self.clear)
        self.button1.place(x=(self.sizex / 7) + 80, y=self.sizey + 20)

        self.image = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

        self.prediction = tk.StringVar()
        label = tk.Label(self.parent, textvariable=self.prediction)
        label.place(x=self.posx + self.sizex + 30, y=self.posy)
        self.prediction.set("Prediction: ---")

    def save(self):
        self.is_new_image = True
        self.image = self.image.resize((28, 28), 1)
        self.image = ImageOps.grayscale(self.image)

    def clear(self):
        self.drawing_area.delete("all")
        self.image = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def b1down(self, event):
        self.b1 = "down"

    def b1up(self, event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self, event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold, self.yold, event.x, event.y, smooth='true', width=15, fill='black')
                self.draw.line(((self.xold, self.yold), (event.x, event.y)), (0, 128, 0), width=15)

        self.xold = event.x
        self.yold = event.y
