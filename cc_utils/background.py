import sys                                          # System bindings
import cv2                                          # OpenCV bindings
import numpy as np
from PIL import Image

class ColorAnalyser():
    def __init__(self, imageLoc):
        self.src = cv2.imread(imageLoc, 1)          # Reads in image source
        self.src = self.src[:,256:-256,:]
        # Empty dictionary container to hold the colour frequencies
        self.colors_count = {}

    def count_colors(self):
        # Splits image Mat into 3 color channels in individual 2D arrays
        (channel_b, channel_g, channel_r) = cv2.split(self.src)

        # Flattens the 2D single channel array so as to make it easier to iterate over it
        channel_b = channel_b.flatten()
        channel_g = channel_g.flatten()  # ""
        channel_r = channel_r.flatten()  # ""

        for i in range(len(channel_b)):
            RGB = "(" + str(channel_r[i]) + "," + \
                str(channel_g[i]) + "," + str(channel_b[i]) + ")"
            if RGB in self.colors_count:
                self.colors_count[RGB] += 1
            else:
                self.colors_count[RGB] = 1

        print("Colours counted")

    def show_colors(self):
        # Sorts dictionary by value
        for keys in sorted(self.colors_count, key=self.colors_count.__getitem__):
            # Prints 'key: value'
            print(keys, ": ", self.colors_count[keys])

        background = int(max(self.colors_count, key=self.colors_count.__getitem__).split(',')[1])
        Image.fromarray(self.src).show()
        self.src = self.src*(self.src>(background+5))
        Image.fromarray(self.src).show()

    def main(self):
        # Checks if an image was actually loaded and errors if it wasn't
        if (self.src is None):
            print("No image data. Check image location for typos")
        else:
            # Counts the amount of instances of RGB values within the image
            self.count_colors()
            # Sorts and shows the colors ordered from least to most often occurance
            self.show_colors()
            # Waits for keypress before closing
            cv2.waitKey(0)


if __name__ == "__main__":
    # Checks if image was given as cli argument
    # if (len(sys.argv) != 2):
    #     print("error: syntax is 'python main.py /example/image/location.jpg'")
    # else:
    path = 'experiments/shtech_A/1-7 87 72.36.jpg'
    Analyser = ColorAnalyser(path)
    Analyser.main()