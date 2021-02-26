import numpy as np

class ColorofPoint:

    def __ColorofPointCorrespondence__(self, img, p):

        ColorofMatch = []

        for i in range(len(p)):
            x_point = p[i, 0]
            y_point = p[i, 1]
            try:
                color = img[x_point, y_point]
            except:
                color = (0,0,0)
            ColorofMatch.append(color)
        return np.array(ColorofMatch)