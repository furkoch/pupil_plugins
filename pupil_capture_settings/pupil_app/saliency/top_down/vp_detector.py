import random
import cv2
import numpy as np
import math

class VP_Detector:

    def __init__(self, input_img):
        self.input_img = input_img
        self.width = self.input_img.shape[1]
        self.height = self.input_img.shape[0]
        self.binalized_img = np.zeros((self.height,self.width), np.uint8)
        self.grid_size_x = 32  # min(self.input_img.shape[0] // 10, self.input_img.shape[1] // 10)
        self.grid_size_y = 24
        self.vanishing_point = None
        self.radius = 10 #10

    def hough_transform(self):
        gray = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

        kernel = np.ones((15, 15), np.uint8)
        opening = gray#cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)

        edges = cv2.Canny(opening, 50, 150, apertureSize=3)  # Canny edge detection
        edges =  cv2.dilate(edges,np.ones((2, 2), np.uint8),iterations = 1)

        hough_lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)  # Hough line detection
        if hough_lines is None: # Lines are represented by rho, theta; convert to endpoint notation
           return  None
        lines = []
        for line in hough_lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                #img = cv2.line(self.input_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                lines.append(((x1, y1), (x2, y2)))
        return lines

    def detect_vp(self):

        hough_lines = self.hough_transform()
        if hough_lines is None:
            print("Hough lines are None")
            return
        size = len(hough_lines) if len(hough_lines)<100 else 100
        lines = random.sample(hough_lines, size) #TODO should include the lines on road
        lines_len = len(lines)
        intersections = []
        for i in range(lines_len):
            src_line = lines[i]
            for j in range(i+1, lines_len):
                dst_line = lines[j]
                if not src_line == dst_line:
                    intersection = self.line_intersection(src_line, dst_line)
                    if intersection:
                        intersections.append(intersection)
        if intersections:
            self.vanishing_point = self.find_vanishing_point(intersections)
            if self.vanishing_point is not None:
                cv2.circle(self.binalized_img, self.vanishing_point, 10, (255, 255, 255), -1)
                cv2.circle(self.input_img, self.vanishing_point, 10, (0,0,255), -1)
        return self.vanishing_point

    def get_binalized_saliency_map(self):
        cv2.circle(self.binalized_img, self.vanishing_point, self.radius, (255, 255, 255), -1)
        return self.binalized_img

    def det(self, a, b):
        return a[0] * b[1] - a[1] * b[0]

    def line_intersection(self, line1, line2):
        x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        div = self.det(x_diff, y_diff)
        if div == 0:
            return None  # Lines don't cross

        d = (self.det(*line1), self.det(*line2))
        x = self.det(d, x_diff) / div
        y = self.det(d, y_diff) / div
        return x, y


    # Given intersections, find the grid where most intersections occur and treat as vanishing point
    def find_vanishing_point(self, intersections):

        image_height = self.input_img.shape[0]
        image_width = self.input_img.shape[1]

        # grid_size_x = 32 #min(self.input_img.shape[0] // 10, self.input_img.shape[1] // 10)
        # grid_size_y = 24

        grid_rows = (image_width  // self.grid_size_x) + 1
        grid_columns = (image_height // self.grid_size_y) + 1

        # Current cell with most intersection points
        max_intersections = 0
        best_cell = None

        for i in range(grid_rows):
            for j in range(grid_columns):
                cell_left = i * self.grid_size_x
                cell_right = (i + 1) * self.grid_size_x
                cell_bottom = j * self.grid_size_y
                cell_top = (j + 1) * self.grid_size_y
                #cv2.rectangle(self.input_img, (cell_left, cell_bottom), (cell_right, cell_top), (0, 0, 255), 10)

                current_intersections = 0  # Number of intersections in the current cell
                for x, y in intersections:
                    if cell_left < x < cell_right and cell_bottom < y < cell_top:
                        current_intersections += 1

                # Current cell has more intersections that previous cell (better)
                if current_intersections > max_intersections:
                    max_intersections = current_intersections
                    xb = ((cell_left + cell_right) // 2)
                    yb = ((cell_bottom + cell_top) // 2)
                    best_cell = (xb,yb)
        return best_cell

    def detect_vp_old(self):
        hough_lines = self.hough_transform()
        if hough_lines is None:
            print("Hough lines are None")
            return
        size = len(hough_lines) if len(hough_lines)<100 else 100
        lines = random.sample(hough_lines, size) #TODO should include the lines on road
        lines_len = len(lines)
        intersections = []
        max_intersections = 0
        current_intersections = 0
        intersect_pts = {}

        for i in range(lines_len):
            src_line = lines[i]
            for j in range(i+1, lines_len):
                dst_line = lines[j]
                if not src_line == dst_line:
                    pt = self.line_intersection(src_line, dst_line)
                    if pt:
                        x, y = pt
                        x,y = int(x),int(y)
                        intersect_pts[str(x)+","+str(y)] = 0

        for i in range(lines_len):
            src_line = lines[i]
            for j in range(i+1, lines_len):
                dst_line = lines[j]
                if not src_line == dst_line:
                    pt = self.line_intersection(src_line, dst_line)
                    if pt:
                        x, y = pt
                        x, y = int(x), int(y)
                        intersect_pts[str(x)+","+str(y)] += 1

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # initialize
        if frame is None:
            print("Frame is None")
            break

        vp_detector = VP_Detector(frame)
        vp_img = vp_detector.detect_vp()

        if vp_img is not None:
            cv2.imshow('VP', vp_img)
        #binalized_image = vp_detector.get_binalized_saliency_map()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()






