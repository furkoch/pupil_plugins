
class Data_Smoothing:
    def normalize_gaze_by_quantity(self,gazes, nQuant):
        raw_x = sum([i['norm_pos'][0] for i in gazes]) / nQuant
        raw_y = sum([i['norm_pos'][1] for i in gazes]) / nQuant
        return raw_x, raw_y

    def normalize_gaze_by_length(self,gazes):
        return self.normalize_gaze_by_quantity(gazes, len(gazes))

    def smooth_norm_pos(self, raw_x, raw_y):
        # We use the most recent gaze position on the surface
        # raw_x, raw_y = gaze_on_screen[-1]['norm_pos']

        # smoothing out the gaze so the mouse has smoother movement
        self.smooth_x += 0.35 * (raw_x - self.smooth_x)
        self.smooth_y += 0.35 * (raw_y - self.smooth_y)

        x = self.smooth_x
        y = self.smooth_y

        y = 1 - y  # inverting y so it shows up correctly on screen

        print("X': {}\t,Y' : {}".format(x, y))

        x *= int(self.x_dim)
        y *= int(self.y_dim)
        # PyMouse or MacOS bugfix - can not go to extreme corners because of hot corners?

        x = min(self.x_dim, max(0, x))
        y = min(self.y_dim, max(0, y))

        return x,y
