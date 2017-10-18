import sys
sys.path.append('C:\\Users\\drivesense\\pupil_capture_settings\\pupil_app')
import cv2
import numpy as np
import itertools
import math
import saliency.itti_koch_model.gabor_kernel

class Itti_Koch_Saliency_Model:

    def __init__(self, input_image, width, height):
        self.input_image = input_image
        self.width = width
        self.height = height
        self.prev_frame = None #For motion detection
        self.saliency_map = None
        #Features are stored here
        self.maps = {'intensity': [],
                     'colors': {'b': [], 'g': [], 'r': [], 'y': []},
                     'orientations': {'0': [], '45': [], '90': [], '135': []}}

    def extract_intensity_img(self, blue_ch, green_ch, red_ch):
        img = (blue_ch + green_ch + red_ch) / 3.0
        return np.asfarray(img, dtype='float64')

    def extract_color_channels(self, r_img, g_img, b_img, img_intensity):
        #Broadly tuned color-channels
        red_channel = r_img - (g_img + b_img) / 2.
        green_channel = g_img - (r_img + b_img) / 2.
        blue_channel = b_img - (r_img + g_img) / 2.
        yellow_channel = (r_img + g_img) / 2. - np.absolute(r_img - g_img) / 2. - b_img
        # red_channel = np.zeros(r_img.shape)
        # green_channel = np.zeros(r_img.shape)
        # blue_channel = np.zeros(r_img.shape)
        # yellow_channel = np.zeros(r_img.shape)
        # for y in range(0, height, 1):
        #     for x in range(0, width, 1):
        #         red_channel[y][x] = np.float64(r_img[y][x]) if(r_img[y][x]> 0.1 * amax) else 0.
        # x<=0.1*amax
        #
        # b, g, r = map(lambda x: np.float64(x) if (x > 0.1 * amax) else 0., [b, g, r])
        # nb, ng, nr = map(lambda x, y, z: max(x - (y + z) / 2., 0.), [b, g, r], [r, r, g], [g, b, b])
        # ny = max(((r + g) / 2. - math.fabs(r - g) / 2. - b), 0.)
        #
        # if i != 0.0:
        #     return map(lambda x: x / np.float64(i), [nb, ng, nr, ny])
        # else:
        #     return nb, ng, nr, ny
        amax = np.max(img_intensity)
        #img_intensity_cpy = np.copy(img_intensity)
        #img_intensity_cpy[img_intensity_cpy==0] = 1.0
        for img in [red_channel, green_channel, blue_channel, yellow_channel]:
            img[img <= 0.1 * amax] = 0
            img /= amax

        return red_channel, green_channel, blue_channel, yellow_channel

    def get_conv_gabor(self, src, theta):
        kernel = cv2.getGaborKernel((8, 8), 4, theta, 8, 1)
        return cv2.filter2D(src, cv2.CV_32F, kernel)

    def construct_gaussian_pyramid(self, inputImage):
        b, g, r = cv2.split(inputImage)
        for scale in range(1, 9):
            if scale < 2:
                continue
            b, g, r = map(cv2.pyrDown, [b, g, r])
            img_intensity = self.extract_intensity_img(r, g, b)
            red_channel, green_channel, blue_channel, yellow_channel = self.extract_color_channels(r, g, b, img_intensity)
            self.maps['intensity'].append(img_intensity)
            self.maps['colors']['r'].append(red_channel)
            self.maps['colors']['g'].append(green_channel)
            self.maps['colors']['b'].append(blue_channel)
            self.maps['colors']['y'].append(yellow_channel)

            for (orientation, index) in zip(sorted(self.maps['orientations'].keys()), range(4)):
                self.maps['orientations'][orientation].append(self.get_conv_gabor(img_intensity, np.pi * index / 4))

    def generate_feature_map(self):
        # scale index for center-surround calculation | (center, surround)
        # index of 0 ~ 6 is meaned 2 ~ 8 in thesis (Ich)
        cs_index = ((0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6))
        maps = {'intensity': [],
                'colors': {'bg': [], 'ry': []},
                'orientations': {'0': [], '45': [], '90': [], '135': []}}

        for c, s in cs_index:
            maps['intensity'].append(self.__scale_diff(self.maps['intensity'][c], self.maps['intensity'][s]))
            for key in maps['orientations'].keys():
                maps['orientations'][key].append(
                    self.__scale_diff(self.maps['orientations'][key][c], self.maps['orientations'][key][s]))
            for key in maps['colors'].keys():
                maps['colors'][key].append(self.__scale_color_diff(self.maps['colors'][key[0]][c], self.maps['colors'][key[0]][s],
                    self.maps['colors'][key[1]][c], self.maps['colors'][key[1]][s]))
        self.maps = maps

    def __scale_diff(self, c, s):
        c_size = tuple(reversed(c.shape))
        return cv2.absdiff(c, cv2.resize(s, c_size, None, 0, 0, cv2.INTER_NEAREST))
    #
    def __scale_color_diff(self, c1, s1, c2, s2):
        c_size = tuple(reversed(c1.shape))
        return cv2.absdiff(c1 - c2, cv2.resize(s2 - s1, c_size, None, 0, 0, cv2.INTER_NEAREST))

    def range_normalize(self, src_img):
        # Compute the min and max of the input img
        min_loc, max_loc, dummy1, dummy2 = cv2.minMaxLoc(src_img)
        return src_img / (max_loc - min_loc) + min_loc / (
        min_loc - max_loc) if max_loc != min_loc else src_img - min_loc

    def avg_local_max(self, src):
        # step size
        stepsize = saliency.itti_koch_model.gabor_kernel.default_step_local
        width = src.shape[1]
        height = src.shape[0]
        # find the local minima and maxima
        num_local = 0
        loc_min_mean = 0
        loc_max_mean = 0

        for y in range(0, height - stepsize, stepsize):
            for x in range(0, width - stepsize, stepsize):
                loc_img = src[y:y + stepsize, x:x + stepsize]
                loc_min, loc_max, dummy1, dummy2 = cv2.minMaxLoc(loc_img)
                loc_max_mean += loc_max
                loc_min_mean += loc_min
                num_local += 1
        return loc_max_mean / num_local

        ## normalization specific for the saliency map model

    def normalize_maps(self, src):
        dst = self.range_normalize(src)
        lmaxmean = self.avg_local_max(dst)
        normcoeff = (1 - lmaxmean) * (1 - lmaxmean)
        return dst * normcoeff

        ## normalizing feature maps

    def normalize_feature_maps(self, FM):
        nmf = list()
        for i in range(0, 6):
            normalizedImage = self.normalize_maps(FM[i])
            nownfm = cv2.resize(normalizedImage, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            nmf.append(nownfm)
        return nmf
    def generate_conspicuity_map(self):
        intensity = self.__scale_add(list(map(self.normalize_maps, self.maps['intensity'])))
        for key in self.maps['colors'].keys():
            self.maps['colors'][key] = list(map(self.normalize_maps, self.maps['colors'][key]))
        color = self.__scale_add(
            [self.maps['colors']['bg'][x] + self.maps['colors']['ry'][x] for x in range(len(self.maps['colors']['bg']))])
        orientation = np.zeros(intensity.shape)
        for key in self.maps['orientations'].keys():
            orientation += self.__scale_add(list(map(self.normalize_maps, self.maps['orientations'][key])))
        return {'intensity': intensity, 'color': color, 'orientation': orientation}

    def __scale_add(self, srcs):
        buf = np.zeros(srcs[0].shape)
        for x in srcs:
            buf += cv2.resize(x, tuple(reversed(buf.shape)), interpolation=cv2.INTER_LINEAR)
        return buf

    def generate_saliency_map(self, srcs):
        srcs = list(map(self.normalize_maps, [srcs[key] for key in srcs.keys()]))
        return srcs[0] / 3. + srcs[1] / 3. + srcs[2] / 3.

    def detect_saliency(self):
        self.construct_gaussian_pyramid(self.input_image)
        self.generate_feature_map()
        conspicuity_maps = self.generate_conspicuity_map()

        self.saliency_map = cv2.resize(self.generate_saliency_map(conspicuity_maps), tuple(reversed(self.input_image.shape[0:2])))

        cv2.imshow('Input image', cv2.flip(self.saliency_map, 1))

        # pyr_level = 1
        # for int_img in self.maps['colors']['ry']:
        #     cv2.imshow(str(pyr_level), int_img)
        #     pyr_level+=1

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # initialize
        if frame is None:
            print("Frame is None")
            break

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        saliency_model = Itti_Koch_Saliency_Model(frame,frame_width, frame_height)
        saliency_model.detect_saliency()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
