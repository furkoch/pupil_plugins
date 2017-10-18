import cv2
import numpy as np
class SR_Saliency:

    def __init__(self):
        pass

    def compute_saliency(self, img, img_width = 128):
        img = cv2.resize(img, (img_width, img_width * img.shape[0] / img.shape[1]))

        c = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        mag = np.sqrt(c[:, :, 0] ** 2 + c[:, :, 1] ** 2)
        spectralResidual = np.exp(np.log(mag) - cv2.boxFilter(np.log(mag), -1, (3, 3)))

        c[:, :, 0] = c[:, :, 0] * spectralResidual / mag
        c[:, :, 1] = c[:, :, 1] * spectralResidual / mag
        c = cv2.dft(c, flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))
        mag = c[:, :, 0] ** 2 + c[:, :, 1] ** 2

        cv2.normalize(cv2.GaussianBlur(mag, (9, 9), 3, 3), mag, 0., 1., cv2.NORM_MINMAX)
        return mag


if __name__ == "__main__":
    sr_saliency = SR_Saliency()
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        saliency_map = sr_saliency.compute_saliency(frame)

        # Display the resulting frame
        cv2.imshow('frame', saliency_map)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()