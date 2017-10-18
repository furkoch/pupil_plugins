import sys
sys.path.append('C:\\Users\\drivesense\\pupil_capture_settings\\pupil_app')
import cv2
import numpy as np
import saliency.itti_koch_model.gabor_kernel

class Itti_Koch_Saliency:

    def __init__(self, input_image, width, height):
        self.input_image = input_image
        self.width = width
        self.height = height
        self.prev_frame = None
        self.saliency_map = None
        #initialize gabor kernels
        self.gabor_kernel_0 = np.array(saliency.itti_koch_model.gabor_kernel.GaborKernel_0)
        self.gabor_kernel_45 = np.array(saliency.itti_koch_model.gabor_kernel.GaborKernel_45)
        self.gabor_kernel_90 = np.array(saliency.itti_koch_model.gabor_kernel.GaborKernel_90)
        self.gabor_kernel_135 = np.array(saliency.itti_koch_model.gabor_kernel.GaborKernel_135)

    def extract_rgbi(self, input_image):
        # convert scale of array elements
        src = np.float32(input_image) * 1. / 255
        # split
        (B, G, R) = cv2.split(src)
        # extract an intensity image
        I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # return
        return R, G, B, I


    #Construct a Gaussian pyramid and store in a list
    def create_gaussian_pyramid(self, src_img):
        dst_image = list()
        dst_image.append(src_img)
        for i in range(1, 9):
            gaus_layer = cv2.pyrDown(dst_image[i - 1])
            dst_image.append(gaus_layer)
        return dst_image

    def center_surround_diff(self, gaus_maps):
        dst = list()
        for c in range(2, 5):
            img_size = gaus_maps[c].shape
            img_size = (img_size[1], img_size[0])  ## (width, height)
            tmp = cv2.resize(gaus_maps[c + 3], img_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(gaus_maps[c], tmp)
            dst.append(nowdst)
            tmp = cv2.resize(gaus_maps[c + 4], img_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(gaus_maps[c], tmp)
            dst.append(nowdst)
        return dst

        #constructing a Gaussian pyramid + taking center-surround differences
    def gaussian_pyramid_csd(self, src_img):
        gaussian_maps = self.create_gaussian_pyramid(src_img)
        dst = self.center_surround_diff(gaussian_maps)
        return dst

        ## intensity feature maps
    def intensity_feature_maps(self, I):
        return self.gaussian_pyramid_csd(I)

    #color feature maps
    def color_feature_maps(self, R, G, B):
        # max(R,G,B)
        tmp1 = cv2.max(R, G)
        rgb_max = cv2.max(B, tmp1)
        rgb_max[rgb_max <= 0] = 0.0001  # prevent dividing by 0
        # min(R,G)
        rgb_min = cv2.min(R, G)
        # rg_channel = (R-G)/max(R,G,B)
        rg_channel = (R - G) / rgb_max
        # BY = (B-min(R,G)/max(R,G,B)
        by_channel = (B - rgb_min) / rgb_max
        # clamp nagative values to 0
        rg_channel[rg_channel < 0] = 0
        by_channel[by_channel < 0] = 0
        # obtain feature maps in the same way as intensity
        rg_feature_map = self.gaussian_pyramid_csd(rg_channel)
        by_feature_map = self.gaussian_pyramid_csd(by_channel)

        return rg_feature_map, by_feature_map

    #orientation feature maps
    def orientation_feature_maps(self, src):
        # creating a Gaussian pyramid
        GaussianI = self.create_gaussian_pyramid(src)
        # convoluting a Gabor filter with an intensity image to extract oriemtation features
        gabor_output_0   = [ np.empty((1,1)), np.empty((1,1)) ]  # dummy data: any kinds of np.array()s are OK
        gabor_output_45  = [ np.empty((1,1)), np.empty((1,1)) ]
        gabor_output_90  = [ np.empty((1,1)), np.empty((1,1)) ]
        gabor_output_135 = [ np.empty((1,1)), np.empty((1,1)) ]
        for j in range(2,9):
            gabor_output_0.append(cv2.filter2D(GaussianI[j], cv2.CV_32F, self.gabor_kernel_0))
            gabor_output_45.append(cv2.filter2D(GaussianI[j], cv2.CV_32F, self.gabor_kernel_45))
            gabor_output_90.append(cv2.filter2D(GaussianI[j], cv2.CV_32F, self.gabor_kernel_90))
            gabor_output_135.append(cv2.filter2D(GaussianI[j], cv2.CV_32F, self.gabor_kernel_135))
        # calculating center-surround differences for every orientation
        csd_0   = self.center_surround_diff(gabor_output_0)
        csd_45  = self.center_surround_diff(gabor_output_45)
        csd_90  = self.center_surround_diff(gabor_output_90)
        csd_135 = self.center_surround_diff(gabor_output_135)
        # concatenate
        dst = list(csd_0)
        dst.extend(csd_45)
        dst.extend(csd_90)
        dst.extend(csd_135)
        # return
        return dst

        ## motion feature maps

    def get_motion_feature_maps(self, src):
        # convert scale
        I8U = np.uint8(255 * src)
        cv2.waitKey(10)
        # calculating optical flows
        if self.prev_frame is not None:
            farne_pyr_scale = saliency.itti_koch_model.gabor_kernel.farne_pyr_scale
            farne_levels = saliency.itti_koch_model.gabor_kernel.farne_levels
            farne_winsize = saliency.itti_koch_model.gabor_kernel.farne_winsize
            farne_iterations = saliency.itti_koch_model.gabor_kernel.farne_iterations
            farne_poly_n = saliency.itti_koch_model.gabor_kernel.farne_poly_n
            farne_poly_sigma = saliency.itti_koch_model.gabor_kernel.farne_poly_sigma
            farne_flags = saliency.itti_koch_model.gabor_kernel.farne_flags
            flow = cv2.calcOpticalFlowFarneback(prev=self.prev_frame,
                                                next=I8U,pyr_scale=farne_pyr_scale,
                                                levels=farne_levels,
                                                winsize=farne_winsize,
                                                iterations=farne_iterations,
                                                poly_n=farne_poly_n,
                                                poly_sigma=farne_poly_sigma,
                                                flags=farne_flags,
                                                flow=None)
            flowx = flow[..., 0]
            flowy = flow[..., 1]
        else:
            flowx = np.zeros(I8U.shape)
            flowy = np.zeros(I8U.shape)
        # create Gaussian pyramids
        dst_x = self.gaussian_pyramid_csd(flowx)
        dst_y = self.gaussian_pyramid_csd(flowy)
        # update the current frame
        self.prev_frame = np.uint8(I8U)
        return dst_x, dst_y

    # conspicuity maps
    ## standard range normalization
    def range_normalize(self, src_img):
        minn, maxx, dummy1, dummy2 = cv2.minMaxLoc(src_img)
        if maxx!=minn:
            dst = src_img / (maxx - minn) + minn / (minn - maxx)
        else:
            dst = src_img - minn
        return dst

        ## computing an average of local maxima

    def avg_local_max(self, src):
        # size
        stepsize = saliency.itti_koch_model.gabor_kernel.default_step_local
        width = src.shape[1]
        height = src.shape[0]
        # find local maxima
        numlocal = 0
        lmaxmean = 0
        for y in range(0, height - stepsize, stepsize):
            for x in range(0, width - stepsize, stepsize):
                localimg = src[y:y + stepsize, x:x + stepsize]
                lmin, lmax, dummy1, dummy2 = cv2.minMaxLoc(localimg)
                lmaxmean += lmax
                numlocal += 1
        # averaging over all the local regions
        return lmaxmean / numlocal

        ## normalization specific for the saliency map model
    def normalization(self, src):
        dst = self.range_normalize(src)
        lmaxmean = self.avg_local_max(dst)
        normcoeff = (1 - lmaxmean) * (1 - lmaxmean)
        return dst * normcoeff

    ## normalizing feature maps
    def normalize_feature_maps(self, FM):
        nmf = list()
        for i in range(0, 6):
            normalizedImage = self.normalization(FM[i])
            nownfm = cv2.resize(normalizedImage, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            nmf.append(nownfm)
        return nmf


    ## intensity conspicuity map
    def intensity_conspicuity_map(self, IFM):
        NIFM = self.normalize_feature_maps(IFM)
        ICM = sum(NIFM)
        return ICM

    ## color conspicuity map
    def color_conspicuity_map(self, CFM_RG, CFM_BY):
        # extracting a conspicuity map for every color opponent pair
        ccm_rg = self.intensity_conspicuity_map(CFM_RG)
        ccm_by = self.intensity_conspicuity_map(CFM_BY)
        # merge
        ccm = ccm_rg + ccm_by
        return ccm

    ## orientation conspicuity map
    def orientation_conspicuity_map(self, OFM):
        OCM = np.zeros((self.height, self.width))
        for i in range(0, 4):
            # slicing
            nowofm = OFM[i * 6:(i + 1) * 6]  # angle = i*45
            # extracting a conspicuity map for every angle
            NOFM = self.intensity_conspicuity_map(nowofm)
            # normalize
            NOFM2 = self.normalization(NOFM)
            # accumulate
            OCM += NOFM2
        return OCM

    ## motion conspicuity map
    def motion_conspicuity_map(self, MFM_X, MFM_Y):
        return self.color_conspicuity_map(MFM_X, MFM_Y)

    def get_saliency_map(self):
        # definitions
        src_img = self.input_image
        size = src_img.shape
        width = size[1] #self.width
        height = size[0] #self.height
        # check
        #        if(width != self.width or height != self.height):
        #            sys.exit("size mismatch")
        # extracting individual color channels
        R, G, B, I = self.extract_rgbi(src_img)
        # extracting feature maps
        IFM = self.intensity_feature_maps(I)
        CFM_RG, CFM_BY = self.color_feature_maps(R, G, B)
        OFM = self.orientation_feature_maps(I)
        mfm_x, mfm_y = self.get_motion_feature_maps(I)
        # extracting conspicuity maps
        ICM = self.intensity_conspicuity_map(IFM)
        CCM = self.color_conspicuity_map(CFM_RG, CFM_BY)
        OCM = self.orientation_conspicuity_map(OFM)
        MCM = self.motion_conspicuity_map(mfm_x, mfm_y)
        # adding all the conspicuity maps to form a saliency map
        wi = saliency.itti_koch_model.gabor_kernel.weight_intensity
        wc = saliency.itti_koch_model.gabor_kernel.weight_color
        wo = saliency.itti_koch_model.gabor_kernel.weight_orientation
        wm = saliency.itti_koch_model.gabor_kernel.weight_motion
        SMMat = wi * ICM + wc * CCM + wo * OCM + wm * MCM
        # normalize
        normalizedSM = self.range_normalize(SMMat)
        normalizedSM2 = normalizedSM.astype(np.float32)
        smoothedSM = cv2.bilateralFilter(normalizedSM2, 7, 3, 1.55)
        self.saliency_map = cv2.resize(smoothedSM, (width, height), interpolation=cv2.INTER_NEAREST)
        src_img = self.saliency_map
        # return
        return self.saliency_map

    def get_binalized_saliency_map(self):
        # get a saliency map
        if self.saliency_map is None:
            self.saliency_map = self.get_saliency_map()
        # convert scale
        SM_I8U = np.uint8(255 * self.saliency_map)
        # binarize
        thresh, binarized_saliency_map = cv2.threshold(SM_I8U, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized_saliency_map

    def get_heatmap(self):
        if self.saliency_map is None:
            self.saliency_map = self.get_saliency_map()




    def get_salient_region(self):
        src = self.input_image
        # get a binarized saliency map
        binarized_SM = self.get_binalized_saliency_map()
        # GrabCut
        img = src.copy()
        mask =  np.where((binarized_SM!=0), cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        rect = (0,0,1,1)  # dummy
        iterCount = 1
        cv2.grabCut(img, mask=mask, rect=rect, bgdModel=bgdmodel, fgdModel=fgdmodel, iterCount=iterCount, mode=cv2.GC_INIT_WITH_MASK)
        # post-processing
        mask_out = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img,img,mask=mask_out)
        return output

    #     Mat mSource_Gray, mBlobHeatMap, mHeatMap;
    #     mSource_Gray = imread(FileName_S.c_str(), 0);
    #     threshold(mSource_Gray, mSource_Gray, 254, 255, THRESH_BINARY);
    #     imshow("Source Image", mSource_Gray);
    #
    #     Mat mDist, mBlobDist;
    #     distanceTransform(mSource_Gray, mDist, CV_DIST_L2, 3);
    #     normalize(mDist, mDist, 0, 1., cv::NORM_MINMAX);
    #     mDist.convertTo(mDist, CV_8UC1, 255, 0);
    #     imshow("mDist", mDist);
    #
    #     vector < vector < Point > > contours;
    #     vector < Vec4i > hierarchy;
    #
    #     Mat mBlobMask = Mat::zeros(mSource_Gray.size(), CV_8UC1);
    #     for (size_t i = 0; i < contours.size(); i++ )
    #     {
    #         drawContours(mBlobMask, contours, (int)
    #         i, Scalar(255), -1);
    #         mDist.copyTo(mBlobDist, mBlobMask);
    #         applyColorMap(mBlobDist, mBlobHeatMap, COLORMAP_JET);
    #         GaussianBlur(mBlobHeatMap, mBlobHeatMap, Size(21, 21), 0, 0);
    #         mBlobHeatMap.copyTo(mHeatMap, mBlobMask);
    #     }

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # initialize
        if frame is None:
            print("Frame is None")
            break

        ik_saliency = Itti_Koch_Saliency(frame, frame.shape[1], frame.shape[0])
        saliency_map = ik_saliency.get_saliency_map()
        binarized_SM = ik_saliency.get_binalized_saliency_map()
        contours = []
        # dst = cv2.distanceTransform(binarized_SM, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) # cv2.CV_DIST_L2, mask_size=3)
        #
        # dst = np.float32( dst)
        # cv2.imshow('Saliency map', dst)

        # for i in range(0,100):
        #     x,y = random.randint(0, 479),random.randint(0, 639)
        #     contours.append((x,y))
        #

        #imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(binarized_SM, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        #cv2.GaussianBlur(frame, (10, 10), 0)

        cv2.imshow('Saliency map', frame)
        #cv2.imshow('Binalized saliency map', binarized_SM)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()