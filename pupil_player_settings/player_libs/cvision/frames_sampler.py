import cv2
import sys
sys.path.append('C:\\Projects\\repo\\sensorics\\EyeTracker\\pupil\\pupil_src\\shared_modules')
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings\\player_libs')
from cvision.img_compare import *
import numpy as np
from numpy import array
import json
import time
from methods import denormalize,normalize

class Frames_Sampler:
    def __init__(self, rec_dir):
        self.rec_dir = rec_dir
        self.similarity_treshold = 0.8
        #Bin fixations by frames on load
        fixations_by_frame = [[] for x in self.get_timestamps()]
        self.fixations = self.get_fixations()
        for f in self.fixations:
            for idx in range(f['start_frame_index'], f['end_frame_index'] + 1):
                if f['aoi'] != 'Unlabeled':
                    fixations_by_frame[idx].append(f)
        self.fixations_by_frame = fixations_by_frame
        print(len(self.fixations_by_frame))

        # frame info for optic-flow
        self.prev_frame_idx = -1
        self.past_fixation_positions = []
        self.prev_gray = None


    def find_frame_rate(self):
        # Start default camera
        video = cv2.VideoCapture(self.rec_dir + '\\world.mp4');

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        # With webcam get(CV_CAP_PROP_FPS) does not work.
        # Let's see for ourselves.

        if int(major_ver) < 3:
            fps = video.get(cv2.CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else:
            fps = video.get(cv2.CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        # # Number of frames to capture
        # num_frames = 120;
        #
        # print ("Capturing {0} frames".format(num_frames))
        #
        # # Start time
        # start = time.time()
        #
        # # Grab a few frames
        # for i in range(0, num_frames):
        #     ret, frame = video.read()
        #
        # # End time
        # end = time.time()
        #
        # # Time elapsed
        # seconds = end - start
        # print ("Time taken : {0} seconds".format(seconds))
        #
        # # Calculate frames per second
        # fps = num_frames / seconds;
        # print ("Estimated frames per second : {0}".format(fps))

        # Release video
        video.release()
        return fps

    def get_timestamps(self):
        #get timestamps from recorded g_pool
        timestamps = []
        with open(self.rec_dir + "\\timestamps.csv") as csv_file:
            for timestamp in csv_file:
                timestamps.append(json.loads(timestamp))
        return  timestamps

    def get_fixations(self):
        fixations = []
        with open(self.rec_dir+"\\fixations.csv") as csv_file:
            for fixation in csv_file:
                fixations.append(json.loads(fixation))
        return fixations

    def get_aoi(self,aoi):
        if aoi is None:
            return None
        arr = aoi.split("/")
        if arr is not None or len(arr)>0:
            return arr[-1]

    def play_frames_by_optic_flow(self, timeframe=4.0):
        cap = cv2.VideoCapture(self.rec_dir + '\\world.mp4')
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while not cap.isOpened():
            cap = cv2.VideoCapture(self.rec_dir + '\\world.mp4')
            cv2.waitKey(1000)
        frame_idx = 0
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)


        # criteria and window size params for calcOpticalFlowPyrLK
        lk_params = dict(winSize=(90, 90),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        frame_fixations = []
        csv_file = open(self.rec_dir + "\\fixation_sequences.csv", 'w')
        #we calculate optic flow for every frame in order to find the fixation position in current frame
        init_fixation_idx = 0
        fixation_idx = 0
        while True:
            flag, frame = cap.read()
            if flag:
                frame_fixations = self.fixations_by_frame[frame_idx]
                updated_past_fixations = []
                #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = frame
                img_shape = img.shape[:-1][::-1]

                succeeding_frame = frame_idx - self.prev_frame_idx == 1
                same_frame = frame_idx == self.prev_frame_idx
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if self.past_fixation_positions and succeeding_frame:
                    past_screen_fixation = np.array(
                        [denormalize(ng['norm_pos'], img_shape, flip_y=True) for ng in self.past_fixation_positions], dtype=np.float32)

                    new_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_img, past_screen_fixation, None, minEigThreshold=0.005, **lk_params)


                    for fixation, new_gaze_pt, s, e in zip(self.past_fixation_positions, new_pts, status, err):
                        if s:
                            # print "norm,updated",gaze['norm_gaze'], normalize(new_gaze_pt,img_shape[:-1],flip_y=True)
                            fixation['norm_pos'] = normalize(new_gaze_pt, img_shape, flip_y=True)
                            updated_past_fixations.append(fixation)
                        else:
                            # logger.debug("dropping gaze")
                            # Since we will replace self.past_gaze_positions later,
                            # not appedning tu updated_past_gaze is like deliting this data point.
                            pass
                else:
                    # we must be seeking, do not try to do optical flow, or pausing: see below.
                    pass

                if same_frame:
                    # re-use last result
                    frame_fixations[:] = self.past_fixation_positions[:]
                else:
                    if frame_fixations:
                        now = frame_fixations[0]['timestamp']
                        cutoff = now - timeframe
                        updated_past_fixations = [g for g in updated_past_fixations if g['timestamp'] > cutoff]
                        # inject the scan path gaze points into recent_gaze_positions
                    frame_fixations[:] = updated_past_fixations + frame_fixations
                    frame_fixations.sort(key=lambda x: x['timestamp'])
                #write the fixation sequence

                for fixation in frame_fixations:
                    if fixation['start_frame_index'] == self.fixations[fixation_idx]['start_frame_index']:
                        print(len(frame_fixations))



                # if first_fixation_ts + timeframe <
                #
                #
                # fixation_sequence = json.dumps({'frame_num':frame_idx}) #, 'fixations': frame_fixations
                # csv_file.write(fixation_sequence+"\n")
                # csv_file.flush()


                # update info for next frame.
                self.prev_gray = gray_img
                self.prev_frame_idx = frame_idx
                self.past_fixation_positions = frame_fixations
                frame_idx+=1
                #print(str(frame_idx)+" / "+ str(frame_count))
            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                # print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(500)

            if cv2.waitKey(10) == 27:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break
        cv2.destroyAllWindows()
        print("Finished reading and writing frames")
        csv_file.close()

    def read_frame(self, frame_num):
        cap = cv2.VideoCapture(self.rec_dir + '\\world.mp4')
        while not cap.isOpened():
            cap = cv2.VideoCapture(self.rec_dir + '\\world.mp4')
            cv2.waitKey(1000)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        flag, frame = cap.read()
        print(frame)
        return {'flag':flag,'frame':frame,'gray':cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)}

    def read_fixation_sequence(self):
        cap = cv2.VideoCapture(self.rec_dir + '\\world.mp4')
        fixation_sequences = []
        while not cap.isOpened():
            cap = cv2.VideoCapture(self.rec_dir + '\\world.mp4')
            cv2.waitKey(1000)
        with open(self.rec_dir + "\\fixation_sequences.csv") as csv_file:
            for fixation_seq in csv_file:
                fixation_seq = json.loads(fixation_seq)
                frame_num = fixation_seq['frame_num']
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                fixation_seq['flag'], fixation_seq['frame'] = cap.read()
                fixation_sequences.append(fixation_seq)
        return fixation_sequences


    def sequence_frames_by_optic_flow(self, timeframe = 4.0):
        cap = cv2.VideoCapture(self.rec_dir + '\\world.mp4')
        while not cap.isOpened():
            cap = cv2.VideoCapture(self.rec_dir + '\\world.mp4')
            cv2.waitKey(1000)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # criteria and window size params for calcOpticalFlowPyrLK
        lk_params = dict(winSize=(90, 90),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

        #We divide the video into frame-sequences of length 4 sec
        segment_len = timeframe * fps
        segment_idx = 0
        in_between_fixations = {}
        with open(self.rec_dir+"\\fixation_sequences.csv", 'w') as csvfile:
            while segment_idx < frame_count:
                fixation_lines = []
                in_between_fixations[segment_idx] = []
                #print(str(segment_idx)+" - "+str(segment_idx+segment_len)+" : "+str(frame_count))
                fixation_idx = 0

                frame_num = min(segment_idx + segment_len,frame_count)-1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

                flag, frame = cap.read()
                gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                is_inbetween_fixation = segment_idx>segment_len and len(in_between_fixations[segment_idx-segment_len])>0
                merged_fixations = in_between_fixations[segment_idx-segment_len] if segment_idx>=segment_len else []

                while fixation_idx<len(self.fixations) and (self.fixations[fixation_idx]['end_frame_index'] <= frame_num or
                                                                (self.fixations[fixation_idx]['start_frame_index'] <= frame_num and self.fixations[fixation_idx]['end_frame_index'] >= frame_num)):

                    fixation = self.fixations[fixation_idx]
                    if (segment_idx <= fixation['start_frame_index'] and fixation['end_frame_index'] > 0):
                        merged_fixations.append(fixation)
                    fixation_idx+=1

                fixation_idx = 0 #important

                while fixation_idx < len(merged_fixations):

                    fixation = merged_fixations[fixation_idx]

                    print("{},segment:{}, start:{}, end:{}".format(self.get_aoi(fixation['aoi']), segment_idx,
                                                                   fixation['start_frame_index'],
                                                                   fixation['end_frame_index']))

                    projected_frame = fixation['end_frame_index']

                    if fixation['start_frame_index'] <= frame_num and fixation['end_frame_index'] >= frame_num:
                        in_between_fixations[segment_idx].append(fixation)
                        projected_frame = fixation['start_frame_index']

                    cap.set(cv2.CAP_PROP_POS_FRAMES, projected_frame)

                    fixation_flag, fixation_frame = cap.read()
                    img_shape = fixation_frame.shape[:-1][::-1]
                    fixation_gray_img = cv2.cvtColor(fixation_frame, cv2.COLOR_BGR2GRAY)

                    past_screen_fixation = np.array([denormalize(ng['norm_pos'], img_shape, flip_y=True) for ng in
                                                     [fixation]], dtype=np.float32)

                    # print("Before: {}".format(fixation['norm_pos']))

                    new_pts, status, err = cv2.calcOpticalFlowPyrLK(fixation_gray_img, gray_img, past_screen_fixation,
                                                                    None, minEigThreshold=0.005, **lk_params)

                    fixation['norm_pos'] = list(normalize(list(new_pts[0]), img_shape, flip_y=True))
                    # print("After: {}".format(fixation['norm_pos']))
                    fixation_lines.append(self.fixations[fixation_idx])
                    fixation_idx+=1

                json_str = json.dumps({'frame_num':frame_num, 'fixations':fixation_lines})
                #cur_iter = fixation_idx<len(self.fixations)-1 and self.fixations[fixation_idx]['end_frame_index'] < frame_num
                csvfile.write(json_str+"\n")

                # for fixation_line in fixation_lines:
                #     print("{},segment:{}, start:{}, end:{}".format(self.get_aoi(fixation_line['aoi']), segment_idx,
                #                                                    fixation_line['start_frame_index'],
                #                                                    fixation_line['end_frame_index']))

                #print('\n')

                segment_idx += segment_len
        print("Fixation sequences created and stored in {}".format(self.rec_dir+"\\fixation_sequences.csv"))



    def sample_frames_by_similarity(self):
        cap = cv2.VideoCapture(self.rec_dir + '\\world.mp4')
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(frame_num)

        while not cap.isOpened():
            cap = cv2.VideoCapture(self.rec_dir + '\\world.mp4')
            cv2.waitKey(1000)
        frame_num = 1 #should be zero if we want to store frames in RAM
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #container for sampled frames
        #[{frame:[{frame_number : next_frame}]}]
        #sampled_frames = []

        csv_file = open(self.rec_dir+"\\frame_samplings.csv", 'w')
        print("Started reading and writing frames to {}\\frame_samplings.csv".format(self.rec_dir))

        while True:
            flag, frame = cap.read()
            if flag:
                init_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #frame_sample = {frame_num:[]}
                #frame_num_init = frame_num
                csv_file.write(str(frame_num)+"\n")
                csv_file.flush()
                frame_num += 1
                while True:
                    new_flag, new_frame = cap.read()
                    if new_flag:
                        next_frame =  cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                        str_sim = struct_sim(init_frame, next_frame)
                        if str_sim < self.similarity_treshold:
                            break
                        #next_frames = {frame_num:new_frame}
                        #frame_sample[frame_num_init].append(next_frames)
                        frame_num += 1
                    else:
                        # The next frame is not ready, so we try to read it again
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                        # print("frame is not ready")
                        # It is better to wait for a while for the next frame to be ready
                        cv2.waitKey(500)

                    if cv2.waitKey(10) == 27:
                        break
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        # If the number of captured frames is equal to the total number of frames,
                        # we stop
                        break
                #sampled_frames.append(frame_sample)
            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                # print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(500)

            if cv2.waitKey(10) == 27:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break
        cv2.destroyAllWindows()
        print("Finished reading and writing frames")
        csv_file.close()

rec_dirs = ["003", "004","v2_001","v2_002"] #,

def read_write_frame_similarity(rec_dir):
    frame_sampler = Frames_Sampler(rec_dir)
    #frame_sampler.sample_frames_by_optic_flow()
    frame_sampler.sequence_frames_by_optic_flow()
    # fixation_sequences = frame_sampler.read_fixation_sequence()
    # for seq in fixation_sequences:
    #     print("{}\n",seq['fixations'])
if __name__ == "__main__":
    file_dir = "C:\\Users\\drivesense\\recordings\\2017_07_18\\"
    read_write_frame_similarity(file_dir + rec_dirs[0])