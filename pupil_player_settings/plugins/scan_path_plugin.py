import cv2
import sys
from player_methods import transparent_circle
sys.path.append('F:\\Projects\\repo\\sensorics\\EyeTracker\\pupil\\pupil_src\\shared_modules')
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings')
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings\\player_libs')
import threading
from plugin import Plugin
import numpy as np
from pyglui import ui
from methods import denormalize,normalize
from  player_libs.csv_rw import CSV_RW
from  scanpath.aoi_provider import AOI_Provider
from itertools import chain
from math import atan, tan
from operator import itemgetter
import csv
import json
# import player_libs.cvision.img_compare
import logging
logger = logging.getLogger(__name__)


class Scan_Path_Detector(Plugin):
    uniqueness = "not_unique"

    def __init__(self, g_pool,timeframe=.5,max_dispersion = 1.0,min_duration = 0.15,h_fov=78, v_fov=50,show_fixations = False,seq_frame_num=5):
        super().__init__(g_pool)

        #let the plugin work after most other plugins.
        self.order = .6
        self.menu = None

        #user settings
        self.timeframe = timeframe
        self.rec_dir = self.g_pool.rec_dir
        self.min_duration = min_duration
        self.max_dispersion = max_dispersion
        self.h_fov = h_fov
        self.v_fov = v_fov
        self.show_fixations = show_fixations

        self.dispersion_slider_min = 0.
        self.dispersion_slider_max = 3.
        self.dispersion_slider_stp = 0.05

        self.pix_per_degree = float(self.g_pool.capture.frame_size[0]) / h_fov
        self.img_size = self.g_pool.capture.frame_size
        self.fixations_to_display = []
        logger.info("Classifying fixations.")
        self.notify_all({'subject': 'fixations_should_recalculate', 'delay': .5})

        #algorithm working data
        self.prev_frame_idx = -1
        self.past_gaze_positions = []
        self.past_fixation_positions = []
        self.prev_gray = None
        self.seq_frame_num = seq_frame_num

        # call encoder
        self.aoi_provider = AOI_Provider(self.rec_dir)


    def recent_events(self, events):
        frame = events.get('frame')
        events['fixations'] = self.g_pool.fixations_by_frame[frame.index]

        img = frame.img
        img_shape = img.shape[:-1][::-1] # width,height

        succeeding_frame = frame.index-self.prev_frame_idx == 1
        same_frame = frame.index == self.prev_frame_idx
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #vars for calcOpticalFlowPyrLK
        lk_params = dict( winSize  = (90, 90),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

        updated_past_gaze = []
        updated_past_fixations = []

        if self.past_fixation_positions and succeeding_frame:
            past_screen_fixation = np.array(
                [denormalize(ng['norm_pos'], img_shape, flip_y=True) for ng in self.past_fixation_positions],
                dtype=np.float32)
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_img, past_screen_fixation, None, minEigThreshold=0.005, **lk_params)

            for fixation, new_gaze_pt, s, e in zip(self.past_fixation_positions, new_pts, status, err):
                if s:
                    # print "norm,updated",gaze['norm_gaze'], normalize(new_gaze_pt,img_shape[:-1],flip_y=True)
                    fixation['norm_pos'] = normalize(new_gaze_pt, img_shape, flip_y=True)
                    updated_past_fixations.append(fixation)
                    # logger.debug("updated gaze")
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
            events['fixations'][:] = self.past_fixation_positions[:]
        else:
            if events.get('fixations', []):
                now = events.get('fixations', [])[0]['timestamp']

                cutoff = now - self.timeframe
                updated_past_fixations = [g for g in updated_past_fixations if g['timestamp'] > cutoff]


                # inject the scan path gaze points into recent_gaze_positions
            events.get('fixations', [])[:] = updated_past_fixations + events.get('fixations',[])
            events.get('fixations', []).sort(key=lambda x: x['timestamp'])


        fixation_pts = []

        for pt in events.get('fixations', []):
            denorm_pos = denormalize(pt['norm_pos'], frame.img.shape[:-1][::-1], flip_y=True)
            fixation_pts.append(denorm_pos)
            transparent_circle(frame.img, denorm_pos, 20, color=(0.2, 0.0, 0.7, 0.5), thickness=2)

            try:
                aoi_label = pt['aoi_label'] #self.aoi_encoder.hex_hash(pt['aoi']) #self.aoi_encoder.encode_aoi(pt['aoi'])
                cv2.putText(img, aoi_label, tuple(map(int, denorm_pos)), cv2.FONT_HERSHEY_SIMPLEX, .7, (185, 224, 236),
                            1, lineType=cv2.LINE_AA)
            except:
                #logger.info(str(pt['timestamp']))
                pass
                #pt['aoi'] = None


            # aoi = self.aoi_encoder.encode_aoi(pt['aoi'])
            # if aoi is not None:
            #     cv2.putText(img, aoi, tuple(map(int, denorm_pos)), cv2.FONT_HERSHEY_SIMPLEX, .7, (185, 224, 236), 1, lineType=cv2.LINE_AA)

        if fixation_pts:
            pts = np.array([fixation_pts], dtype=np.int32)
            cv2.polylines(frame.img, pts, isClosed=False, color= (0.25*255,0.7*255,0,0.2*255), thickness=2, lineType=cv2.LINE_AA)
        else:
            pass
            #Visualize the gaze-points

        #update info for next frame.
        self.prev_gray = gray_img
        self.prev_frame_idx = frame.index
        self.past_gaze_positions = events['gaze_positions']
        self.past_fixation_positions = events.get('fixations', [])

    def on_scanpath_build(self):
        thread = threading.Thread(target=self.build_scanpath_by_all_fixations, args=())
        thread.daemon = True  # Daemonize thread
        thread.start()



    def build_scanpath_by_all_fixations(self):
        i=0
        with open(self.rec_dir + "\\fixations.csv", 'w') as csvfile:
            fixation_line = ""
            len_fix = len(self.fixations)
            logger.info("There are " + str(len_fix) + " fixations")
            for fixation in self.fixations:
                fixation = json.dumps(fixation)
                new_line = "\n" if i < len_fix -1 else ""
                csvfile.write(fixation+new_line)
                i+=1
        j=0
        with open(self.rec_dir + "\\timestamps.csv", 'w') as csvfile_ts:
            timestamp_line = ""
            len_ts = len(self.g_pool.timestamps)
            logger.info("There are "+str(len_ts)+" timestamps")
            for timestamp in self.g_pool.timestamps:
                timestamp = json.dumps(timestamp)
                new_line = "\n" if j < len_ts - 1 else ""
                csvfile_ts.write(timestamp + new_line)
                j+=1
        logger.info("All fixations and timestamps are written and stored in fixations.csv and timestamps.csv file")


    def build_scanpath_by_frame_similarity(self):
        i = 0
        with open(self.aoi_provider.file_dir + "\\fixation_sequences.csv", 'w') as csvfile:
            fixation_line = ""
            len_fix = len(self.fixations)
            for fix_index in  range(0,len_fix):
                current_fixation = self.fixations[fix_index]
                current_end_frame_index = current_fixation['end_frame_index']
                current_img = self.g_pool.fixations_by_frame[current_end_frame_index].img
                for next_fix_index in range(fix_index, len_fix-1):
                    next_fixation = self.fixations[next_fix_index]
                    next_end_frame_index = next_fixation['end_frame_index']
                    next_img = self.g_pool.fixations_by_frame[next_end_frame_index].img
                    #Compute frame-similarity
                    str_sim = struct_sim(current_img, next_img)
                    if str_sim < 0.8:
                        break
                    print(str(next_end_frame_index))
        logger.info("All scanpaths are written and stored in fixation_sequences.csv file")

    def build_scanpath_by_fixed_sampling(self):
        fix_len = len(self.g_pool.fixations_by_frame)
        with open(self.aoi_provider.file_dir+"\\fixation_sequences.csv", 'w') as csvfile:
            for frame_indx in range(0,fix_len,self.seq_frame_num):
                j=frame_indx
                fixation_line = ""
                while (j<self.seq_frame_num + frame_indx) and (self.seq_frame_num+frame_indx<fix_len):
                    for fixation_by_frame in self.g_pool.fixations_by_frame[j]:
                        last_line = (j==self.seq_frame_num + frame_indx-1) or (self.seq_frame_num+frame_indx==fix_len-1)
                        fixation_line+=fixation_by_frame['aoi'] + ("\n" if last_line else ",")
                    j+=1
                if j==frame_indx:
                    for k in range(frame_indx,fix_len):
                        for fixation_by_frame in self.g_pool.fixations_by_frame[j]:
                            fixation_line += fixation_by_frame['aoi'] + ("\n" if (k==fix_len-1) else ",")
                fixation_line = fixation_line.strip('\n')
                if fixation_line.strip(' \t\r')!="":
                    csvfile.write(fixation_line)
            logger.info("All scanpathes are written and stored in fixation_sequences.csv file")

    def on_sample_frames_by_similarity(self):
        thread = threading.Thread(target=self.sample_frames_by_similarity, args=())
        thread.daemon = True
        thread.start()

    def sample_frames_by_similarity(self):
        cap = cv2.VideoCapture(self.rec_dir+'\\world.mp4')
        while (cap.isOpened()):
            print('reading')
            ret, frame = cap.read()
            print(ret)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def _classify(self):
        logger.info("Reclassifying fixations.")
        gaze_data = list(chain(*self.g_pool.gaze_positions_by_frame))

        # lets assume we need data for at least 30% of the duration
        sample_threshold = self.min_duration * self.g_pool.capture.frame_rate * .33
        dispersion_threshold = self.max_dispersion
        duration_threshold = self.min_duration
        self.notify_all({'subject': 'fixations_changed'})

        def dist_deg(p1, p2):
            return np.sqrt(((p1[0] - p2[0]) * self.h_fov) ** 2 + ((p1[1] - p2[1]) * self.v_fov) ** 2)

        fixations = []
        try:
            gaze_el = gaze_data.pop(0)
            fixation_support = [gaze_el]
            fixation_support[-1]['aoi'] = gaze_el['aoi']
            fixation_support[-1]['aoi_label'] = self.aoi_provider.get_aoi(gaze_el['aoi'])
        except IndexError:
            logger.warning("This recording has no gaze data. Aborting")
            return
        while True:
            fixation_centroid = sum([p['norm_pos'][0] for p in fixation_support]) / len(fixation_support), sum(
                [p['norm_pos'][1] for p in fixation_support]) / len(fixation_support)
            dispersion = max([dist_deg(fixation_centroid, p['norm_pos']) for p in fixation_support])

            if dispersion < dispersion_threshold and gaze_data:
                # so far all samples inside the threshold, lets add a new candidate
                gaze_poped = gaze_data.pop(0)
                fixation_support += [gaze_poped]
                # fixation_support[-1]['aoi'] = gaze_poped['aoi']
                # fixation_support[-1]['aoi_label'] = self.aoi_provider.get_aoi(gaze_poped['aoi'])
            else:
                if gaze_data:
                    # last added point will break dispersion threshold for current candidate fixation. So we conclude sampling for this fixation.
                    last_sample = fixation_support.pop(-1)
                if fixation_support:
                    duration = fixation_support[-1]['timestamp'] - fixation_support[0]['timestamp']
                    if duration > duration_threshold and len(fixation_support) > sample_threshold:
                        # long enough for fixation: we classifiy this fixation candidate as fixation
                        # calculate character of fixation
                        fixation_centroid = sum([p['norm_pos'][0] for p in fixation_support]) / len(
                            fixation_support), sum([p['norm_pos'][1] for p in fixation_support]) / len(fixation_support)
                        dispersion = max([dist_deg(fixation_centroid, p['norm_pos']) for p in fixation_support])
                        confidence = sum(g['confidence'] for g in fixation_support) / len(fixation_support)

                        # avg pupil size  = mean of (mean of pupil size per gaze ) for all gaze points of support
                        avg_pupil_size = sum(
                            [sum([p['diameter'] for p in g['base_data']]) / len(g['base_data']) for g in
                             fixation_support]) / len(fixation_support)
                        new_fixation = {'topic': 'fixation',
                                        'id': len(fixations),
                                        'norm_pos': fixation_centroid,
                                        'base_data': fixation_support,
                                        'duration': duration,
                                        'aoi':next((item['aoi'] for item in fixation_support if item['aoi'] is not None), 'Unlabeled'),
                                        'dispersion': dispersion,
                                        'start_frame_index': fixation_support[0]['index'],
                                        'mid_frame_index': fixation_support[len(fixation_support) // 2]['index'],
                                        'end_frame_index': fixation_support[-1]['index'],
                                        'pix_dispersion': dispersion * self.pix_per_degree,
                                        'timestamp': fixation_support[0]['timestamp'],
                                        'pupil_diameter': avg_pupil_size,
                                        'confidence': confidence}
                        if(new_fixation['aoi']!='Unlabeled'):
                            new_fixation['aoi_label'] = self.aoi_provider.get_aoi(new_fixation['aoi'])
                            fixations.append(new_fixation)
                if gaze_data:
                    # start a new fixation candite
                    fixation_support = [last_sample]
                else:
                    break

        self.fixations = fixations
        # gather some statisics for debugging and feedback.
        total_fixation_time = sum([f['duration'] for f in fixations])
        total_video_time = self.g_pool.timestamps[-1] - self.g_pool.timestamps[0]
        fixation_count = len(fixations)
        logger.info(
            "Detected {} Fixations. Total duration of fixations: {:.2f}sec total time of video {:0.2f}sec ".format(
                fixation_count, total_fixation_time, total_video_time))

        # now lets bin fixations into frames. Fixations may be repeated this way as they span muliple frames
        fixations_by_frame = [[] for x in self.g_pool.timestamps]
        for f in fixations:
            for idx in range(f['start_frame_index'], f['end_frame_index'] + 1):
                if f['aoi']!='Unlabeled':
                    fixations_by_frame[idx].append(f)
        self.g_pool.fixations_by_frame = fixations_by_frame

    def init_gui(self):
        # initialize the menu
        def set_sequence_frames(seq_frame_num):
            self.seq_frame_num = seq_frame_num

        def on_aoi_encode():
            self.aoi_provider.rw_aoi_binary()
            logger.info("All AOI's are hashed by hex-32")

        self.menu = ui.Scrolling_Menu('Scan Path')
        self.menu.append(ui.Button('Close', self.unset_alive))
        #Scan-path properties
        scanpath_menu = ui.Growing_Menu('ScanPath Properties')
        scanpath_menu.append(ui.Slider('timeframe', self, min=0, step=0.1, max=5, label="Timeframe in sec"))
        scanpath_menu.append(ui.Button('Encode AOI manually', on_aoi_encode))
        scanpath_menu.append(ui.Slider('seq_frame_num', self, min=1, step=1, max=20, label='Number of frames in a fixation sequence',
                                       setter=set_sequence_frames))
        scanpath_menu.append(ui.Button('Save Fixations and Timestamps', self.on_scanpath_build))
        #scanpath_menu.append(ui.Button('Sample frames by similarity', self.on_sample_frames_by_similarity))

        self.menu.append(scanpath_menu)

        def set_h_fov(new_fov):
            self.h_fov = new_fov
            self.pix_per_degree = float(self.g_pool.capture.frame_size[0]) / new_fov
            self.notify_all({'subject': 'fixations_should_recalculate', 'delay': 1.})

        def set_v_fov(new_fov):
            self.v_fov = new_fov
            self.pix_per_degree = float(self.g_pool.capture.frame_size[1]) / new_fov
            self.notify_all({'subject': 'fixations_should_recalculate', 'delay': 1.})

        def set_duration(new_value):
            self.min_duration = new_value
            self.notify_all({'subject': 'fixations_should_recalculate', 'delay': 1.})

        def set_dispersion(new_value):
            self.max_dispersion = new_value
            self.notify_all({'subject': 'fixations_should_recalculate', 'delay': 1.})

        def jump_next_fixation(_):
            ts = self.last_frame_ts
            for f in self.fixations:
                if f['timestamp'] > ts:
                    self.g_pool.capture.seek_to_frame(f['mid_frame_index'])
                    self.g_pool.new_seek = True
                    return
            logger.error('could not seek to next fixation.')

        fixation_menu = ui.Growing_Menu('Fixation Properties')
        fixation_menu.append(ui.Info_Text("Fixation Properties"))
        fixation_menu.append(ui.Info_Text("Press the export button or type 'e' to start the export."))
        fixation_menu.append(ui.Slider('min_duration', self, min=0.0, step=0.05, max=1.0, label='Duration threshold',
                                   setter=set_duration))
        fixation_menu.append(ui.Slider('max_dispersion', self,
                                   min=self.dispersion_slider_min,
                                   step=self.dispersion_slider_stp,
                                   max=self.dispersion_slider_max,
                                   label='Dispersion threshold',
                                   setter=set_dispersion))
        fixation_menu.append(
            ui.Slider('h_fov', self, min=5, step=1, max=180, label='Horizontal FOV of scene camera', setter=set_h_fov))
        fixation_menu.append(
            ui.Slider('v_fov', self, min=5, step=1, max=180, label='Vertical FOV of scene camera', setter=set_v_fov))
        self.menu.append(fixation_menu)
        self.g_pool.gui.append(self.menu)


    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def get_gazes(self):
        pass
    # pts = []
    # for pt in events.get('gaze_positions', []):
    #     if pt['confidence'] >= self.g_pool.min_data_confidence:
    #         denorm_pos = denormalize(pt['norm_pos'], frame.img.shape[:-1][::-1], flip_y=True)
    #         pts.append(denorm_pos)
    #         transparent_circle(frame.img, denorm_pos, radius=self.radius, color=(self.b, self.g, self.r, self.a), thickness=thickness)
    #         aoi = self.aoi_encoder.encode_aoi(pt['aoi'])
    #         if aoi is not None:
    #             cv2.putText(img, aoi, tuple(map(int, denorm_pos)), cv2.FONT_HERSHEY_SIMPLEX, .7,
    #                     (185, 224, 236), 1, lineType=cv2.LINE_AA)
    #
    # bgra = (self.pb * 255, self.pg * 255, self.pr * 255, self.pa * 255)
    # if pts:
    #     pts = np.array([pts], dtype=np.int32)
    #     cv2.polylines(frame.img, pts, isClosed=False, color=bgra, thickness=self.pthickness, lineType=cv2.LINE_AA)

    def on_notify(self, notification):
        if notification['subject'] == 'gaze_positions_changed':
            logger.info('Gaze postions changed. Recalculating.')
            self._classify()
        elif notification['subject'] == 'fixations_should_recalculate':
            self._classify()
        #elif notification['subject'] == "should_export":
            #self.export_fixations(notification['range'], notification['export_dir'])

    def unset_alive(self):
        self.alive = False

    def get_init_dict(self):
        return {'timeframe':self.timeframe}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
