import unittest
import sys
sys.path.append('F:\\Projects\\repo\\sensorics\\EyeTracker\\pupil\\pupil_src\\shared_modules')
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings')
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings\\player_libs')
from  player_libs.csv_rw import CSV_RW
# from player_libs.cvision.frames_sampler import Frames_Sampler
# from  scanpath.aoi_provider import AOI_Provider
# from file_methods import save_object, load_object

class PlayerTests(unittest.TestCase):

    player_path = "F:\\Soft\\pupil_v0912_windows_x64\\pupil_v0912_windows_x64\\pupil_player_v0.9.12\\pupil_player.exe"

    def setUp(self):
        self.rec_dir = "C:\\Users\\drivesense\\recordings\\2017_09_19\\001"
        # self.aoi_provider = AOI_Provider(self.rec_dir)
        # self.csv_rw = self.aoi_provider.csv_rw


    def tes_run_player(self):
        import os
        os.system(self.player_path + " " + self.rec_dir)

    def tes_rec_gazes(self):
        rec_gazes = self.csv_rw.rec_gazes
        self.assertIsNotNone(rec_gazes,"Rec-data is null")
        for gaze in rec_gazes:
            self.assertIsNotNone(gaze)
            self.assertEqual(gaze.srf,"screen")
            self.assertTrue(gaze.aoi!="")

    def tes_sample_frames_by_similarity(self):
        frame_editor = Frames_Sampler(self.rec_dir)
        frame_editor.find_frame_rate()
        # frame_editor.sample_frames_by_similarity()

    def tes_rec_timestamps(self):
        rec_timestamps = self.csv_rw.rec_timestamps
        self.assertIsNotNone(rec_timestamps)
        for key,val in rec_timestamps.items():
            self.assertIsNotNone(key)
            self.assertIsNotNone(val)
            self.assertTrue(key.strip()!="")

    def tes_read_pupil_data(self):
        pupil_file = self.csv_rw.fileDir + "\\pupil_data"
        print(pupil_file)
        rec_data = load_object(pupil_file)
        for key, gaze_poses in rec_data.items():
            if key == "gaze_positions":
                for gp in gaze_poses:
                    try:
                        print(gp['aoi'])
                    except:
                        pass

    def tes_read_fixation_sequence(self):
        fix_sequences = self.aoi_provider.read_fixation_sequence()
        for seq in fix_sequences:
            print(str(len(seq))+' =>'+str(seq))


    def tes_rec_aoi(self):
        rec_data = self.aoi_provider.rw_aoi_binary()

        for key, gaze_poses in rec_data.items():
            if key == "gaze_positions":
                for gp in gaze_poses:
                    print(gp['aoi'])
                    if gp['aoi'] is not None:
                        ts = str(gp['timestamp'])
                        self.assertEqual(gp['aoi'], self.csv_rw.rec_timestamps[ts])




if __name__ == '__main__':
    unittest.main()