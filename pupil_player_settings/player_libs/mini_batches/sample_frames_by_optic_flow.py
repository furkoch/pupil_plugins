import sys
sys.path.append('C:\\Projects\\repo\\sensorics\\EyeTracker\\pupil\\pupil_src\\shared_modules')
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings')
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings\\player_libs')
from player_libs.cvision.frames_sampler import Frames_Sampler

rec_dirs = ["003", "004","v2_001","v2_002"] #,

def read_write_frame_similarity(rec_dir):
    frame_sampler = Frames_Sampler(rec_dir)
    frame_sampler.sample_frames_by_optic_flow()

if __name__ == "__main__":
    file_dir = "C:\\Users\\drivesense\\recordings\\2017_07_18\\"
    read_write_frame_similarity(file_dir + rec_dirs[0])
    # for rec_dir in rec_dirs:
    #     read_write_frame_similarity(file_dir + rec_dir)