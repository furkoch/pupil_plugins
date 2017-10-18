import sys
sys.path.append('C:\\Projects\\repo\\sensorics\\EyeTracker\\pupil\\pupil_src\\shared_modules')
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings')
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings\\player_libs')
from player_libs.cvision.frames_sampler import Frames_Sampler

rec_dirs = ["004","v2_001","v2_002"] #"003",

def read_write_frame_similarity(rec_dir):
    frame_editor = Frames_Sampler(rec_dir)
    frame_editor.sample_frames_by_similarity()

def main():
    file_dir = "C:\\Users\\drivesense\\recordings\\2017_07_18\\"
    for rec_dir in rec_dirs:
        read_write_frame_similarity(file_dir+rec_dir)

if __name__ == "__main__":
    main()