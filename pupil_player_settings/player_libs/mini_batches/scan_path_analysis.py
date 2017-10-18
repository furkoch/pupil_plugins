import sys
sys.path.append('C:\\Projects\\repo\\sensorics\\EyeTracker\\pupil\\pupil_src\\shared_modules')
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings')
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings\\player_libs')
from player_libs.cvision.frames_sampler import Frames_Sampler
import json

rec_dirs = ["003","004"]

def read_fixation_sequences(rec_dir):
    fixations = []
    with open(rec_dir) as csv_file:
        for fixation in csv_file:
            fixations.append(json.loads(fixation))
    return fixations

def dump_fixation_sequences():
    file_dir = "C:\\Users\\drivesense\\recordings\\2017_07_18\\"
    fixation_sequences = []
    for rec_dir in rec_dirs:
        fixation_sequence = read_fixation_sequences(file_dir + rec_dir + "\\fixation_sequences.csv")
        fixation_sequences.append(fixation_sequence)
    return fixation_sequences



if __name__ == "__main__":
    pass




