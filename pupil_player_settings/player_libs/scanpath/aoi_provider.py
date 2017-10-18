import sys
sys.path.append('F:\\Projects\\repo\\sensorics\\EyeTracker\\pupil\\pupil_src\\shared_modules')
from file_methods import save_object, load_object
import string
from collections import deque
import _hashlib
from player_libs.csv_rw import CSV_RW

class AOI_Provider:

    def __init__(self,rec_dir):
        self.encoded_aoi_list = {}
        annotation_list = [au+str(dt) for dt in [i for i in range(1,10)] for au in list(string.ascii_uppercase)]
        self.annotation_pool = deque(annotation_list)
        self.rec_dir = rec_dir
        self.csv_rw = CSV_RW(rec_dir)
        self.csv_rw.read_file()
        self.file_dir = self.csv_rw.fileDir

    def rw_aoi_binary(self):
        pupil_file =  self.csv_rw.fileDir + "\\pupil_data"
        rec_data = load_object(pupil_file)
        for key, gaze_poses in rec_data.items():
            if key == "gaze_positions":
                for gp in gaze_poses:
                    try:
                        ts = str(gp['timestamp'])
                        gp['aoi'] = self.csv_rw.rec_timestamps[ts]
                    except:
                        gp['aoi'] = None

        save_object(rec_data, pupil_file)
        return rec_data

    def read_fixation_sequence(self):
        filepath = self.csv_rw.fileDir+"\\fixation_sequences.csv"
        fixation_sequences = []
        with open(filepath,'r') as fix_seq:
            for line in fix_seq:
                seq = [self.get_aoi(aoi) for aoi in line.split(',') if aoi!=""]
                fixation_sequences.append(seq)
        return  fixation_sequences



    def hex_hash(self,str):
        return _hashlib.openssl_sha1(str.encode('utf-8')).hexdigest()

    def get_aoi(self,aoi):
        if aoi is None:
            return None
        arr = aoi.split("/")
        if arr is not None or len(arr)>0:
            return arr[-1]

    def encode_aoi(self, nextAOI):
        if nextAOI not in self.encoded_aoi_list.keys():
            self.encoded_aoi_list[nextAOI] = self.annotation_pool.popleft()
        return self.encoded_aoi_list[nextAOI]

    def decode_aoi(self, aoi_label):
        inv_dic = {v: k for k, v in self.encoded_aoi_list.items()}
        return inv_dic[aoi_label]

if __name__=="__main__":
    aoi_prov = AOI_Provider("C:\\Users\\drivesense\\recordings\\2017_09_19\\001")
    aoi_prov.rw_aoi_binary()
    print("Finished writing AOI's")


