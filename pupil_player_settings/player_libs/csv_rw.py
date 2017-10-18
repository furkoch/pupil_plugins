from player_libs.gaze_position import GazePosition

class CSV_RW:

    LABELS_FILENAME = "aoi_labels.csv"
    FRAME_SAMPLING_FILENAME = "frame_samplings.csv"
    PUPIL_DATA_FILENAME = "pupil_data"
    LINE_DELIMITER = "\n"
    COLUMN_DELIMITER = ","


    def __init__(self, fileDir):
        self.fileDir = fileDir
        self.file_path_labels = fileDir + "\\" + CSV_RW.LABELS_FILENAME
        self.file_path_frame_samplings = fileDir + "\\" + CSV_RW.FRAME_SAMPLING_FILENAME
        self.rec_timestamps = {}
        self.rec_gazes = []

    def read_file(self):
        lines = []
        with open(self.file_path_labels) as fp:
            for line in fp:
                line = line.replace(CSV_RW.LINE_DELIMITER, "")
                splited_arr = line.split(CSV_RW.COLUMN_DELIMITER)
                if len(splited_arr)>0:
                    srf = splited_arr[0]
                    aoi = splited_arr[1]
                    gaze_position = GazePosition(srf, aoi) # gaze_position.timestamps = [float(el) for el in gpArr[2:]]
                    for el in splited_arr[2:]:
                        #el = float(el)
                        self.rec_timestamps[el] = aoi
                        gaze_position.timestamps.append(el)
                    self.rec_gazes.append(gaze_position)

