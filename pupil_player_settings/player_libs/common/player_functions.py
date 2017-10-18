import json

def get_timestamps(rec_dir):
    # get timestamps from recorded g_pool
    timestamps = []
    with open(rec_dir + "\\timestamps.csv") as csv_file:
        for timestamp in csv_file:
            timestamps.append(json.loads(timestamp))
    return timestamps


def get_fixations(rec_dir):
    fixations = []
    with open(rec_dir + "\\fixations.csv") as csv_file:
        for fixation in csv_file:
            fixations.append(json.loads(fixation))
    return fixations


def get_aoi(aoi):
    if aoi is None:
        return None
    arr = aoi.split("/")
    if arr is not None or len(arr) > 0:
        return arr[-1]