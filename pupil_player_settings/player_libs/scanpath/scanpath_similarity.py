import sys
sys.path.append('C:\\Projects\\repo\\sensorics\\EyeTracker\\pupil\\pupil_src\\shared_modules')
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings\\player_libs')
import numpy as np
from numpy import array
import player_libs.common.player_functions as pf

class Scanpath_Similarity:
    def __init__(self, recDirs):
        self.recDirs = recDirs

    def similarity_index(self, sourceScanpath, targetScanpath, ed, percent = False):
        m = len(sourceScanpath)
        n = len(targetScanpath)
        ed = ed/max(m,n)
        return 100*(1 - ed) if percent else 1 - ed

    def compute_edit_distance(self):
        sim_arr = np.zeros((len(self.recDirs), len(self.recDirs)))
        for i in range(0,len(self.recDirs)):
            sourceFixations = pf.get_fixations(self.recDirs[i])
            for j in range(0,len(self.recDirs)):
                targetFixations = pf.get_fixations(self.recDirs[j])
                desim = self.edit_distance_algorithm_substitution_matrix(sourceFixations, targetFixations)
                desim /= max(len(sourceFixations),len(targetFixations))
                sim_arr[i][j] = 1 - desim
                sim_arr[j][i] =  sim_arr[i][j]
        return sim_arr

    def edit_distance_algorithm(self, sourceScanpath, targetScanpath):
        # Create a table to store results of subproblems
        m = len(sourceScanpath)
        n = len(targetScanpath)
        #print("{}:{}".format(m,n))
        dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

        # Fill d[][] in bottom up manner
        for i in range(m + 1):
            for j in range(n + 1):

                # If first string is empty, only option is to
                # isnert all characters of second string
                if i == 0:
                    dp[i][j] = j  # Min. operations = j

                # If second string is empty, only option is to
                # remove all characters of second string
                elif j == 0:
                    dp[i][j] = i  # Min. operations = i

                # If last characters are same, ignore last char
                # and recur for remaining string
                elif sourceScanpath[i - 1]['aoi'] == targetScanpath[j - 1]['aoi']:
                    dp[i][j] = dp[i - 1][j - 1]

                # If last character are different, consider all
                # possibilities and find minimum
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                       dp[i - 1][j],  # Remove
                                       dp[i - 1][j - 1])  # Replace

        return  dp[m][n]

    def edit_distance_algorithm_frame_compare(self, sourceScanpath, targetScanpath):
        # Create a table to store results of subproblems
        m = len(sourceScanpath)
        n = len(targetScanpath)

        dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

        # Fill d[][] in bottom up manner
        for i in range(m + 1):
            for j in range(n + 1):

                if i == 0:
                    dp[i][j] = j  # Min. operations = j

                elif j == 0:
                    dp[i][j] = i  # Min. operations = i

                #compare the frame indices as well
                elif sourceScanpath[i - 1]['aoi'] == targetScanpath[j - 1]['aoi']:
                    if (sourceScanpath[i - 1]['start_frame_index']<=targetScanpath[i - 1]['start_frame_index'] and sourceScanpath[i - 1]['end_frame_index']>=targetScanpath[i - 1]['start_frame_index']) or \
                            (targetScanpath[i - 1]['start_frame_index']<=sourceScanpath[i - 1]['start_frame_index'] and targetScanpath[i - 1]['end_frame_index']>=sourceScanpath[i - 1]['start_frame_index']):
                        dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                       dp[i - 1][j],  # Remove
                                       dp[i - 1][j - 1])  # Replace

        return  dp[m][n]

    def calculate_substitution_cost(self, sourceFixation, targetFixation, method = "euclidean", alpha = 0.001):
        xs,ys = sourceFixation['norm_pos']
        xt, yt = targetFixation['norm_pos']
        dist = sqrt(pow(xs - xt, 2) + pow(ys - yt, 2)) if method == "euclidiean" else (abs(xs - xt) + abs(ys - yt))
        return alpha * dist

    def edit_distance_algorithm_substitution_matrix(self, sourceScanpath, targetScanpath, cost_type='euclidean'):
        # Create a table to store results of subproblems
        m = len(sourceScanpath)
        n = len(targetScanpath)

        dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

        # Fill d[][] in bottom up manner
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j  # Min. operations = j
                elif j == 0:
                    dp[i][j] = i  # Min. operations = i

                #compare the frame indices as well
                elif sourceScanpath[i - 1]['aoi'] == targetScanpath[j - 1]['aoi']:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    replacement_cost = self.calculate_substitution_cost(sourceScanpath[i - 1], targetScanpath[j - 1], cost_type)
                    dp[i][j] =  min(1 + dp[i][j - 1],  # Insert
                                    1 + dp[i - 1][j],  # Remove
                                    replacement_cost + dp[i - 1][j - 1])  # Replace
        return dp[m][n]



if __name__ == "__main__":
    file_dir = "C:\\Users\\drivesense\\recordings\\2017_07_18\\"
    rec_dirs = [file_dir+"003", file_dir+"004"] # "v2_001", "v2_002"
    scanpath_sim = Scanpath_Similarity(rec_dirs)
    sim_matrix = scanpath_sim.compute_edit_distance()
    print(sim_matrix)


