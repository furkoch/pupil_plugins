import sys
sys.path.append('C:\\Users\\drivesense\\pupil_player_settings\\player_libs')
import numpy as np

class Needleman_Wunsch_Alignment:

    def __init__(self, sourceScanpath, targetScanpath, match_award = 10, mismatch_penalty=-5, gap_penalty=-5):
        self.sourceScanpath = sourceScanpath
        self.targetScanpath = targetScanpath

        self.nSourceScanpath, nTargetScanpath = len(self.sourceScanpath), len(self.targetScanpath)

        self.match_award = match_award
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty


    def compute_score_matrix(self):

        #dp table that later be used to traceback
        score_matrix = np.zeros((self.nSourceScanpath + 1, self.nTargetScanpath + 1))

        for i in range(0,self.nSourceScanpath+1):
            score_matrix[i][0] = self.gap_penalty * i

        for j in range(0,self.nTargetScanpath+1):
            score_matrix[0][j] = self.gap_penalty * j

        for i in range(1, self.nSourceScanpath + 1):
            for j in range(1, self.nTargetScanpath + 1):
                match_score = score_matrix[i - 1][j - 1] + self.get_match_score(self.sourceScanpath[i-1], self.targetScanpath[j-1])
                delete_score = score_matrix[i - 1][j] + self.gap_penalty
                insert_score = score_matrix[i][j - 1] + self.gap_penalty

                score_matrix[i][j] = max(match_score, delete_score, insert_score)

        return score_matrix

    def traceback(self, score_matrix):
        # Traceback and compute the alignment
        alignSource, alignTarget = '', ''
        i, j = self.nSourceScanpath, self.nTargetScanpath  # start from the bottom right cell

        while i > 0 and j > 0:  # end toching the top or the left edge
            score_current = score_matrix[i][j]
            score_diagonal = score_matrix[i - 1][j - 1]
            score_up = score_matrix[i][j - 1]
            score_left = score_matrix[i - 1][j]

            if score_current == score_diagonal + self.get_match_score(self.sourceScanpath[i-1], self.targetScanpath[j-1]):
                alignSource += self.sourceScanpath[i - 1]
                alignTarget += self.targetScanpath[j - 1]
                i -= 1
                j -= 1
            elif score_current == score_left + self.gap_penalty:
                alignSource += self.sourceScanpath[i - 1]
                alignTarget += '-'
                i -= 1
            elif score_current == score_up + self.gap_penalty:
                alignSource += '-'
                alignTarget += self.targetScanpath[j - 1]
                j -= 1

        # Finish tracing up to the top left cell
        while i > 0:
            alignSource += self.sourceScanpath[i - 1]
            alignTarget += '-'
            i -= 1
        while j > 0:
            alignSource += '-'
            alignTarget += self.targetScanpath[j - 1]
            j -= 1

    def finalize(self, alignSource, alignTarget):
        alignSource = alignSource[::-1]  # reverse sequence 1
        alignTarget = alignTarget[::-1]  # reverse sequence 2

        i, j = 0, 0

        # calcuate identity, score and aligned sequeces
        symbol = ''
        found = 0
        score = 0
        identity = 0
        for i in range(0, len(alignSource)):
            # if two AAs are the same, then output the letter
            if alignSource[i] == alignTarget[i]:
                symbol = symbol + alignSource[i]
                identity = identity + 1
                score += self.get_match_score(alignSource[i], alignTarget[i])

            # if they are not identical and none of them is gap
            elif alignSource[i] != alignTarget[i] and alignSource[i] != '-' and alignTarget[i] != '-':
                score += self.get_match_score(alignSource[i], alignTarget[i])
                symbol += ' '
                found = 0

            # if one of them is a gap, output a space
            elif alignSource[i] == '-' or alignTarget[i] == '-':
                symbol += ' '
                score += self.gap_penalty

        identity = float(identity) / len(alignSource) * 100

        print('Identity = {} percent'.format(identity))
        print('Score = {}'.format(score))
        print(alignSource)
        print(symbol)
        print(alignTarget)

        def get_match_score(self, sourceScanpath, targetScanpath):
            if sourceScanpath['aoi'] == targetScanpath['aoi']:
                return self.match_award
            elif sourceScanpath['aoi'] == '-' or targetScanpath['aoi'] == '-':
                return self.gap_penalty
            else:
                return self.mismatch_penalty








