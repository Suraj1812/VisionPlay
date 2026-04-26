\
\
\
\
\
\
\
   
import cv2
import numpy as np
from collections import deque

_RESIZE = (160, 90)                               

class CameraCutDetector:

    def __init__(self, hard_threshold: float = 28.0, hist_threshold: float = 0.32):
        self.hard_threshold = hard_threshold
        self.hist_threshold = hist_threshold

        self._prev_gray = None
        self._prev_hist = None

                                     
        self._diff_history: deque[float] = deque(maxlen=60)
        self._hist_diff_history: deque[float] = deque(maxlen=60)

                                                       
        self._consecutive_high_diff = 0

               
        self.total_cuts = 0
        self.cut_frames: list[int] = []
        self._frame_id = 0

    def is_cut(self, frame: np.ndarray) -> bool:
                                                                              
        self._frame_id += 1

                             
        small = cv2.resize(frame, _RESIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                                                         
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        cv2.normalize(hist, hist)

        is_cut = False

        if self._prev_gray is not None and self._prev_hist is not None:
                                                
            mean_diff = float(np.mean(cv2.absdiff(self._prev_gray, gray)))

                                              
            hist_diff = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))

            self._diff_history.append(mean_diff)
            self._hist_diff_history.append(hist_diff)

                                                            
            if len(self._diff_history) > 10:
                arr = np.array(self._diff_history)
                adaptive_thresh = float(np.mean(arr) + 3.0 * np.std(arr))
                effective_thresh = min(adaptive_thresh, self.hard_threshold)
            else:
                effective_thresh = self.hard_threshold

            if mean_diff > effective_thresh and hist_diff > self.hist_threshold:
                self._consecutive_high_diff += 1
                                                                               
                                                                             
                                                                                         
                if self._consecutive_high_diff <= 2:
                    is_cut = True
                    self.total_cuts += 1
                    self.cut_frames.append(self._frame_id)
            else:
                self._consecutive_high_diff = 0

        self._prev_gray = gray
        self._prev_hist = hist
        return is_cut

    def is_replay_segment(self) -> bool:
\
\
           
        if len(self.cut_frames) < 3:
            return False
        recent = self.cut_frames[-3:]
        return (recent[-1] - recent[-2]) < 60 and (recent[-2] - recent[-3]) < 60

    def reset(self):
        self._prev_gray = None
        self._prev_hist = None
        self._diff_history.clear()
        self._hist_diff_history.clear()
        self._consecutive_high_diff = 0
        self.total_cuts = 0
        self.cut_frames.clear()
        self._frame_id = 0
