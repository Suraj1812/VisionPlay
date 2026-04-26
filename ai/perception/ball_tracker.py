from __future__ import annotations

import cv2
import numpy as np
import math

                         
_EYE23 = np.eye(2, 3, dtype=np.float32)
_DOWNSCALE = 0.25                                        

class SpatioTemporalBallTracker:

    def __init__(self, fps: float = 25.0):
        self.fps = max(fps, 1.0)
        self.dt = 1.0 / self.fps

                                                 
        self._prev_gray_small = None
        self._prev_pts = None
        self._ego_transform = _EYE23.copy()

                                                                         
        self._kf = self._create_kalman()
        self._kf_initialized = False

                        
        self.last_pos = None                          
        self.last_size = 14.0                                           
        self.frames_lost = 0
        self.max_lost = int(fps * 1.2)                                 
        self.total_detections = 0

                                           
        self._trajectory: list[tuple[float, float, int]] = []                      
        self._max_trajectory = 90

                     
        self._max_speed_px = fps * 40                                     

    def _create_kalman(self) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(6, 2)
        dt = self.dt
                                                 
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.zeros((2, 6), dtype=np.float32)
        kf.measurementMatrix[0, 0] = 1
        kf.measurementMatrix[1, 1] = 1
                                                                        
        kf.processNoiseCov = np.diag([1, 1, 5, 5, 50, 50]).astype(np.float32) * 0.05
                                                                    
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        return kf

    def add_frame(self, frame: np.ndarray):
        pass                             

    def predict(self, frame: np.ndarray, yolo_ball_bbox=None) -> list | None:
        h, w = frame.shape[:2]

        prev_gray_small = self._prev_gray_small.copy() if self._prev_gray_small is not None else None
        small = cv2.resize(frame, (0, 0), fx=_DOWNSCALE, fy=_DOWNSCALE, interpolation=cv2.INTER_AREA)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        ego = self._estimate_ego_motion_fast(gray_small)

                                
        if self._kf_initialized:
            state = self._kf.predict()
            pred_cx, pred_cy = float(state[0, 0]), float(state[1, 0])

                                                     
            if self.last_pos is not None:
                pt = np.array([[[pred_cx, pred_cy]]], dtype=np.float32)
                warped = cv2.transform(pt, ego)
                pred_cx, pred_cy = float(warped[0, 0, 0]), float(warped[0, 0, 1])
                self._kf.statePost[0, 0] = pred_cx
                self._kf.statePost[1, 0] = pred_cy
        else:
            pred_cx, pred_cy = w / 2.0, h / 2.0

                             
        if yolo_ball_bbox:
            cx = (yolo_ball_bbox[0] + yolo_ball_bbox[2]) / 2.0
            cy = (yolo_ball_bbox[1] + yolo_ball_bbox[3]) / 2.0
            bw = yolo_ball_bbox[2] - yolo_ball_bbox[0]
            bh = yolo_ball_bbox[3] - yolo_ball_bbox[1]

                                                                
            if self._kf_initialized:
                dist = math.hypot(cx - pred_cx, cy - pred_cy)
                if dist > self._max_speed_px * 2:
                                                                   
                    return self._handle_lost(pred_cx, pred_cy)

                                                         
            self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

            meas = np.array([[np.float32(cx)], [np.float32(cy)]])
            if not self._kf_initialized:
                self._kf.statePost = np.array(
                    [[cx], [cy], [0], [0], [0], [9.8 * self.dt]], dtype=np.float32
                )                                  
                self._kf.statePre = self._kf.statePost.copy()
                self._kf_initialized = True
            self._kf.correct(meas)

            self.last_size = max(bw, bh, 8.0)
            self.last_pos = [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2]
            self.frames_lost = 0
            self.total_detections += 1
            self._add_to_trajectory(cx, cy)

            return self.last_pos + [0.90]

        motion_ball_bbox = self._detect_motion_ball_candidate(
            gray_small=gray_small,
            prev_gray_small=prev_gray_small,
            pred_cx=pred_cx,
            pred_cy=pred_cy,
            frame_width=w,
            frame_height=h,
        )
        if motion_ball_bbox is not None:
            cx = (motion_ball_bbox[0] + motion_ball_bbox[2]) / 2.0
            cy = (motion_ball_bbox[1] + motion_ball_bbox[3]) / 2.0
            bw = motion_ball_bbox[2] - motion_ball_bbox[0]
            bh = motion_ball_bbox[3] - motion_ball_bbox[1]

            self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
            meas = np.array([[np.float32(cx)], [np.float32(cy)]])
            if not self._kf_initialized:
                self._kf.statePost = np.array(
                    [[cx], [cy], [0], [0], [0], [9.8 * self.dt]], dtype=np.float32
                )
                self._kf.statePre = self._kf.statePost.copy()
                self._kf_initialized = True
            self._kf.correct(meas)

            self.last_size = max(bw, bh, 8.0)
            self.last_pos = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
            self.frames_lost = 0
            self.total_detections += 1
            self._add_to_trajectory(cx, cy)
            return self.last_pos + [0.42]

        return self._handle_lost(pred_cx, pred_cy)

    def _detect_motion_ball_candidate(
        self,
        gray_small: np.ndarray,
        prev_gray_small: np.ndarray | None,
        pred_cx: float,
        pred_cy: float,
        frame_width: int,
        frame_height: int,
    ) -> list[float] | None:
        if prev_gray_small is None:
            return None

        diff = cv2.absdiff(gray_small, prev_gray_small)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, binary = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        small_h, small_w = gray_small.shape[:2]
        roi_left = int(small_w * 0.30)
        roi_top = int(small_h * 0.18)
        roi_right = int(small_w * 0.70)
        roi_bottom = int(small_h * 0.86)
        roi = binary[roi_top:roi_bottom, roi_left:roi_right]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        best_score = 0.0
        predicted_small_x = pred_cx * _DOWNSCALE
        predicted_small_y = pred_cy * _DOWNSCALE

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2.0 or area > 90.0:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width < 2 or height < 2 or width > 14 or height > 14:
                continue

            aspect_ratio = width / max(float(height), 1.0)
            if aspect_ratio < 0.45 or aspect_ratio > 2.2:
                continue

            contour_perimeter = max(cv2.arcLength(contour, True), 1.0)
            circularity = (4.0 * math.pi * area) / (contour_perimeter * contour_perimeter)
            if circularity < 0.22:
                continue

            cx = roi_left + x + (width / 2.0)
            cy = roi_top + y + (height / 2.0)
            dist = math.hypot(cx - predicted_small_x, cy - predicted_small_y)
            center_lane_bias = 1.0 - min(abs(cx - (small_w / 2.0)) / max(small_w * 0.25, 1.0), 1.0)
            motion_strength = float(roi[y:y + height, x:x + width].mean()) / 255.0
            score = circularity * 1.7 + motion_strength * 1.3 + center_lane_bias * 0.7 - (dist / max(small_w, 1.0)) * 0.9
            if score <= best_score:
                continue

            best_score = score
            best_box = [
                (roi_left + x) / _DOWNSCALE,
                (roi_top + y) / _DOWNSCALE,
                (roi_left + x + width) / _DOWNSCALE,
                (roi_top + y + height) / _DOWNSCALE,
            ]

        if best_score < 0.72:
            return None

        return best_box

    def _handle_lost(self, pred_cx: float, pred_cy: float) -> list | None:
        if not self._kf_initialized or self.last_pos is None:
            return None

        self.frames_lost += 1
        if self.frames_lost > self.max_lost:
            return None

                                                 
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 20.0
        meas = np.array([[np.float32(pred_cx)], [np.float32(pred_cy)]])
        self._kf.correct(meas)

        half = self.last_size / 2.0
        self.last_pos = [pred_cx - half, pred_cy - half, pred_cx + half, pred_cy + half]
        self._add_to_trajectory(pred_cx, pred_cy)

        confidence = max(0.30, 0.85 - (0.06 * self.frames_lost))
        return self.last_pos + [confidence]

    def _estimate_ego_motion_fast(self, gray_small: np.ndarray) -> np.ndarray:
                                                                      
        if self._prev_gray_small is None:
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray_small, maxCorners=100, qualityLevel=0.01, minDistance=15
            )
            self._prev_gray_small = gray_small
            return _EYE23.copy()

        if self._prev_pts is None or len(self._prev_pts) < 6:
            self._prev_pts = cv2.goodFeaturesToTrack(
                self._prev_gray_small, maxCorners=100, qualityLevel=0.01, minDistance=15
            )
        if self._prev_pts is None or len(self._prev_pts) < 6:
            self._prev_gray_small = gray_small
            return _EYE23.copy()

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray_small, gray_small, self._prev_pts, None,
            winSize=(15, 15), maxLevel=2
        )
        good_old = self._prev_pts[status == 1]
        good_new = curr_pts[status == 1]

        if len(good_old) < 6:
            self._prev_gray_small = gray_small
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray_small, maxCorners=100, qualityLevel=0.01, minDistance=15
            )
            return _EYE23.copy()

                                                                
        transform, _ = cv2.estimateAffinePartial2D(
            good_old / _DOWNSCALE, good_new / _DOWNSCALE
        )
        if transform is None:
            transform = _EYE23.copy()

        self._prev_gray_small = gray_small
        self._prev_pts = good_new.reshape(-1, 1, 2)
        self._ego_transform = transform
        return transform

    def _add_to_trajectory(self, cx: float, cy: float):
        self._trajectory.append((cx, cy, self.total_detections))
        if len(self._trajectory) > self._max_trajectory:
            self._trajectory.pop(0)

    def get_trajectory(self) -> list[tuple[float, float]]:
        return [(x, y) for x, y, _ in self._trajectory]

    def get_velocity(self) -> tuple[float, float]:
                                                                      
        if self._kf_initialized:
            return float(self._kf.statePost[2, 0]), float(self._kf.statePost[3, 0])
        return 0.0, 0.0

    def get_acceleration(self) -> tuple[float, float]:
                                                                            
        if self._kf_initialized:
            return float(self._kf.statePost[4, 0]), float(self._kf.statePost[5, 0])
        return 0.0, 0.0

    def reset(self):
        self._prev_gray_small = None
        self._prev_pts = None
        self._ego_transform = _EYE23.copy()
        self._kf = self._create_kalman()
        self._kf_initialized = False
        self.last_pos = None
        self.last_size = 14.0
        self.frames_lost = 0
        self.total_detections = 0
        self._trajectory.clear()
