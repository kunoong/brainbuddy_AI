import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import math

class DynamicFeatureCalculator:
    """동적 특징 계산기 (시계열 기반)"""
    
    def __init__(self):
        pass
    
    def calculate_stability(self, values: List[float], window_size: int = 5) -> float:
        """안정성 계산 (표준편차 기반)"""
        if len(values) < window_size:
            return 0.0
        
        recent_values = values[-window_size:]
        return 1.0 / (1.0 + np.std(recent_values))
    
    def calculate_jitter(self, x_values: List[float], y_values: List[float]) -> float:
        """시선 불안정성 계산"""
        if len(x_values) < 2 or len(y_values) < 2:
            return 0.0
        
        # 연속된 점들 간의 거리 변화
        distances = []
        for i in range(1, len(x_values)):
            dist = math.sqrt((x_values[i] - x_values[i-1])**2 + 
                           (y_values[i] - y_values[i-1])**2)
            distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def calculate_saccade_frequency(self, gaze_x: List[float], gaze_y: List[float], 
                                  threshold: float = 50.0) -> float:
        """사케이드(급속 안구 운동) 빈도 계산"""
        if len(gaze_x) < 2:
            return 0.0
        
        saccade_count = 0
        for i in range(1, len(gaze_x)):
            movement = math.sqrt((gaze_x[i] - gaze_x[i-1])**2 + 
                               (gaze_y[i] - gaze_y[i-1])**2)
            if movement > threshold:
                saccade_count += 1
        
        return saccade_count / len(gaze_x)
    
    def is_frontal_gaze(self, gaze_x: float, gaze_y: float, 
                       screen_center_x: float = 960, screen_center_y: float = 540,
                       threshold: float = 200) -> int:
        """정면 주시 여부 판단"""
        distance_from_center = math.sqrt((gaze_x - screen_center_x)**2 + 
                                       (gaze_y - screen_center_y)**2)
        return 1 if distance_from_center <= threshold else 0
    
    def detect_blink(self, l_ear: float, r_ear: float, threshold: float = 0.25) -> int:
        """눈 깜빡임 감지"""
        avg_ear = (l_ear + r_ear) / 2
        return 1 if avg_ear < threshold else 0
    
    def calculate_fixation_duration(self, gaze_positions: List[Tuple[float, float]], 
                                  threshold: float = 50.0) -> float:
        """고정 응시 시간 계산"""
        if len(gaze_positions) < 2:
            return 0.0
        
        fixation_duration = 0
        current_fixation_length = 1
        
        for i in range(1, len(gaze_positions)):
            prev_x, prev_y = gaze_positions[i-1]
            curr_x, curr_y = gaze_positions[i]
            
            distance = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            
            if distance <= threshold:
                current_fixation_length += 1
            else:
                if current_fixation_length > fixation_duration:
                    fixation_duration = current_fixation_length
                current_fixation_length = 1
        
        return fixation_duration
    
    def calculate_person_dynamic_features(self, person_data: pd.DataFrame) -> Dict:
        """개인별 동적 특징 계산"""
        
        # 시계열 데이터 정렬
        person_data = person_data.sort_values('filename')
        
        # 시선 데이터
        gaze_x_list = person_data['gaze_x'].tolist()
        gaze_y_list = person_data['gaze_y'].tolist()
        gaze_positions = list(zip(gaze_x_list, gaze_y_list))
        
        # 머리 데이터
        head_yaw_list = person_data['head_yaw'].tolist()
        head_pitch_list = person_data['head_pitch'].tolist()
        head_roll_list = person_data['head_roll'].tolist()
        
        # EAR 데이터
        l_ear_list = person_data['l_EAR'].tolist()
        r_ear_list = person_data['r_EAR'].tolist()
        
        # 거리 데이터 
        distance_list = person_data['cam_distance'].tolist()
        
        dynamic_features = {
            # 안정성 특징
            'gaze_stability': self.calculate_stability(gaze_x_list + gaze_y_list),
            'head_stability': self.calculate_stability(head_yaw_list + head_pitch_list + head_roll_list),
            
            # 시선 특징
            'gaze_jitter': self.calculate_jitter(gaze_x_list, gaze_y_list),
            'saccade_frequency': self.calculate_saccade_frequency(gaze_x_list, gaze_y_list),
            'fixation_duration': self.calculate_fixation_duration(gaze_positions),
            
            # 이진 특징들의 평균 (확률)
            'gaze_direction_prob': np.mean([self.is_frontal_gaze(x, y) for x, y in gaze_positions]),
            'blink_frequency': np.mean([self.detect_blink(l, r) for l, r in zip(l_ear_list, r_ear_list)]),
            
            # 거리 변화
            'distance_change': np.std(distance_list) if len(distance_list) > 1 else 0,
        }
        
        return dynamic_features
