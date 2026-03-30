import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import math

class JSONFeatureExtractor:
    """JSON 파일에서 집중도 관련 특징 추출"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.valid_postures = ['C', 'D', 'U']
        self.class_mapping = {
            'F': 2,  # 집중
            'S': 0, 'N': 0,  # 비집중  
            'D': 1, 'A': 1   # 주의산만
        }
        
    def find_all_json_files(self) -> List[Path]:
        """모든 JSON 파일 경로 찾기"""
        json_files = []
        
        print(f"기본 경로: {self.base_path}")
        
        # 001~148 폴더 탐색
        for person_id in range(1, 149):
            person_folder = f"{person_id:03d}"
            
            # Monitor JSON 파일들
            monitor_json_dir = self.base_path / person_folder / "T1" / "Monitor" / "json_rgb"
            if monitor_json_dir.exists():
                json_files.extend(list(monitor_json_dir.glob("*.json")))
            
            # Laptop JSON 파일들 
            laptop_json_dir = self.base_path / person_folder / "T1" / "Laptop" / "json_rgb"
            if laptop_json_dir.exists():
                json_files.extend(list(laptop_json_dir.glob("*.json")))
        
        print(f"총 {len(json_files)}개 JSON 파일 발견")
        return json_files
    
    def extract_static_features(self, json_data: Dict) -> Optional[Dict]:
        """정적 특징 추출 (JSON 구조에 맞게 수정)"""
        try:
            # 필수 키 존재 확인
            if 'Annotations' not in json_data:
                return None
                
            anno = json_data['Annotations']
            
            # pose와 distance가 Annotations 안에 있음
            pose = anno.get('pose', {})
            distance = anno.get('distance', {})
            
            # 필수 필드 확인
            required_fields = ['condition', 'posture', 'metaid', 'inst', 'name']
            for field in required_fields:
                if field not in anno:
                    return None
            
            # pose와 distance 필드 확인
            if not pose or 'head' not in pose or 'point' not in pose:
                return None
            if not distance or 'cam' not in distance:
                return None
            
            # 눈 중심 좌표 추출
            l_eye_x, l_eye_y, r_eye_x, r_eye_y = None, None, None, None
            l_iris_rx, l_iris_ry, r_iris_rx, r_iris_ry = None, None, None, None
            
            annotations = anno.get('annotations', [])
            
            for annotation in annotations:
                label = annotation.get('label', '')
                
                if label == 'l_center' and 'points' in annotation:
                    points = annotation['points']
                    if points and len(points) > 0:
                        l_eye_x, l_eye_y = points[0][0], points[0][1]
                        
                elif label == 'r_center' and 'points' in annotation:
                    points = annotation['points']
                    if points and len(points) > 0:
                        r_eye_x, r_eye_y = points[0][0], points[0][1]
                        
                elif label == 'l_iris':
                    l_iris_rx = annotation.get('rx', 0)
                    l_iris_ry = annotation.get('ry', 0)
                    
                elif label == 'r_iris':
                    r_iris_rx = annotation.get('rx', 0)
                    r_iris_ry = annotation.get('ry', 0)
            
            # EAR 계산
            l_EAR = (l_iris_ry * 2) / (l_iris_rx * 2) if l_iris_rx and l_iris_rx > 0 else 0.3
            r_EAR = (r_iris_ry * 2) / (r_iris_rx * 2) if r_iris_rx and r_iris_rx > 0 else 0.3
            
            # 머리 포즈 데이터 추출
            head_data = pose.get('head', [0, 0, 0])
            head_roll, head_pitch, head_yaw = head_data[0], head_data[1], head_data[2]
            
            # 시선 데이터 추출
            point_data = pose.get('point', [0, 0])
            gaze_x, gaze_y = point_data[0], point_data[1]
            
            head_tilt_angle = math.sqrt(head_roll**2 + head_pitch**2)
            
            static_features = {
                # 머리 방향
                'head_yaw': float(head_yaw),
                'head_pitch': float(head_pitch),
                'head_roll': float(head_roll),
                
                # 거리
                'cam_distance': float(distance.get('cam', 60)),
                
                # 시선 방향
                'gaze_x': float(gaze_x),
                'gaze_y': float(gaze_y),
                'gaze_z': 0.0,
                
                # 눈 중심 좌표
                'l_eye_x': float(l_eye_x) if l_eye_x is not None else 0.0,
                'l_eye_y': float(l_eye_y) if l_eye_y is not None else 0.0,
                'r_eye_x': float(r_eye_x) if r_eye_x is not None else 0.0,
                'r_eye_y': float(r_eye_y) if r_eye_y is not None else 0.0,
                
                # EAR
                'l_EAR': float(l_EAR),
                'r_EAR': float(r_EAR),
                
                # 머리 기울기
                'head_tilt_angle': float(head_tilt_angle),
                
                # 메타 정보
                'condition': str(anno['condition']),
                'posture': str(anno['posture']),
                'metaid': str(anno['metaid']),
                'inst': str(anno['inst']),
                'filename': str(anno['name'])
            }
            
            return static_features
            
        except Exception as e:
            print(f"특징 추출 중 오류: {e}")
            return None
    
    def process_all_json_files(self) -> List[Dict]:
        """모든 JSON 파일 처리"""
        json_files = self.find_all_json_files()
        
        all_features = []
        failed_count = 0
        
        print("JSON 파일 처리 중...")
        
        for i, json_file in enumerate(json_files):
            if i % 1000 == 0:
                print(f"진행률: {i}/{len(json_files)} ({i/len(json_files)*100:.1f}%)")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # 유효한 자세와 상태만 필터링
                anno = json_data['Annotations']
                if (anno['posture'] in self.valid_postures and 
                    anno['condition'] in self.class_mapping):
                    
                    # 정적 특징 추출
                    features = self.extract_static_features(json_data)
                    
                    # features가 None이 아닐 때만 라벨 추가
                    if features is not None:
                        features['label_3class'] = self.class_mapping[anno['condition']]
                        all_features.append(features)
                    else:
                        failed_count += 1
                
            except Exception as e:
                failed_count += 1
                continue
        
        print(f"\n처리 완료: {len(all_features)}개 성공, {failed_count}개 실패")
        return all_features
