import pandas as pd
import numpy as np
from json_feature_extractor import JSONFeatureExtractor
from feature_calculator import DynamicFeatureCalculator
from pathlib import Path
import pickle

class DataProcessor:
    """데이터 통합 처리기"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.json_extractor = JSONFeatureExtractor(base_path)
        self.feature_calculator = DynamicFeatureCalculator()
        
        self.output_dir = Path("./processed_data")
        self.output_dir.mkdir(exist_ok=True)
    
    def process_complete_dataset(self):
        """전체 데이터셋 처리"""
        print("=== JSON 특징 기반 3클래스 집중도 데이터 처리 시작 ===")
        
        # 1. JSON에서 정적 특징 추출
        print("\n1단계: JSON 정적 특징 추출...")
        static_features = self.json_extractor.process_all_json_files()
        
        if not static_features:
            print("추출된 특징이 없습니다!")
            return
        
        # DataFrame 생성
        df_static = pd.DataFrame(static_features)
        print(f"정적 특징 데이터: {len(df_static)}개 샘플")
        
        # 2. 개인별 동적 특징 계산
        print("\n2단계: 개인별 동적 특징 계산...")
        dynamic_features = []
        
        unique_persons = df_static['metaid'].unique()
        print(f"고유 인물 수: {len(unique_persons)}명")
        
        for i, person_id in enumerate(unique_persons):
            if i % 10 == 0:
                print(f"동적 특징 계산 진행률: {i}/{len(unique_persons)}")
            
            person_data = df_static[df_static['metaid'] == person_id]
            
            # 각 condition별로 동적 특징 계산
            for condition in person_data['condition'].unique():
                condition_data = person_data[person_data['condition'] == condition]
                
                if len(condition_data) >= 3:  # 최소 3개 샘플 필요
                    dynamic_feat = self.feature_calculator.calculate_person_dynamic_features(condition_data)
                    
                    # 대표 정적 특징 (평균)
                    representative_static = {
                        'metaid': person_id,
                        'condition': condition,
                        'posture': condition_data['posture'].mode().iloc[0],
                        'inst': condition_data['inst'].mode().iloc[0],
                        'label_3class': condition_data['label_3class'].iloc[0],
                        
                        # 정적 특징 평균
                        'head_yaw_mean': condition_data['head_yaw'].mean(),
                        'head_pitch_mean': condition_data['head_pitch'].mean(),
                        'head_roll_mean': condition_data['head_roll'].mean(),
                        'cam_distance_mean': condition_data['cam_distance'].mean(),
                        'gaze_x_mean': condition_data['gaze_x'].mean(),
                        'gaze_y_mean': condition_data['gaze_y'].mean(),
                        'l_eye_x_mean': condition_data['l_eye_x'].mean(),
                        'l_eye_y_mean': condition_data['l_eye_y'].mean(),
                        'r_eye_x_mean': condition_data['r_eye_x'].mean(),
                        'r_eye_y_mean': condition_data['r_eye_y'].mean(),
                        'l_EAR_mean': condition_data['l_EAR'].mean(),
                        'r_EAR_mean': condition_data['r_EAR'].mean(),
                        'head_tilt_angle_mean': condition_data['head_tilt_angle'].mean(),
                        
                        # 정적 특징 표준편차
                        'head_yaw_std': condition_data['head_yaw'].std(),
                        'head_pitch_std': condition_data['head_pitch'].std(),
                        'head_roll_std': condition_data['head_roll'].std(),
                        'gaze_x_std': condition_data['gaze_x'].std(),
                        'gaze_y_std': condition_data['gaze_y'].std(),
                    }
                    
                    # 정적 + 동적 특징 결합
                    combined_features = {**representative_static, **dynamic_feat}
                    dynamic_features.append(combined_features)
        
        # 3. 최종 데이터셋 생성
        print("\n3단계: 최종 데이터셋 생성...")
        df_final = pd.DataFrame(dynamic_features)
        
        print(f"최종 데이터셋: {len(df_final)}개 샘플")
        
        # 4. 데이터셋 저장
        self.save_dataset(df_final)
        
        # 5. 통계 정보 출력
        self.print_final_statistics(df_final)
        
        print("\n=== 데이터 처리 완료 ===")
    
    def save_dataset(self, df: pd.DataFrame):
        """최종 데이터셋 저장"""
        
        # CSV 저장
        csv_path = self.output_dir / "json_features_3class_dataset.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV 저장 완료: {csv_path}")
        
        # 메타데이터 저장
        metadata = {
            'total_samples': len(df),
            'unique_persons': df['metaid'].nunique(),
            'class_distribution': df['label_3class'].value_counts().to_dict(),
            'posture_distribution': df['posture'].value_counts().to_dict(),
            'device_distribution': df['inst'].value_counts().to_dict(),
            'feature_columns': df.columns.tolist()
        }
        
        metadata_path = self.output_dir / "dataset_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"메타데이터 저장 완료: {metadata_path}")
    
    def print_final_statistics(self, df: pd.DataFrame):
        """최종 통계 출력"""
        print(f"\n=== 최종 데이터셋 정보 ===")
        print(f"총 샘플 수: {len(df)}")
        print(f"고유 인물 수: {df['metaid'].nunique()}명")
        print(f"특징 개수: {len(df.columns) - 5}개")  # 메타 컬럼 제외
        
        print(f"\n클래스 분포:")
        class_names = {0: '비집중', 1: '주의산만', 2: '집중'}
        for label, count in df['label_3class'].value_counts().sort_index().items():
            percentage = count / len(df) * 100
            print(f"  {class_names[label]}: {count}개 ({percentage:.1f}%)")
        
        print(f"\n자세 분포: {df['posture'].value_counts().to_dict()}")
        print(f"디바이스 분포: {df['inst'].value_counts().to_dict()}")
        
        # 주요 특징 통계
        print(f"\n주요 특징 통계:")
        key_features = ['gaze_stability', 'head_stability', 'gaze_jitter', 
                       'fixation_duration', 'saccade_frequency']
        
        for feature in key_features:
            if feature in df.columns:
                mean_val = df[feature].mean()
                std_val = df[feature].std()
                print(f"  {feature}: 평균={mean_val:.3f}, 표준편차={std_val:.3f}")

def main():
    # 기준 경로 설정
    base_path = r"C:\Users\user\Downloads\126.디스플레이 중심 안구 움직임 영상 데이터\01-1.정식개방데이터\Training\02.라벨링데이터\TL"
    
    processor = DataProcessor(base_path)
    processor.process_complete_dataset()

if __name__ == "__main__":
    main()
