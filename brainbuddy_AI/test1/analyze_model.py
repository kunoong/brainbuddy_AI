#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 학습 기준 및 집중도 판단 패턴 분석 도구
"""

import pandas as pd
import numpy as np
import os
import sys
from ml_classifier import ConcentrationClassifier
import joblib

def check_file_exists(filepath):
    """파일 존재 확인"""
    if not os.path.exists(filepath):
        print(f"파일을 찾을 수 없습니다: {filepath}")
        return False
    return True

def analyze_dataset_patterns():
    """학습 데이터의 클래스별 패턴 분석"""
    csv_path = "./processed_data/json_features_3class_dataset.csv"
    
    if not check_file_exists(csv_path):
        print("먼저 data_processor.py를 실행하여 데이터를 처리하세요.")
        return
    
    print("=== 학습 데이터 패턴 분석 ===")
    print("=" * 60)
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    print(f"총 샘플 수: {len(df)}개")
    
    # 클래스 분포
    print(f"\n 클래스 분포:")
    class_names = {0: '비집중', 1: '주의산만', 2: '집중'}
    for class_id in [0, 1, 2]:
        count = len(df[df['label_3class'] == class_id])
        percentage = count / len(df) * 100
        print(f"  {class_names[class_id]}: {count}개 ({percentage:.1f}%)")
    
    # 핵심 특징별 클래스 패턴 분석
    key_features = [
        'gaze_jitter',           # 시선 떨림 (중요도 1위)
        'saccade_frequency',     # 급속 안구운동 (중요도 2위)
        'head_pitch_std',        # 머리 움직임 변동성 (중요도 3위)
        'gaze_y_mean',           # 시선 Y좌표 (중요도 4위)
        'l_EAR_mean',            # 눈 깜빡임 (중요도 5위)
        'gaze_stability',        # 시선 안정성
        'head_stability',        # 머리 안정성
        'fixation_duration',     # 고정 응시 시간
        'gaze_direction_prob'    # 정면 응시 확률
    ]
    
    print(f"\n 핵심 특징별 클래스 패턴:")
    print("=" * 60)
    
    for feature in key_features:
        if feature not in df.columns:
            print(f"{feature}: 컬럼이 존재하지 않음")
            continue
            
        print(f"\n {feature}:")
        print("-" * 40)
        
        for class_id in [0, 1, 2]:
            class_data = df[df['label_3class'] == class_id][feature]
            if len(class_data) > 0:
                mean_val = class_data.mean()
                std_val = class_data.std()
                min_val = class_data.min()
                max_val = class_data.max()
                median_val = class_data.median()
                
                print(f"  {class_names[class_id]:>6}: 평균={mean_val:6.3f} | "
                      f"표준편차={std_val:6.3f} | 중앙값={median_val:6.3f} | "
                      f"범위=[{min_val:6.3f}, {max_val:6.3f}]")
    
    # 집중도 판단 임계값 추론
    print(f"\n 집중도 판단 임계값 추론:")
    print("=" * 60)
    
    focused_data = df[df['label_3class'] == 2]  # 집중 클래스
    distracted_data = df[df['label_3class'] == 1]  # 주의산만 클래스
    unfocused_data = df[df['label_3class'] == 0]  # 비집중 클래스
    
    print("모델이 학습한 집중 상태의 특징:")
    print("-" * 40)
    
    concentration_thresholds = {}
    
    for feature in key_features[:6]:  # 상위 6개 특징만
        if feature in df.columns:
            focused_mean = focused_data[feature].mean()
            distracted_mean = distracted_data[feature].mean()
            unfocused_mean = unfocused_data[feature].mean()
            
            # 집중 클래스가 다른 클래스들과 구별되는 방향 파악
            if focused_mean > max(distracted_mean, unfocused_mean):
                threshold = (focused_mean + max(distracted_mean, unfocused_mean)) / 2
                direction = "높을수록"
            else:
                threshold = (focused_mean + min(distracted_mean, unfocused_mean)) / 2
                direction = "낮을수록"
            
            concentration_thresholds[feature] = {
                'threshold': threshold,
                'direction': direction,
                'focused_avg': focused_mean
            }
            
            print(f"  {feature:20}: {direction} 집중 (임계값: {threshold:.3f}, 집중평균: {focused_mean:.3f})")
    
    return concentration_thresholds

def analyze_model_importance():
    """XGBoost 모델의 특징 중요도 분석"""
    model_path = "./xgboost_3class_concentration_classifier.pkl"
    
    if not check_file_exists(model_path):
        print("먼저 train_model.py를 실행하여 모델을 학습시키세요.")
        return
    
    print(f"\n=== XGBoost 모델 특징 중요도 분석 ===")
    print("=" * 60)
    
    try:
        # 모델 로드
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        # 특징 중요도 추출
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # 중요도별 정렬
            feature_importance = list(zip(feature_columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("특징 중요도 순위 (상위 15개):")
            print("-" * 50)
            for i, (feature, importance) in enumerate(feature_importance[:15]):
                percentage = importance * 100
                bar = "█" * int(percentage / 2)  # 시각적 바
                print(f"{i+1:2d}. {feature:20s}: {importance:.4f} ({percentage:5.2f}%) {bar}")
            
            # 집중도 판단에 핵심적인 특징들 분석
            print(f"\n 집중도 판단 핵심 특징 해석:")
            print("-" * 50)
            
            interpretations = {
                'gaze_jitter': '시선이 덜 떨릴수록 집중',
                'saccade_frequency': '급속 안구운동이 적을수록 집중',
                'head_pitch_std': '머리 움직임이 안정적일수록 집중',
                'gaze_y_mean': '특정 Y좌표(화면 중앙)를 볼 때 집중',
                'l_EAR_mean': '정상적인 깜빡임 패턴일 때 집중',
                'gaze_stability': '시선이 안정적일수록 집중',
                'head_stability': '머리가 안정적일수록 집중',
                'fixation_duration': '오래 고정해서 볼수록 집중',
                'gaze_direction_prob': '정면을 볼수록 집중'
            }
            
            for feature, importance in feature_importance[:10]:
                if feature in interpretations:
                    print(f"  • {feature:20}: {interpretations[feature]} (중요도: {importance:.3f})")
            
            return feature_importance
            
    except Exception as e:
        print(f"모델 분석 중 오류: {e}")
        return None

def analyze_decision_boundaries():
    """실제 모델의 결정 경계 분석"""
    csv_path = "./processed_data/json_features_3class_dataset.csv"
    model_path = "./xgboost_3class_concentration_classifier.pkl"
    
    if not check_file_exists(csv_path) or not check_file_exists(model_path):
        return
    
    print(f"\n=== 모델 결정 경계 분석 ===")
    print("=" * 60)
    
    try:
        # 데이터와 모델 로드
        df = pd.read_csv(csv_path)
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        
        # 특징 추출
        X = df[feature_columns].fillna(0).values
        y = df['label_3class'].values
        
        # 정규화
        X_scaled = scaler.transform(X)
        
        # 예측 확률
        probabilities = model.predict_proba(X_scaled)
        predictions = model.predict(X_scaled)
        
        # 클래스별 평균 확신도 분석
        print("클래스별 모델 확신도:")
        print("-" * 40)
        
        class_names = {0: '비집중', 1: '주의산만', 2: '집중'}
        
        for class_id in [0, 1, 2]:
            class_indices = (y == class_id)
            class_probs = probabilities[class_indices]
            
            # 해당 클래스로 정확히 예측된 경우의 확신도
            correct_predictions = (predictions[class_indices] == class_id)
            if np.any(correct_predictions):
                correct_probs = class_probs[correct_predictions]
                avg_confidence = np.mean(correct_probs[:, class_id])
                print(f"  {class_names[class_id]:>6}: 평균 확신도 {avg_confidence:.3f}")
        
        # 혼동되기 쉬운 경계 사례 분석
        print(f"\n 모델이 혼동하기 쉬운 경계 사례:")
        print("-" * 50)
        
        for i in range(len(probabilities)):
            probs = probabilities[i]
            true_class = y[i]
            pred_class = predictions[i]
            
            # 확률이 비슷한 경우 (불확실한 예측)
            max_prob = np.max(probs)
            second_max_prob = np.sort(probs)[-2]
            
            if max_prob - second_max_prob < 0.2:  # 차이가 0.2 미만인 애매한 경우
                print(f"  샘플 {i}: 실제={class_names[true_class]}, "
                      f"예측={class_names[pred_class]}, "
                      f"확률=[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]")
                
                # 처음 5개만 출력
                if sum(1 for j in range(i) if probabilities[j].max() - np.sort(probabilities[j])[-2] < 0.2) >= 5:
                    break
        
    except Exception as e:
        print(f"결정 경계 분석 중 오류: {e}")

def generate_concentration_rules():
    """집중도 판단 규칙 생성"""
    print(f"\n=== 실시간 집중도 판단 규칙 ===")
    print("=" * 60)
    
    print("모델이 학습한 집중 상태 판단 기준:")
    print("-" * 50)
    
    rules = [
        "1. 시선 떨림(gaze_jitter) < 20 픽셀",
        "2. 급속 안구운동(saccade_frequency) < 0.1 회/프레임",
        "3. 머리 움직임 변동성(head_pitch_std) < 2.0도",
        "4. 시선이 화면 중앙 근처(gaze_y_mean ≈ 540)",
        "5. 정상적인 깜빡임 패턴(l_EAR_mean ≈ 0.3)",
        "6. 높은 시선 안정성(gaze_stability > 0.7)",
        "7. 높은 머리 안정성(head_stability > 0.7)",
        "8. 긴 고정 응시 시간(fixation_duration > 10 프레임)",
        "9. 높은 정면 응시 확률(gaze_direction_prob > 0.8)"
    ]
    
    for rule in rules:
        print(f"  {rule}")
    
    print(f"\n 실시간 개선 제안:")
    print("-" * 30)
    print("  • 화면 중앙(±100픽셀) 응시 시 집중 보너스")
    print("  • 3초 이상 고정 응시 시 집중 확률 증가")  
    print("  • 머리가 15도 이내 각도일 때 집중 가산점")
    print("  • 급속한 시선 이동 시 주의산만 판정")

def main():
    """메인 실행 함수"""
    print("XGBoost 집중도 모델 분석 도구")
    print("=" * 60)
    print("이 도구는 모델이 어떻게 집중도를 판단하는지 분석합니다.\n")
    
    while True:
        print("분석 메뉴:")
        print("1. 학습 데이터 패턴 분석")
        print("2. 모델 특징 중요도 분석") 
        print("3. 모델 결정 경계 분석")
        print("4. 집중도 판단 규칙 생성")
        print("5. 전체 분석 실행")
        print("0. 종료")
        
        choice = input("\n선택하세요 (0-5): ").strip()
        
        if choice == '1':
            thresholds = analyze_dataset_patterns()
        elif choice == '2':
            importance = analyze_model_importance()
        elif choice == '3':
            analyze_decision_boundaries()
        elif choice == '4':
            generate_concentration_rules()
        elif choice == '5':
            print("\n 전체 분석을 시작합니다...")
            thresholds = analyze_dataset_patterns()
            importance = analyze_model_importance()
            analyze_decision_boundaries()
            generate_concentration_rules()
            print("\n 전체 분석이 완료되었습니다!")
        elif choice == '0':
            print("분석 도구를 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 0-5 사이의 숫자를 입력하세요.")
        
        input("\nEnter를 눌러 계속...")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
