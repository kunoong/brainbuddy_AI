import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml_classifier import ConcentrationClassifier

def train_xgboost_concentration_model():
    """XGBoost 기반 집중도 모델 학습 (불균형 데이터 보완)"""
    
    print("=== XGBoost 기반 3클래스 집중도 모델 학습 ===")
    print("불균형 데이터 보완 기능 포함\n")
    
    # 1. 데이터 로드
    print("1단계: 처리된 데이터셋 로드...")
    csv_path = "./processed_data/json_features_3class_dataset.csv"
    
    try:
        df = pd.read_csv(csv_path)
        print(f"데이터 로드 완료: {len(df)}개 샘플")
    except FileNotFoundError:
        print(f"데이터 파일을 찾을 수 없습니다: {csv_path}")
        print("먼저 data_processor.py를 실행하여 데이터를 처리하세요.")
        return
    
    # 2. 분류기 초기화 및 특징 준비
    print("\n2단계: 특징 준비...")
    classifier = ConcentrationClassifier()
    X, y, feature_columns = classifier.prepare_features(df)
    
    print(f"특징 벡터 차원: {X.shape[1]}")
    print(f"총 샘플 수: {len(X)}")
    
    # 3. 개인별 데이터 분할
    print("\n3단계: 개인별 데이터 분할...")
    unique_persons = df['metaid'].unique()
    np.random.seed(42)
    train_persons = np.random.choice(unique_persons, 
                                   size=int(len(unique_persons) * 0.8), 
                                   replace=False)
    
    train_mask = df['metaid'].isin(train_persons)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[~train_mask]
    y_test = y[~train_mask]
    
    print(f"학습 데이터: {len(X_train)}개 ({len(train_persons)}명)")
    print(f"테스트 데이터: {len(X_test)}개 ({len(unique_persons) - len(train_persons)}명)")
    
    # 4. 고급 데이터 전처리 (불균형 보완)
    print("\n4단계: 불균형 데이터 보완...")
    
    # 리샘플링 방법 선택
    print("사용할 리샘플링 방법을 선택하세요:")
    print("1. auto (자동 선택)")
    print("2. smote (기본 SMOTE)")
    print("3. borderline (Borderline SMOTE)")
    print("4. adasyn (ADASYN)")
    print("5. smote_tomek (SMOTE + Tomek)")
    
    choice = input("선택 (1-5, Enter=자동): ").strip()
    method_map = {
        '1': 'auto', '2': 'smote', '3': 'borderline', 
        '4': 'adasyn', '5': 'smote_tomek', '': 'auto'
    }
    method = method_map.get(choice, 'auto')
    
    X_train_processed, y_train_processed = classifier.prepare_data_advanced(X_train, y_train, method)
    
    # 5. XGBoost 모델 학습
    print("\n5단계: XGBoost 모델 학습...")
    
    tune_params = input("하이퍼파라미터 튜닝을 수행하시겠습니까? (y/N): ").lower().strip()
    tune_hyperparams = tune_params in ['y', 'yes']
    
    if tune_hyperparams:
        print("하이퍼파라미터 튜닝은 시간이 오래 걸릴 수 있습니다 (10-30분)")
    
    # XGBoost 학습
    print("XGBoost 학습 중...")
    training_results = classifier.train_xgboost_simple(X_train_processed, y_train_processed)

        
    print(f"XGBoost CV 점수 (F1-Macro): {training_results['cv_mean']:.4f} (+/- {training_results['cv_std']*2:.4f})")
    
    # 6. 모델 평가
    print("\n6단계: 모델 성능 평가...")
    evaluation = classifier.evaluate_advanced(X_test, y_test, feature_columns)
    
    # 7. 결과 출력
    print(f"\n=== 최종 평가 결과 ===")
    print(f"모델: {training_results['model_type']}")
    print(f"정확도: {evaluation['accuracy']:.4f}")
    print(f"F1 Score (Macro): {evaluation['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {evaluation['f1_weighted']:.4f}")
    
    # 클래스별 성능
    print(f"\n클래스별 성능:")
    report = evaluation['classification_report']
    for class_name in ['비집중', '주의산만', '집중']:
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            print(f"  {class_name:>6}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    # 혼동 행렬
    print(f"\n혼동 행렬:")
    cm = evaluation['confusion_matrix']
    print("         예측")
    print("실제    비집중  주의산만  집중")
    class_names_list = ['비집중', '주의산만', '집중']
    for i, true_class in enumerate(class_names_list):
        row = f"{true_class:>6}  "
        for j in range(3):
            row += f"{cm[i][j]:>6}  "
        print(row)
    
    # 8. 모델 저장
    print(f"\n7단계: 모델 저장...")
    model_path = "./xgboost_3class_concentration_classifier.pkl"
    classifier.save_model(model_path, feature_columns)
    
    print(f"\nXGBoost 모델 학습 완료!")
    print(f"모델 저장 경로: {model_path}")
    print("inference.py로 실시간 테스트를 진행할 수 있습니다.")

if __name__ == "__main__":
    train_xgboost_concentration_model()
