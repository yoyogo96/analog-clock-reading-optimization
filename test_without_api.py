"""
API 키 없이 평가 시스템 테스트
"""

import json
import os
from evaluation_system import SeparateEvaluationSystem

def create_mock_predictions(ground_truth):
    """모의 예측 결과 생성 (API 키 없이 테스트용)"""
    predictions = []
    
    for i, truth in enumerate(ground_truth):
        # 다양한 성능을 시뮬레이션
        if i % 4 == 0:  # 정확한 예측
            pred = {
                'hour': truth['hour'],
                'minute': truth['minute'],
                'confidence': 0.9
            }
        elif i % 4 == 1:  # 시간만 틀림
            pred = {
                'hour': (truth['hour'] + 1) % 24,
                'minute': truth['minute'], 
                'confidence': 0.7
            }
        elif i % 4 == 2:  # 분만 틀림
            pred = {
                'hour': truth['hour'],
                'minute': (truth['minute'] + 10) % 60,
                'confidence': 0.6
            }
        else:  # 둘 다 틀림
            pred = {
                'hour': (truth['hour'] + 2) % 24,
                'minute': (truth['minute'] + 15) % 60,
                'confidence': 0.4
            }
        
        predictions.append(pred)
    
    return predictions

def main():
    print("=" * 50)
    print("Clock Reading Evaluation System Test")
    print("=" * 50)
    
    # 메타데이터 로드
    with open('dataset/metadata.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 처음 50개 샘플로 테스트
    test_samples = dataset[:50]
    
    print(f"Testing with {len(test_samples)} samples...")
    
    # 모의 예측 결과 생성
    mock_predictions = create_mock_predictions(test_samples)
    
    # 평가 시스템 실행
    evaluator = SeparateEvaluationSystem()
    evaluation = evaluator.comprehensive_evaluation(mock_predictions, test_samples)
    
    # 결과 출력
    print("\n" + "=" * 30)
    print("EVALUATION RESULTS")
    print("=" * 30)
    
    print(f"총 샘플 수: {evaluation['total_samples']}")
    print(f"시간 정확도: {evaluation['hour_metrics']['accuracy']:.2%}")
    print(f"분 정확도: {evaluation['minute_metrics']['accuracy']:.2%}")
    print(f"전체 매칭: {evaluation['combined_metrics']['exact_match_accuracy']:.2%}")
    
    print(f"\n평균 시간 오차: {evaluation['hour_metrics']['mean_absolute_error']:.2f}시간")
    print(f"평균 분 오차: {evaluation['minute_metrics']['mean_absolute_error']:.2f}분")
    print(f"5분 허용오차 정확도: {evaluation['minute_metrics']['tolerance_5min_rate']:.2%}")
    print(f"10분 허용오차 정확도: {evaluation['minute_metrics']['tolerance_10min_rate']:.2%}")
    
    # 시계 타입별 성능
    print(f"\n" + "=" * 30)
    print("시계 타입별 성능")
    print("=" * 30)
    
    for clock_type, metrics in evaluation['by_clock_type'].items():
        print(f"\n{clock_type.upper()} 시계 ({metrics['sample_count']}개):")
        print(f"  시간 정확도: {metrics['hour_metrics']['accuracy']:.2%}")
        print(f"  분 정확도: {metrics['minute_metrics']['accuracy']:.2%}")
        print(f"  전체 매칭: {metrics['combined_metrics']['exact_match_accuracy']:.2%}")
    
    # 보고서 생성
    print(f"\n" + "=" * 30)
    print("보고서 생성 중...")
    print("=" * 30)
    
    try:
        evaluator.generate_report(evaluation, "test_evaluation_report.json")
        print("✓ 평가 보고서 생성 완료: test_evaluation_report.json/txt")
    except Exception as e:
        print(f"✗ 보고서 생성 실패: {e}")
    
    try:
        evaluator.plot_results(evaluation, "test_plots")
        print("✓ 시각화 차트 생성 완료: test_plots/")
    except Exception as e:
        print(f"✗ 시각화 생성 실패: {e}")
    
    print(f"\n" + "=" * 50)
    print("평가 시스템 테스트 완료!")
    print("=" * 50)
    
    print("\n실제 GPT-4o 테스트를 위해서는 OpenAI API 키가 필요합니다.")
    print("API 키 설정 후 다음을 실행하세요:")
    print("export OPENAI_API_KEY='your-api-key'")
    print("python main_pipeline.py --samples 100 --baseline-samples 20 --final-samples 30")

if __name__ == "__main__":
    main()