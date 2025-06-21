"""
TextGrad 없이 GPT-4o 시간 읽기 테스트
"""

import os
import json
from gpt4o_time_reader import GPT4oTimeReader
from evaluation_system import SeparateEvaluationSystem

def main():
    print("=" * 50)
    print("GPT-4o Clock Reading Test")
    print("=" * 50)
    
    # API 키 설정
    api_key = os.getenv('OPENAI_API_KEY')
    
    # 메타데이터 로드
    with open('dataset/metadata.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 처음 20개 샘플로 테스트
    test_samples = dataset[:20]
    image_paths = [os.path.join("dataset", sample['filename']) for sample in test_samples]
    
    print(f"Testing GPT-4o with {len(test_samples)} samples...")
    print("Sample images:")
    for i, sample in enumerate(test_samples):
        print(f"  {i+1}. {sample['filename']} - {sample['time_string']} ({sample['clock_type']})")
    
    # GPT-4o 시간 읽기
    reader = GPT4oTimeReader(api_key)
    
    print("\nReading times with GPT-4o...")
    predictions = reader.batch_read_times(image_paths)
    
    # 결과 출력
    print("\n" + "=" * 30)
    print("RESULTS")
    print("=" * 30)
    
    for i, (pred, truth) in enumerate(zip(predictions, test_samples)):
        print(f"\n{i+1}. {truth['filename']} ({truth['clock_type']})")
        print(f"   실제: {truth['hour']:02d}:{truth['minute']:02d}")
        print(f"   예측: {pred.get('hour', 'N/A'):02d}:{pred.get('minute', 'N/A'):02d}")
        print(f"   신뢰도: {pred.get('confidence', 'N/A')}")
        
        # 정확도 체크
        hour_correct = pred.get('hour') == truth['hour']
        minute_correct = pred.get('minute') == truth['minute']
        exact_match = hour_correct and minute_correct
        
        print(f"   시간 {'✓' if hour_correct else '✗'} 분 {'✓' if minute_correct else '✗'} 전체 {'✓' if exact_match else '✗'}")
    
    # 평가
    evaluator = SeparateEvaluationSystem()
    evaluation = evaluator.comprehensive_evaluation(predictions, test_samples)
    
    print("\n" + "=" * 30)
    print("EVALUATION SUMMARY")
    print("=" * 30)
    
    print(f"시간 정확도: {evaluation['hour_metrics']['accuracy']:.1%}")
    print(f"분 정확도: {evaluation['minute_metrics']['accuracy']:.1%}")
    print(f"전체 매칭: {evaluation['combined_metrics']['exact_match_accuracy']:.1%}")
    
    # 시계 타입별
    print(f"\n시계 타입별 성능:")
    for clock_type, metrics in evaluation['by_clock_type'].items():
        if metrics['sample_count'] > 0:
            print(f"  {clock_type}: {metrics['combined_metrics']['exact_match_accuracy']:.1%} ({metrics['sample_count']}개)")
    
    # 결과 저장 (numpy 배열 제거)
    safe_evaluation = {}
    for key, value in evaluation.items():
        if key == 'hour_metrics' or key == 'minute_metrics':
            safe_value = {}
            for sub_key, sub_value in value.items():
                if sub_key not in ['confusion_matrix', 'error_distribution']:
                    safe_value[sub_key] = sub_value
            safe_evaluation[key] = safe_value
        elif key != 'by_clock_type':
            safe_evaluation[key] = value
        else:
            safe_clock_types = {}
            for clock_type, metrics in value.items():
                safe_metrics = {}
                for metric_key, metric_value in metrics.items():
                    if metric_key != 'hour_metrics' and metric_key != 'minute_metrics':
                        safe_metrics[metric_key] = metric_value
                    else:
                        safe_sub_metrics = {}
                        for sub_key, sub_value in metric_value.items():
                            if sub_key not in ['confusion_matrix', 'error_distribution']:
                                safe_sub_metrics[sub_key] = sub_value
                        safe_metrics[metric_key] = safe_sub_metrics
                safe_clock_types[clock_type] = safe_metrics
            safe_evaluation[key] = safe_clock_types
    
    with open('gpt4o_test_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'predictions': predictions,
            'ground_truth': test_samples,
            'evaluation': safe_evaluation
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과가 'gpt4o_test_results.json'에 저장되었습니다.")

if __name__ == "__main__":
    main()