"""
빠른 성능 분석 - 프롬프트 최적화 효과 확인
"""

import os
import json
from gpt4o_time_reader import GPT4oTimeReader
from evaluation_system import SeparateEvaluationSystem

def analyze_prompt_performance():
    api_key = os.getenv('OPENAI_API_KEY')
    
    # 기본 프롬프트 (초기)
    baseline_prompt = """Analyze this analog clock image and determine the exact time.

Response format:
{
    "hour": hour(0-23),
    "minute": minute(0-59), 
    "confidence": confidence(0.0-1.0)
}

Instructions:
- Short thick hand = HOUR hand
- Long thin hand = MINUTE hand
- Read minute hand first: multiply the number it points to by 5
- Read hour hand: if between two numbers, use the smaller one
- Convert to 24-hour format
- Respond only in JSON format"""

    # 개선된 프롬프트 (최적화 후)
    improved_prompt = """Analyze this analog clock image and determine the exact time.

Response format:
```json
{
    "hour": hour(0-23),
    "minute": minute(0-59),
    "confidence": confidence(0.0-1.0)
}
```

Instructions:

1. **Hand Identification:**
   - **HOUR Hand:** This is the short and thick hand.
   - **MINUTE Hand:** This is the long and thin hand.

2. **Reading Methodology:**

   - **Minute Calculation:**
     - Identify the number the MINUTE hand is pointing to on the clock face.
     - Use the formula: Minute = Number × 5.
     - Example: If the MINUTE hand points to 3, the calculation is 3 × 5 = 15 minutes.

   - **Hour Calculation:**
     - Identify the number the HOUR hand is pointing to or is between.
     - If the HOUR hand is directly on or between two numbers, choose the smaller number.
     - Adjust for 24-hour format if necessary (e.g., 1 PM becomes 13 in 24-hour format).

3. **Common Mistakes to Avoid:**
   - Ensure the correct identification of the HOUR and MINUTE hands based on their size.
   - Always use the multiply-by-5 formula for the minute calculation.
   - When the HOUR hand is between two numbers, always select the smaller number.

4. **Confidence Level:**
   - Assign a confidence score based on the clarity of the hand positions and your certainty in the reading. Use a scale from 0.0 (uncertain) to 1.0 (completely certain).

5. **JSON Output:**
   - Ensure the response is formatted as valid JSON, following the structure exactly as shown above.

Use these refined instructions to accurately determine and format the time from the analog clock image."""

    # 데이터셋 로드
    with open('dataset/metadata.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 작은 샘플로 빠른 테스트 (10개만)
    test_samples = dataset[:10]
    image_paths = [os.path.join("dataset", sample['filename']) for sample in test_samples]
    
    reader = GPT4oTimeReader(api_key)
    evaluator = SeparateEvaluationSystem()
    
    print("🔍 프롬프트 최적화 성능 분석")
    print("=" * 50)
    print(f"테스트 샘플: {len(test_samples)}개")
    print()
    
    # 1. 기본 프롬프트 테스트
    print("📊 1. 기본 프롬프트 (TextGrad 시작 전)")
    print("프롬프트 미리보기:", baseline_prompt[:100] + "...")
    print()
    
    baseline_predictions = reader.batch_read_times(image_paths, baseline_prompt)
    baseline_eval = evaluator.comprehensive_evaluation(baseline_predictions, test_samples)
    
    print("📈 기본 프롬프트 성능:")
    print(f"  🎯 전체 정확한 매칭: {baseline_eval['combined_metrics']['exact_match_accuracy']:.1%}")
    print(f"  🕐 시간 정확도: {baseline_eval['hour_metrics']['accuracy']:.1%}")
    print(f"  ⏱️  분 정확도: {baseline_eval['minute_metrics']['accuracy']:.1%}")
    print(f"  📊 시간 평균 오차: {baseline_eval['hour_metrics']['mean_absolute_error']:.1f}시간")
    print(f"  📊 분 평균 오차: {baseline_eval['minute_metrics']['mean_absolute_error']:.1f}분")
    print()
    
    # 2. 개선된 프롬프트 테스트
    print("🚀 2. 개선된 프롬프트 (TextGrad 최적화 후)")
    print("프롬프트 미리보기:", improved_prompt[:100] + "...")
    print()
    
    improved_predictions = reader.batch_read_times(image_paths, improved_prompt)
    improved_eval = evaluator.comprehensive_evaluation(improved_predictions, test_samples)
    
    print("📈 개선된 프롬프트 성능:")
    print(f"  🎯 전체 정확한 매칭: {improved_eval['combined_metrics']['exact_match_accuracy']:.1%}")
    print(f"  🕐 시간 정확도: {improved_eval['hour_metrics']['accuracy']:.1%}")
    print(f"  ⏱️  분 정확도: {improved_eval['minute_metrics']['accuracy']:.1%}")
    print(f"  📊 시간 평균 오차: {improved_eval['hour_metrics']['mean_absolute_error']:.1f}시간")
    print(f"  📊 분 평균 오차: {improved_eval['minute_metrics']['mean_absolute_error']:.1f}분")
    print()
    
    # 3. 개선 효과 분석
    exact_improvement = improved_eval['combined_metrics']['exact_match_accuracy'] - baseline_eval['combined_metrics']['exact_match_accuracy']
    hour_improvement = improved_eval['hour_metrics']['accuracy'] - baseline_eval['hour_metrics']['accuracy']
    minute_improvement = improved_eval['minute_metrics']['accuracy'] - baseline_eval['minute_metrics']['accuracy']
    
    print("🎉 TextGrad 최적화 효과:")
    print(f"  🎯 전체 매칭 개선: {exact_improvement:+.1%}")
    print(f"  🕐 시간 정확도 개선: {hour_improvement:+.1%}")
    print(f"  ⏱️  분 정확도 개선: {minute_improvement:+.1%}")
    print()
    
    # 4. 실패 사례 분석
    print("🔍 실패 사례 분석:")
    failed_count = 0
    for i, (pred_base, pred_imp, truth) in enumerate(zip(baseline_predictions, improved_predictions, test_samples)):
        base_correct = pred_base.get('hour') == truth['hour'] and pred_base.get('minute') == truth['minute']
        imp_correct = pred_imp.get('hour') == truth['hour'] and pred_imp.get('minute') == truth['minute']
        
        if not base_correct and not imp_correct:
            failed_count += 1
            print(f"  📸 {truth['filename']}: 실제 {truth['hour']:02d}:{truth['minute']:02d}")
            print(f"     기본: {pred_base.get('hour', '?'):02d}:{pred_base.get('minute', '?'):02d}")
            print(f"     개선: {pred_imp.get('hour', '?'):02d}:{pred_imp.get('minute', '?'):02d}")
    
    print(f"\n📊 여전히 실패한 샘플: {failed_count}/{len(test_samples)}개")
    
    # 5. 결론
    if exact_improvement > 0 or hour_improvement > 0 or minute_improvement > 0:
        print("\n✅ TextGrad 최적화 성공!")
        print("   프롬프트가 개선되어 성능이 향상되었습니다.")
    else:
        print("\n⚠️  추가 최적화 필요")
        print("   더 많은 훈련 데이터나 다른 최적화 전략이 필요할 수 있습니다.")

if __name__ == "__main__":
    analyze_prompt_performance()