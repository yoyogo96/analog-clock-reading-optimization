"""
최적화된 프롬프트 테스트
"""

import os
import json
from gpt4o_time_reader import GPT4oTimeReader
from evaluation_system import SeparateEvaluationSystem

def main():
    api_key = os.getenv('OPENAI_API_KEY')
    
    # 최적화된 프롬프트
    optimized_prompt = """Examine the provided analog clock image and accurately determine the current time. Follow the detailed instructions to ensure precision in your time reading.

**Response Format:**
```json
{
    "hour": hour(0-23),
    "minute": minute(0-59),
    "confidence": confidence(0.0-1.0)
}
```

**Instructions:**

1. **Identify and Differentiate Clock Hands:**
   - The **short, thick hand** is the **HOUR** hand.
   - The **long, thin hand** is the **MINUTE** hand.
   - Double-check the hand shapes to avoid confusion between them.

2. **Calculate Minutes:**
   - Focus on the MINUTE hand first, as it provides a more precise indicator of time.
   - Determine the number it points to on the clock face.
   - Multiply this number by 5 to calculate the accurate minute value.
     - **Example:** If the minute hand points at the number 3, calculate 3 × 5 = 15 minutes.
   - Ensure that the calculated minute falls within the range of 0 to 59.

3. **Determine Hours:**
   - Observe the positioning of the HOUR hand.
   - If the HOUR hand is exactly on a number, that is the hour.
   - If the HOUR hand is between two numbers, select the smaller number.
     - **Example:** If the hour hand is between 4 and 5, assign the hour as 4.
   - For times in the afternoon or evening, add 12 to the hour to convert to a 24-hour format.
     - **Example:** 3 PM is recorded as 15.
   - Consider subtle positioning and ensure the hour is correct relative to the minute hand's position.

4. **Prevent Common Mistakes:**
   - Verify that the hands are not confused with each other.
   - Confirm that minute calculations are in increments of 5.
   - Reassess the hour reading, especially if it seems inconsistent with the minute hand's position.

5. **Assign Confidence Level:**
   - Evaluate the clarity of the clock image and your certainty in the reading.
   - Assign a confidence level between 0.0 (least confident) and 1.0 (most confident).

Respond strictly in the specified JSON format, ensuring accuracy and reflection of your analysis."""

    # 기본 프롬프트
    baseline_prompt = """이 아날로그 시계 이미지를 보고 정확한 시간을 읽어주세요.

응답 형식:
{
    "hour": 시간(0-23),
    "minute": 분(0-59),
    "confidence": 확신도(0.0-1.0)
}

주의사항:
- 시침(짧고 굵은 바늘)과 분침(길고 얇은 바늘)을 정확히 구분하세요
- 시침의 위치로 시간을, 분침의 위치로 분을 읽으세요
- 시간은 24시간 형식으로 답변하세요
- JSON 형식으로만 답변하세요"""

    # 데이터셋 로드
    with open('dataset/metadata.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    test_samples = dataset[:20]  # 20개 샘플로 테스트
    image_paths = [os.path.join("dataset", sample['filename']) for sample in test_samples]
    
    reader = GPT4oTimeReader(api_key)
    evaluator = SeparateEvaluationSystem()
    
    print("🔬 최적화된 프롬프트 vs 기본 프롬프트 비교 테스트")
    print("=" * 60)
    
    # 기본 프롬프트 테스트
    print("📊 기본 프롬프트 테스트...")
    baseline_predictions = reader.batch_read_times(image_paths, baseline_prompt)
    baseline_eval = evaluator.comprehensive_evaluation(baseline_predictions, test_samples)
    
    print(f"기본 프롬프트 성능:")
    print(f"  전체 매칭: {baseline_eval['combined_metrics']['exact_match_accuracy']:.1%}")
    print(f"  시간 정확도: {baseline_eval['hour_metrics']['accuracy']:.1%}")
    print(f"  분 정확도: {baseline_eval['minute_metrics']['accuracy']:.1%}")
    
    # 최적화된 프롬프트 테스트
    print(f"\n🚀 최적화된 프롬프트 테스트...")
    optimized_predictions = reader.batch_read_times(image_paths, optimized_prompt)
    optimized_eval = evaluator.comprehensive_evaluation(optimized_predictions, test_samples)
    
    print(f"최적화된 프롬프트 성능:")
    print(f"  전체 매칭: {optimized_eval['combined_metrics']['exact_match_accuracy']:.1%}")
    print(f"  시간 정확도: {optimized_eval['hour_metrics']['accuracy']:.1%}")
    print(f"  분 정확도: {optimized_eval['minute_metrics']['accuracy']:.1%}")
    
    # 개선 효과 계산
    exact_improvement = optimized_eval['combined_metrics']['exact_match_accuracy'] - baseline_eval['combined_metrics']['exact_match_accuracy']
    hour_improvement = optimized_eval['hour_metrics']['accuracy'] - baseline_eval['hour_metrics']['accuracy']
    minute_improvement = optimized_eval['minute_metrics']['accuracy'] - baseline_eval['minute_metrics']['accuracy']
    
    print(f"\n📈 개선 효과:")
    print(f"  전체 매칭: {exact_improvement:+.1%}")
    print(f"  시간 정확도: {hour_improvement:+.1%}")
    print(f"  분 정확도: {minute_improvement:+.1%}")
    
    # 결과 저장
    comparison_result = {
        'baseline': {
            'exact_match': baseline_eval['combined_metrics']['exact_match_accuracy'],
            'hour_accuracy': baseline_eval['hour_metrics']['accuracy'],
            'minute_accuracy': baseline_eval['minute_metrics']['accuracy']
        },
        'optimized': {
            'exact_match': optimized_eval['combined_metrics']['exact_match_accuracy'],
            'hour_accuracy': optimized_eval['hour_metrics']['accuracy'],
            'minute_accuracy': optimized_eval['minute_metrics']['accuracy']
        },
        'improvement': {
            'exact_match': exact_improvement,
            'hour_accuracy': hour_improvement,
            'minute_accuracy': minute_improvement
        },
        'optimized_prompt': optimized_prompt
    }
    
    with open('prompt_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 결과가 'prompt_comparison_results.json'에 저장되었습니다.")
    
    if exact_improvement > 0:
        print(f"\n🎉 프롬프트 최적화 성공! {exact_improvement:.1%} 개선")
    else:
        print(f"\n🤔 추가 최적화가 필요할 수 있습니다.")

if __name__ == "__main__":
    main()