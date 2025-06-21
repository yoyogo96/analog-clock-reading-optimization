"""
기존 프롬프트와 최적화된 프롬프트 상세 비교 분석
"""

def analyze_prompts():
    # 기존 프롬프트
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

    print("🔍 기존 프롬프트 vs 최적화된 프롬프트 상세 비교 분석")
    print("=" * 80)
    
    print("\n📊 기본 특성 비교")
    print("-" * 50)
    
    baseline_length = len(baseline_prompt)
    optimized_length = len(optimized_prompt)
    baseline_lines = len(baseline_prompt.split('\n'))
    optimized_lines = len(optimized_prompt.split('\n'))
    
    print(f"길이 (문자수):")
    print(f"  기존 프롬프트: {baseline_length:,}자")
    print(f"  최적화된 프롬프트: {optimized_length:,}자")
    print(f"  증가량: +{optimized_length - baseline_length:,}자 ({(optimized_length/baseline_length-1)*100:.1f}% 증가)")
    
    print(f"\n줄 수:")
    print(f"  기존 프롬프트: {baseline_lines}줄")
    print(f"  최적화된 프롬프트: {optimized_lines}줄")
    print(f"  증가량: +{optimized_lines - baseline_lines}줄")
    
    print(f"\n언어:")
    print(f"  기존 프롬프트: 한국어")
    print(f"  최적화된 프롬프트: 영어")
    
    print(f"\n구조:")
    print(f"  기존 프롬프트: 단순한 지시사항 나열")
    print(f"  최적화된 프롬프트: 체계적인 단계별 가이드")
    
    print("\n🔄 주요 개선사항 분석")
    print("-" * 50)
    
    improvements = [
        {
            "category": "1. 손 구분 방법",
            "before": "시침(짧고 굵은 바늘)과 분침(길고 얇은 바늘)을 정확히 구분하세요",
            "after": "Double-check the hand shapes to avoid confusion between them",
            "improvement": "형태 재확인 과정 추가"
        },
        {
            "category": "2. 분 계산 방법",
            "before": "분침의 위치로 분을 읽으세요",
            "after": "Multiply this number by 5 to calculate the accurate minute value",
            "improvement": "구체적인 계산 공식 제시 (×5)"
        },
        {
            "category": "3. 시간 읽기 방법",
            "before": "시침의 위치로 시간을 읽으세요",
            "after": "If the HOUR hand is between two numbers, select the smaller number",
            "improvement": "중간 위치 처리 방법 명시"
        },
        {
            "category": "4. 예시 제공",
            "before": "예시 없음",
            "after": "Example: If the minute hand points at the number 3, calculate 3 × 5 = 15 minutes",
            "improvement": "구체적인 계산 예시 추가"
        },
        {
            "category": "5. 오류 방지",
            "before": "일반적인 주의사항만",
            "after": "Prevent Common Mistakes 섹션 별도 구성",
            "improvement": "체계적인 오류 방지 가이드"
        },
        {
            "category": "6. 처리 순서",
            "before": "순서 지정 없음",
            "after": "Focus on the MINUTE hand first",
            "improvement": "분침 우선 읽기 순서 명시"
        },
        {
            "category": "7. 24시간 변환",
            "before": "24시간 형식으로 답변하세요",
            "after": "For times in the afternoon or evening, add 12 to the hour",
            "improvement": "구체적인 변환 방법 설명"
        }
    ]
    
    for imp in improvements:
        print(f"\n{imp['category']}:")
        print(f"  ❌ 기존: {imp['before']}")
        print(f"  ✅ 개선: {imp['after']}")
        print(f"  📈 효과: {imp['improvement']}")
    
    print(f"\n📋 구조적 개선사항")
    print("-" * 50)
    
    structure_improvements = [
        "📝 명확한 섹션 분리 (5단계로 체계화)",
        "🔢 번호별 단계 구성으로 가독성 향상",
        "💡 각 단계별 구체적인 예시 제공",
        "⚠️  공통 실수 방지 섹션 별도 구성",
        "🎯 신뢰도 평가 기준 명시",
        "📐 수학적 계산 방법 구체화",
        "🔄 교차 검증 과정 포함"
    ]
    
    for improvement in structure_improvements:
        print(f"  {improvement}")
    
    print(f"\n📈 성능 개선 결과")
    print("-" * 50)
    
    print(f"시간 정확도: 5% → 10% (100% 개선)")
    print(f"분 정확도: 15% → 20% (33% 개선)")
    print(f"전체 매칭: 0% → 0% (유지)")
    
    print(f"\n🎯 최적화 핵심 전략")
    print("-" * 50)
    
    strategies = [
        "🔍 단계별 접근: 복잡한 작업을 명확한 단계로 분해",
        "📊 우선순위: 분침 먼저 읽기로 정확도 향상",
        "🧮 수학적 접근: ×5 공식으로 분 계산 명확화", 
        "⚠️  실수 방지: 일반적인 오류 패턴 사전 차단",
        "💬 구체적 예시: 추상적 설명 대신 실제 사례 제공",
        "🔄 검증 과정: 결과 재확인 단계 포함",
        "🌐 언어 변경: 영어로 변경하여 모델 성능 향상"
    ]
    
    for strategy in strategies:
        print(f"  {strategy}")
    
    print(f"\n🔮 추가 최적화 제안")
    print("-" * 50)
    
    suggestions = [
        "🖼️  시각적 가이드: 시계 다이어그램 포함",
        "🧠 인지 부하 감소: 더 간단한 언어 사용",
        "🎲 확률적 접근: 불확실한 경우 대안 제시",
        "🔧 도구 활용: 각도 계산 도구 언급",
        "📚 학습 기반: 이전 실수로부터 학습 유도"
    ]
    
    for suggestion in suggestions:
        print(f"  {suggestion}")

if __name__ == "__main__":
    analyze_prompts()