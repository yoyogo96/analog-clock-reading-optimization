"""
TextGrad 없이 수동 프롬프트 최적화
GPT-4o를 이용한 반복적 프롬프트 개선
"""

import os
import json
import random
from typing import List, Dict, Tuple
from gpt4o_time_reader import GPT4oTimeReader
from evaluation_system import SeparateEvaluationSystem
import openai

class ManualPromptOptimizer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.time_reader = GPT4oTimeReader(api_key)
        self.evaluator = SeparateEvaluationSystem()
        
        # 초기 프롬프트들
        self.initial_prompts = [
            """이 아날로그 시계 이미지를 보고 정확한 시간을 읽어주세요.

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
- JSON 형식으로만 답변하세요""",

            """아날로그 시계를 정확히 읽어주세요.

분석 방법:
1. 짧고 굵은 바늘(시침): 시간을 나타냄
2. 길고 얇은 바늘(분침): 분을 나타냄
3. 분침이 가리키는 숫자 × 5 = 분
4. 시침은 정시와 다음 시 사이에 위치

응답 형식:
{
    "hour": 시간(0-23),
    "minute": 분(0-59),
    "confidence": 확신도(0.0-1.0)
}

JSON으로만 답변하세요.""",

            """Clock Reading Instructions:

Step 1: Identify the hands
- Short, thick hand = HOUR hand
- Long, thin hand = MINUTE hand

Step 2: Read the minute hand first
- Find which number the long hand points to
- Multiply by 5 to get minutes (12=60=0, 1=5, 2=10, 3=15, etc.)

Step 3: Read the hour hand
- The short hand shows the hour
- If it's between two numbers, use the smaller number
- Convert to 24-hour format if needed

Response format:
{
    "hour": hour(0-23),
    "minute": minute(0-59),
    "confidence": confidence(0.0-1.0)
}

Respond only in JSON format."""
        ]
    
    def evaluate_prompt(self, prompt: str, test_data: List[Dict], max_samples: int = 15) -> Tuple[float, Dict]:
        """프롬프트 평가"""
        # 랜덤 샘플 선택
        test_samples = random.sample(test_data, min(max_samples, len(test_data)))
        image_paths = [os.path.join("dataset", sample['filename']) for sample in test_samples]
        
        print(f"Testing prompt with {len(test_samples)} samples...")
        
        # 예측 수행
        predictions = self.time_reader.batch_read_times(image_paths, prompt)
        
        # 평가
        evaluation = self.evaluator.comprehensive_evaluation(predictions, test_samples)
        
        return evaluation['combined_metrics']['exact_match_accuracy'], evaluation
    
    def generate_improved_prompt(self, current_prompt: str, evaluation_results: Dict, failed_examples: List[Dict]) -> str:
        """GPT-4o를 이용한 프롬프트 개선"""
        
        # 실패 사례 분석
        error_analysis = f"""
현재 성능:
- 시간 정확도: {evaluation_results['hour_metrics']['accuracy']:.1%}
- 분 정확도: {evaluation_results['minute_metrics']['accuracy']:.1%}
- 전체 매칭: {evaluation_results['combined_metrics']['exact_match_accuracy']:.1%}
- 평균 시간 오차: {evaluation_results['hour_metrics']['mean_absolute_error']:.1f}시간
- 평균 분 오차: {evaluation_results['minute_metrics']['mean_absolute_error']:.1f}분

실패 사례들:
"""
        
        for i, example in enumerate(failed_examples[:5]):  # 최대 5개 사례
            error_analysis += f"- 실제: {example['true_hour']:02d}:{example['true_minute']:02d}, 예측: {example['pred_hour']:02d}:{example['pred_minute']:02d}\n"
        
        improvement_prompt = f"""당신은 프롬프트 엔지니어링 전문가입니다. 아날로그 시계 읽기 프롬프트를 개선해야 합니다.

현재 프롬프트:
{current_prompt}

{error_analysis}

문제점 분석:
1. 시침과 분침 구분이 어려움
2. 분 계산 오류 (분침 위치 × 5)
3. 시간 계산 오류 (시침이 두 숫자 사이에 있을 때)
4. 24시간 형식 변환 문제

개선된 프롬프트를 작성해주세요. 다음 요구사항을 만족해야 합니다:
1. 시침과 분침을 명확히 구분하는 방법 제시
2. 단계별 읽기 방법 명시
3. 일반적인 실수 방지 방법 포함
4. JSON 응답 형식 유지
5. 더 구체적이고 정확한 지침 제공

개선된 프롬프트만 출력하세요:"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": improvement_prompt}],
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    def collect_failed_examples(self, predictions: List[Dict], ground_truth: List[Dict]) -> List[Dict]:
        """실패 사례 수집"""
        failed_examples = []
        
        for pred, truth in zip(predictions, ground_truth):
            pred_hour = pred.get('hour', -1)
            pred_minute = pred.get('minute', -1)
            true_hour = truth['hour']
            true_minute = truth['minute']
            
            if pred_hour != true_hour or pred_minute != true_minute:
                failed_examples.append({
                    'filename': truth['filename'],
                    'true_hour': true_hour,
                    'true_minute': true_minute,
                    'pred_hour': pred_hour,
                    'pred_minute': pred_minute,
                    'confidence': pred.get('confidence', 0)
                })
        
        return failed_examples
    
    def optimize_prompt(self, dataset: List[Dict], num_iterations: int = 3) -> str:
        """프롬프트 최적화 메인 로직"""
        print("=" * 60)
        print("🚀 아날로그 시계 프롬프트 최적화 시작!")
        print("=" * 60)
        
        # 데이터 분할 (70% 훈련, 30% 검증)
        random.shuffle(dataset)
        split_idx = int(len(dataset) * 0.7)
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]
        
        print(f"훈련 데이터: {len(train_data)}개, 검증 데이터: {len(val_data)}개")
        
        best_prompt = None
        best_score = 0.0
        optimization_history = []
        
        # 초기 프롬프트들 평가
        print(f"\n📊 초기 프롬프트 평가...")
        for i, prompt in enumerate(self.initial_prompts):
            print(f"\n--- 초기 프롬프트 {i+1} 테스트 ---")
            print(f"프롬프트:\n{prompt}\n")
            score, evaluation = self.evaluate_prompt(prompt, val_data, 10)
            
            print(f"성능: {score:.1%} (시간: {evaluation['hour_metrics']['accuracy']:.1%}, 분: {evaluation['minute_metrics']['accuracy']:.1%})")
            
            if score > best_score:
                best_score = score
                best_prompt = prompt
                print("🎯 새로운 최고 성능!")
        
        print(f"\n🏆 최고 초기 프롬프트 성능: {best_score:.1%}")
        
        # 반복적 개선
        current_prompt = best_prompt
        
        for iteration in range(num_iterations):
            print(f"\n{'='*50}")
            print(f"🔄 최적화 반복 {iteration + 1}/{num_iterations}")
            print(f"{'='*50}")
            
            # 현재 프롬프트로 훈련 데이터 평가
            train_score, train_evaluation = self.evaluate_prompt(current_prompt, train_data, 20)
            
            # 실패 사례 수집
            train_samples = random.sample(train_data, 20)
            train_image_paths = [os.path.join("dataset", s['filename']) for s in train_samples]
            train_predictions = self.time_reader.batch_read_times(train_image_paths, current_prompt)
            failed_examples = self.collect_failed_examples(train_predictions, train_samples)
            
            print(f"훈련 성능: {train_score:.1%}")
            print(f"실패 사례: {len(failed_examples)}개")
            
            # 프롬프트 개선
            print("🛠️  프롬프트 개선 중...")
            try:
                improved_prompt = self.generate_improved_prompt(current_prompt, train_evaluation, failed_examples)
                
                print(f"\n📝 개선된 프롬프트:\n{'-'*50}\n{improved_prompt}\n{'-'*50}")
                
                # 개선된 프롬프트 검증
                print("✅ 개선된 프롬프트 검증 중...")
                new_score, new_evaluation = self.evaluate_prompt(improved_prompt, val_data, 15)
                
                print(f"개선 후 성능: {new_score:.1%} (변화: {new_score - best_score:+.1%})")
                
                if new_score > best_score:
                    best_score = new_score
                    best_prompt = improved_prompt
                    current_prompt = improved_prompt
                    print("🎉 성능 개선 성공!")
                else:
                    print("📈 성능 개선 없음, 기존 프롬프트 유지")
                
                optimization_history.append({
                    'iteration': iteration + 1,
                    'train_score': train_score,
                    'val_score': new_score,
                    'improvement': new_score - best_score,
                    'prompt': improved_prompt
                })
                
            except Exception as e:
                print(f"❌ 프롬프트 개선 실패: {e}")
                break
        
        # 최종 결과
        print(f"\n{'='*60}")
        print("🎯 최적화 완료!")
        print(f"{'='*60}")
        print(f"최종 성능: {best_score:.1%}")
        
        # 최종 평가
        final_score, final_evaluation = self.evaluate_prompt(best_prompt, val_data, len(val_data))
        
        print(f"최종 검증 성능:")
        print(f"  전체 매칭: {final_evaluation['combined_metrics']['exact_match_accuracy']:.1%}")
        print(f"  시간 정확도: {final_evaluation['hour_metrics']['accuracy']:.1%}")
        print(f"  분 정확도: {final_evaluation['minute_metrics']['accuracy']:.1%}")
        
        # 결과 저장
        with open('optimization_history.json', 'w', encoding='utf-8') as f:
            json.dump(optimization_history, f, ensure_ascii=False, indent=2)
        
        with open('optimized_prompt.txt', 'w', encoding='utf-8') as f:
            f.write(best_prompt)
        
        print(f"\n📁 결과 저장:")
        print(f"  - optimization_history.json")
        print(f"  - optimized_prompt.txt")
        
        return best_prompt

def main():
    api_key = os.getenv('OPENAI_API_KEY')
    
    # 데이터셋 로드
    with open('dataset/metadata.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 최적화 실행
    optimizer = ManualPromptOptimizer(api_key)
    optimized_prompt = optimizer.optimize_prompt(dataset, num_iterations=3)
    
    print(f"\n🎉 최적화 완료!")
    print(f"최적화된 프롬프트:")
    print("-" * 40)
    print(optimized_prompt)

if __name__ == "__main__":
    main()