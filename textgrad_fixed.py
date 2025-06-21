"""
Python 3.8 호환 TextGrad 대체 구현
"""

import openai
import json
import os
import random
from typing import List, Dict, Any, Optional
from gpt4o_time_reader import GPT4oTimeReader
from evaluation_system import SeparateEvaluationSystem

class Variable:
    """TextGrad Variable 대체 클래스"""
    def __init__(self, value: str, requires_grad: bool = False, role_description: str = ""):
        self.value = value
        self.requires_grad = requires_grad
        self.role_description = role_description
        self.gradient = None
    
    def backward(self, feedback: str):
        """역전파 시뮬레이션 - GPT-4o로 프롬프트 개선"""
        if not self.requires_grad:
            return
        
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        improvement_prompt = f"""You are a prompt engineering expert. Improve the following prompt based on the feedback.

Current prompt:
{self.value}

Feedback:
{feedback}

Please provide an improved version that addresses the issues mentioned in the feedback. Focus on making the instructions clearer and more specific for analog clock reading.

Improved prompt:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": improvement_prompt}],
                max_tokens=800,
                temperature=0.7
            )
            
            self.value = response.choices[0].message.content.strip()
            print(f"✨ 프롬프트가 개선되었습니다!")
            
        except Exception as e:
            print(f"❌ 프롬프트 개선 실패: {e}")

class TextGradOptimizer:
    """TextGrad 스타일 최적화기"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        os.environ['OPENAI_API_KEY'] = api_key
        self.time_reader = GPT4oTimeReader(api_key)
        self.evaluator = SeparateEvaluationSystem()
        
        # 초기 프롬프트들
        self.initial_prompts = [
            """Analyze this analog clock image and determine the exact time.

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
- Respond only in JSON format""",

            """Read this analog clock carefully following these steps:

Step 1: Identify the hands
- HOUR hand: Short and thick
- MINUTE hand: Long and thin

Step 2: Read the MINUTE hand
- Find which number the long hand points to
- Calculate: number × 5 = minutes
- Examples: points to 3 = 15 minutes, points to 6 = 30 minutes

Step 3: Read the HOUR hand  
- The short hand indicates the hour
- If between two numbers, choose the smaller number
- Example: between 2 and 3 = hour is 2

Step 4: Format response
{
    "hour": hour(0-23),
    "minute": minute(0-59),
    "confidence": confidence(0.0-1.0)
}

JSON only.""",

            """ANALOG CLOCK READING PROTOCOL

🔍 HAND IDENTIFICATION:
- Hour hand: SHORT and THICK
- Minute hand: LONG and THIN

📐 MINUTE CALCULATION:
- Locate the long (minute) hand
- Identify the number it points to (1-12)
- Formula: Number × 5 = Minutes
- Special cases: 12 = 0 minutes

🕐 HOUR CALCULATION:
- Locate the short (hour) hand
- Read the number it has passed or is on
- If between numbers, use the lower number
- Account for hand movement: hour hand moves gradually

⚠️ COMMON MISTAKES TO AVOID:
- Don't confuse hour and minute hands
- Don't use the number the hour hand points to directly
- Don't ignore that hour hand moves between hours

📊 OUTPUT FORMAT:
{
    "hour": hour(0-23),
    "minute": minute(0-59),
    "confidence": confidence(0.0-1.0)
}

Return JSON only."""
        ]
    
    def evaluate_prompt(self, prompt: str, test_data: List[Dict], max_samples: int = 15) -> tuple:
        """프롬프트 평가"""
        test_samples = random.sample(test_data, min(max_samples, len(test_data)))
        image_paths = [os.path.join("dataset", sample['filename']) for sample in test_samples]
        
        predictions = self.time_reader.batch_read_times(image_paths, prompt)
        evaluation = self.evaluator.comprehensive_evaluation(predictions, test_samples)
        
        return evaluation['combined_metrics']['exact_match_accuracy'], evaluation, predictions, test_samples
    
    def create_loss_feedback(self, evaluation: Dict, predictions: List[Dict], ground_truth: List[Dict]) -> str:
        """손실 기반 피드백 생성"""
        
        # 오류 분석
        hour_errors = []
        minute_errors = []
        error_examples = []
        
        for pred, truth in zip(predictions, ground_truth):
            pred_hour = pred.get('hour', -1)
            pred_minute = pred.get('minute', -1)
            true_hour = truth['hour']
            true_minute = truth['minute']
            
            if pred_hour != true_hour or pred_minute != true_minute:
                hour_error = abs(pred_hour - true_hour) if pred_hour >= 0 else 12
                minute_error = abs(pred_minute - true_minute) if pred_minute >= 0 else 30
                
                hour_errors.append(hour_error)
                minute_errors.append(minute_error)
                
                error_examples.append(f"True: {true_hour:02d}:{true_minute:02d}, Predicted: {pred_hour:02d}:{pred_minute:02d}")
        
        # 피드백 생성
        feedback = f"""Current performance analysis:

ACCURACY METRICS:
- Hour accuracy: {evaluation['hour_metrics']['accuracy']:.1%}
- Minute accuracy: {evaluation['minute_metrics']['accuracy']:.1%}
- Exact match: {evaluation['combined_metrics']['exact_match_accuracy']:.1%}

ERROR ANALYSIS:
- Average hour error: {sum(hour_errors)/len(hour_errors) if hour_errors else 0:.1f} hours
- Average minute error: {sum(minute_errors)/len(minute_errors) if minute_errors else 0:.1f} minutes

FAILED EXAMPLES:
{chr(10).join(error_examples[:5])}

ISSUES TO ADDRESS:
1. Hour hand vs minute hand confusion
2. Incorrect minute calculation (should be position × 5)
3. Hour reading when hand is between numbers
4. 24-hour format conversion errors

IMPROVEMENT NEEDED:
- Clearer hand identification instructions
- More explicit minute calculation steps
- Better hour reading guidance
- Emphasize common mistake prevention"""

        return feedback
    
    def optimize(self, dataset: List[Dict], num_iterations: int = 3, samples_per_iter: int = 15):
        """TextGrad 스타일 최적화"""
        
        print("=" * 60)
        print("🚀 TextGrad 스타일 프롬프트 최적화 시작!")
        print("=" * 60)
        
        # 데이터 분할
        random.shuffle(dataset)
        split_idx = int(len(dataset) * 0.7)
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]
        
        print(f"훈련: {len(train_data)}개, 검증: {len(val_data)}개")
        
        # 초기 프롬프트 평가
        best_prompt_var = None
        best_score = 0.0
        
        print("\n📊 초기 프롬프트 평가...")
        for i, prompt in enumerate(self.initial_prompts):
            print(f"\n--- 프롬프트 {i+1} ---")
            print(f"{prompt}\n")
            
            score, eval_result, _, _ = self.evaluate_prompt(prompt, val_data, 10)
            print(f"성능: {score:.1%}")
            
            if score > best_score:
                best_score = score
                best_prompt_var = Variable(prompt, requires_grad=True, role_description="analog clock reading prompt")
                print("🎯 새 최고 성능!")
        
        # 모든 프롬프트가 0%인 경우 첫 번째 프롬프트를 기본으로 사용
        if best_prompt_var is None:
            best_prompt_var = Variable(self.initial_prompts[0], requires_grad=True, role_description="analog clock reading prompt")
            print("⚠️  모든 초기 프롬프트가 0% 성능, 첫 번째 프롬프트로 최적화 시작")
        
        print(f"\n🏆 최고 초기 성능: {best_score:.1%}")
        
        # 최적화 반복
        optimization_history = []
        
        for iteration in range(num_iterations):
            print(f"\n{'='*50}")
            print(f"🔄 최적화 반복 {iteration + 1}/{num_iterations}")
            print(f"{'='*50}")
            
            # 현재 프롬프트 평가
            train_score, train_eval, train_preds, train_gt = self.evaluate_prompt(
                best_prompt_var.value, train_data, samples_per_iter
            )
            
            print(f"현재 훈련 성능: {train_score:.1%}")
            
            # 손실 피드백 생성
            feedback = self.create_loss_feedback(train_eval, train_preds, train_gt)
            print(f"피드백 생성 완료")
            
            # 역전파 수행 (프롬프트 개선)
            print("🛠️  역전파 수행 중...")
            old_prompt = best_prompt_var.value
            best_prompt_var.backward(feedback)
            
            if best_prompt_var.value != old_prompt:
                print(f"\n📝 개선된 프롬프트:\n{'-'*50}")
                print(best_prompt_var.value)
                print('-'*50)
                
                # 검증
                new_score, new_eval, _, _ = self.evaluate_prompt(best_prompt_var.value, val_data, 15)
                improvement = new_score - best_score
                
                print(f"\n검증 성능: {new_score:.1%} (변화: {improvement:+.1%})")
                
                if new_score > best_score:
                    best_score = new_score
                    print("🎉 성능 개선!")
                else:
                    print("📈 성능 유지")
                
                optimization_history.append({
                    'iteration': iteration + 1,
                    'old_score': train_score,
                    'new_score': new_score,
                    'improvement': improvement,
                    'prompt': best_prompt_var.value
                })
            else:
                print("프롬프트 변화 없음")
        
        # 최종 결과
        print(f"\n{'='*60}")
        print("🎯 최적화 완료!")
        print(f"{'='*60}")
        
        final_score, final_eval, _, _ = self.evaluate_prompt(best_prompt_var.value, val_data, len(val_data))
        
        print(f"최종 성능:")
        print(f"  전체 매칭: {final_eval['combined_metrics']['exact_match_accuracy']:.1%}")
        print(f"  시간 정확도: {final_eval['hour_metrics']['accuracy']:.1%}")
        print(f"  분 정확도: {final_eval['minute_metrics']['accuracy']:.1%}")
        
        # 결과 저장
        with open('textgrad_optimization_history.json', 'w', encoding='utf-8') as f:
            json.dump(optimization_history, f, ensure_ascii=False, indent=2)
        
        with open('textgrad_optimized_prompt.txt', 'w', encoding='utf-8') as f:
            f.write(best_prompt_var.value)
        
        print(f"\n📁 결과 저장 완료")
        
        return best_prompt_var.value

def main():
    api_key = os.getenv('OPENAI_API_KEY')
    
    # 데이터셋 로드
    with open('dataset/metadata.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # TextGrad 스타일 최적화
    optimizer = TextGradOptimizer(api_key)
    optimized_prompt = optimizer.optimize(dataset, num_iterations=3, samples_per_iter=15)
    
    print(f"\n🎉 TextGrad 스타일 최적화 완료!")

if __name__ == "__main__":
    main()