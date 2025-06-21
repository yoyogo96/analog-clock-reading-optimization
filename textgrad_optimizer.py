"""
TextGrad를 사용한 프롬프트 최적화
"""

import textgrad as tg
import json
import os
import numpy as np
from typing import List, Dict, Tuple
from gpt4o_time_reader import GPT4oTimeReader
import random

class TimeReadingOptimizer:
    def __init__(self, api_key: str = None):
        # TextGrad 엔진 설정
        tg.set_backward_engine("gpt-4o")
        
        self.time_reader = GPT4oTimeReader(api_key)
        
        # 초기 프롬프트들
        self.initial_prompts = [
            """이 시계 이미지를 보고 정확한 시간을 읽어주세요.

응답 형식:
{
    "hour": 시간(0-23),
    "minute": 분(0-59),
    "confidence": 확신도(0.0-1.0)
}

주의사항:
- 아날로그 시계의 경우 시침과 분침의 위치를 정확히 판단하세요
- 디지털 시계의 경우 숫자를 정확히 읽으세요
- 시간은 24시간 형식으로 답변하세요
- JSON 형식으로만 답변하세요""",
            
            """시계 이미지를 분석하여 현재 시간을 정확히 판독해주세요.

반드시 다음 JSON 형식으로 답변하세요:
{
    "hour": 시간값(0-23 사이 정수),
    "minute": 분값(0-59 사이 정수),
    "confidence": 확신도(0.0-1.0 사이 실수)
}

분석 방법:
- 아날로그 시계: 시침(짧고 두꺼운 바늘)과 분침(길고 얇은 바늘)의 각도를 정확히 측정
- 디지털 시계: 표시된 숫자를 정확히 인식
- 한글 시계: 텍스트 내용을 정확히 해석

24시간 형식으로 답변하고, JSON 외의 다른 텍스트는 포함하지 마세요.""",
            
            """Clock Time Reading Task

Analyze the clock image and determine the exact time shown.

Output Format (JSON only):
{
    "hour": integer (0-23),
    "minute": integer (0-59),
    "confidence": float (0.0-1.0)
}

Analysis Guidelines:
- Analog clocks: Identify hour hand (short, thick) and minute hand (long, thin) positions
- Digital clocks: Read displayed numbers accurately
- Text clocks: Interpret written time expressions
- Use 24-hour format
- Provide only JSON response, no additional text"""
        ]
    
    def create_loss_function(self, predictions: List[Dict], ground_truth: List[Dict]) -> tg.Variable:
        """손실 함수 생성"""
        total_loss = 0.0
        valid_predictions = 0
        
        for pred, truth in zip(predictions, ground_truth):
            if pred.get('hour', -1) >= 0 and pred.get('minute', -1) >= 0:
                hour_error = abs(pred['hour'] - truth['hour'])
                minute_error = abs(pred['minute'] - truth['minute'])
                
                # 시간 오차는 더 큰 가중치 적용
                loss = hour_error * 2.0 + minute_error * 1.0
                total_loss += loss
                valid_predictions += 1
        
        if valid_predictions > 0:
            avg_loss = total_loss / valid_predictions
        else:
            avg_loss = 100.0  # 높은 페널티
        
        return tg.Variable(avg_loss, requires_grad=True, role_description="time reading loss")
    
    def evaluate_prompt(self, prompt: str, test_data: List[Dict], max_samples: int = 20) -> Tuple[float, List[Dict]]:
        """프롬프트 평가"""
        # 테스트 샘플 선택
        test_samples = random.sample(test_data, min(max_samples, len(test_data)))
        image_paths = [os.path.join("dataset", sample['filename']) for sample in test_samples]
        
        # 예측 수행
        predictions = self.time_reader.batch_read_times(image_paths, prompt)
        
        # 평가 지표 계산
        evaluation = self.time_reader.evaluate_results(predictions, test_samples)
        
        return evaluation['exact_match_accuracy'], predictions
    
    def optimize_prompt(self, train_data: List[Dict], val_data: List[Dict], 
                       num_iterations: int = 5, samples_per_iter: int = 10) -> str:
        """프롬프트 최적화"""
        best_prompt = self.initial_prompts[0]
        best_score = 0.0
        optimization_history = []
        
        # 각 초기 프롬프트 평가
        print("Evaluating initial prompts...")
        for i, prompt in enumerate(self.initial_prompts):
            score, _ = self.evaluate_prompt(prompt, val_data, samples_per_iter)
            print(f"Initial prompt {i+1} score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_prompt = prompt
        
        print(f"Best initial prompt score: {best_score:.3f}")
        
        # TextGrad 변수 생성
        prompt_var = tg.Variable(best_prompt, requires_grad=True, 
                                role_description="time reading prompt")
        
        # 최적화 반복
        for iteration in range(num_iterations):
            print(f"\nOptimization iteration {iteration + 1}/{num_iterations}")
            
            # 현재 프롬프트로 예측
            current_prompt = prompt_var.value
            score, predictions = self.evaluate_prompt(current_prompt, train_data, samples_per_iter)
            
            print(f"Current score: {score:.3f}")
            
            # 손실 계산 (정확도를 손실로 변환)
            loss_value = 1.0 - score
            loss = tg.Variable(loss_value, requires_grad=True)
            
            # 그래디언트 계산을 위한 피드백 생성
            feedback = f"""Current prompt achieved {score:.1%} accuracy on time reading task.
            
Key issues observed:
- Hour detection accuracy needs improvement
- Minute precision could be better
- Confidence calibration may be off

Improve the prompt to:
1. Better guide hour/minute distinction in analog clocks
2. Improve number recognition in digital clocks
3. Handle edge cases more robustly
4. Maintain JSON format consistency

Current prompt:
{current_prompt}

Generate an improved version that addresses these issues."""
            
            # 역전파를 통한 프롬프트 업데이트
            try:
                loss.backward(feedback)
                
                # 새로운 프롬프트 검증
                new_score, _ = self.evaluate_prompt(prompt_var.value, val_data, samples_per_iter)
                print(f"New prompt score: {new_score:.3f}")
                
                if new_score > best_score:
                    best_score = new_score
                    best_prompt = prompt_var.value
                    print("✓ Improvement found!")
                else:
                    print("- No improvement, keeping previous best")
                    prompt_var.value = best_prompt
                
            except Exception as e:
                print(f"Optimization error: {e}")
                break
            
            optimization_history.append({
                'iteration': iteration + 1,
                'score': score,
                'best_score': best_score,
                'prompt': current_prompt
            })
        
        # 최적화 결과 저장
        with open('optimization_history.json', 'w', encoding='utf-8') as f:
            json.dump(optimization_history, f, ensure_ascii=False, indent=2)
        
        return best_prompt
    
    def run_optimization(self, dataset_path: str = "dataset/metadata.json"):
        """전체 최적화 프로세스 실행"""
        # 데이터셋 로드
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 데이터 분할 (70% 훈련, 30% 검증)
        random.shuffle(dataset)
        split_idx = int(len(dataset) * 0.7)
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
        
        # 프롬프트 최적화
        optimized_prompt = self.optimize_prompt(train_data, val_data)
        
        # 최종 평가
        print("\n=== Final Evaluation ===")
        final_score, final_predictions = self.evaluate_prompt(optimized_prompt, val_data, len(val_data))
        print(f"Final optimized prompt score: {final_score:.3f}")
        
        # 최적화된 프롬프트 저장
        with open('optimized_prompt.txt', 'w', encoding='utf-8') as f:
            f.write(optimized_prompt)
        
        return optimized_prompt, final_score

if __name__ == "__main__":
    optimizer = TimeReadingOptimizer()
    
    if os.path.exists("dataset/metadata.json"):
        optimized_prompt, final_score = optimizer.run_optimization()
        print(f"\nOptimization completed!")
        print(f"Final score: {final_score:.3f}")
        print(f"Optimized prompt saved to 'optimized_prompt.txt'")
    else:
        print("Dataset not found. Please run dataset_generator.py first.")