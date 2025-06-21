"""
GPT-4o를 사용한 시계 시간 읽기
"""

import openai
import base64
import json
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

class GPT4oTimeReader:
    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY')
        )
        
        # 기본 프롬프트
        self.base_prompt = """이 시계 이미지를 보고 정확한 시간을 읽어주세요.

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
- JSON 형식으로만 답변하세요"""
    
    def encode_image(self, image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def read_time_from_image(self, image_path: str, prompt: Optional[str] = None) -> Dict:
        """이미지에서 시간 읽기"""
        base64_image = self.encode_image(image_path)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt or self.base_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        try:
            # JSON 응답 파싱 (코드 블록 제거)
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:]  # ```json 제거
            if content.endswith('```'):
                content = content[:-3]  # ``` 제거
            content = content.strip()
            
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            # JSON 파싱 실패시 기본값 반환
            return {
                "hour": -1,
                "minute": -1,
                "confidence": 0.0,
                "error": "Failed to parse response",
                "raw_response": response.choices[0].message.content
            }
    
    def batch_read_times(self, image_paths: List[str], prompt: Optional[str] = None) -> List[Dict]:
        """여러 이미지에서 시간 읽기"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            try:
                result = self.read_time_from_image(image_path, prompt)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "hour": -1,
                    "minute": -1,
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        return results
    
    def evaluate_results(self, results: List[Dict], ground_truth: List[Dict]) -> Dict:
        """결과 평가"""
        correct_hours = 0
        correct_minutes = 0
        correct_both = 0
        total = len(results)
        
        hour_errors = []
        minute_errors = []
        
        for result, truth in zip(results, ground_truth):
            pred_hour = result.get('hour', -1)
            pred_minute = result.get('minute', -1)
            true_hour = truth['hour']
            true_minute = truth['minute']
            
            if pred_hour == true_hour:
                correct_hours += 1
            else:
                hour_errors.append(abs(pred_hour - true_hour))
            
            if pred_minute == true_minute:
                correct_minutes += 1
            else:
                minute_errors.append(abs(pred_minute - true_minute))
            
            if pred_hour == true_hour and pred_minute == true_minute:
                correct_both += 1
        
        return {
            "total_samples": total,
            "hour_accuracy": correct_hours / total,
            "minute_accuracy": correct_minutes / total,
            "exact_match_accuracy": correct_both / total,
            "avg_hour_error": sum(hour_errors) / len(hour_errors) if hour_errors else 0,
            "avg_minute_error": sum(minute_errors) / len(minute_errors) if minute_errors else 0,
            "hour_errors": hour_errors,
            "minute_errors": minute_errors
        }

if __name__ == "__main__":
    # 테스트 실행
    reader = GPT4oTimeReader()
    
    # 샘플 이미지로 테스트 (데이터셋 생성 후 실행)
    if os.path.exists("dataset/metadata.json"):
        with open("dataset/metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 처음 10개 샘플로 테스트
        test_samples = metadata[:10]
        image_paths = [os.path.join("dataset", sample['filename']) for sample in test_samples]
        
        results = reader.batch_read_times(image_paths)
        evaluation = reader.evaluate_results(results, test_samples)
        
        print("Evaluation Results:")
        print(f"Hour Accuracy: {evaluation['hour_accuracy']:.2%}")
        print(f"Minute Accuracy: {evaluation['minute_accuracy']:.2%}")
        print(f"Exact Match Accuracy: {evaluation['exact_match_accuracy']:.2%}")
    else:
        print("Dataset not found. Please run dataset_generator.py first.")