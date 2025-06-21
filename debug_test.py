"""
GPT-4o 디버깅 테스트
"""

import os
import json
import openai
import base64

def test_single_image():
    # API 키 설정
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    
    # 이미지 인코딩
    image_path = "dataset/clock_0000_digital.png"
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    prompt = """이 시계 이미지를 보고 정확한 시간을 읽어주세요.

응답 형식:
{
    "hour": 시간(0-23),
    "minute": 분(0-59),
    "confidence": 확신도(0.0-1.0)
}

주의사항:
- 디지털 시계의 경우 숫자를 정확히 읽으세요
- 시간은 24시간 형식으로 답변하세요
- JSON 형식으로만 답변하세요"""
    
    print(f"Testing image: {image_path}")
    print(f"Prompt: {prompt}")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
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
        
        print(f"\nRaw response: {response.choices[0].message.content}")
        
        # JSON 파싱 시도 (코드 블록 제거)
        try:
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:]  # ```json 제거
            if content.endswith('```'):
                content = content[:-3]  # ``` 제거
            content = content.strip()
            
            result = json.loads(content)
            print(f"Parsed result: {result}")
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print("Raw content:", response.choices[0].message.content)
            
    except Exception as e:
        print(f"API call failed: {e}")

if __name__ == "__main__":
    test_single_image()