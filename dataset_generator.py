"""
가상 시계 데이터셋 생성기
다양한 시간 표현 방식으로 시계 데이터를 생성합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
import os
from typing import List, Tuple, Dict
import random

class ClockDatasetGenerator:
    def __init__(self, output_dir: str = "dataset"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_analog_clock(self, hour: int, minute: int, size: int = 256) -> Image.Image:
        """아날로그 시계 이미지 생성"""
        img = Image.new('RGB', (size, size), 'white')
        draw = ImageDraw.Draw(img)
        
        # 원형 시계 테두리
        center = size // 2
        radius = center - 20
        draw.ellipse([center-radius, center-radius, center+radius, center+radius], 
                    outline='black', width=3)
        
        # 시간 표시 (12, 3, 6, 9)
        for i in range(0, 12, 3):
            angle = np.radians(i * 30 - 90)
            x = center + (radius - 15) * np.cos(angle)
            y = center + (radius - 15) * np.sin(angle)
            number = 12 if i == 0 else i
            draw.text((x-5, y-5), str(number), fill='black')
        
        # 시침 (짧고 굵음)
        hour_angle = np.radians((hour % 12) * 30 + minute * 0.5 - 90)
        hour_length = radius * 0.5
        hour_x = center + hour_length * np.cos(hour_angle)
        hour_y = center + hour_length * np.sin(hour_angle)
        draw.line([center, center, hour_x, hour_y], fill='black', width=4)
        
        # 분침 (길고 얇음)
        minute_angle = np.radians(minute * 6 - 90)
        minute_length = radius * 0.8
        minute_x = center + minute_length * np.cos(minute_angle)
        minute_y = center + minute_length * np.sin(minute_angle)
        draw.line([center, center, minute_x, minute_y], fill='black', width=2)
        
        # 중심점
        draw.ellipse([center-3, center-3, center+3, center+3], fill='black')
        
        return img
    
    def generate_digital_clock(self, hour: int, minute: int, size: int = 256) -> Image.Image:
        """디지털 시계 이미지 생성"""
        img = Image.new('RGB', (size, size), 'black')
        draw = ImageDraw.Draw(img)
        
        # 시간 텍스트 (24시간 형식과 12시간 형식 랜덤)
        if random.choice([True, False]):
            time_str = f"{hour:02d}:{minute:02d}"
        else:
            hour_12 = hour % 12
            if hour_12 == 0:
                hour_12 = 12
            am_pm = "AM" if hour < 12 else "PM"
            time_str = f"{hour_12}:{minute:02d} {am_pm}"
        
        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), time_str)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 중앙에 텍스트 배치
        x = (size - text_width) // 2
        y = (size - text_height) // 2
        
        draw.text((x, y), time_str, fill='green')
        
        return img
    
    def generate_word_clock(self, hour: int, minute: int, size: int = 256) -> Image.Image:
        """문자로 된 시계 이미지 생성"""
        img = Image.new('RGB', (size, size), 'white')
        draw = ImageDraw.Draw(img)
        
        # 시간을 문자로 변환
        hour_words = {
            1: "한시", 2: "두시", 3: "세시", 4: "네시", 5: "다섯시", 6: "여섯시",
            7: "일곱시", 8: "여덟시", 9: "아홉시", 10: "열시", 11: "열한시", 12: "열두시",
            13: "한시", 14: "두시", 15: "세시", 16: "네시", 17: "다섯시", 18: "여섯시",
            19: "일곱시", 20: "여덟시", 21: "아홉시", 22: "열시", 23: "열한시", 0: "열두시"
        }
        
        if minute == 0:
            time_text = f"{hour_words[hour]} 정각"
        elif minute == 30:
            time_text = f"{hour_words[hour]} 반"
        else:
            time_text = f"{hour_words[hour]} {minute}분"
        
        # 텍스트 중앙 배치
        bbox = draw.textbbox((0, 0), time_text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size - text_width) // 2
        y = (size - text_height) // 2
        
        draw.text((x, y), time_text, fill='black')
        
        return img
    
    def generate_dataset(self, num_samples: int = 1000) -> List[Dict]:
        """데이터셋 생성 (아날로그 시계만)"""
        dataset = []
        
        for i in range(num_samples):
            # 랜덤 시간 생성
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            
            # 아날로그 시계만 생성
            clock_type = 'analog'
            img = self.generate_analog_clock(hour, minute)
            
            # 이미지 저장
            filename = f"clock_{i:04d}_{clock_type}.png"
            filepath = os.path.join(self.output_dir, filename)
            img.save(filepath)
            
            # 메타데이터 저장
            dataset.append({
                'filename': filename,
                'clock_type': clock_type,
                'hour': hour,
                'minute': minute,
                'time_string': f"{hour:02d}:{minute:02d}"
            })
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
        
        # 메타데이터 JSON 파일로 저장
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        return dataset

if __name__ == "__main__":
    generator = ClockDatasetGenerator()
    dataset = generator.generate_dataset(500)  # 500개 샘플 생성
    print(f"Dataset generated with {len(dataset)} samples")