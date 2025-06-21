"""
빠른 프롬프트 최적화 (적은 샘플로)
"""

import os
import json
import random
from textgrad_fixed import TextGradOptimizer

def main():
    api_key = os.getenv('OPENAI_API_KEY')
    
    # 데이터셋 로드
    with open('dataset/metadata.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print("🚀 빠른 프롬프트 최적화 시작!")
    print(f"전체 데이터셋: {len(dataset)}개")
    
    # 적은 샘플로 빠른 최적화
    optimizer = TextGradOptimizer(api_key)
    
    # 샘플 수를 줄여서 빠르게 테스트
    optimized_prompt = optimizer.optimize(
        dataset, 
        num_iterations=2,  # 반복 횟수 줄임
        samples_per_iter=8  # 샘플 수 줄임
    )
    
    print(f"\n🎉 빠른 최적화 완료!")
    print(f"\n최종 최적화된 프롬프트:")
    print("="*60)
    print(optimized_prompt)
    print("="*60)

if __name__ == "__main__":
    main()