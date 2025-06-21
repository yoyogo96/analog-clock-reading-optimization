"""
전체 시계 시간 읽기 파이프라인
데이터셋 생성 → GPT-4o 시간 읽기 → TextGrad 최적화 → 평가
"""

import os
import json
import argparse
from datetime import datetime
from dataset_generator import ClockDatasetGenerator
from gpt4o_time_reader import GPT4oTimeReader
from textgrad_optimizer import TimeReadingOptimizer
from evaluation_system import SeparateEvaluationSystem

class TimeReadingPipeline:
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.results = {}
    
    def step1_generate_dataset(self, num_samples: int = 500):
        """Step 1: 가상 시계 데이터셋 생성"""
        print("=" * 50)
        print("Step 1: Generating Clock Dataset")
        print("=" * 50)
        
        generator = ClockDatasetGenerator()
        dataset = generator.generate_dataset(num_samples)
        
        print(f"✓ Generated {len(dataset)} clock images")
        self.results['dataset_size'] = len(dataset)
        return dataset
    
    def step2_baseline_evaluation(self, dataset: list, sample_size: int = 50):
        """Step 2: 기본 프롬프트로 성능 평가"""
        print("\n" + "=" * 50)
        print("Step 2: Baseline Evaluation")
        print("=" * 50)
        
        reader = GPT4oTimeReader(self.openai_api_key)
        evaluator = SeparateEvaluationSystem()
        
        # 테스트 샘플 선택
        test_samples = dataset[:sample_size]
        image_paths = [os.path.join("dataset", sample['filename']) for sample in test_samples]
        
        print(f"Testing baseline with {len(test_samples)} samples...")
        predictions = reader.batch_read_times(image_paths)
        
        # 평가
        baseline_eval = evaluator.comprehensive_evaluation(predictions, test_samples)
        
        print(f"✓ Baseline Hour Accuracy: {baseline_eval['hour_metrics']['accuracy']:.2%}")
        print(f"✓ Baseline Minute Accuracy: {baseline_eval['minute_metrics']['accuracy']:.2%}")
        print(f"✓ Baseline Exact Match: {baseline_eval['combined_metrics']['exact_match_accuracy']:.2%}")
        
        # 보고서 저장
        evaluator.generate_report(baseline_eval, "baseline_evaluation.json")
        evaluator.plot_results(baseline_eval, "baseline_plots")
        
        self.results['baseline'] = baseline_eval
        return baseline_eval
    
    def step3_optimize_prompt(self, dataset: list):
        """Step 3: TextGrad로 프롬프트 최적화"""
        print("\n" + "=" * 50)
        print("Step 3: Prompt Optimization with TextGrad")
        print("=" * 50)
        
        optimizer = TimeReadingOptimizer(self.openai_api_key)
        
        try:
            optimized_prompt, final_score = optimizer.run_optimization()
            
            print(f"✓ Prompt optimization completed")
            print(f"✓ Optimized prompt score: {final_score:.2%}")
            
            self.results['optimization'] = {
                'final_score': final_score,
                'optimized_prompt': optimized_prompt
            }
            
            return optimized_prompt
            
        except Exception as e:
            print(f"✗ Optimization failed: {e}")
            print("Using best initial prompt as fallback...")
            
            # 최고 초기 프롬프트 선택
            best_prompt = optimizer.initial_prompts[0]
            self.results['optimization'] = {
                'final_score': 0.0,
                'optimized_prompt': best_prompt,
                'error': str(e)
            }
            
            return best_prompt
    
    def step4_final_evaluation(self, dataset: list, optimized_prompt: str, sample_size: int = 100):
        """Step 4: 최적화된 프롬프트로 최종 평가"""
        print("\n" + "=" * 50)
        print("Step 4: Final Evaluation with Optimized Prompt")
        print("=" * 50)
        
        reader = GPT4oTimeReader(self.openai_api_key)
        evaluator = SeparateEvaluationSystem()
        
        # 테스트 샘플 선택 (기본 평가와 다른 샘플 사용)
        test_samples = dataset[-sample_size:] if len(dataset) >= sample_size else dataset
        image_paths = [os.path.join("dataset", sample['filename']) for sample in test_samples]
        
        print(f"Testing optimized prompt with {len(test_samples)} samples...")
        predictions = reader.batch_read_times(image_paths, optimized_prompt)
        
        # 평가
        final_eval = evaluator.comprehensive_evaluation(predictions, test_samples)
        
        print(f"✓ Final Hour Accuracy: {final_eval['hour_metrics']['accuracy']:.2%}")
        print(f"✓ Final Minute Accuracy: {final_eval['minute_metrics']['accuracy']:.2%}")
        print(f"✓ Final Exact Match: {final_eval['combined_metrics']['exact_match_accuracy']:.2%}")
        
        # 보고서 저장
        evaluator.generate_report(final_eval, "final_evaluation.json")
        evaluator.plot_results(final_eval, "final_plots")
        
        self.results['final'] = final_eval
        return final_eval
    
    def step5_comparison_report(self):
        """Step 5: 최종 비교 보고서 생성"""
        print("\n" + "=" * 50)
        print("Step 5: Generating Comparison Report")
        print("=" * 50)
        
        if 'baseline' not in self.results or 'final' not in self.results:
            print("✗ Cannot generate comparison - missing evaluation results")
            return
        
        baseline = self.results['baseline']
        final = self.results['final']
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': self.results['dataset_size'],
            'baseline_results': {
                'hour_accuracy': baseline['hour_metrics']['accuracy'],
                'minute_accuracy': baseline['minute_metrics']['accuracy'],
                'exact_match': baseline['combined_metrics']['exact_match_accuracy']
            },
            'final_results': {
                'hour_accuracy': final['hour_metrics']['accuracy'],
                'minute_accuracy': final['minute_metrics']['accuracy'],
                'exact_match': final['combined_metrics']['exact_match_accuracy']
            },
            'improvements': {
                'hour_accuracy': final['hour_metrics']['accuracy'] - baseline['hour_metrics']['accuracy'],
                'minute_accuracy': final['minute_metrics']['accuracy'] - baseline['minute_metrics']['accuracy'],
                'exact_match': final['combined_metrics']['exact_match_accuracy'] - baseline['combined_metrics']['exact_match_accuracy']
            }
        }
        
        # 비교 보고서 저장
        with open('comparison_report.json', 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        # 텍스트 요약
        report = f"""
시계 시간 읽기 프롬프트 최적화 결과 비교
=========================================

데이터셋 크기: {comparison['dataset_size']}개

기본 프롬프트 성능:
- 시간 정확도: {comparison['baseline_results']['hour_accuracy']:.2%}
- 분 정확도: {comparison['baseline_results']['minute_accuracy']:.2%}
- 전체 매칭: {comparison['baseline_results']['exact_match']:.2%}

최적화된 프롬프트 성능:
- 시간 정확도: {comparison['final_results']['hour_accuracy']:.2%}
- 분 정확도: {comparison['final_results']['minute_accuracy']:.2%}
- 전체 매칭: {comparison['final_results']['exact_match']:.2%}

개선 효과:
- 시간 정확도: {comparison['improvements']['hour_accuracy']:+.2%}
- 분 정확도: {comparison['improvements']['minute_accuracy']:+.2%}
- 전체 매칭: {comparison['improvements']['exact_match']:+.2%}

최적화된 프롬프트:
{self.results.get('optimization', {}).get('optimized_prompt', 'N/A')}
"""
        
        with open('comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print("✓ Comparison report saved")
    
    def run_full_pipeline(self, num_samples: int = 500, 
                         baseline_samples: int = 50, 
                         final_samples: int = 100):
        """전체 파이프라인 실행"""
        print("Starting Time Reading Optimization Pipeline")
        print(f"Parameters: {num_samples} samples, baseline: {baseline_samples}, final: {final_samples}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: 데이터셋 생성
            dataset = self.step1_generate_dataset(num_samples)
            
            # Step 2: 기본 평가
            baseline_eval = self.step2_baseline_evaluation(dataset, baseline_samples)
            
            # Step 3: 프롬프트 최적화
            optimized_prompt = self.step3_optimize_prompt(dataset)
            
            # Step 4: 최종 평가
            final_eval = self.step4_final_evaluation(dataset, optimized_prompt, final_samples)
            
            # Step 5: 비교 보고서
            self.step5_comparison_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "=" * 50)
            print("Pipeline Completed Successfully!")
            print("=" * 50)
            print(f"Total duration: {duration}")
            print(f"Generated files:")
            print("- dataset/ (clock images)")
            print("- baseline_evaluation.json/txt")
            print("- optimized_prompt.txt")
            print("- final_evaluation.json/txt")
            print("- comparison_report.json/txt")
            print("- baseline_plots/ and final_plots/")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description='Clock Time Reading Optimization Pipeline')
    parser.add_argument('--samples', type=int, default=500, 
                       help='Number of clock images to generate')
    parser.add_argument('--baseline-samples', type=int, default=50,
                       help='Number of samples for baseline evaluation')
    parser.add_argument('--final-samples', type=int, default=100,
                       help='Number of samples for final evaluation')
    parser.add_argument('--api-key', type=str, 
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # API 키 확인
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key is required.")
        print("Set OPENAI_API_KEY environment variable or use --api-key argument")
        return
    
    # 파이프라인 실행
    pipeline = TimeReadingPipeline(api_key)
    success = pipeline.run_full_pipeline(
        num_samples=args.samples,
        baseline_samples=args.baseline_samples,
        final_samples=args.final_samples
    )
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")

if __name__ == "__main__":
    main()