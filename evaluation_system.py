"""
시간과 분을 분리한 평가 시스템
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from datetime import datetime

class SeparateEvaluationSystem:
    def __init__(self):
        self.results_history = []
    
    def calculate_hour_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """시간(hour) 관련 메트릭 계산"""
        hour_correct = 0
        hour_errors = []
        hour_confusion_matrix = np.zeros((24, 24))  # 24시간 confusion matrix
        
        for pred, truth in zip(predictions, ground_truth):
            pred_hour = pred.get('hour', -1)
            true_hour = truth['hour']
            
            if pred_hour >= 0:  # 유효한 예측
                if pred_hour == true_hour:
                    hour_correct += 1
                else:
                    error = abs(pred_hour - true_hour)
                    # 시간은 원형이므로 12시간 이상 차이나면 반대방향으로 계산
                    if error > 12:
                        error = 24 - error
                    hour_errors.append(error)
                
                # Confusion matrix 업데이트
                hour_confusion_matrix[true_hour][pred_hour] += 1
        
        total_samples = len(predictions)
        
        return {
            'accuracy': hour_correct / total_samples if total_samples > 0 else 0,
            'error_rate': len(hour_errors) / total_samples if total_samples > 0 else 0,
            'mean_absolute_error': np.mean(hour_errors) if hour_errors else 0,
            'std_error': np.std(hour_errors) if hour_errors else 0,
            'max_error': max(hour_errors) if hour_errors else 0,
            'confusion_matrix': hour_confusion_matrix,
            'error_distribution': hour_errors
        }
    
    def calculate_minute_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """분(minute) 관련 메트릭 계산"""
        minute_correct = 0
        minute_errors = []
        minute_tolerance_5 = 0  # 5분 이내 오차
        minute_tolerance_10 = 0  # 10분 이내 오차
        
        for pred, truth in zip(predictions, ground_truth):
            pred_minute = pred.get('minute', -1)
            true_minute = truth['minute']
            
            if pred_minute >= 0:  # 유효한 예측
                if pred_minute == true_minute:
                    minute_correct += 1
                else:
                    error = abs(pred_minute - true_minute)
                    # 분도 원형이므로 30분 이상 차이나면 반대방향으로 계산
                    if error > 30:
                        error = 60 - error
                    minute_errors.append(error)
                    
                    # 허용 오차 계산
                    if error <= 5:
                        minute_tolerance_5 += 1
                    if error <= 10:
                        minute_tolerance_10 += 1
        
        total_samples = len(predictions)
        
        return {
            'accuracy': minute_correct / total_samples if total_samples > 0 else 0,
            'error_rate': len(minute_errors) / total_samples if total_samples > 0 else 0,
            'mean_absolute_error': np.mean(minute_errors) if minute_errors else 0,
            'std_error': np.std(minute_errors) if minute_errors else 0,
            'max_error': max(minute_errors) if minute_errors else 0,
            'tolerance_5min_rate': (minute_correct + minute_tolerance_5) / total_samples if total_samples > 0 else 0,
            'tolerance_10min_rate': (minute_correct + minute_tolerance_10) / total_samples if total_samples > 0 else 0,
            'error_distribution': minute_errors
        }
    
    def calculate_combined_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """전체 시간 매칭 메트릭"""
        exact_match = 0
        time_errors = []  # 분 단위 총 오차
        
        for pred, truth in zip(predictions, ground_truth):
            pred_hour = pred.get('hour', -1)
            pred_minute = pred.get('minute', -1)
            true_hour = truth['hour']
            true_minute = truth['minute']
            
            if pred_hour >= 0 and pred_minute >= 0:
                if pred_hour == true_hour and pred_minute == true_minute:
                    exact_match += 1
                else:
                    # 총 시간 오차를 분 단위로 계산
                    pred_total_minutes = pred_hour * 60 + pred_minute
                    true_total_minutes = true_hour * 60 + true_minute
                    error = abs(pred_total_minutes - true_total_minutes)
                    
                    # 하루 경계 처리 (12시간 이상 차이나면 반대방향)
                    if error > 12 * 60:
                        error = 24 * 60 - error
                    
                    time_errors.append(error)
        
        total_samples = len(predictions)
        
        return {
            'exact_match_accuracy': exact_match / total_samples if total_samples > 0 else 0,
            'mean_time_error_minutes': np.mean(time_errors) if time_errors else 0,
            'std_time_error_minutes': np.std(time_errors) if time_errors else 0
        }
    
    def analyze_by_clock_type(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """시계 타입별 성능 분석"""
        clock_types = ['analog', 'digital', 'word']
        type_analysis = {}
        
        for clock_type in clock_types:
            type_predictions = []
            type_ground_truth = []
            
            for pred, truth in zip(predictions, ground_truth):
                if truth.get('clock_type') == clock_type:
                    type_predictions.append(pred)
                    type_ground_truth.append(truth)
            
            if type_predictions:
                hour_metrics = self.calculate_hour_metrics(type_predictions, type_ground_truth)
                minute_metrics = self.calculate_minute_metrics(type_predictions, type_ground_truth)
                combined_metrics = self.calculate_combined_metrics(type_predictions, type_ground_truth)
                
                type_analysis[clock_type] = {
                    'sample_count': len(type_predictions),
                    'hour_metrics': hour_metrics,
                    'minute_metrics': minute_metrics,
                    'combined_metrics': combined_metrics
                }
        
        return type_analysis
    
    def comprehensive_evaluation(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """종합 평가"""
        evaluation_result = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(predictions),
            'hour_metrics': self.calculate_hour_metrics(predictions, ground_truth),
            'minute_metrics': self.calculate_minute_metrics(predictions, ground_truth),
            'combined_metrics': self.calculate_combined_metrics(predictions, ground_truth),
            'by_clock_type': self.analyze_by_clock_type(predictions, ground_truth)
        }
        
        self.results_history.append(evaluation_result)
        return evaluation_result
    
    def generate_report(self, evaluation_result: Dict, save_path: str = "evaluation_report.json"):
        """평가 보고서 생성"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        
        # 텍스트 요약 생성
        report_text = f"""
시계 시간 읽기 성능 평가 보고서
================================

전체 샘플 수: {evaluation_result['total_samples']}

시간(Hour) 성능:
- 정확도: {evaluation_result['hour_metrics']['accuracy']:.2%}
- 평균 절대 오차: {evaluation_result['hour_metrics']['mean_absolute_error']:.2f}시간
- 최대 오차: {evaluation_result['hour_metrics']['max_error']}시간

분(Minute) 성능:
- 정확도: {evaluation_result['minute_metrics']['accuracy']:.2%}
- 평균 절대 오차: {evaluation_result['minute_metrics']['mean_absolute_error']:.2f}분
- 5분 허용 오차 정확도: {evaluation_result['minute_metrics']['tolerance_5min_rate']:.2%}
- 10분 허용 오차 정확도: {evaluation_result['minute_metrics']['tolerance_10min_rate']:.2%}

전체 시간 매칭:
- 정확한 매칭 정확도: {evaluation_result['combined_metrics']['exact_match_accuracy']:.2%}
- 평균 시간 오차: {evaluation_result['combined_metrics']['mean_time_error_minutes']:.2f}분

시계 타입별 성능:
"""
        
        for clock_type, metrics in evaluation_result['by_clock_type'].items():
            report_text += f"""
{clock_type.upper()} 시계 ({metrics['sample_count']}개 샘플):
  - 시간 정확도: {metrics['hour_metrics']['accuracy']:.2%}
  - 분 정확도: {metrics['minute_metrics']['accuracy']:.2%}
  - 전체 매칭: {metrics['combined_metrics']['exact_match_accuracy']:.2%}
"""
        
        with open(save_path.replace('.json', '.txt'), 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text
    
    def plot_results(self, evaluation_result: Dict, save_dir: str = "plots"):
        """결과 시각화"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 시간별 정확도 비교
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        clock_types = list(evaluation_result['by_clock_type'].keys())
        hour_accuracies = [evaluation_result['by_clock_type'][ct]['hour_metrics']['accuracy'] 
                          for ct in clock_types]
        minute_accuracies = [evaluation_result['by_clock_type'][ct]['minute_metrics']['accuracy'] 
                            for ct in clock_types]
        
        x = np.arange(len(clock_types))
        width = 0.35
        
        plt.bar(x - width/2, hour_accuracies, width, label='Hour Accuracy', alpha=0.8)
        plt.bar(x + width/2, minute_accuracies, width, label='Minute Accuracy', alpha=0.8)
        
        plt.xlabel('Clock Type')
        plt.ylabel('Accuracy')
        plt.title('Hour vs Minute Accuracy by Clock Type')
        plt.xticks(x, clock_types)
        plt.legend()
        plt.ylim(0, 1)
        
        # 2. 오차 분포
        plt.subplot(1, 2, 2)
        hour_errors = evaluation_result['hour_metrics']['error_distribution']
        minute_errors = evaluation_result['minute_metrics']['error_distribution']
        
        if hour_errors and minute_errors:
            plt.hist(hour_errors, bins=12, alpha=0.7, label='Hour Errors', density=True)
            plt.hist(minute_errors, bins=30, alpha=0.7, label='Minute Errors', density=True)
            plt.xlabel('Error (hours/minutes)')
            plt.ylabel('Density')
            plt.title('Error Distribution')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 시간 Confusion Matrix (24시간)
        plt.figure(figsize=(12, 10))
        hour_cm = evaluation_result['hour_metrics']['confusion_matrix']
        
        # 데이터가 있는 경우만 플롯
        if np.sum(hour_cm) > 0:
            sns.heatmap(hour_cm, annot=True, fmt='g', cmap='Blues', 
                       xticklabels=range(24), yticklabels=range(24))
            plt.xlabel('Predicted Hour')
            plt.ylabel('True Hour')
            plt.title('Hour Prediction Confusion Matrix')
            plt.savefig(os.path.join(save_dir, 'hour_confusion_matrix.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # 테스트 실행
    evaluator = SeparateEvaluationSystem()
    
    # 샘플 데이터로 테스트
    if os.path.exists("dataset/metadata.json"):
        from gpt4o_time_reader import GPT4oTimeReader
        
        reader = GPT4oTimeReader()
        
        with open("dataset/metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 처음 20개 샘플로 테스트
        test_samples = metadata[:20]
        image_paths = [os.path.join("dataset", sample['filename']) for sample in test_samples]
        
        print("Reading times from sample images...")
        predictions = reader.batch_read_times(image_paths)
        
        print("Evaluating results...")
        evaluation = evaluator.comprehensive_evaluation(predictions, test_samples)
        
        print("Generating report...")
        evaluator.generate_report(evaluation)
        
        print("Creating plots...")
        evaluator.plot_results(evaluation)
        
        print("Evaluation completed!")
    else:
        print("Dataset not found. Please run dataset_generator.py first.")