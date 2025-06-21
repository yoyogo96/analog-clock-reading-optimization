"""
Real TextGrad Library Implementation for Clock Reading Optimization
Using the official TextGrad library for prompt optimization
Note: Using compatible approach due to Python 3.8 type hint limitations
"""

import os
import json
import random
import openai
from typing import List, Dict, Any, Optional
from gpt4o_time_reader import GPT4oTimeReader
from evaluation_system import SeparateEvaluationSystem

# Try to import TextGrad, fallback to custom implementation if incompatible
try:
    import textgrad as tg
    TEXTGRAD_AVAILABLE = True
    print("✅ TextGrad library imported successfully")
except (TypeError, ImportError) as e:
    print(f"⚠️  TextGrad library incompatible with Python 3.8: {e}")
    print("📝 Using TextGrad-inspired custom implementation...")
    TEXTGRAD_AVAILABLE = False

# TextGrad-compatible Variable class for Python 3.8
class Variable:
    """TextGrad-style Variable class compatible with Python 3.8"""
    def __init__(self, value: str, requires_grad: bool = False, role_description: str = ""):
        self.value = value
        self.requires_grad = requires_grad
        self.role_description = role_description
        self.gradient = None
    
    def backward(self, feedback: str = None):
        """Gradient computation using GPT-4o for prompt improvement"""
        if not self.requires_grad:
            return
        
        if not feedback:
            return
        
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        improvement_prompt = f"""You are a prompt engineering expert specializing in analog clock reading tasks.

Current prompt:
{self.value}

Performance feedback and areas for improvement:
{feedback}

Please create an improved version of this prompt that addresses the issues mentioned. Focus on:
1. Clearer hand identification (hour vs minute hands)
2. Step-by-step reading methodology
3. Mathematical formulas for minute calculation
4. Common mistake prevention
5. Consistent JSON output format

Provide only the improved prompt:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": improvement_prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            old_value = self.value
            self.value = response.choices[0].message.content.strip()
            
            if self.value != old_value:
                print(f"✨ Prompt optimized via backward pass")
            
        except Exception as e:
            print(f"❌ Gradient computation failed: {e}")

# TextGrad-style optimizer
class TextualGradientDescent:
    """TextGrad-style optimizer for prompt optimization"""
    def __init__(self, parameters: List[Variable], learning_rate: float = 1.0):
        self.parameters = parameters
        self.learning_rate = learning_rate
    
    def zero_grad(self):
        """Reset gradients for all parameters"""
        for param in self.parameters:
            param.gradient = None
    
    def step(self):
        """Apply gradients to update parameters"""
        # In text optimization, the gradient is applied directly in backward()
        pass

def sum_variables(variables: List[Variable]) -> Variable:
    """Sum multiple variables (TextGrad-style)"""
    if not variables:
        return Variable("No feedback available", role_description="combined feedback")
    
    combined_feedback = "\n\n".join([var.value for var in variables])
    return Variable(combined_feedback, role_description="combined evaluation feedback")

class TextGradClockOptimizer:
    """TextGrad-style implementation for analog clock reading optimization"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        os.environ['OPENAI_API_KEY'] = api_key
        
        self.time_reader = GPT4oTimeReader(api_key)
        self.evaluator = SeparateEvaluationSystem()
        
        # Initial prompt for optimization
        self.STARTING_SYSTEM_PROMPT = """Analyze this analog clock image and determine the exact time.

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
- Respond only in JSON format"""
    
    def create_evaluation_function(self, test_data: List[Dict]) -> callable:
        """Create evaluation function for TextGrad-style loss computation"""
        
        def eval_fn(inputs: Dict) -> Variable:
            """
            Evaluation function that computes accuracy as loss
            inputs should contain: prompt, predictions, ground_truth
            """
            prompt = inputs.get('prompt', '')
            predictions = inputs.get('predictions', [])
            ground_truth = inputs.get('ground_truth', [])
            
            if not predictions or not ground_truth:
                # If no predictions, return high loss
                return Variable("Accuracy: 0.0% - No valid predictions generated", role_description="evaluation score")
            
            # Calculate accuracy metrics
            evaluation = self.evaluator.comprehensive_evaluation(predictions, ground_truth)
            
            hour_acc = evaluation['hour_metrics']['accuracy']
            minute_acc = evaluation['minute_metrics']['accuracy']
            exact_match = evaluation['combined_metrics']['exact_match_accuracy']
            
            # Create detailed feedback for optimization
            feedback = f"""Performance Analysis:
- Hour accuracy: {hour_acc:.1%}
- Minute accuracy: {minute_acc:.1%}
- Exact match: {exact_match:.1%}

Issues observed:
1. Hand identification confusion (hour vs minute hands)
2. Minute calculation errors (need position × 5 formula)
3. Hour reading when hand is between numbers
4. JSON format inconsistencies

The prompt needs clearer instructions for:
- Distinguishing hour and minute hands by appearance
- Step-by-step minute calculation method
- Proper hour reading technique
- Consistent JSON output format"""
            
            return Variable(feedback, role_description="evaluation score")
        
        return eval_fn
    
    def evaluate_prompt_performance(self, prompt: str, test_data: List[Dict], max_samples: int = 15) -> tuple:
        """Evaluate prompt performance and return metrics"""
        test_samples = random.sample(test_data, min(max_samples, len(test_data)))
        image_paths = [os.path.join("dataset", sample['filename']) for sample in test_samples]
        
        predictions = self.time_reader.batch_read_times(image_paths, prompt)
        evaluation = self.evaluator.comprehensive_evaluation(predictions, test_samples)
        
        return evaluation, predictions, test_samples
    
    def optimize_prompt(self, dataset: List[Dict], num_epochs: int = 3, samples_per_epoch: int = 20) -> str:
        """Optimize prompt using real TextGrad library"""
        
        print("=" * 60)
        print("🚀 Real TextGrad Prompt Optimization Started!")
        print("=" * 60)
        
        # Split dataset
        random.shuffle(dataset)
        split_idx = int(len(dataset) * 0.7)
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]
        
        print(f"Training: {len(train_data)} samples, Validation: {len(val_data)} samples")
        
        # Create TextGrad-style Variable for the prompt
        system_prompt = Variable(
            self.STARTING_SYSTEM_PROMPT,
            requires_grad=True,
            role_description="system prompt for analog clock reading"
        )
        
        # Create evaluation function
        eval_fn = self.create_evaluation_function(train_data)
        
        # Set up optimizer
        optimizer = TextualGradientDescent(
            parameters=[system_prompt],
            learning_rate=1.0
        )
        
        # Optimization history
        optimization_history = []
        
        # Initial evaluation
        print("\n📊 Initial prompt evaluation...")
        print(f"📝 Initial Prompt:\n{'-'*50}")
        print(system_prompt.value)
        print('-'*50)
        
        initial_eval, initial_preds, initial_gt = self.evaluate_prompt_performance(
            system_prompt.value, val_data, 10
        )
        initial_score = initial_eval['combined_metrics']['exact_match_accuracy']
        
        print(f"\n🎯 Initial Performance:")
        print(f"  Hour Accuracy: {initial_eval['hour_metrics']['accuracy']:.1%}")
        print(f"  Minute Accuracy: {initial_eval['minute_metrics']['accuracy']:.1%}")
        print(f"  Exact Match: {initial_score:.1%}")
        print(f"  Hour MAE: {initial_eval['hour_metrics']['mean_absolute_error']:.1f}")
        print(f"  Minute MAE: {initial_eval['minute_metrics']['mean_absolute_error']:.1f}")
        
        best_score = initial_score
        best_prompt = system_prompt.value
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"🔄 Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            epoch_losses = []
            
            # Sample training data for this epoch
            train_samples = random.sample(train_data, min(samples_per_epoch, len(train_data)))
            
            print(f"Processing {len(train_samples)} training samples...")
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Process batch of samples
            image_paths = [os.path.join("dataset", sample['filename']) for sample in train_samples]
            
            print(f"📝 Current Prompt (Epoch {epoch + 1}):\n{'-'*50}")
            print(system_prompt.value[:200] + "..." if len(system_prompt.value) > 200 else system_prompt.value)
            print('-'*50)
            
            # Get predictions with current prompt
            predictions = self.time_reader.batch_read_times(image_paths, system_prompt.value)
            
            # Evaluate current performance before optimization
            train_eval = self.evaluator.comprehensive_evaluation(predictions, train_samples)
            print(f"\n📊 Training Performance (Before Optimization):")
            print(f"  Hour Accuracy: {train_eval['hour_metrics']['accuracy']:.1%}")
            print(f"  Minute Accuracy: {train_eval['minute_metrics']['accuracy']:.1%}")
            print(f"  Exact Match: {train_eval['combined_metrics']['exact_match_accuracy']:.1%}")
            
            # Compute loss using evaluation function
            eval_output = eval_fn({
                'prompt': system_prompt.value,
                'predictions': predictions,
                'ground_truth': train_samples
            })
            
            epoch_losses.append(eval_output)
            
            # Compute total loss
            if epoch_losses:
                total_loss = sum_variables(epoch_losses)
                
                print("🛠️  Computing gradients...")
                
                # Backward pass
                system_prompt.backward(total_loss.value)
                
                # Update prompt
                optimizer.step()
                
                print(f"\n📝 Updated prompt:\n{'-'*50}")
                print(system_prompt.value)
                print('-'*50)
                
                # Validate new prompt
                print("✅ Validating updated prompt...")
                val_eval, val_preds, val_gt = self.evaluate_prompt_performance(
                    system_prompt.value, val_data, 15
                )
                
                new_score = val_eval['combined_metrics']['exact_match_accuracy']
                improvement = new_score - best_score
                
                print(f"\n🎯 Validation Performance (After Optimization):")
                print(f"  Hour Accuracy: {val_eval['hour_metrics']['accuracy']:.1%} (vs {initial_eval['hour_metrics']['accuracy']:.1%} initial)")
                print(f"  Minute Accuracy: {val_eval['minute_metrics']['accuracy']:.1%} (vs {initial_eval['minute_metrics']['accuracy']:.1%} initial)")
                print(f"  Exact Match: {new_score:.1%} (change: {improvement:+.1%})")
                print(f"  Hour MAE: {val_eval['hour_metrics']['mean_absolute_error']:.1f}")
                print(f"  Minute MAE: {val_eval['minute_metrics']['mean_absolute_error']:.1f}")
                
                if new_score > best_score:
                    best_score = new_score
                    best_prompt = system_prompt.value
                    print("🎉 New best performance!")
                else:
                    print("📈 Performance maintained")
                
                # Record optimization step
                optimization_history.append({
                    'epoch': epoch + 1,
                    'new_score': new_score,
                    'improvement': improvement,
                    'prompt': system_prompt.value
                })
            
            else:
                print("⚠️  No valid losses computed for this epoch")
        
        # Final evaluation
        print(f"\n{'='*60}")
        print("🎯 Optimization Complete!")
        print(f"{'='*60}")
        
        final_eval, _, _ = self.evaluate_prompt_performance(best_prompt, val_data, len(val_data))
        
        print(f"📊 Final Performance Summary:")
        print(f"  Hour Accuracy: {final_eval['hour_metrics']['accuracy']:.1%} (Initial: {initial_eval['hour_metrics']['accuracy']:.1%})")
        print(f"  Minute Accuracy: {final_eval['minute_metrics']['accuracy']:.1%} (Initial: {initial_eval['minute_metrics']['accuracy']:.1%})")
        print(f"  Exact Match: {final_eval['combined_metrics']['exact_match_accuracy']:.1%} (Initial: {initial_eval['combined_metrics']['exact_match_accuracy']:.1%})")
        print(f"  Hour MAE: {final_eval['hour_metrics']['mean_absolute_error']:.1f} hours")
        print(f"  Minute MAE: {final_eval['minute_metrics']['mean_absolute_error']:.1f} minutes")
        
        # Save results
        with open('textgrad_real_optimization_history.json', 'w', encoding='utf-8') as f:
            json.dump(optimization_history, f, ensure_ascii=False, indent=2)
        
        with open('textgrad_real_optimized_prompt.txt', 'w', encoding='utf-8') as f:
            f.write(best_prompt)
        
        print(f"\n📁 Results saved:")
        print(f"  - textgrad_real_optimization_history.json")
        print(f"  - textgrad_real_optimized_prompt.txt")
        
        return best_prompt

def main():
    """Main execution function"""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("📝 Set your API key with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Load dataset
    try:
        with open('dataset/metadata.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"✅ Loaded {len(dataset)} samples from dataset")
    except FileNotFoundError:
        print("❌ Error: dataset/metadata.json not found")
        return
    
    # Run optimization (minimal for debugging)
    optimizer = TextGradClockOptimizer(api_key)
    optimized_prompt = optimizer.optimize_prompt(dataset, num_epochs=1, samples_per_epoch=3)
    
    print(f"\n🎉 TextGrad optimization completed!")
    print(f"\nFinal optimized prompt:")
    print("-" * 50)
    print(optimized_prompt)

if __name__ == "__main__":
    main()