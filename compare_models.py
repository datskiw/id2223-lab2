#!/usr/bin/env python3
"""
Simple model comparison script.
Measures: 1) Inference speed, 2) Response quality (basic check)
Uses local models for fast testing.
"""
from llama_cpp import Llama
import time
import os

# Test models
MODELS = [
    {
        "name": "Llama 3.2 1B",
        "path": "models/llama3-1B-finetome-q8_0.gguf",
        "format": "llama3"
    },
    {
        "name": "Llama 3.2 3B",
        "path": "models/llama3-3B-finetome-q8_0.gguf",
        "format": "llama3"
    },
    {
        "name": "Qwen 2 0.5B",
        "path": "models/qwen2-0.5b-finetome-q8_0.gguf",
        "format": "chatml"
    }
]

# Test questions
TEST_QUESTIONS = [
    "What is the weather today?",
    "Will it rain tomorrow?",
    "What's the weather in Paris?"
]

def build_prompt(message, format_type="llama3"):
    """Build prompt in correct format."""
    system_prompt = "You are a helpful weather assistant. Weather data: Temp 15°C, overcast, 20% rain chance."
    
    if format_type == "chatml":
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    else:  # llama3
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def test_model(model_config):
    """Test a single model and return metrics."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_config['name']}")
    print(f"{'='*60}")
    
    # Load model from local file
    model_path = model_config['path']
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    print(f"Loading model from {model_path}...")
    try:
        llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, n_batch=128)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Test inference speed
    print("\nTesting inference speed...")
    speeds = []
    stop_tokens = ["<|im_end|>", "<|end_of_text|>"] if model_config['format'] == "chatml" else ["<|eot_id|>", "<|end_of_text|>"]
    
    test_prompt = build_prompt("What is the weather today?", model_config['format'])
    
    for i in range(3):  # Run 3 times for average
        start = time.time()
        output = llm(test_prompt, max_tokens=50, temperature=0.7, stop=stop_tokens)
        elapsed = time.time() - start
        speeds.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    avg_speed = sum(speeds) / len(speeds)
    print(f"  Average speed: {avg_speed:.2f}s")
    
    # Test response quality (simple check: does it mention weather data?)
    print("\nTesting response quality...")
    quality_scores = []
    
    for question in TEST_QUESTIONS:
        prompt = build_prompt(question, model_config['format'])
        output = llm(prompt, max_tokens=100, temperature=0.7, stop=stop_tokens)
        response = output["choices"][0]["text"].strip()
        
        # Simple quality check: does response contain relevant info?
        has_temp = any(word in response.lower() for word in ["temp", "temperature", "15", "°c", "celsius"])
        has_condition = any(word in response.lower() for word in ["overcast", "cloudy", "weather", "rain"])
        score = 1 if (has_temp or has_condition) else 0
        quality_scores.append(score)
        
        print(f"  Q: {question}")
        print(f"    A: {response[:80]}...")
        print(f"    Score: {score}/1")
    
    avg_quality = sum(quality_scores) / len(quality_scores) * 100
    print(f"  Average quality score: {avg_quality:.0f}%")
    
    return {
        "name": model_config['name'],
        "speed": avg_speed,
        "quality": avg_quality,
        "size": model_config.get('size', 'N/A')
    }

def main():
    """Run comparison."""
    print("Model Comparison Tool")
    print("=" * 60)
    print("Metrics: 1) Inference Speed (seconds), 2) Response Quality (%)")
    print("=" * 60)
    
    results = []
    for model in MODELS:
        result = test_model(model)
        if result:
            results.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Speed (s)':<12} {'Quality (%)':<12}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['name']:<20} {r['speed']:<12.2f} {r['quality']:<12.0f}")
    
    # Find best
    fastest = min(results, key=lambda x: x['speed'])
    best_quality = max(results, key=lambda x: x['quality'])
    
    print("\nBest Performance:")
    print(f"  Fastest: {fastest['name']} ({fastest['speed']:.2f}s)")
    print(f"  Best Quality: {best_quality['name']} ({best_quality['quality']:.0f}%)")

if __name__ == "__main__":
    main()

