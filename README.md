# 🕐 Analog Clock Reading Optimization with GPT-4o and TextGrad

> Improving AI's analog clock reading capabilities through automated prompt optimization

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com/)
[![TextGrad](https://img.shields.io/badge/TextGrad-Compatible-orange.svg)](https://github.com/zou-group/textgrad)

## 🎯 Project Overview

Generative AI models struggle with reading analog clocks accurately. This project addresses this challenge through:

1. **📊 Synthetic Dataset Generation**: Creates diverse analog clock images
2. **🤖 GPT-4o Integration**: Leverages vision capabilities for time reading  
3. **🔧 Automated Prompt Optimization**: Uses TextGrad-style optimization
4. **📈 Comprehensive Evaluation**: Separate analysis for hours and minutes

## 🏆 Key Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Hour Accuracy** | 5% | 10% | **+100%** 🚀 |
| **Minute Accuracy** | 15% | 20% | **+33%** 📈 |
| **Prompt Length** | 239 chars | 2,021 chars | **+746%** |

## ✨ Features

- **🎯 Analog Clock Focus**: Specialized dataset for analog clock reading challenges
- **📊 Separated Evaluation**: Independent analysis of hour and minute accuracy
- **🤖 Automated Optimization**: TextGrad-style prompt improvement system
- **📈 Detailed Analytics**: Performance analysis with confusion matrices
- **🔧 Python 3.8 Compatible**: Resolved TextGrad compatibility issues

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/clock_api.git
cd clock_api
pip install -r requirements.txt
```

### 2. Environment Setup
```bash
cp .env.example .env
# Edit .env file to add your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

### 3. Generate Dataset
```bash
python dataset_generator.py
```

### 4. Run Optimization
```bash
# Quick test
python test_optimized_prompt.py

# Full optimization pipeline
python textgrad_fixed.py
```

## 📁 Project Structure

```
clock_api/
├── 📊 dataset_generator.py      # Synthetic analog clock generation
├── 🤖 gpt4o_time_reader.py     # GPT-4o vision integration
├── 🔧 textgrad_fixed.py        # TextGrad-style optimization
├── 📈 evaluation_system.py     # Comprehensive evaluation metrics
├── 🧪 test_optimized_prompt.py # Baseline vs optimized comparison
├── 📋 requirements.txt         # Dependencies
└── 📖 README.md               # Documentation
```

## 📊 Evaluation Metrics

### Hour Analysis
- **Accuracy**: Exact hour match rate
- **MAE**: Mean absolute error in hours
- **Confusion Matrix**: 24-hour prediction visualization

### Minute Analysis  
- **Accuracy**: Exact minute match rate
- **MAE**: Mean absolute error in minutes
- **Tolerance Rates**: 5-minute and 10-minute error tolerance

### Combined Analysis
- **Exact Match**: Perfect time matching rate
- **Total Error**: Combined time error in minutes

## 🔧 Optimization Techniques

The project implements several advanced optimization strategies:

1. **📐 Mathematical Approach**: `number × 5 = minutes` formula
2. **📋 Step-by-Step Breakdown**: Systematic 5-stage process
3. **💡 Concrete Examples**: Real cases instead of abstract descriptions
4. **⚠️ Error Prevention**: Proactive common mistake blocking
5. **🔄 Validation Process**: Result verification steps
6. **🎯 Priority Order**: Minute-hand-first reading strategy

## 🧪 Experimental Results

```python
# Baseline vs Optimized Performance
baseline = {
    "hour_accuracy": 5%,
    "minute_accuracy": 15%,
    "exact_match": 0%
}

optimized = {
    "hour_accuracy": 10%,    # +100% improvement
    "minute_accuracy": 20%,  # +33% improvement  
    "exact_match": 0%        # Maintained
}
```

## 🔬 Technical Implementation

- **TextGrad Compatibility**: Resolved Python 3.8 type hint issues
- **Automated Feedback**: GPT-4o generates optimization feedback
- **Iterative Improvement**: Multi-stage prompt refinement
- **Performance Tracking**: Real-time optimization monitoring

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4o vision capabilities
- TextGrad team for prompt optimization framework
- Python community for excellent libraries

---

**🚀 Ready to improve AI's analog clock reading? Start with the quick setup guide above!**