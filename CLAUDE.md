# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Clock Time Reading optimization project using GPT-4o and TextGrad. Addresses the challenge of AI models reading clock times accurately through prompt optimization and comprehensive evaluation.

## Development Setup

### Environment Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env to add OPENAI_API_KEY
```

### Common Commands

#### Full Pipeline
```bash
python main_pipeline.py --samples 500 --baseline-samples 50 --final-samples 100
```

#### Individual Components
```bash
# Generate clock dataset
python dataset_generator.py

# Test GPT-4o time reading
python gpt4o_time_reader.py

# Optimize prompts with TextGrad
python textgrad_optimizer.py

# Run detailed evaluation
python evaluation_system.py
```

## Architecture

### Core Components

1. **Dataset Generator** (`dataset_generator.py`)
   - Creates analog, digital, and Korean text clocks
   - Generates 500+ diverse time representations
   - Outputs images + metadata JSON

2. **GPT-4o Time Reader** (`gpt4o_time_reader.py`)
   - Handles image-to-text time reading via OpenAI API
   - Supports batch processing for efficiency
   - Returns structured JSON responses

3. **TextGrad Optimizer** (`textgrad_optimizer.py`)
   - Implements automatic prompt optimization
   - Uses gradient-based feedback for prompt improvement
   - Maintains optimization history

4. **Evaluation System** (`evaluation_system.py`)
   - Separate evaluation for hours and minutes
   - Clock type-specific performance analysis
   - Generates confusion matrices and error distributions

5. **Main Pipeline** (`main_pipeline.py`)
   - Orchestrates end-to-end workflow
   - Provides comparison reports
   - Handles error recovery

### Data Flow
```
Images → GPT-4o → Predictions → Evaluation → Optimization → Improved Prompts
```

## Key Features

- **Separate Hour/Minute Evaluation**: Critical for understanding specific weaknesses
- **Multiple Clock Types**: Analog, digital, Korean text representation
- **Automated Optimization**: TextGrad-powered prompt improvement
- **Comprehensive Reporting**: JSON + text reports with visualizations

## API Requirements

- OpenAI API key required for GPT-4o access
- TextGrad uses GPT-4o for backward engine
- Estimated cost: ~$5-10 for full pipeline run