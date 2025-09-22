# Heat Protection Sleeve Classification System

A computer vision system that uses Google Gemini Vision API to classify heat protection sleeves in automotive engine images as **OK**, **Broken**, or **Uncertain**.

## 🎯 Overview

This system performs automated visual inspection of heat protection sleeves using AI-powered image classification. It's designed for quality control in automotive manufacturing, helping identify damaged or missing heat protection components that could lead to safety issues.

## 📁 Repository Structure

```
Computer Vision/
├── README.md                           # This file
├── .env.example                        # Environment variables template
├── requirements.txt                    # Python dependencies
├── heat_protection_classifier.py       # Core classification engine
├── batch_tester.py                    # Batch processing tool
├── results_analyzer.py                # Performance analysis tool
├── Archiv/                            # Training and test images
│   ├── correct/                       # Images with OK sleeves
│   └── notcorrect/                    # Images with broken sleeves
├── results/                           # Analysis outputs
│   ├── latest_results.json            # Most recent batch results
│   ├── confusion_matrix.png           # Performance visualization
│   └── performance_report.html        # Detailed analysis report
└── example_files_cache.json           # Cached example images (auto-generated)
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- Google Gemini API key
- Virtual environment (recommended)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Computer Vision"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your API key
# API_KEY_2=your_google_gemini_api_key_here
```

### 4. Run Classification

#### Single Image Classification
```bash
python heat_protection_classifier.py path/to/your/image.jpg
```

#### Batch Processing
```bash
python batch_tester.py --directories "Archiv/correct" "Archiv/notcorrect"
```

#### Results Analysis
```bash
python results_analyzer.py --results-file results/latest_results.json
```

## 🔧 Components

### Core Classifier (`heat_protection_classifier.py`)

The main classification engine that:
- Uses few-shot learning with positive and negative examples
- Implements retry logic for API reliability
- Caches uploaded images to reduce API calls
- Returns structured JSON responses with confidence scores

**Key Features:**
- Automatic example image caching (48-hour TTL)
- Exponential backoff retry mechanism
- Detailed evidence summaries for each classification

### Batch Tester (`batch_tester.py`)

Processes multiple images and generates comprehensive results:

```bash
python batch_tester.py --directories "dir1" "dir2" [--output results]
```

**Output Files:**
- `latest_results.json` - Complete results with metadata
- `latest_results.csv` - Tabular format for analysis
- `raw_results_[timestamp].json` - Timestamped backup

### Results Analyzer (`results_analyzer.py`)

Generates performance metrics and visualizations:

```bash
python results_analyzer.py --results-file results/latest_results.json
```

**Generates:**
- Confusion matrix visualization
- Confidence score analysis
- Binary performance metrics (excluding "Uncertain")
- HTML performance report
- Per-class precision, recall, and F1-scores

## 📊 Performance Metrics

The system provides comprehensive performance analysis:

### Overall Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class performance metrics
- **Confidence Analysis**: Distribution of prediction confidence scores

### Binary Analysis
Focused analysis excluding "Uncertain" predictions:
- **OK Precision**: Reliability when predicting "OK"
- **Broken Recall**: Ability to detect actual defects
- **Trade-off Analysis**: Balance between false positives and false negatives

## 🎛️ Configuration

### Environment Variables (.env)
```bash
API_KEY_2=your_google_gemini_api_key_here
```

### Example Images
The system uses few-shot learning with these example categories:
- **Positive Examples** (OK sleeves): `Archiv/correct/`
- **Negative Examples** (Broken sleeves): `Archiv/notcorrect/`

## 📈 Usage Examples

### Basic Classification
```python
from heat_protection_classifier import HeatProtectionClassifier

classifier = HeatProtectionClassifier()
result = classifier.classify_image("path/to/image.jpg")
print(result)
```

### Batch Processing with Custom Output
```bash
python batch_tester.py \
  --directories "test_images/set1" "test_images/set2" \
  --output "custom_results"
```

### Analysis with Custom Thresholds
```python
from results_analyzer import ResultsAnalyzer

analyzer = ResultsAnalyzer()
results = analyzer.analyze_results("results/latest_results.json")
```

## 🔍 Understanding Results

### Classification Output
Each classification returns:
```json
{
  "classification": "OK | Broken | Uncertain",
  "confidence": 0.95,
  "evidence_summary": "Detailed visual analysis...",
  "observations": {
    "sleeve_presence": "present | absent | occluded",
    "coverage_in_heat_zone": "continuous | partial_gap | absent",
    "integrity": ["no_damage", "fray", "tear", "hole", "burn_char"],
    "positioning": ["well_positioned", "slipped_back", "loose_end"],
    "hot_zone_cues": ["exhaust_metal_nearby", "turbo_housing"],
    "cosmetics": ["dusty", "clean", "oily"]
  }
}
```

### Ground Truth Assignment
- Images in `Archiv/correct/` → Ground Truth: "OK"
- Images in `Archiv/notcorrect/` → Ground Truth: "Broken"

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   ValueError: API_KEY_2 not found in .env file
   ```
   Solution: Ensure `.env` file exists with valid API key

2. **503 Service Unavailable**
   ```
   Classification failed: 503 UNAVAILABLE
   ```
   Solution: The system automatically retries. If persistent, try reducing batch size.

3. **Cache Issues**
   ```
   Cache missing required fields (name, uri, mime_type)
   ```
   Solution: Delete `example_files_cache.json` to force refresh

### Performance Optimization

- **Reduce API Calls**: The system caches example images for 48 hours
- **Batch Size**: Process images in smaller batches if experiencing timeouts
- **Retry Logic**: Built-in exponential backoff handles temporary API issues

## 📋 Requirements

See `requirements.txt` for complete dependency list. Key packages:
- `google-genai` - Google Gemini API client
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `scikit-learn` - Performance metrics
- `python-dotenv` - Environment management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the generated performance reports
3. Examine the console output for detailed error messages
4. Create an issue with sample images and error logs
