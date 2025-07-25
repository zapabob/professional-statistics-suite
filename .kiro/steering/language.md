---
inclusion: always
---

# Professional Statistics Suite Development Standards

## Mandatory Code Patterns

### Edition-Based Feature Gating
ALWAYS implement edition checks before any feature execution:

```python
# -*- coding: utf-8 -*-
from config import check_feature_permission

def statistical_feature():
    if not check_feature_permission('feature_name'):
        raise PermissionError("Feature requires higher edition")
    # Continue with implementation
```

**Edition Capabilities:**
- `Lite`: Basic descriptive statistics, simple visualizations
- `Standard`: Advanced statistics, basic ML algorithms  
- `Professional`: Full statistical suite, AI integration, custom reports
- `GPU Accelerated`: GPU computing, advanced AI models, parallel processing

### Required Module Import Order
Load modules in this exact sequence to ensure proper dependency resolution:

1. `config.py` - Edition management and licensing (MUST BE FIRST)
2. `ai_integration.py` - Unified LLM interface and AI orchestration
3. `data_preprocessing.py` - Data validation and cleaning pipelines
4. `advanced_statistics.py` - Statistical computation engines
5. `professional_reports.py` - Report generation and export

### Code Style Requirements
- File encoding: `# -*- coding: utf-8 -*-` (mandatory header)
- Python execution: Use `py -3` command on Windows systems
- Conditional imports with graceful fallbacks:

```python
try:
    if check_feature_permission('gpu_acceleration'):
        import torch, cupy
    if check_feature_permission('advanced_ai'):
        import transformers, faiss
except ImportError:
    pass  # Always provide CPU fallback
```

## Core Architecture Patterns

### Data Processing Pipeline
All statistical operations must follow this validation pattern:

```python
def validate_statistical_data(data):
    """Validate data quality: normality, outliers, missing values"""
    # Log validation results to logs/ directory
    # Return comprehensive validation report
```

**Performance Standards:**
- Process large datasets (>100MB) in chunks to prevent memory issues
- Save operation state to `checkpoints/` for resumable long-running tasks
- Display progress indicators for operations exceeding 5 seconds
- Clean up intermediate results after processing completion

### AI Integration Architecture
Use `ai_integration.py` as the unified interface for all AI operations:

**Supported Providers:** OpenAI, Ollama, LM Studio, Kobold.cpp
**Required Features:**
- RAG implementation using sentence-transformers + faiss-cpu
- Sandbox execution for AI-generated statistical code
- Statistical methodology validation before execution
- Conversation history persistence for context continuity
- Graceful degradation when AI services are unavailable

### Security and Licensing Framework
**License Validation (Critical):**
- Query `booth_licenses.db` before ANY feature access
- Verify machine binding for trial license enforcement
- Log all license violations to `logs/` with automatic rotation
- Use `booth_protection.py` for PyInstaller executable builds
- Apply feature gates consistently across ALL statistical modules

## Project Structure Standards

### Required Directory Layout
Maintain this exact directory structure:

```
logs/          # Auto-rotating application logs with timestamps
reports/       # Generated statistical reports (ISO 8601 timestamps)
checkpoints/   # Resumable operation state for long-running tasks
templates/     # Reusable report templates and configurations
qr_codes/      # Booth system integration and licensing QR codes
```

### Error Handling Standards
Implement consistent error handling across all modules:

```python
import logging

try:
    # Statistical operations here
except Exception as e:
    logging.error(f"Statistical operation failed: {e}")
    raise  # Always re-raise to maintain error propagation
```

## User Interface Guidelines

### UX Requirements
- Display progress feedback for any operation exceeding 5 seconds
- Dynamically hide/disable features based on current edition license
- Show user-friendly error messages (log technical details separately)
- Ensure full keyboard navigation accessibility
- Maintain consistent professional color scheme throughout

### Code Quality Standards
**Documentation Requirements:**
- Include mathematical formulas in statistical function docstrings
- Use comprehensive type annotations for all parameters and returns
- Implement detailed logging for debugging and audit trail purposes
- Provide actionable error messages with clear solution suggestions

**Testing Requirements:**
- Create unit tests using known statistical datasets for validation
- Implement integration tests for AI provider connections and licensing
- Test edition-based feature gating across all statistical modules