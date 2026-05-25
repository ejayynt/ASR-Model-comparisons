# ASR Model Comparisons

## Project Overview
This repository contains a reproducible benchmarking pipeline designed to evaluate the performance of three Automatic Speech Recognition (ASR) systems: **Deepgram (Nova-3)**, **Sarvam (Saaras)**, and **Whisper V3 (via Groq LPUs)**. 

The evaluation strictly focuses on Entity Recovery Rate (ERR) for location extraction in highly degraded, real-world acoustic environments (e.g., live traffic, code-switching between Hinglish and Kannada). The core objective is to determine the optimal ASR architecture for a production backend relying on Latin-script location string matching.

## Repository Structure
* `/data` - Contains the custom dataset of 21 `.wav` audio samples recorded in live Bangalore traffic, alongside `ground_truth.json`.
* `main.py` - The core execution pipeline that routes audio to the APIs, calculates latency, computes Error Rates, and generates the output dataset.
* `requirements.txt` - Required Python dependencies.
* `results.csv` - The aggregated output metrics and raw model transcripts (UTF-8 Encoded).
* `Report.pdf` - The final architectural analysis, failure modes, and system design recommendations.

## Core Findings
* **Acoustic vs. System Failure:** While models like Sarvam demonstrated high acoustic accuracy in traffic, their automated transliteration into native Indic scripts (Devanagari) fundamentally breaks Latin-based Entity Recovery pipelines, registering a 0% ERR.
* **Deepgram (Baseline):** Achieved the highest ERR (23.8%) by forcing Latin-script (English) phonetic outputs, making it the most viable for direct backend integration without a secondary transliteration microservice.
* **Whisper V3:** Exhibited catastrophic hallucination loops when exposed to high-decibel traffic noise combined with code-switching.

## Setup and Installation

1. **Clone the repository and navigate to the directory:**
   ```bash
   git clone https://github.com/ejayynt/ASR-Model-comparisons.git
   cd ASR-Model-comparisons
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   The script requires API keys for the respective models. Ensure these are set in your environment before execution:
   * `DEEPGRAM_API_KEY`
   * `SARVAM_API_KEY`
   * `GROQ_API_KEY`

## Execution
To run the full evaluation pipeline:
```bash
python main.py
```

## Output
The script generates a `results.csv` file encoded in `utf-8-sig` to preserve native Devanagari and Kannada scripts without Mojibake corruption on standard OS environments. See `Report.pdf` for a comprehensive breakdown of these outputs.
