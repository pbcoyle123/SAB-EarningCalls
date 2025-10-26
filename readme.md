# Earnings Call Classification and Attribution Analysis

This repository analyzes earnings call transcripts to identify and study self-attribution bias (SAB) and related behavioral patterns in corporate communications through a three-stage pipeline.

## Pipeline Overview

**GPT Classification** (`gpt_classification/`) - Five separate scripts run by a main file to parse, structure, and classify earnings call transcripts with labels for topic, sentiment, temporal orientation, and attribution type. The OpenAI model is used, requiring an API key to be configured in `config.json`.

**Embedding Analysis** (`embedding_analysis/`) - After labels are generated, this module performs embedding analysis to understand features around attribution labels, test if embedding-based approaches can classify the labels, and conduct experiments to determine if hypothesized SAB firms and periods are more accurately captured by the GPT classification approach, an embedding approach, or a combined feature set.

**Audio Analysis** (`audio_analysis/`) - Transcribes audio files from earnings calls, aligns these with existing transcript files, and creates features around the already-classified attribution labels from earlier steps so that audio features for these labels can be investigated.

## Setup

1. Install dependencies: `pip install -r gpt_classification/requirements.txt`
2. Configure your OpenAI API key in `gpt_classification/config.json`
3. Run the classification pipeline via `gpt_classification/main.py`

Results are stored in the `output/` directory with classification results, embedding analysis reports, visualizations, and audio processing outputs.