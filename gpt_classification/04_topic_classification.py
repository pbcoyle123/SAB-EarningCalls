# -*- coding: utf-8 -*-
"""
04_topic_classification.py

Topic Classification Script with Structured Outputs
==================================================

OVERVIEW:
This script processes earnings call transcripts through a 3-pass pipeline to classify content 
into granular topics. Each transcript file completes all 3 passes before moving to the next file.

EXECUTION ORDER:
main()
 For each transcript file:
    process_transcript_file()  # Runs all 3 passes for one file
       PASS 1: classify_topics_structured()  # LLM: classify paragraph into 1-5 topics
       PASS 2: attribute_sentences_structured()  # LLM: assign sentences to topics (multi-topic paragraphs)
          rejoin_same_topic_sentences()  # Merge consecutive sentences with same topic
       PASS 3: apply_contextual_refinement_pass_single_file()  # Refine ambiguous classifications
           needs_contextual_refinement()  # Check if snippet needs refinement
           perform_contextual_refinement_structured()  # LLM: refine with prev/next context
    merge_meeting_logistics_snippets()  # Merge sequential logistics snippets (optimization)
    create_csv_output()  # Apply deduplication, CSV formatting, safe encoding
    Save individual CSV file (complete, optimized representation)
 create_processing_summary_csv()  # Generate overview stats across all files

CORE METHODS:
- query_llm_structured(): Core LLM interface with Pydantic schema validation
- decode_json_text(): Clean unicode escapes and encoding issues
- safe_csv_format(): Handle CSV formatting for special characters
- extract_speaker_team(): Determine if speaker is management or analyst

OUTPUT: 
- Individual CSV files per transcript (complete, optimized representation)
- Processing summary CSV (overview statistics across all files)
- Metadata files (duplicates analysis, processing details)
"""

import openai
import pandas as pd
import json
import os
import glob
import nltk
import time
from tqdm import tqdm
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import re
from itertools import groupby
import difflib  # Add difflib for detailed text comparison

# Verify NLTK punkt data is available (should be installed by setup)
try:
    test_sentences = nltk.sent_tokenize("This is a test. This should work.")
    print(f" NLTK sentence tokenizer ready: {len(test_sentences)} sentences detected")
except Exception as e:
    print(f" NLTK punkt data not available: {e}")
    print("  Attempting emergency download...")
    try:
        nltk.download('punkt', quiet=True)
        test_sentences = nltk.sent_tokenize("Emergency test.")
        print(" Emergency punkt download successful")
    except Exception as e2:
        print(f" CRITICAL: NLTK punkt unavailable - script will fail: {e2}")
        raise RuntimeError("NLTK punkt data required but not available")

# Set your OpenAI API key from config file
from config import get_openai_api_key
openai.api_key = get_openai_api_key()

#  API key (commented out):
# if you want to use env variable, use this:
#openai.api_key = os.getenv('OPENAI_API_KEY')

# Specify the GPT model
GPT_MODEL = "gpt-4.1-mini"

# Update the output directory path (updated for parallel_cpu structure)
OUTPUT_DIR = "04"  # Points to parallel_cpu/04/ for step 4 output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create metadata subdirectory for auxiliary files
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")
os.makedirs(METADATA_DIR, exist_ok=True)

# Input directory for paragraph-segmented transcripts (updated for parallel_cpu structure)
DATA_DIR = "03"  # Points to parallel_cpu/03/ where previous script outputs

# Mapping between categories and underlying topics
category_topic_mapping = {
    "Financial Performance Metrics": ["Revenue", "Earnings Per Share (EPS)", "Gross Margin", "Operating Margin", 
                                     "Cost of Goods Sold", "Operating Expenses", "Guidance", "Financial Performance"],
    "Balance Sheet and Cash Flow Metrics": ["Cash Flow from Operations", "Capital Expenditures", 
                                            "Debt Levels and Financing", "Balance Sheet Metrics", "Dividends and Buybacks"],
    "Operational Efficiency and Management": ["Cost Management and Efficiency", "Mergers and Acquisitions (M&A)", 
                                              "Strategic Initiatives", "Management Changes", "Workforce", 
                                              "Product/Service Updates", "Innovation and R&D"],
    "Market and Industry Analysis": ["Industry Trends", "Market Share and Competition", "Customer Acquisitions"],
    "Geographical and Economic Considerations": ["Economic Factors", "Foreign Exchange", "Geographic Performance"],
    "Regulation and Risk": ["Regulatory Changes", "Risk Factors", "Environmental, Social and Governance"],
    "Meeting Logistics": ["Meeting Logistics (Introductions, Transitions, Call Structure, Greetings, Closing Remarks)"],
    "Others": ["Others"]
}

# Flatten list of topics for prompt
topics_list = [topic for topics in category_topic_mapping.values() for topic in topics]

# ========================================
# PYDANTIC SCHEMAS FOR STRUCTURED OUTPUTS
# ========================================

class TopicClassification(BaseModel):
    topic: str = Field(description="The specific topic name")
    category: str = Field(description="The category this topic belongs to")
    confidence: int = Field(ge=0, le=100, description="Confidence score 0-100%")
    topic_coverage: int = Field(ge=0, le=100, description="Percentage of paragraph covered by this topic")
    temporal_context: Literal["Retrospective", "Current", "Forward-looking"] = Field(description="Temporal context of the topic")
    content_sentiment: Literal["Positive", "Negative", "Neutral", "Uncertain"] = Field(description="Sentiment of the content")
    speaker_tone: Literal["Positive", "Negative", "Neutral", "Uncertain"] = Field(description="Tone of the speaker")

class InitialClassificationResponse(BaseModel):
    topics: List[TopicClassification] = Field(min_length=1, max_length=5, description="List of classified topics")
    total_coverage: int = Field(ge=0, le=100, description="Total coverage percentage")

class SentenceAttribution(BaseModel):
    position: int = Field(ge=0, description="Position of sentence in paragraph (0-based index)")
    attributed_topic: str = Field(description="The topic this sentence is attributed to")
    confidence: int = Field(ge=0, le=100, description="Confidence in the attribution")
    temporal_context: str = Field(description="Temporal context for this sentence")
    is_fallback: bool = Field(default=False, description="Whether this attribution is a fallback assignment")

class SentenceAttributionResponse(BaseModel):
    sentence_attributions: List[SentenceAttribution] = Field(description="List of sentence attributions")


# ========================================
# STRUCTURED LLM QUERY FUNCTIONS
# ========================================

def query_llm_structured(prompt: str, schema_class: BaseModel, retries: int = 3, stats_tracker: dict = None):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification_response",
                        "schema": schema_class.model_json_schema()
                    }
                },
                temperature=0.1,
                top_p=0.2
            )
            
            # Track successful LLM call
            if stats_tracker:
                stats_tracker['Total_LLM_Calls'] += 1
                if schema_class == InitialClassificationResponse:
                    stats_tracker['Topic_Classification_Calls'] += 1
                elif schema_class == SentenceAttributionResponse:
                    stats_tracker['Sentence_Attribution_Calls'] += 1
                elif hasattr(schema_class, '__name__') and 'ContextualRefinement' in schema_class.__name__:
                    stats_tracker['Contextual_Refinement_Calls'] += 1
            
            # Parse and validate the response
            response_text = response.choices[0].message.content
            return schema_class.model_validate_json(response_text)
            
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            # Track failed LLM call
            if stats_tracker:
                stats_tracker['Error_Count'] += 1
            if attempt == retries - 1:
                # Return a default response if all attempts fail
                if schema_class == InitialClassificationResponse:
                    return InitialClassificationResponse(
                        topics=[
                            TopicClassification(
                                topic="Uncategorized",
                                category="Others",
                                confidence=0,
                                topic_coverage=0,
                                temporal_context="Current",
                                content_sentiment="Neutral",
                                speaker_tone="Uncertain"
                            )
                        ],
                        total_coverage=0
                    )
                elif schema_class == SentenceAttributionResponse:
                    return SentenceAttributionResponse(sentence_attributions=[])
            
            # Wait before retrying
            time.sleep(2)

def classify_topics_structured(company: str, speaker: str, text: str, stats_tracker: dict = None) -> tuple:
    """
    Convert original classify_with_variable_topics to use structured outputs
    
    Args:
        company: Company name/symbol
        speaker: Speaker name
        text: Text to classify
        
    Returns:
        Tuple of (topics_with_scores, total_coverage, needs_refinement, refinement_reason)
    """
    # Validate inputs
    if not text or not isinstance(text, str) or not text.strip():
        print(f"Warning: Empty or invalid text provided for classification")
        return ([{
            "topic": "Uncategorized", 
            "category": "Uncategorized", 
            "confidence": 0,
            "topic_coverage": 0,
            "temporal_context": "Unknown",
            "content_sentiment": "Neutral",
            "speaker_tone": "Uncertain"
        }], 0, False, "empty_text")
    
    # Clean the text
    text = text.strip()
    
    # Create the prompt for structured output
    prompt = f"""You are analyzing an earnings call transcript paragraph for topic classification.

AVAILABLE TOPICS: {'; '.join(topics_list)}

PARAGRAPH: {text}

SPEAKER: {speaker} from {company}

CLASSIFICATION GUIDELINES:
1. Identify ONLY the most relevant topics that are meaningfully discussed in this paragraph (not merely mentioned in passing).
2. MAXIMUM 5 topics allowed - focus on the most important ones.
3. For very short or transitional statements, focus on the main intent.
4. For introductions, meeting logistics, greetings, call structure descriptions, or closing remarks, classify as "Meeting Logistics (Introductions, Transitions, Call Structure, Greetings, Closing Remarks)".
5. "Guidance" refers ONLY to future projections, not explanations of past results.
6. For paragraphs discussing products or services, focus on whether the main point is about product features (Product/Service Updates) or strategic business decisions (Strategic Initiatives).

For each topic you identify:
1. Include only topics that are MEANINGFULLY discussed (not merely mentioned in passing)
2. Assign a confidence score (0-100%) to each topic
3. Estimate what percentage of the paragraph's content is covered by each topic
4. Ensure the topics you identify collectively cover the paragraph's content
5. LIMIT TO MAXIMUM 5 TOPICS

TEMPORAL CONTEXT (time orientation - NOT sentiment):
MUST be EXACTLY one of these 3 time-based options:
- Retrospective: discussing past events or results
- Current: describing present conditions or ongoing situations  
- Forward-looking: providing future projections or expectations

IMPORTANT: Use ONLY "Retrospective", "Current", or "Forward-looking" for temporal context. Do NOT use "Neutral".

CONTENT SENTIMENT (based ONLY on objective facts being reported):
- Positive: reporting growth, exceeding targets, new opportunities, strong performance
- Negative: reporting decline, missing targets, challenges, losses
- Neutral: factual statements without clear positive/negative implications
NOTE: Content sentiment must be EXACTLY one of: "Positive", "Negative", or "Neutral"

SPEAKER TONE (how information is delivered, regardless of content):
MUST be EXACTLY one of these 3 options only:
- Positive: upbeat, enthusiastic, or confident delivery
- Negative: downbeat, concerned, or anxious delivery  
- Neutral: balanced, matter-of-fact delivery

IMPORTANT: Use ONLY "Positive", "Negative", or "Neutral" for speaker tone - no other labels allowed.

Remember: content sentiment and speaker tone are independent.

Provide your classification in the required JSON format."""

    # Query the LLM with structured output
    classification_response = query_llm_structured(prompt, InitialClassificationResponse, stats_tracker=stats_tracker)
    
    # Convert to the original format for compatibility
    topics_with_scores = []
    for topic_class in classification_response.topics:
        # Find category based on topic
        category = next((cat for cat, topics in category_topic_mapping.items() 
                        if any(t.lower() == topic_class.topic.lower() for t in topics)), "Uncategorized")
        
        topics_with_scores.append({
            "topic": topic_class.topic,
            "category": category,
            "confidence": topic_class.confidence,
            "topic_coverage": topic_class.topic_coverage,
            "temporal_context": topic_class.temporal_context,
            "content_sentiment": topic_class.content_sentiment,
            "speaker_tone": topic_class.speaker_tone
        })
    
    total_coverage = classification_response.total_coverage
    
    # Determine if refinement is needed based on patterns (preserve original logic)
    needs_refinement = False
    refinement_reason = None
    
    # Check for patterns that suggest refinement is needed
    if total_coverage < 80:
        needs_refinement = True
        refinement_reason = "low_coverage"
    elif len(topics_with_scores) >= 4:
        needs_refinement = True
        refinement_reason = "high_topic_count"
    elif len(topics_with_scores) > 1 and topics_with_scores[0]['confidence'] - topics_with_scores[1]['confidence'] < 15:
        needs_refinement = True
        refinement_reason = "ambiguous_confidence"
    elif topics_with_scores and topics_with_scores[0]['confidence'] < 50:
        needs_refinement = True
        refinement_reason = "low_primary_confidence"
    # UPDATED: Only flag short text if it's NOT meeting logistics AND has low confidence
    elif (len(text.split()) < 20 and 
          topics_with_scores and 
          topics_with_scores[0]['topic'] != "Meeting Logistics (Introductions, Transitions, Call Structure, Greetings, Closing Remarks)" and
          topics_with_scores[0]['confidence'] < 70):
        needs_refinement = True
        refinement_reason = "short_text_low_confidence"
    
    return topics_with_scores, total_coverage, needs_refinement, refinement_reason

def attribute_sentences_structured(text: str, topics_with_scores: List[dict], stats_tracker: dict = None) -> Optional[List[dict]]:
    """
    Convert original attribute_sentences_to_topics to use structured outputs
    
    Args:
        text: The paragraph text
        topics_with_scores: List of topics with their scores
        
    Returns:
        List of sentence attributions or None if not applicable
    """
    # Only process if we have multiple topics with significant coverage
    if len(topics_with_scores) < 2:
        # For single topic case, return None to indicate no sentence attribution needed
        return None
        
    # Split the paragraph into sentences
    sentences = nltk.sent_tokenize(text)
    
    # For very short paragraphs or single sentences, can't meaningfully subdivide
    if len(sentences) < 2:
        return None
    
    # Create a prompt with numbered sentences
    numbered_sentences = [f"Sentence {i+1}: \"{sentence}\"" for i, sentence in enumerate(sentences)]
    
    # Update the coverage info to include temporal context
    coverage_info = ", ".join([
        f"{t['topic']} ({t['topic_coverage']}%, {t.get('temporal_context', 'Unknown')})" 
        for t in topics_with_scores
    ])
    
    numbered_sentences_text = '\n'.join(numbered_sentences)
    topics_text = ', '.join([t['topic'] for t in topics_with_scores])
    
    prompt = f"""You are analyzing a paragraph from an earnings call transcript that discusses multiple topics: {coverage_info}

The paragraph has been split into numbered sentences:
{numbered_sentences_text}

Available topics: {topics_text}

For EACH numbered sentence, identify which ONE topic from the list it most closely relates to.
Include the temporal context (Forward-looking, Current, or Retrospective) from the matched topic.

IMPORTANT: 
- You MUST return exactly {len(sentences)} sentence attributions
- Return ONLY the position (0-{len(sentences)-1}), topic, confidence, and temporal context
- Do NOT return sentence text - only the classification metadata
- Each sentence position must appear exactly once

Provide your response in JSON format with a list of sentence attributions."""

    # Query the LLM with structured output
    try:
        attribution_response = query_llm_structured(prompt, SentenceAttributionResponse, stats_tracker=stats_tracker)
        
        # Convert to the original format for compatibility
        sentence_topic_mapping = []
        
        # Track which positions we've processed to detect duplicates and missing
        processed_positions = set()
        
        for attribution in attribution_response.sentence_attributions:
            # Validate position is within expected range
            if attribution.position < 0 or attribution.position >= len(sentences):
                print(f"WARNING: Invalid position {attribution.position} for sentence attribution (expected 0-{len(sentences)-1})")
                continue
                
            # Check for duplicate positions
            if attribution.position in processed_positions:
                print(f"WARNING: Duplicate position {attribution.position} in LLM response - skipping")
                continue
                
            # Get the original sentence at this position (systematic mapping)
            original_sentence = sentences[attribution.position]
            
            # Find the valid topic from our list
            topic_obj = None
            valid_topic = attribution.attributed_topic
            
            if not any(t["topic"].lower() == attribution.attributed_topic.lower() for t in topics_with_scores):
                # If exact topic not found, find closest match
                best_topic_match = None
                for t in topics_with_scores:
                    if attribution.attributed_topic.lower() in t["topic"].lower() or t["topic"].lower() in attribution.attributed_topic.lower():
                        best_topic_match = t["topic"]
                        topic_obj = t
                        break
                
                if best_topic_match:
                    valid_topic = best_topic_match
                else:
                    # Fallback to primary topic
                    valid_topic = topics_with_scores[0]["topic"]
                    topic_obj = topics_with_scores[0]
            else:
                # Find the matching topic object
                topic_obj = next((t for t in topics_with_scores if t["topic"].lower() == attribution.attributed_topic.lower()), None)
            
            # Use LLM confidence if provided, otherwise use topic confidence
            confidence = attribution.confidence if hasattr(attribution, 'confidence') and attribution.confidence > 0 else (topic_obj["confidence"] if topic_obj else 0)
            # Use returned temporal context or fallback to topic's temporal context
            final_temporal = attribution.temporal_context if attribution.temporal_context != "Unknown" else topic_obj.get("temporal_context", "Unknown")
            
            # ALWAYS use the original sentence text from systematic mapping
            sentence_topic_mapping.append({
                "sentence": original_sentence,  # Systematic mapping from position to original sentence
                "attributed_topic": valid_topic,
                "position": attribution.position,
                "confidence": confidence,
                "temporal_context": final_temporal,
                "is_fallback": attribution.is_fallback
            })
            
            processed_positions.add(attribution.position)
        
        # Check if we've missed any sentences and assign them the primary topic
        mapped_positions = processed_positions  # Use the positions we actually processed
        missing_positions = set(range(len(sentences))) - mapped_positions
        
        if missing_positions:
            print(f"WARNING: LLM missed {len(missing_positions)} sentences at positions: {sorted(missing_positions)}")
            
        for i in missing_positions:
                sentence = sentences[i]  # Get the original sentence at this position
                primary_topic = topics_with_scores[0]
                sentence_topic_mapping.append({
                    "sentence": sentence,
                    "attributed_topic": primary_topic["topic"],
                    "position": i,
                    "confidence": primary_topic["confidence"],
                    "temporal_context": primary_topic.get("temporal_context", "Unknown"),
                    "is_fallback": True
                })
                print(f"     Added fallback for position {i}: '{sentence[:50]}...'")
        
        # Final validation - ensure we have exactly the right number of sentences
        if len(sentence_topic_mapping) != len(sentences):
            print(f"WARNING: Sentence count mismatch: expected {len(sentences)}, got {len(sentence_topic_mapping)}")
            return None
        
        # Sort by position for coherent ordering
        sentence_topic_mapping.sort(key=lambda x: x["position"])
        
        return sentence_topic_mapping
    
    except Exception as e:
        print(f"Error in sentence attribution: {e}")
        return None

def rejoin_same_topic_sentences(sentence_attributions: List[dict]) -> List[dict]:
    """Rejoin consecutive sentences that have the same attributed topic"""
    if not sentence_attributions:
        return []
    
    rejoined_snippets = []
    current_group = [sentence_attributions[0]]  # Start with first sentence
    
    for i in range(1, len(sentence_attributions)):
        current_sentence = sentence_attributions[i]
        previous_sentence = sentence_attributions[i-1]
        
        # Check if same topic and consecutive positions
        if (current_sentence['attributed_topic'] == previous_sentence['attributed_topic'] and 
            current_sentence['position'] == previous_sentence['position'] + 1):
            # Add to current group
            current_group.append(current_sentence)
        else:
            # Process current group and start new group
            rejoined_snippets.append(merge_sentence_group(current_group))
            current_group = [current_sentence]
    
    # Don't forget the last group
    rejoined_snippets.append(merge_sentence_group(current_group))
    
    return rejoined_snippets

def merge_sentence_group(sentence_group: List[dict]) -> dict:
    """Merge a group of consecutive sentences with same topic"""
    if len(sentence_group) == 1:
        return sentence_group[0]  # Single sentence, return as-is
    
    # Merge multiple sentences
    merged_sentence = " ".join([s['sentence'] for s in sentence_group])
    first_sentence = sentence_group[0]
    
    return {
        'sentence': merged_sentence,
        'attributed_topic': first_sentence['attributed_topic'],
        'position': first_sentence['position'],  # Use position of first sentence
        'confidence': first_sentence['confidence'],
        'temporal_context': first_sentence['temporal_context'],
        'is_fallback': any(s['is_fallback'] for s in sentence_group),  # True if any is fallback
        'sentence_count': len(sentence_group)  # Track how many were merged
    }

# ========================================
# CSV FORMATTING AND EXPORT
# ========================================

def safe_csv_format(text):
    if pd.isna(text) or text == "" or text is None:
        return ""
    
    # Convert to string and handle special characters
    text = str(text)
    
    # Handle unicode escapes (like \u2019 for apostrophes)
    try:
        # Try to decode JSON unicode escapes
        text = text.encode().decode('unicode_escape')
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If that fails, try JSON loads to handle unicode
        try:
            text = json.loads(f'"{text}"')
        except (json.JSONDecodeError, ValueError):
            # If all else fails, keep original text
            pass
    
    # Escape quotes by doubling them
    text = text.replace('"', '""')
    
    # Wrap in quotes if contains commas, quotes, or newlines
    if ',' in text or '"' in text or '\n' in text:
        text = f'"{text}"'
        
    return text

def create_csv_output(all_snippets: List[dict]) -> pd.DataFrame:
    """
    Create DataFrame with proper CSV formatting
    
    Args:
        all_snippets: List of snippet dictionaries
        
    Returns:
        Formatted DataFrame ready for CSV export
    """
    if not all_snippets:
        return pd.DataFrame()
    
    # Add snippet order based on original paragraph order (not processing order)
    # Sort snippets by their original order first
    all_snippets_sorted = sorted(all_snippets, key=lambda x: (
        x.get('Company', ''), x.get('Quarter', ''), x.get('Date', ''), 
        x.get('Section', ''), x.get('Original_Paragraph_Order', 0)
    ))
    
    # DEDUPLICATION: Remove any potential duplicates based on snippet content and speaker
    seen_snippets = set()
    deduplicated_snippets = []
    duplicate_snippets = []  # NEW: Track duplicates for analysis
    
    for snippet in all_snippets_sorted:
        # Create a unique key based on content, speaker, and section
        unique_key = (
            snippet.get('Company', ''),
            snippet.get('Quarter', ''), 
            snippet.get('Date', ''),
            snippet.get('Speaker_Name', ''),
            snippet.get('Section', ''),
            snippet.get('Snippet', '')[:100],  # First 100 chars to identify duplicates
            snippet.get('Original_Paragraph_Order', 0)
        )
        
        if unique_key not in seen_snippets:
            seen_snippets.add(unique_key)
            deduplicated_snippets.append(snippet)
        else:
            print(f"WARNING: Removed duplicate snippet for {snippet.get('Speaker_Name', '')} in {snippet.get('Section', '')}")
            # NEW: Capture duplicate for analysis
            # Find the original snippet that was already added
            original_snippet = None
            for existing_snippet in deduplicated_snippets:
                existing_key = (
                    existing_snippet.get('Company', ''),
                    existing_snippet.get('Quarter', ''), 
                    existing_snippet.get('Date', ''),
                    existing_snippet.get('Speaker_Name', ''),
                    existing_snippet.get('Section', ''),
                    existing_snippet.get('Snippet', '')[:100],
                    existing_snippet.get('Original_Paragraph_Order', 0)
                )
                if existing_key == unique_key:
                    original_snippet = existing_snippet
                    break
            
            duplicate_snippets.append({
                'duplicate_type': 'removed_duplicate',
                'duplicate_key': str(unique_key),
                'original_snippet_data': original_snippet.copy() if original_snippet else None,
                'duplicate_snippet_data': snippet.copy()  # Full copy of all metadata
            })
    
    print(f"STATS: Deduplication: {len(all_snippets_sorted)} -> {len(deduplicated_snippets)} snippets")
    
    # NEW: Save duplicates to CSV for analysis if any found
    if duplicate_snippets:
        print(f"SAVE: Found {len(duplicate_snippets)} duplicates - saving for analysis...")
        
        # Flatten duplicate data for CSV
        duplicate_rows = []
        for i, dup in enumerate(duplicate_snippets):
            base_data = {
                'Duplicate_Index': i + 1,
                'Duplicate_Type': dup['duplicate_type'],
                'Duplicate_Key_Preview': dup['duplicate_key'][:200] + "..." if len(dup['duplicate_key']) > 200 else dup['duplicate_key']
            }
            
            # Add original snippet metadata with 'Original_' prefix
            if dup['original_snippet_data']:
                for key, value in dup['original_snippet_data'].items():
                    base_data[f'Original_{key}'] = value
            
            # Add duplicate snippet metadata with 'Duplicate_' prefix  
            if dup['duplicate_snippet_data']:
                for key, value in dup['duplicate_snippet_data'].items():
                    base_data[f'Duplicate_{key}'] = value
            
            duplicate_rows.append(base_data)
        
        # Create duplicates DataFrame
        duplicates_df = pd.DataFrame(duplicate_rows)
        
        # Save duplicates to CSV  
        file_count = len(set([str(dup['duplicate_snippet_data'].get('Company', '')) + '_' + 
                             str(dup['duplicate_snippet_data'].get('Quarter', '')) + '_' + 
                             str(dup['duplicate_snippet_data'].get('Date', '')) for dup in duplicate_snippets]))
        duplicates_filename = f"duplicate_snippets_analysis_{file_count}_files.csv"
        duplicates_path = os.path.join(METADATA_DIR, duplicates_filename)
        duplicates_df.to_csv(duplicates_path, index=False, encoding='utf-8')
        print(f"   Duplicates analysis saved to: {duplicates_path}")
    
    # Then assign sequential order
    for i, snippet in enumerate(deduplicated_snippets, 1):
        snippet['Snippet_Order'] = i
    
    # Create DataFrame from deduplicated snippets
    df = pd.DataFrame(deduplicated_snippets)
    
    # Reorder columns to put Snippet_Order first
    if 'Snippet_Order' in df.columns:
        cols = ['Snippet_Order'] + [col for col in df.columns if col != 'Snippet_Order']
        df = df[cols]
    
    # Apply CSV formatting to text columns
    text_columns = ['Snippet', 'Original_Paragraph', 'Refinement_Reason', 'Speaker_Name', 
                   'Contextual_Refinement_Reasoning', 'Contextual_Refinement_Reason']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_csv_format)
    
    # Apply formatting to topic columns that might contain text
    topic_columns = [col for col in df.columns if 'Topic' in col and not col.endswith('_Confidence') and not col.endswith('_Coverage')]
    for col in topic_columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_csv_format)
    
    return df

def create_content_mismatches_csv(validation_report: dict, debug_info: dict):
    if not validation_report.get('speaker_content_mismatches'):
        print("No content mismatches to export to CSV")
        return
    
    print("INFO: Creating detailed CSV comparison of content mismatches...")
    
    # Extract mismatches data
    mismatches_data = []
    
    for mismatch in validation_report['speaker_content_mismatches']:
        section_name = f"{mismatch['section'][0]}_{mismatch['section'][1]}_{mismatch['section'][2]}_{mismatch['section'][3]}"
        speaker_name = mismatch['speaker'].split(' (order:')[0]  # Remove order info
        
        # Find original and reconstructed content
        original_content = ""
        reconstructed_content = ""
        
        # Find in debug_info
        if section_name in debug_info.get('original_speakers', {}):
            for speaker_data in debug_info['original_speakers'][section_name]:
                if speaker_data['speaker_name'] == speaker_name:
                    original_content = speaker_data['dialogue_text']
                    break
        
        if section_name in debug_info.get('reconstructed_speakers', {}):
            for speaker_data in debug_info['reconstructed_speakers'][section_name]:
                if speaker_data['speaker_name'] == speaker_name:
                    reconstructed_content = speaker_data['merged_text']
                    break
        
        # Clean reconstructed content by removing quotes and fixing duplications
        reconstructed_content_cleaned = clean_reconstructed_content(reconstructed_content)
        
        # Calculate differences
        original_words = original_content.split()
        reconstructed_words = reconstructed_content_cleaned.split()
        reconstructed_original_words = reconstructed_content.split()
        
        # Simple diff highlighting (basic version)
        difference_summary = generate_difference_summary(original_content, reconstructed_content_cleaned)
        
        # Generate detailed diff analysis
        raw_diff = extract_simple_differences(original_content, reconstructed_content)
        cleaned_diff = extract_simple_differences(original_content, reconstructed_content_cleaned)
        
        mismatches_data.append({
            'Mismatch_ID': len(mismatches_data) + 1,
            'Company': mismatch['section'][0],
            'Quarter': mismatch['section'][1], 
            'Year': mismatch['section'][2],
            'Section': mismatch['section'][3],
            'Speaker_Name': speaker_name,
            'Original_Length_Chars': len(original_content),
            'Original_Length_Words': len(original_words),
            'Reconstructed_Length_Chars': len(reconstructed_content),
            'Reconstructed_Length_Words': len(reconstructed_original_words),
            'Cleaned_Reconstructed_Length_Chars': len(reconstructed_content_cleaned),
            'Cleaned_Reconstructed_Length_Words': len(reconstructed_words),
            'Length_Diff_Chars': len(reconstructed_content) - len(original_content),
            'Length_Diff_Words': len(reconstructed_original_words) - len(original_words),
            'Cleaned_Length_Diff_Chars': len(reconstructed_content_cleaned) - len(original_content),
            'Cleaned_Length_Diff_Words': len(reconstructed_words) - len(original_words),
            'Has_Quote_Issues': '"' in reconstructed_content and not '"' in original_content,
            'Has_Duplication_Issues': has_duplication_issues(reconstructed_content),
            'Difference_Summary': difference_summary,
            # Detailed diff analysis
            'Raw_Similarity_Percent': raw_diff['similarity_percentage'],
            'Cleaned_Similarity_Percent': cleaned_diff['similarity_percentage'],
            'Raw_Change_Type': raw_diff['change_type'],
            'Cleaned_Change_Type': cleaned_diff['change_type'],
            'Raw_Added_Words': raw_diff['added_words'],
            'Raw_Removed_Words': raw_diff['removed_words'],
            'Cleaned_Added_Words': cleaned_diff['added_words'],
            'Cleaned_Removed_Words': cleaned_diff['removed_words'],
            'Raw_Differences': raw_diff['simple_diff'],
            'Cleaned_Differences': cleaned_diff['simple_diff'],
            # Original content columns
            'Original_Content': original_content,
            'Reconstructed_Content_Raw': reconstructed_content,
            'Reconstructed_Content_Cleaned': reconstructed_content_cleaned,
            'Original_Preview_100': original_content[:100] + "..." if len(original_content) > 100 else original_content,
            'Reconstructed_Preview_100': reconstructed_content[:100] + "..." if len(reconstructed_content) > 100 else reconstructed_content,
            'Cleaned_Preview_100': reconstructed_content_cleaned[:100] + "..." if len(reconstructed_content_cleaned) > 100 else reconstructed_content_cleaned
        })
    
    # Create DataFrame
    mismatches_df = pd.DataFrame(mismatches_data)
    
    # Generate filename
    mismatches_filename = f"content_mismatches_detailed_comparison_{len(validation_report['speaker_content_mismatches'])}_mismatches.csv"
    mismatches_path = os.path.join(METADATA_DIR, mismatches_filename)
    
    # Save to CSV
    mismatches_df.to_csv(mismatches_path, index=False, encoding='utf-8')
    
    print(f"STATS: Content mismatches CSV saved to: {mismatches_path}")
    print(f"   - {len(mismatches_data)} mismatches analyzed")
    print(f"   - {sum(1 for m in mismatches_data if m['Has_Quote_Issues'])} have quote issues")
    print(f"   - {sum(1 for m in mismatches_data if m['Has_Duplication_Issues'])} have duplication issues")
    print(f"   - Average raw similarity: {sum(m['Raw_Similarity_Percent'] for m in mismatches_data) / len(mismatches_data):.1f}%")
    print(f"   - Average cleaned similarity: {sum(m['Cleaned_Similarity_Percent'] for m in mismatches_data) / len(mismatches_data):.1f}%")
    print(f"   - CSV includes detailed diff analysis with added/removed words and similarity scores")
    
    return mismatches_path

def clean_reconstructed_content(content: str) -> str:
    """
    Clean reconstructed content by removing quotes and fixing known issues
    
    Args:
        content: Raw reconstructed content
        
    Returns:
        Cleaned content
    """
    if not content:
        return content
    
    # Remove surrounding quotes that were added by CSV formatting
    cleaned = content.strip()
    
    # Remove quotes that wrap individual sentences/snippets
    # This is a simplified approach - we'll remove quotes that appear at snippet boundaries
    import re
    
    # Remove quotes around sentences
    cleaned = re.sub(r'"([^"]+)"', r'\1', cleaned)
    
    # Remove duplicate consecutive sentences (basic approach)
    sentences = cleaned.split('. ')
    deduplicated_sentences = []
    prev_sentence = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence != prev_sentence:
            deduplicated_sentences.append(sentence)
            prev_sentence = sentence
    
    cleaned = '. '.join(deduplicated_sentences)
    
    # Fix spacing issues
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

def has_duplication_issues(content: str) -> bool:
    """
    Check if content has obvious duplication issues
    
    Args:
        content: Content to check
        
    Returns:
        True if duplication issues detected
    """
    if not content:
        return False
    
    # Look for repeated phrases
    sentences = content.split('. ')
    seen_sentences = set()
    
    for sentence in sentences:
        sentence = sentence.strip().lower()
        if sentence and len(sentence) > 20:  # Only check substantial sentences
            if sentence in seen_sentences:
                return True
            seen_sentences.add(sentence)
    
    return False

def generate_difference_summary(original: str, reconstructed: str) -> str:
    """
    Generate a summary of differences between original and reconstructed content
    
    Args:
        original: Original content
        reconstructed: Reconstructed content
        
    Returns:
        Summary of differences
    """
    if not original or not reconstructed:
        return "Missing content"
    
    issues = []
    
    # Length difference
    len_diff = len(reconstructed) - len(original)
    if abs(len_diff) > 10:
        if len_diff > 0:
            issues.append(f"Reconstructed is {len_diff} chars longer")
        else:
            issues.append(f"Reconstructed is {abs(len_diff)} chars shorter")
    
    # Word count difference
    original_words = len(original.split())
    reconstructed_words = len(reconstructed.split())
    word_diff = reconstructed_words - original_words
    if abs(word_diff) > 5:
        if word_diff > 0:
            issues.append(f"Reconstructed has {word_diff} more words")
        else:
            issues.append(f"Reconstructed has {abs(word_diff)} fewer words")
    
    # Quote issues
    if '"' in reconstructed and reconstructed.count('"') > original.count('"'):
        issues.append("Extra quotes in reconstructed")
    
    # Basic duplication check
    if has_duplication_issues(reconstructed):
        issues.append("Potential duplicated content")
    
    # Similarity check (very basic)
    original_clean = ' '.join(original.split())
    reconstructed_clean = ' '.join(reconstructed.split())
    
    if original_clean == reconstructed_clean:
        return "Perfect match"
    elif not issues:
        issues.append("Minor formatting differences")
    
    return "; ".join(issues) if issues else "No major issues detected"

def generate_detailed_diff(original: str, reconstructed: str) -> dict:
    """
    Generate detailed differences between original and reconstructed text using difflib
    
    Args:
        original: Original text
        reconstructed: Reconstructed text
        
    Returns:
        Dictionary with detailed diff information
    """
    if not original or not reconstructed:
        return {
            "diff_type": "missing_text",
            "diff_html": "One of the texts is empty",
            "diff_unified": "One of the texts is empty",
            "added_text": "",
            "removed_text": "",
            "word_changes": 0,
            "char_changes": 0
        }
    
    # Split into words for word-level comparison
    original_words = original.split()
    reconstructed_words = reconstructed.split()
    
    # Generate unified diff (good for showing line-by-line changes)
    unified_diff = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        reconstructed.splitlines(keepends=True),
        fromfile='original',
        tofile='reconstructed',
        lineterm='',
        n=3
    ))
    
    # Generate HTML diff (visual representation)
    html_diff = difflib.HtmlDiff()
    html_table = html_diff.make_table(
        original.splitlines(),
        reconstructed.splitlines(),
        fromdesc='Original',
        todesc='Reconstructed',
        context=True,
        numlines=2
    )
    
    # Word-level sequence matcher
    word_matcher = difflib.SequenceMatcher(None, original_words, reconstructed_words)
    
    # Character-level sequence matcher  
    char_matcher = difflib.SequenceMatcher(None, original, reconstructed)
    
    # Extract added and removed text
    added_words = []
    removed_words = []
    
    for tag, i1, i2, j1, j2 in word_matcher.get_opcodes():
        if tag == 'delete':
            removed_words.extend(original_words[i1:i2])
        elif tag == 'insert':
            added_words.extend(reconstructed_words[j1:j2])
        elif tag == 'replace':
            removed_words.extend(original_words[i1:i2])
            added_words.extend(reconstructed_words[j1:j2])
    
    # Count changes
    word_changes = len([op for op in word_matcher.get_opcodes() if op[0] != 'equal'])
    char_changes = len([op for op in char_matcher.get_opcodes() if op[0] != 'equal'])
    
    # Determine diff type
    ratio = char_matcher.ratio()
    if ratio > 0.95:
        diff_type = "minor_differences"
    elif ratio > 0.8:
        diff_type = "moderate_differences"
    elif ratio > 0.5:
        diff_type = "major_differences"
    else:
        diff_type = "completely_different"
    
    return {
        "diff_type": diff_type,
        "similarity_ratio": round(ratio, 3),
        "diff_html": html_table,
        "diff_unified": '\n'.join(unified_diff),
        "added_text": ' '.join(added_words),
        "removed_text": ' '.join(removed_words),
        "word_changes": word_changes,
        "char_changes": char_changes,
        "word_additions": len(added_words),
        "word_removals": len(removed_words)
    }

def extract_simple_differences(original: str, reconstructed: str) -> dict:
    """
    Extract simple, readable differences for CSV display
    
    Args:
        original: Original text
        reconstructed: Reconstructed text
        
    Returns:
        Dictionary with simple difference descriptions
    """
    diff_info = generate_detailed_diff(original, reconstructed)
    
    # Simple descriptions for CSV
    differences = []
    
    if diff_info["word_additions"] > 0:
        differences.append(f"+{diff_info['word_additions']} words")
    if diff_info["word_removals"] > 0:
        differences.append(f"-{diff_info['word_removals']} words")
    
    if diff_info["added_text"]:
        added_preview = diff_info["added_text"][:100] + "..." if len(diff_info["added_text"]) > 100 else diff_info["added_text"]
        differences.append(f"Added: '{added_preview}'")
    
    if diff_info["removed_text"]:
        removed_preview = diff_info["removed_text"][:100] + "..." if len(diff_info["removed_text"]) > 100 else diff_info["removed_text"]
        differences.append(f"Removed: '{removed_preview}'")
    
    return {
        "simple_diff": "; ".join(differences) if differences else "No major differences detected",
        "similarity_percentage": round(diff_info["similarity_ratio"] * 100, 1),
        "change_type": diff_info["diff_type"],
        "added_words": diff_info["word_additions"],
        "removed_words": diff_info["word_removals"]
    }

# ========================================
# MAIN PROCESSING FUNCTIONS
# ========================================

def extract_speaker_team(speaker: str, transcript: dict) -> str:
    """Extract team membership from JSON metadata"""
    # Check if speaker is in management_team
    if "metadata" in transcript and "management_team" in transcript["metadata"]:
        for member in transcript["metadata"]["management_team"]:
            if member.lower() in speaker.lower() or speaker.lower() in member.lower():
                return "management_team"
    
    # Check if speaker is in external_members (analysts)
    if "metadata" in transcript and "external_members" in transcript["metadata"]:
        for member in transcript["metadata"]["external_members"]:
            if member.lower() in speaker.lower() or speaker.lower() in member.lower():
                return "external_members"
    
    # Default fallback
    return "unknown"

def extract_speaker_role(speaker: str) -> str:
    """Extract role/title from speaker name"""
    speaker_upper = speaker.upper()
    
    if 'CEO' in speaker_upper:
        return 'CEO'
    elif 'CFO' in speaker_upper:
        return 'CFO'
    elif 'CTO' in speaker_upper:
        return 'CTO'
    elif 'COO' in speaker_upper:
        return 'COO'
    elif 'PRESIDENT' in speaker_upper:
        return 'President'
    elif 'CHAIRMAN' in speaker_upper:
        return 'Chairman'
    elif 'DIRECTOR' in speaker_upper and 'IR' in speaker_upper:
        return 'IR Director'
    elif 'ANALYST' in speaker_upper:
        return 'Analyst'
    elif 'OPERATOR' in speaker_upper:
        return 'Operator'
    else:
        return ''

def clean_multiple_encoding(text: str) -> str:
    """Handle multiple levels of UTF-8 encoding"""
    if not text:
        return text
    
    # Common multi-encoded patterns
    replacements = {
        'â€™': "'",  # Triple-encoded apostrophe
        'â€œ': '"',  # Triple-encoded open quote
        'â€': '"',   # Triple-encoded close quote
        'â€"': '—',  # Triple-encoded em dash
        'â€"': '–',  # Triple-encoded en dash
    }
    
    cleaned = text
    for encoded, decoded in replacements.items():
        cleaned = cleaned.replace(encoded, decoded)
    
    return cleaned

def decode_json_text(text: str) -> str:
    """Decode JSON unicode escapes and handle special characters including multiple-encoded UTF-8"""
    if not text or not isinstance(text, str):
        return text
    
    # Handle unicode escapes (like \u2019 for apostrophes)
    try:
        # Try to decode JSON unicode escapes
        decoded_text = text.encode().decode('unicode_escape')
        
        # Handle double-encoded UTF-8 by trying to decode it properly
        try:
            # First encode as latin-1 to get raw bytes, then decode as utf-8
            double_decoded = decoded_text.encode('latin-1').decode('utf-8')
            # If successful, use the double-decoded version
            decoded_text = double_decoded
        except (UnicodeDecodeError, UnicodeEncodeError):
            # If double-decoding fails, keep the single-decoded version
            pass
            
        # Apply multiple encoding cleanup for triple+ encoded patterns
        decoded_text = clean_multiple_encoding(decoded_text)
        
        return decoded_text
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If that fails, try JSON loads to handle unicode
        try:
            decoded_text = json.loads(f'"{text}"')
            
            # Try double-decoding on the JSON-decoded text too
            try:
                double_decoded = decoded_text.encode('latin-1').decode('utf-8')
                decoded_text = double_decoded
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
            
            # Apply multiple encoding cleanup for triple+ encoded patterns
            decoded_text = clean_multiple_encoding(decoded_text)
                
            return decoded_text
        except (json.JSONDecodeError, ValueError):
            # If all else fails, apply cleanup and return
            return clean_multiple_encoding(text)

def calculate_word_count(text: str) -> int:
    """Calculate word count for percentage calculations"""
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())

def process_transcript_file(file_path: str) -> tuple:
    """
    Process a single transcript file through the classification pipeline
    
    Args:
        file_path: Path to the JSON transcript file
        
    Returns:
        Tuple of (snippet_results, processing_stats)
    """
    start_time = time.time()
    file_name = os.path.basename(file_path)
    print(f"Processing transcript file: {file_name}")
    
    # Initialize processing statistics
    processing_stats = {
        'File_Name': file_name,
        'File_Path': file_path,
        'Start_Time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
        'End_Time': '',
        'Processing_Time_Seconds': 0,
        'Processing_Time_Minutes': 0,
        'Pass1_Topic_Classification_Time_Seconds': 0,
        'Pass1_Topic_Classification_Time_Minutes': 0,
        'Pass2_Sentence_Attribution_Time_Seconds': 0,
        'Pass2_Sentence_Attribution_Time_Minutes': 0,
        'Pass3_Contextual_Refinement_Time_Seconds': 0,
        'Pass3_Contextual_Refinement_Time_Minutes': 0,
        'Total_LLM_Calls': 0,
        'Topic_Classification_Calls': 0,
        'Sentence_Attribution_Calls': 0,
        'Contextual_Refinement_Calls': 0,
        'Contextual_Refinements_Applied': 0,
        'Total_Paragraphs_Processed': 0,
        'Paragraphs_With_Sentence_Attribution': 0,
        'Total_Snippets_Created': 0,
        'Prepared_Remarks_Paragraphs': 0,
        'QnA_Paragraphs': 0,
        'Total_Companies': 0,
        'Total_Speakers': 0,
        'Total_Transcript_Words': 0,
        'Average_Words_Per_Paragraph': 0,
        'Error_Count': 0,
        'Success_Rate': 0
    }
    
    # Load JSON transcript data
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Determine if it's a single transcript or a list
    transcripts = data if isinstance(data, list) else [data]
    
    all_snippets = []
    companies_set = set()
    speakers_set = set()
    
    # NEW: Global paragraph order counter for proper tracking
    global_paragraph_order = 0
    
    for transcript in transcripts:
        # Extract metadata
        company = transcript["symbol"]
        quarter = transcript["quarter"]
        year = transcript["year"]
        
        # Track companies and speakers
        companies_set.add(company)
        
        # Calculate transcript length
        total_transcript_text = ""
        sections_text = {}
        
        # Collect all text for length calculations
        if "parsed_content" in transcript:
            # Process prepared remarks
            if "prepared_remarks" in transcript["parsed_content"]:
                section_text = ""
                for pr in transcript["parsed_content"]["prepared_remarks"]:
                    paragraph_fields = [key for key in pr.keys() if key.startswith("paragraph_")]
                    for para_field in paragraph_fields:
                        if para_field in pr and isinstance(pr[para_field], str):
                            # Decode JSON text for consistent length calculation
                            decoded_text = decode_json_text(pr[para_field])
                            section_text += " " + decoded_text
                            total_transcript_text += " " + decoded_text
                sections_text["prepared_remarks"] = section_text.strip()
            
            # Process Q&A section
            if "qna_section" in transcript["parsed_content"]:
                section_text = ""
                for qa in transcript["parsed_content"]["qna_section"]:
                    paragraph_fields = [key for key in qa.keys() if key.startswith("paragraph_")]
                    for para_field in paragraph_fields:
                        if para_field in qa and isinstance(qa[para_field], str):
                            # Decode JSON text for consistent length calculation
                            decoded_text = decode_json_text(qa[para_field])
                            section_text += " " + decoded_text
                            total_transcript_text += " " + decoded_text
                sections_text["qna_section"] = section_text.strip()
        
        total_transcript_length = calculate_word_count(total_transcript_text)
        
        # Process prepared remarks section
        if "parsed_content" in transcript and "prepared_remarks" in transcript["parsed_content"]:
            prepared_remarks = transcript["parsed_content"]["prepared_remarks"]
            section_length = calculate_word_count(sections_text.get("prepared_remarks", ""))
            
            for pr in tqdm(prepared_remarks, desc="Processing prepared remarks"):
                speaker = pr["speaker"]
                team = extract_speaker_team(speaker, transcript)
                role = extract_speaker_role(speaker)
                speaker_order = pr.get("order", 0)  # Get speaker order from JSON
                
                # Track speakers
                speakers_set.add(speaker)
                
                # Extract all paragraph fields
                paragraph_fields = [key for key in pr.keys() if key.startswith("paragraph_")]
                paragraph_fields.sort(key=lambda x: int(x.split("_")[1]))
                
                # Process each paragraph
                for para_idx, para_field in enumerate(paragraph_fields):
                    if para_field not in pr or not isinstance(pr[para_field], str) or not pr[para_field].strip():
                        continue
                        
                    # Decode JSON unicode escapes and clean text
                    text = decode_json_text(pr[para_field].strip())
                    
                    # Track paragraph processing
                    processing_stats['Total_Paragraphs_Processed'] += 1
                    processing_stats['Prepared_Remarks_Paragraphs'] += 1
                    
                    # FIXED: Proper paragraph order tracking
                    global_paragraph_order += 1
                    paragraph_number = int(para_field.split("_")[1])  # Extract paragraph number from field name
                    
                    # PASS 1: Initial topic classification
                    pass1_start = time.time()
                    topics_with_scores, total_coverage, needs_refinement, refinement_reason = classify_topics_structured(
                        company, speaker, text, stats_tracker=processing_stats
                    )
                    pass1_end = time.time()
                    processing_stats['Pass1_Topic_Classification_Time_Seconds'] += (pass1_end - pass1_start)
                    
                    # PASS 2: Sentence attribution for multi-topic paragraphs
                    sentence_attributions = None
                    has_sentence_attribution = False
                    
                    # Only do sentence attribution if we have multiple topics and multiple sentences
                    if len(topics_with_scores) >= 2 and len(nltk.sent_tokenize(text)) >= 2:
                        pass2_start = time.time()
                        sentence_attributions = attribute_sentences_structured(text, topics_with_scores, stats_tracker=processing_stats)
                        pass2_end = time.time()
                        processing_stats['Pass2_Sentence_Attribution_Time_Seconds'] += (pass2_end - pass2_start)
                        if sentence_attributions and len(sentence_attributions) > 1:
                            has_sentence_attribution = True
                    
                    # CORRECTED logic with rejoining - ENSURE MUTUALLY EXCLUSIVE:
                    if has_sentence_attribution:
                        # ONLY create sentence-level snippets, no paragraph-level
                        processing_stats['Paragraphs_With_Sentence_Attribution'] += 1
                        
                        # Get rejoined sentence groups
                        rejoined_attributions = rejoin_same_topic_sentences(sentence_attributions)
                        
                        # Create one snippet per rejoined group
                        for attribution in rejoined_attributions:
                            snippet_length = calculate_word_count(attribution['sentence'])
                            
                            snippet_data = {
                                'Company': company,
                                'Quarter': quarter,
                                'Date': year,
                                'Speaker_Name': speaker,
                                'Team': team,
                                'Role': role,
                                'Section': 'prepared_remarks',
                                'Snippet': decode_json_text(attribution['sentence']),
                                'Snippet_Pct_Transcript': round((snippet_length / total_transcript_length) * 100, 4) if total_transcript_length > 0 else 0,
                                'Total_Transcript_Length': total_transcript_length,
                                'Snippet_Pct_Section': round((snippet_length / section_length) * 100, 4) if section_length > 0 else 0,
                                'Total_Section_Length': section_length,
                                'Snippet_Level': 'sentence' if attribution.get('sentence_count', 1) == 1 else 'merged_sentences',
                                'Has_Sentence_Attribution': True,
                                'Primary_Topic': attribution['attributed_topic'],
                                'Primary_Topic_Confidence': attribution['confidence'],
                                'Primary_Topic_Coverage': 100,
                                'Temporal_Context': attribution['temporal_context'],
                                'Content_Sentiment': topics_with_scores[0]['content_sentiment'],
                                'Speaker_Tone': topics_with_scores[0]['speaker_tone'],
                                'Total_Coverage': 100,
                                'Needs_Refinement': needs_refinement,
                                'Refinement_Reason': refinement_reason or '',
                                'Original_Paragraph': text,
                                'Original_Paragraph_Order': global_paragraph_order,
                                'Speaker_Order': speaker_order,
                                'Paragraph_Number': paragraph_number
                            }
                            
                            # Add all topic classifications
                            for i in range(5):
                                topic_prefix = ['Primary', 'Secondary', 'Tertiary', 'Quaternary', 'Quinary'][i]
                                if i < len(topics_with_scores):
                                    topic = topics_with_scores[i]
                                    snippet_data[f'{topic_prefix}_Topic'] = topic['topic']
                                    snippet_data[f'{topic_prefix}_Topic_Confidence'] = topic['confidence']
                                    snippet_data[f'{topic_prefix}_Topic_Coverage'] = topic['topic_coverage']
                                    snippet_data[f'{topic_prefix}_Temporal_Context'] = topic['temporal_context']
                                    snippet_data[f'{topic_prefix}_Content_Sentiment'] = topic['content_sentiment']
                                else:
                                    snippet_data[f'{topic_prefix}_Topic'] = ''
                                    snippet_data[f'{topic_prefix}_Topic_Confidence'] = 0
                                    snippet_data[f'{topic_prefix}_Topic_Coverage'] = 0
                                    snippet_data[f'{topic_prefix}_Temporal_Context'] = ''
                                    snippet_data[f'{topic_prefix}_Content_Sentiment'] = ''
                            
                            all_snippets.append(snippet_data)
                    else:
                        # ONLY create paragraph-level snippet, no sentence-level
                        # Create one snippet for the entire paragraph
                        snippet_length = calculate_word_count(text)
                        
                        snippet_data = {
                            'Company': company,
                            'Quarter': quarter,
                            'Date': year,
                            'Speaker_Name': speaker,
                            'Team': team,
                            'Role': role,
                            'Section': 'prepared_remarks',
                            'Snippet': text,  # Already decoded above
                            'Snippet_Pct_Transcript': round((snippet_length / total_transcript_length) * 100, 4) if total_transcript_length > 0 else 0,
                            'Total_Transcript_Length': total_transcript_length,
                            'Snippet_Pct_Section': round((snippet_length / section_length) * 100, 4) if section_length > 0 else 0,
                            'Total_Section_Length': section_length,
                            'Snippet_Level': 'paragraph',
                            'Has_Sentence_Attribution': False,
                            'Total_Coverage': total_coverage,
                            'Needs_Refinement': needs_refinement,
                            'Refinement_Reason': refinement_reason or '',
                            'Original_Paragraph': text,
                            'Original_Paragraph_Order': global_paragraph_order,
                            'Speaker_Order': speaker_order,
                            'Paragraph_Number': paragraph_number
                        }
                        
                        # Add topic classifications  
                        for i in range(5):
                            topic_prefix = ['Primary', 'Secondary', 'Tertiary', 'Quaternary', 'Quinary'][i]
                            if i < len(topics_with_scores):
                                topic = topics_with_scores[i]
                                snippet_data[f'{topic_prefix}_Topic'] = topic['topic']
                                snippet_data[f'{topic_prefix}_Topic_Confidence'] = topic['confidence']
                                snippet_data[f'{topic_prefix}_Topic_Coverage'] = topic['topic_coverage']
                                snippet_data[f'{topic_prefix}_Temporal_Context'] = topic['temporal_context']
                                snippet_data[f'{topic_prefix}_Content_Sentiment'] = topic['content_sentiment']
                                if i == 0:  # Primary topic
                                    snippet_data['Temporal_Context'] = topic['temporal_context']
                                    snippet_data['Content_Sentiment'] = topic['content_sentiment']
                                    snippet_data['Speaker_Tone'] = topic['speaker_tone']
                            else:
                                snippet_data[f'{topic_prefix}_Topic'] = ''
                                snippet_data[f'{topic_prefix}_Topic_Confidence'] = 0
                                snippet_data[f'{topic_prefix}_Topic_Coverage'] = 0
                                snippet_data[f'{topic_prefix}_Temporal_Context'] = ''
                                snippet_data[f'{topic_prefix}_Content_Sentiment'] = ''
                        
                        all_snippets.append(snippet_data)
        
        # Process Q&A section (same logic as prepared remarks)
        if "parsed_content" in transcript and "qna_section" in transcript["parsed_content"]:
            qna_section = transcript["parsed_content"]["qna_section"]
            section_length = calculate_word_count(sections_text.get("qna_section", ""))
            
            for qa in tqdm(qna_section, desc="Processing Q&A section"):
                speaker = qa["speaker"]
                team = extract_speaker_team(speaker, transcript)
                role = extract_speaker_role(speaker)
                speaker_order = qa.get("order", 0)  # Get speaker order from JSON
                
                # Track speakers
                speakers_set.add(speaker)
                
                # Extract all paragraph fields
                paragraph_fields = [key for key in qa.keys() if key.startswith("paragraph_")]
                paragraph_fields.sort(key=lambda x: int(x.split("_")[1]))
                
                # Process each paragraph
                for para_idx, para_field in enumerate(paragraph_fields):
                    if para_field not in qa or not isinstance(qa[para_field], str) or not qa[para_field].strip():
                        continue
                        
                    # Decode JSON unicode escapes and clean text
                    text = decode_json_text(qa[para_field].strip())
                    
                    # Track paragraph processing
                    processing_stats['Total_Paragraphs_Processed'] += 1
                    
                    # FIXED: Proper paragraph order tracking
                    global_paragraph_order += 1
                    paragraph_number = int(para_field.split("_")[1])  # Extract paragraph number from field name
                    
                    # PASS 1: Initial topic classification
                    pass1_start = time.time()
                    topics_with_scores, total_coverage, needs_refinement, refinement_reason = classify_topics_structured(
                        company, speaker, text, stats_tracker=processing_stats
                    )
                    pass1_end = time.time()
                    processing_stats['Pass1_Topic_Classification_Time_Seconds'] += (pass1_end - pass1_start)
                    
                    # PASS 2: Sentence attribution for multi-topic paragraphs
                    sentence_attributions = None
                    has_sentence_attribution = False
                    
                    # Only do sentence attribution if we have multiple topics and multiple sentences
                    if len(topics_with_scores) >= 2 and len(nltk.sent_tokenize(text)) >= 2:
                        pass2_start = time.time()
                        sentence_attributions = attribute_sentences_structured(text, topics_with_scores, stats_tracker=processing_stats)
                        pass2_end = time.time()
                        processing_stats['Pass2_Sentence_Attribution_Time_Seconds'] += (pass2_end - pass2_start)
                        if sentence_attributions and len(sentence_attributions) > 1:
                            has_sentence_attribution = True
                    
                    # CORRECTED logic with rejoining - ENSURE MUTUALLY EXCLUSIVE:
                    if has_sentence_attribution:
                        # ONLY create sentence-level snippets, no paragraph-level
                        processing_stats['Paragraphs_With_Sentence_Attribution'] += 1
                        
                        # Get rejoined sentence groups
                        rejoined_attributions = rejoin_same_topic_sentences(sentence_attributions)
                        
                        # Create one snippet per rejoined group
                        for attribution in rejoined_attributions:
                            snippet_length = calculate_word_count(attribution['sentence'])
                            
                            snippet_data = {
                                'Company': company,
                                'Quarter': quarter,
                                'Date': year,
                                'Speaker_Name': speaker,
                                'Team': team,
                                'Role': role,
                                'Section': 'qna_section',
                                'Snippet': decode_json_text(attribution['sentence']),
                                'Snippet_Pct_Transcript': round((snippet_length / total_transcript_length) * 100, 4) if total_transcript_length > 0 else 0,
                                'Total_Transcript_Length': total_transcript_length,
                                'Snippet_Pct_Section': round((snippet_length / section_length) * 100, 4) if section_length > 0 else 0,
                                'Total_Section_Length': section_length,
                                'Snippet_Level': 'sentence' if attribution.get('sentence_count', 1) == 1 else 'merged_sentences',
                                'Has_Sentence_Attribution': True,
                                'Primary_Topic': attribution['attributed_topic'],
                                'Primary_Topic_Confidence': attribution['confidence'],
                                'Primary_Topic_Coverage': 100,
                                'Temporal_Context': attribution['temporal_context'],
                                'Content_Sentiment': topics_with_scores[0]['content_sentiment'],
                                'Speaker_Tone': topics_with_scores[0]['speaker_tone'],
                                'Total_Coverage': 100,
                                'Needs_Refinement': needs_refinement,
                                'Refinement_Reason': refinement_reason or '',
                                'Original_Paragraph': text,
                                'Original_Paragraph_Order': global_paragraph_order,
                                'Speaker_Order': speaker_order,
                                'Paragraph_Number': paragraph_number
                            }
                            
                            # Add all topic classifications
                            for i in range(5):
                                topic_prefix = ['Primary', 'Secondary', 'Tertiary', 'Quaternary', 'Quinary'][i]
                                if i < len(topics_with_scores):
                                    topic = topics_with_scores[i]
                                    snippet_data[f'{topic_prefix}_Topic'] = topic['topic']
                                    snippet_data[f'{topic_prefix}_Topic_Confidence'] = topic['confidence']
                                    snippet_data[f'{topic_prefix}_Topic_Coverage'] = topic['topic_coverage']
                                    snippet_data[f'{topic_prefix}_Temporal_Context'] = topic['temporal_context']
                                    snippet_data[f'{topic_prefix}_Content_Sentiment'] = topic['content_sentiment']
                                else:
                                    snippet_data[f'{topic_prefix}_Topic'] = ''
                                    snippet_data[f'{topic_prefix}_Topic_Confidence'] = 0
                                    snippet_data[f'{topic_prefix}_Topic_Coverage'] = 0
                                    snippet_data[f'{topic_prefix}_Temporal_Context'] = ''
                                    snippet_data[f'{topic_prefix}_Content_Sentiment'] = ''
                            
                            all_snippets.append(snippet_data)
                    else:
                        # ONLY create paragraph-level snippet, no sentence-level
                        # Create one snippet for the entire paragraph
                        snippet_length = calculate_word_count(text)
                        
                        snippet_data = {
                            'Company': company,
                            'Quarter': quarter,
                            'Date': year,
                            'Speaker_Name': speaker,
                            'Team': team,
                            'Role': role,
                            'Section': 'qna_section',
                            'Snippet': text,  # Already decoded above
                            'Snippet_Pct_Transcript': round((snippet_length / total_transcript_length) * 100, 4) if total_transcript_length > 0 else 0,
                            'Total_Transcript_Length': total_transcript_length,
                            'Snippet_Pct_Section': round((snippet_length / section_length) * 100, 4) if section_length > 0 else 0,
                            'Total_Section_Length': section_length,
                            'Snippet_Level': 'paragraph',
                            'Has_Sentence_Attribution': False,
                            'Total_Coverage': total_coverage,
                            'Needs_Refinement': needs_refinement,
                            'Refinement_Reason': refinement_reason or '',
                            'Original_Paragraph': text,
                            'Original_Paragraph_Order': global_paragraph_order,
                            'Speaker_Order': speaker_order,
                            'Paragraph_Number': paragraph_number
                        }
                        
                        # Add topic classifications  
                        for i in range(5):
                            topic_prefix = ['Primary', 'Secondary', 'Tertiary', 'Quaternary', 'Quinary'][i]
                            if i < len(topics_with_scores):
                                topic = topics_with_scores[i]
                                snippet_data[f'{topic_prefix}_Topic'] = topic['topic']
                                snippet_data[f'{topic_prefix}_Topic_Confidence'] = topic['confidence']
                                snippet_data[f'{topic_prefix}_Topic_Coverage'] = topic['topic_coverage']
                                snippet_data[f'{topic_prefix}_Temporal_Context'] = topic['temporal_context']
                                snippet_data[f'{topic_prefix}_Content_Sentiment'] = topic['content_sentiment']
                                if i == 0:  # Primary topic
                                    snippet_data['Temporal_Context'] = topic['temporal_context']
                                    snippet_data['Content_Sentiment'] = topic['content_sentiment']
                                    snippet_data['Speaker_Tone'] = topic['speaker_tone']
                            else:
                                snippet_data[f'{topic_prefix}_Topic'] = ''
                                snippet_data[f'{topic_prefix}_Topic_Confidence'] = 0
                                snippet_data[f'{topic_prefix}_Topic_Coverage'] = 0
                                snippet_data[f'{topic_prefix}_Temporal_Context'] = ''
                                snippet_data[f'{topic_prefix}_Content_Sentiment'] = ''
                        
                        all_snippets.append(snippet_data)
    
    # Finalize processing statistics (before Pass 3)
    end_time_before_pass3 = time.time()
    
    # Add Pass 3 timing to total processing time
    total_processing_time = end_time_before_pass3 - start_time + processing_stats.get('Pass3_Contextual_Refinement_Time_Seconds', 0)
    
    processing_stats['End_Time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_before_pass3 + processing_stats.get('Pass3_Contextual_Refinement_Time_Seconds', 0)))
    processing_stats['Processing_Time_Seconds'] = round(total_processing_time, 2)
    processing_stats['Processing_Time_Minutes'] = round(total_processing_time / 60, 2)
    processing_stats['Pass1_Topic_Classification_Time_Minutes'] = round(processing_stats['Pass1_Topic_Classification_Time_Seconds'] / 60, 2)
    processing_stats['Pass2_Sentence_Attribution_Time_Minutes'] = round(processing_stats['Pass2_Sentence_Attribution_Time_Seconds'] / 60, 2)
    # Pass 3 timing is already calculated above
    processing_stats['Total_Companies'] = len(companies_set)
    processing_stats['Total_Speakers'] = len(speakers_set)
    processing_stats['Total_Transcript_Words'] = total_transcript_length
    processing_stats['Average_Words_Per_Paragraph'] = round(total_transcript_length / max(processing_stats['Total_Paragraphs_Processed'], 1), 2)
    processing_stats['Total_Snippets_Created'] = len(all_snippets)
    processing_stats['QnA_Paragraphs'] = processing_stats['Total_Paragraphs_Processed'] - processing_stats['Prepared_Remarks_Paragraphs']
    
    # Calculate success rate
    total_attempts = processing_stats['Total_LLM_Calls'] + processing_stats['Error_Count']
    processing_stats['Success_Rate'] = round((processing_stats['Total_LLM_Calls'] / max(total_attempts, 1)) * 100, 2)
    
    # PASS 3: Apply contextual refinement within this file
    print(f"Pass 3: Contextual refinement for ambiguous snippets...")
    pass3_start = time.time()
    
    all_snippets = apply_contextual_refinement_pass_single_file(all_snippets, processing_stats)
    
    pass3_end = time.time()
    processing_stats['Pass3_Contextual_Refinement_Time_Seconds'] = pass3_end - pass3_start
    processing_stats['Pass3_Contextual_Refinement_Time_Minutes'] = round((pass3_end - pass3_start) / 60, 2)
    
    print(f"Processed {len(all_snippets)} snippets from {len(transcripts)} transcript(s)")
    print(f"Processing completed in {processing_stats['Processing_Time_Minutes']} minutes")
    return all_snippets, processing_stats

def create_processing_summary_csv(all_processing_stats: List[dict]):
    if not all_processing_stats:
        print("No processing statistics to export")
        return
    
    # Create DataFrame from processing stats
    stats_df = pd.DataFrame(all_processing_stats)
    
    # Calculate overall totals
    total_row = {
        'File_Name': 'TOTAL_SUMMARY',
        'File_Path': f'{len(all_processing_stats)} files processed',
        'Start_Time': min([stats['Start_Time'] for stats in all_processing_stats]),
        'End_Time': max([stats['End_Time'] for stats in all_processing_stats]),
        'Processing_Time_Seconds': sum([stats['Processing_Time_Seconds'] for stats in all_processing_stats]),
        'Processing_Time_Minutes': sum([stats['Processing_Time_Minutes'] for stats in all_processing_stats]),
        'Pass1_Topic_Classification_Time_Seconds': sum([stats.get('Pass1_Topic_Classification_Time_Seconds', 0) for stats in all_processing_stats]),
        'Pass1_Topic_Classification_Time_Minutes': sum([stats.get('Pass1_Topic_Classification_Time_Minutes', 0) for stats in all_processing_stats]),
        'Pass2_Sentence_Attribution_Time_Seconds': sum([stats.get('Pass2_Sentence_Attribution_Time_Seconds', 0) for stats in all_processing_stats]),
        'Pass2_Sentence_Attribution_Time_Minutes': sum([stats.get('Pass2_Sentence_Attribution_Time_Minutes', 0) for stats in all_processing_stats]),
        'Pass3_Contextual_Refinement_Time_Seconds': sum([stats.get('Pass3_Contextual_Refinement_Time_Seconds', 0) for stats in all_processing_stats]),
        'Pass3_Contextual_Refinement_Time_Minutes': sum([stats.get('Pass3_Contextual_Refinement_Time_Minutes', 0) for stats in all_processing_stats]),
        'Total_LLM_Calls': sum([stats['Total_LLM_Calls'] for stats in all_processing_stats]),
        'Topic_Classification_Calls': sum([stats['Topic_Classification_Calls'] for stats in all_processing_stats]),
        'Sentence_Attribution_Calls': sum([stats['Sentence_Attribution_Calls'] for stats in all_processing_stats]),
        'Contextual_Refinement_Calls': sum([stats['Contextual_Refinement_Calls'] for stats in all_processing_stats]),
        'Contextual_Refinements_Applied': sum([stats['Contextual_Refinements_Applied'] for stats in all_processing_stats]),
        'Total_Paragraphs_Processed': sum([stats['Total_Paragraphs_Processed'] for stats in all_processing_stats]),
        'Paragraphs_With_Sentence_Attribution': sum([stats['Paragraphs_With_Sentence_Attribution'] for stats in all_processing_stats]),
        'Total_Snippets_Created': sum([stats['Total_Snippets_Created'] for stats in all_processing_stats]),
        'Prepared_Remarks_Paragraphs': sum([stats['Prepared_Remarks_Paragraphs'] for stats in all_processing_stats]),
        'QnA_Paragraphs': sum([stats['QnA_Paragraphs'] for stats in all_processing_stats]),
        'Total_Companies': len(set([stats['Total_Companies'] for stats in all_processing_stats if stats['Total_Companies'] > 0])),
        'Total_Speakers': sum([stats['Total_Speakers'] for stats in all_processing_stats]),
        'Total_Transcript_Words': sum([stats['Total_Transcript_Words'] for stats in all_processing_stats]),
        'Average_Words_Per_Paragraph': round(sum([stats['Total_Transcript_Words'] for stats in all_processing_stats]) / 
                                           max(sum([stats['Total_Paragraphs_Processed'] for stats in all_processing_stats]), 1), 2),
        'Error_Count': sum([stats['Error_Count'] for stats in all_processing_stats]),
        'Success_Rate': round((sum([stats['Total_LLM_Calls'] for stats in all_processing_stats]) / 
                              max(sum([stats['Total_LLM_Calls'] for stats in all_processing_stats]) + 
                                  sum([stats['Error_Count'] for stats in all_processing_stats]), 1)) * 100, 2),
        'Final_Snippets_After_Optimization': sum([stats.get('Final_Snippets_After_Optimization', 0) for stats in all_processing_stats]),
        'Meeting_Logistics_Merged': sum([stats.get('Meeting_Logistics_Merged', 0) for stats in all_processing_stats]),
        'Content_Validation_Passed': sum([1 for stats in all_processing_stats if stats.get('Content_Validation_Passed', False)]),
        'Content_Validation_Failed': sum([1 for stats in all_processing_stats if not stats.get('Content_Validation_Passed', True)]),
        'Total_Content_Validation_Issues': sum([stats.get('Content_Validation_Issues', 0) for stats in all_processing_stats]),
        'Total_Missing_Sections': sum([stats.get('Content_Validation_Missing_Sections', 0) for stats in all_processing_stats]),
        'Total_Extra_Sections': sum([stats.get('Content_Validation_Extra_Sections', 0) for stats in all_processing_stats])
    }
    
    # Add total row
    stats_df = pd.concat([stats_df, pd.DataFrame([total_row])], ignore_index=True)
    
    # Save processing summary
    summary_filename = f"processing_summary_{len(all_processing_stats)}_files.csv"
    summary_path = os.path.join(METADATA_DIR, summary_filename)
    stats_df.to_csv(summary_path, index=False, encoding='utf-8')
    
    print(f"\nProcessing summary saved to: {summary_path}")
    
    # Print key statistics
    print(f"\n{'='*50}")
    print("PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total files processed: {len(all_processing_stats)}")
    print(f"Total processing time: {total_row['Processing_Time_Minutes']:.2f} minutes")
    print(f"  - Pass 1 (Topic Classification): {total_row.get('Pass1_Topic_Classification_Time_Minutes', 0):.2f} minutes")
    print(f"  - Pass 2 (Sentence Attribution): {total_row.get('Pass2_Sentence_Attribution_Time_Minutes', 0):.2f} minutes")
    print(f"  - Pass 3 (Contextual Refinement): {total_row.get('Pass3_Contextual_Refinement_Time_Minutes', 0):.2f} minutes")
    print(f"Total LLM calls: {total_row['Total_LLM_Calls']}")
    print(f"  - Topic classification calls: {total_row['Topic_Classification_Calls']}")
    print(f"  - Sentence attribution calls: {total_row['Sentence_Attribution_Calls']}")
    print(f"  - Contextual refinement calls: {total_row['Contextual_Refinement_Calls']}")
    print(f"Total paragraphs processed: {total_row['Total_Paragraphs_Processed']}")
    print(f"Total snippets created: {total_row['Total_Snippets_Created']}")
    print(f"Contextual refinements applied: {total_row['Contextual_Refinements_Applied']}")
    print(f"Meeting logistics snippets merged: {total_row['Meeting_Logistics_Merged']}")
    print(f"Success rate: {total_row['Success_Rate']:.2f}%")
    print(f"Content validation: {total_row['Content_Validation_Passed']}/{len(all_processing_stats)} files passed")
    if total_row['Total_Content_Validation_Issues'] > 0:
        print(f"Content validation issues: {total_row['Total_Content_Validation_Issues']} total")
        print(f"  - Missing sections: {total_row['Total_Missing_Sections']}")
        print(f"  - Extra sections: {total_row['Total_Extra_Sections']}")
    print(f"Average processing time per file: {total_row['Processing_Time_Minutes']/len(all_processing_stats):.2f} minutes")

def generate_output_filename(all_snippets: List[dict]) -> str:
    """
    Generate output filename based on company, quarter, and year information from snippets
    Each file relates to a specific quarter/year for a single company.
    
    Args:
        all_snippets: List of processed snippets
        
    Returns:
        Formatted filename string
    """
    if not all_snippets:
        return "topic_class_no_data.csv"
    
    # Get company, quarter, year from first snippet (all should be the same)
    first_snippet = all_snippets[0]
    company = first_snippet.get('Company', 'UNKNOWN')
    quarter = first_snippet.get('Quarter', 'UNKNOWN') 
    year = first_snippet.get('Date', 'UNKNOWN')  # 'Date' field contains the year
    
    return f"{company}_{quarter}_{year}_topic_class.csv"

def generate_summary_statistics(df: pd.DataFrame):
    if df.empty:
        print("No data to analyze")
        return
    
    print(f"\n{'='*50}")
    print("SUMMARY STATISTICS")
    print(f"{'='*50}")
    
    print(f"Total snippets: {len(df)}")
    print(f"Total companies: {df['Company'].nunique()}")
    print(f"Total speakers: {df['Speaker_Name'].nunique()}")
    
    # Snippet level analysis
    snippet_level_counts = df['Snippet_Level'].value_counts()
    print(f"\nSnippet Level Distribution:")
    for level, count in snippet_level_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {level}: {count} ({percentage:.2f}%)")
    
    # Sentence attribution analysis
    has_attribution_count = df['Has_Sentence_Attribution'].sum()
    attribution_percentage = (has_attribution_count / len(df)) * 100
    print(f"\nSentence Attribution:")
    print(f"  Snippets with sentence attribution: {has_attribution_count} ({attribution_percentage:.2f}%)")
    
    # Contextual refinement analysis
    if 'Contextual_Refinement_Applied' in df.columns:
        refinement_applied_count = df['Contextual_Refinement_Applied'].sum()
        refinement_percentage = (refinement_applied_count / len(df)) * 100
        print(f"\nContextual Refinement:")
        print(f"  Snippets with contextual refinement applied: {refinement_applied_count} ({refinement_percentage:.2f}%)")
        
        # Refinement reasons
        if 'Contextual_Refinement_Reason' in df.columns:
            refinement_reasons = df['Contextual_Refinement_Reason'].value_counts()
            print(f"  Refinement reasons:")
            for reason, count in refinement_reasons.items():
                percentage = (count / len(df)) * 100
                print(f"    {reason}: {count} ({percentage:.2f}%)")
    
    # Topic analysis
    print(f"\nTop 10 Primary Topics:")
    primary_topics = df['Primary_Topic'].value_counts().head(10)
    for topic, count in primary_topics.items():
        percentage = (count / len(df)) * 100
        print(f"  {topic}: {count} ({percentage:.2f}%)")
    
    # Section analysis
    section_counts = df['Section'].value_counts()
    print(f"\nSection Distribution:")
    for section, count in section_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {section}: {count} ({percentage:.2f}%)")
    
    # Team analysis
    team_counts = df['Team'].value_counts()
    print(f"\nSpeaker Team Distribution:")
    for team, count in team_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {team}: {count} ({percentage:.2f}%)")

def merge_meeting_logistics_snippets(all_snippets: List[dict]) -> List[dict]:
    """
    Merge sequential snippets that are solely Meeting Logistics with confidence > 90%
    Only merge within the same speaker to maintain structure
    
    Args:
        all_snippets: List of snippet dictionaries
        
    Returns:
        List with merged snippets
    """
    if not all_snippets:
        return all_snippets
    
    merged_snippets = []
    
    # Sort to ensure proper ordering for merging
    sorted_snippets = sorted(all_snippets, key=lambda x: (
        x['Company'], x['Quarter'], x['Date'], x['Section'], 
        x.get('Original_Paragraph_Order', 0)  # Use actual paragraph order
    ))
    
    # Group by transcript and section
    for (company, quarter, date, section), section_snippets in groupby(
        sorted_snippets, 
        key=lambda x: (x['Company'], x['Quarter'], x['Date'], x['Section'])
    ):
        section_snippets = list(section_snippets)
        
        # Process snippets in order WITHOUT grouping by speaker
        # We'll track speaker changes to only merge within same speaker runs
        i = 0
        while i < len(section_snippets):
            current_snippet = section_snippets[i]
            
            # Check if this is a meeting logistics snippet with high confidence
            is_logistics = (current_snippet.get('Primary_Topic') == 
                           "Meeting Logistics (Introductions, Transitions, Call Structure, Greetings, Closing Remarks)")
            high_confidence = current_snippet.get('Primary_Topic_Confidence', 0) > 90
            
            if is_logistics and high_confidence:
                # Look for consecutive meeting logistics snippets from same speaker
                merge_group = [current_snippet]
                j = i + 1
                
                while j < len(section_snippets):
                    next_snippet = section_snippets[j]
                    next_is_logistics = (next_snippet.get('Primary_Topic') == 
                                       "Meeting Logistics (Introductions, Transitions, Call Structure, Greetings, Closing Remarks)")
                    next_high_confidence = next_snippet.get('Primary_Topic_Confidence', 0) > 90
                    
                    # Only merge if same speaker and consecutive logistics
                    if (next_is_logistics and next_high_confidence and 
                        next_snippet['Speaker_Name'] == current_snippet['Speaker_Name']):
                        merge_group.append(next_snippet)
                        j += 1
                    else:
                        break
                
                # If we have multiple snippets to merge
                if len(merge_group) > 1:
                    merged_snippet = merge_logistics_group(merge_group)
                    merged_snippets.append(merged_snippet)
                    i = j  # Skip past all merged snippets
                else:
                    merged_snippets.append(current_snippet)
                    i += 1
            else:
                merged_snippets.append(current_snippet)
                i += 1
    
    return merged_snippets

def merge_logistics_group(logistics_group: List[dict]) -> dict:
    """Merge a group of meeting logistics snippets"""
    if len(logistics_group) == 1:
        return logistics_group[0]
    
    # Merge the snippet text
    merged_text = " ".join([snippet['Snippet'] for snippet in logistics_group])
    
    # Use first snippet as base and update relevant fields
    merged_snippet = logistics_group[0].copy()
    merged_snippet['Snippet'] = merged_text
    
    # Update length-related fields
    merged_snippet['Snippet_Pct_Transcript'] = sum([s.get('Snippet_Pct_Transcript', 0) for s in logistics_group])
    merged_snippet['Snippet_Pct_Section'] = sum([s.get('Snippet_Pct_Section', 0) for s in logistics_group])
    
    # Update snippet level
    merged_snippet['Snippet_Level'] = 'merged_logistics'
    
    # Keep highest confidence
    merged_snippet['Primary_Topic_Confidence'] = max([s.get('Primary_Topic_Confidence', 0) for s in logistics_group])
    
    # Merge original paragraphs
    original_paragraphs = [s.get('Original_Paragraph', '') for s in logistics_group if s.get('Original_Paragraph')]
    merged_snippet['Original_Paragraph'] = " ".join(original_paragraphs)
    
    # Add note about merging
    merged_snippet['Refinement_Reason'] = f"merged_{len(logistics_group)}_logistics_snippets"
    
    return merged_snippet

def needs_contextual_refinement(snippet_data: dict, text: str) -> tuple:
    """
    Determine if a snippet needs contextual refinement based on original criteria
    
    Args:
        snippet_data: The snippet dictionary
        text: The snippet text
        
    Returns:
        Tuple of (needs_refinement: bool, reason: str)
    """
    # Short paragraphs (< 20 words)
    if len(text.split()) < 20:
        return True, "short_text"
    
    # Low primary confidence (< 60%)
    primary_confidence = snippet_data.get('Primary_Topic_Confidence', 0)
    if primary_confidence < 60:
        return True, "low_primary_confidence"
    
    # Close secondary confidence (difference < 20%)
    secondary_confidence = snippet_data.get('Secondary_Topic_Confidence', 0)
    if secondary_confidence > 0 and (primary_confidence - secondary_confidence) < 20:
        return True, "close_secondary_confidence"
    
    return False, None

def perform_contextual_refinement_structured(snippet_data: dict, prev_snippet_data: dict = None, next_snippet_data: dict = None, stats_tracker: dict = None) -> dict:
    """
    Perform contextual refinement using structured outputs with surrounding snippet context
    
    Args:
        snippet_data: Current snippet to refine
        prev_snippet_data: Previous snippet data (optional)
        next_snippet_data: Next snippet data (optional)
        stats_tracker: Statistics tracker
        
    Returns:
        Updated snippet data with refined classification
    """
    text = snippet_data.get('Snippet', '')
    company = snippet_data.get('Company', '')
    speaker = snippet_data.get('Speaker_Name', '')
    
    # Build context sections
    prev_context = ""
    if prev_snippet_data:
        prev_text = prev_snippet_data.get('Snippet', '')
        prev_topic = prev_snippet_data.get('Primary_Topic', 'Unknown')
        prev_context = f'PREVIOUS STATEMENT: "{prev_text}"\nPREVIOUS PRIMARY TOPIC: {prev_topic}'
    
    next_context = ""
    if next_snippet_data:
        next_text = next_snippet_data.get('Snippet', '')
        next_topic = next_snippet_data.get('Primary_Topic', 'Unknown')
        next_context = f'NEXT STATEMENT: "{next_text}"\nNEXT PRIMARY TOPIC: {next_topic}'
    
    # Only proceed if we have some context
    if not prev_context and not next_context:
        return snippet_data
    
    # Create contextual refinement prompt
    current_topics = [snippet_data.get('Primary_Topic', '')]
    if snippet_data.get('Secondary_Topic'):
        current_topics.append(snippet_data.get('Secondary_Topic'))
    current_topics_str = ", ".join([t for t in current_topics if t])
    
    prompt = f"""You are refining a topic classification for a SHORT or AMBIGUOUS statement from an earnings call transcript using surrounding context.

CURRENT STATEMENT: "{text}"
CURRENT CLASSIFICATION: {current_topics_str} (Primary confidence: {snippet_data.get('Primary_Topic_Confidence', 0)}%)

{prev_context}

{next_context}

AVAILABLE TOPICS: {'; '.join(topics_list)}

Considering the surrounding context, please determine if the current statement:
1. Should continue/match the topic from the previous statement
2. Should introduce/match the topic that continues in the next statement  
3. Has its own independent topic that's different from current classification
4. Current classification is already correct

For introductions, meeting logistics, greetings, call structure descriptions, or closing remarks, classify as "Meeting Logistics (Introductions, Transitions, Call Structure, Greetings, Closing Remarks)".

CLASSIFICATION GUIDELINES:
- Focus on the main intent and meaning of the current statement
- Consider how it flows with surrounding context
- Prefer consistency when statements are transitional or connective
- Only change classification if context clearly suggests a better topic

TEMPORAL CONTEXT (time orientation - NOT sentiment):
MUST be EXACTLY one of these 3 time-based options:
- Retrospective: discussing past events or results
- Current: describing present conditions or ongoing situations  
- Forward-looking: providing future projections or expectations

IMPORTANT: Use ONLY "Retrospective", "Current", or "Forward-looking" for temporal context. Do NOT use "Neutral".

CONTENT SENTIMENT (based ONLY on objective facts being reported):
- Positive: reporting growth, exceeding targets, new opportunities, strong performance
- Negative: reporting decline, missing targets, challenges, losses
- Neutral: factual statements without clear positive/negative implications
NOTE: Content sentiment must be EXACTLY one of: "Positive", "Negative", or "Neutral"

SPEAKER TONE (how information is delivered, regardless of content):
MUST be EXACTLY one of these 3 options only:
- Positive: upbeat, enthusiastic, or confident delivery
- Negative: downbeat, concerned, or anxious delivery  
- Neutral: balanced, matter-of-fact delivery

IMPORTANT: Use ONLY "Positive", "Negative", or "Neutral" - no other labels allowed.

Remember: content sentiment and speaker tone are independent.

Provide your analysis in the required JSON format with the most appropriate topic classification."""

    # Create a simplified schema for contextual refinement
    class ContextualRefinementResponse(BaseModel):
        relationship: Literal["continue_previous", "introduce_next", "independent", "current_correct"] = Field(description="Relationship to surrounding context")
        refined_topic: str = Field(description="The refined topic classification")
        confidence: int = Field(ge=0, le=100, description="Confidence in the refined classification")
        reasoning: str = Field(description="Brief explanation of the refinement decision")
        temporal_context: Literal["Retrospective", "Current", "Forward-looking"] = Field(description="Temporal context")
        content_sentiment: Literal["Positive", "Negative", "Neutral"] = Field(description="Content sentiment")
        speaker_tone: Literal["Positive", "Negative", "Neutral"] = Field(description="Speaker tone")

    try:
        # Query LLM with structured output
        refinement_response = query_llm_structured(prompt, ContextualRefinementResponse, stats_tracker=stats_tracker)
        
        # Update snippet data with refined classification if it's different
        if refinement_response.relationship != "current_correct":
            # Find category for refined topic
            refined_category = next((cat for cat, topics in category_topic_mapping.items() 
                                   if any(t.lower() == refinement_response.refined_topic.lower() for t in topics)), "Uncategorized")
            
            # Store original classification for comparison
            snippet_data['Original_Primary_Topic'] = snippet_data.get('Primary_Topic', '')
            snippet_data['Original_Primary_Confidence'] = snippet_data.get('Primary_Topic_Confidence', 0)
            snippet_data['Contextual_Refinement_Applied'] = True
            snippet_data['Contextual_Refinement_Reasoning'] = refinement_response.reasoning
            snippet_data['Contextual_Relationship'] = refinement_response.relationship
            
            # Update primary topic with refined classification
            snippet_data['Primary_Topic'] = refinement_response.refined_topic
            snippet_data['Primary_Topic_Confidence'] = refinement_response.confidence
            snippet_data['Primary_Topic_Coverage'] = 100  # Assume full coverage for single-topic refinement
            snippet_data['Temporal_Context'] = refinement_response.temporal_context
            snippet_data['Content_Sentiment'] = refinement_response.content_sentiment
            snippet_data['Speaker_Tone'] = refinement_response.speaker_tone
            
            # Clear secondary topics since contextual refinement typically results in single topic
            for prefix in ['Secondary', 'Tertiary', 'Quaternary', 'Quinary']:
                snippet_data[f'{prefix}_Topic'] = ''
                snippet_data[f'{prefix}_Topic_Confidence'] = 0
                snippet_data[f'{prefix}_Topic_Coverage'] = 0
                snippet_data[f'{prefix}_Temporal_Context'] = ''
                snippet_data[f'{prefix}_Content_Sentiment'] = ''
        else:
            # Mark that contextual review was performed but no change needed
            snippet_data['Contextual_Refinement_Applied'] = False
            snippet_data['Contextual_Refinement_Reason'] = "Current classification confirmed by context"
            
    except Exception as e:
        print(f"Error in contextual refinement: {e}")
        snippet_data['Contextual_Refinement_Applied'] = False
        snippet_data['Contextual_Refinement_Reason'] = f"Refinement failed: {str(e)}"
    
    return snippet_data

def apply_contextual_refinement_pass_single_file(all_snippets: List[dict], stats_tracker: dict = None) -> List[dict]:
    """
    Apply contextual refinement to snippets within a single file
    
    Args:
        all_snippets: List of all snippets from this file
        stats_tracker: Statistics tracker
        
    Returns:
        List of snippets with contextual refinement applied
    """
    if not all_snippets:
        return all_snippets
    
    refined_snippets = []
    refinement_count = 0
    
    # Sort to ensure proper ordering for context
    sorted_snippets = sorted(all_snippets, key=lambda x: (
        x['Company'], x['Quarter'], x['Date'], x['Section'], 
        x.get('Original_Paragraph_Order', 0)  # Use actual paragraph order for proper context
    ))
    
    # Process each transcript section
    for (company, quarter, date, section), section_snippets in groupby(
        sorted_snippets, 
        key=lambda x: (x['Company'], x['Quarter'], x['Date'], x['Section'])
    ):
        section_snippets = list(section_snippets)
        
        # Apply contextual refinement within each section
        for i, snippet in enumerate(section_snippets):
            text = snippet.get('Snippet', '')
            needs_refinement, reason = needs_contextual_refinement(snippet, text)
            
            if needs_refinement:
                # Get previous and next snippets for context
                prev_snippet = section_snippets[i-1] if i > 0 else None
                next_snippet = section_snippets[i+1] if i < len(section_snippets) - 1 else None
                
                # Only refine if we have at least one context snippet
                if prev_snippet or next_snippet:
                    refined_snippet = perform_contextual_refinement_structured(
                        snippet, prev_snippet, next_snippet, stats_tracker
                    )
                    refined_snippet['Contextual_Refinement_Reason'] = reason
                    refined_snippets.append(refined_snippet)
                    
                    if refined_snippet.get('Contextual_Refinement_Applied', False):
                        refinement_count += 1
                else:
                    # No context available, keep original
                    snippet['Contextual_Refinement_Applied'] = False
                    snippet['Contextual_Refinement_Reason'] = f"no_context_available_{reason}"
                    refined_snippets.append(snippet)
            else:
                # No refinement needed
                snippet['Contextual_Refinement_Applied'] = False
                snippet['Contextual_Refinement_Reason'] = "no_refinement_needed"
                refined_snippets.append(snippet)
    
    print(f"Applied contextual refinement to {refinement_count} snippets in this file")
    
    # Track refinement statistics
    if stats_tracker:
        stats_tracker['Contextual_Refinements_Applied'] = refinement_count
        stats_tracker['Contextual_Refinement_Calls'] = refinement_count  # One LLM call per refinement
    
    return refined_snippets

# Removed apply_contextual_refinement_pass() - no longer needed after refactor to single-file processing

def extract_metadata_from_json(file_path: str) -> dict:
    """
    Extract metadata (company, quarter, year) from JSON file without full processing
    
    Args:
        file_path: Path to the JSON transcript file
        
    Returns:
        Dictionary with metadata or None if extraction fails
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Determine if it's a single transcript or a list
        transcripts = data if isinstance(data, list) else [data]
        
        if not transcripts:
            return None
            
        # Extract metadata from first transcript
        transcript = transcripts[0]
        
        metadata = {
            'company': transcript.get("symbol", "UNKNOWN"),
            'quarter': transcript.get("quarter", "UNKNOWN"),
            'year': transcript.get("year", "UNKNOWN"),
            'file_path': file_path
        }
        
        return metadata
        
    except Exception as e:
        print(f"Warning: Could not extract metadata from {file_path}: {e}")
        return None

def predict_output_filename(metadata: dict) -> str:
    """
    Predict the output filename based on metadata (same logic as generate_output_filename)
    
    Args:
        metadata: Dictionary with company, quarter, year
        
    Returns:
        Predicted output filename
    """
    if not metadata:
        return "topic_class_no_data.csv"
    
    company = metadata.get('company', 'UNKNOWN')
    quarter = metadata.get('quarter', 'UNKNOWN')
    year = metadata.get('year', 'UNKNOWN')  # Note: year becomes 'Date' in snippets
    
    return f"{company}_{quarter}_{year}_topic_class.csv"

def check_already_processed(input_file_path: str, output_file_path: str) -> dict:
    """
    Check if a file has already been processed by comparing input and output files
    
    Args:
        input_file_path: Path to input JSON file
        output_file_path: Path to expected output CSV file
        
    Returns:
        Dictionary with check results
    """
    result = {
        'already_processed': False,
        'output_exists': False,
        'input_newer': False,
        'input_mtime': None,
        'output_mtime': None,
        'input_mtime_str': '',
        'output_mtime_str': ''
    }
    
    try:
        # Check if output file exists
        if os.path.exists(output_file_path):
            result['output_exists'] = True
            result['already_processed'] = True
            
            # Get modification times
            input_mtime = os.path.getmtime(input_file_path)
            output_mtime = os.path.getmtime(output_file_path)
            
            result['input_mtime'] = input_mtime
            result['output_mtime'] = output_mtime
            result['input_mtime_str'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(input_mtime))
            result['output_mtime_str'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(output_mtime))
            
            # Check if input is newer (might want to reprocess)
            if input_mtime > output_mtime:
                result['input_newer'] = True
                
    except Exception as e:
        print(f"Warning: Error checking file timestamps for {input_file_path}: {e}")
    
    return result

def main():
    print("=" * 60)
    print("TOPIC CLASSIFICATION WITH STRUCTURED OUTPUTS")
    print("=" * 60)
    
    # Find all paragraph-segmented JSON files
    transcript_files = glob.glob(os.path.join(DATA_DIR, "*_paragraphs.json"))
    
    if not transcript_files:
        print(f"No paragraph-segmented transcript files found in {DATA_DIR}")
        print("Please ensure files are named with pattern '*_paragraphs.json'")
        return
    
    print(f"Found {len(transcript_files)} transcript files to process:")
    for file in transcript_files:
        print(f"  - {os.path.basename(file)}")
    
    # Check for already processed files
    print(f"\nChecking for already processed files...")
    files_to_process = []
    files_skipped = []
    
    for file_path in transcript_files:
        # Extract metadata to predict output filename
        metadata = extract_metadata_from_json(file_path)
        if not metadata:
            print(f"Warning: Could not extract metadata from {os.path.basename(file_path)}, will process")
            files_to_process.append(file_path)
            continue
        
        # Predict output filename
        predicted_filename = predict_output_filename(metadata)
        predicted_output_path = os.path.join(OUTPUT_DIR, predicted_filename)
        
        # Check if already processed
        check_result = check_already_processed(file_path, predicted_output_path)
        
        if check_result['already_processed']:
            files_skipped.append({
                'input_file': file_path,
                'output_file': predicted_output_path,
                'company': metadata['company'],
                'quarter': metadata['quarter'],
                'year': metadata['year'],
                'input_mtime_str': check_result['input_mtime_str'],
                'output_mtime_str': check_result['output_mtime_str'],
                'input_newer': check_result['input_newer']
            })
            print(f"  SKIPPED: {os.path.basename(file_path)} -> {predicted_filename} (output exists: {check_result['output_mtime_str']})")
        else:
            files_to_process.append(file_path)
    
    # Report skipping summary
    if files_skipped:
        print(f"\n{'='*60}")
        print(f"SKIPPED FILES SUMMARY")
        print(f"{'='*60}")
        print(f"Skipped {len(files_skipped)} already processed files:")
        for skipped in files_skipped:
            status = " [INPUT NEWER]" if skipped['input_newer'] else ""
            print(f"  - {skipped['company']}_{skipped['quarter']}_{skipped['year']}: Output from {skipped['output_mtime_str']}{status}")
        print(f"To reprocess these files, delete the corresponding CSV files in {OUTPUT_DIR}")
        
        if any(s['input_newer'] for s in files_skipped):
            print(f"\nNOTE: {sum(1 for s in files_skipped if s['input_newer'])} skipped files have newer input than output.")
            print(f"Consider deleting those output files if you want to reprocess with updated inputs.")
    
    if not files_to_process:
        print(f"\nAll files have already been processed. No new processing needed.")
        print(f"Total transcript files found: {len(transcript_files)}")
        print(f"Already processed: {len(files_skipped)}")
        return
    
    print(f"\n{'='*60}")
    print(f"PROCESSING PLAN")
    print(f"{'='*60}")
    print(f"Total files found: {len(transcript_files)}")
    print(f"Already processed (skipped): {len(files_skipped)}")
    print(f"Files to process: {len(files_to_process)}")
    print(f"Processing will begin shortly...")
    
    # Process only the files that need processing
    all_snippets = []
    all_processing_stats = []
    
    for file_path in files_to_process:
        try:
            file_snippets, file_stats = process_transcript_file(file_path)
            
            # Apply final optimizations before saving individual file
            if file_snippets:
                print(f"Applying final optimizations for {os.path.basename(file_path)}...")
                
                # Apply meeting logistics merging to individual file
                original_count = len(file_snippets)
                file_snippets = merge_meeting_logistics_snippets(file_snippets)
                merged_count = original_count - len(file_snippets)
                if merged_count > 0:
                    print(f"  Merged {merged_count} meeting logistics snippets")
                
                # Update stats with final optimized count
                file_stats['Final_Snippets_After_Optimization'] = len(file_snippets)
                file_stats['Meeting_Logistics_Merged'] = merged_count
                
                print(f"Saving individual CSV for {os.path.basename(file_path)}...")
                individual_df = create_csv_output(file_snippets)
                individual_filename = generate_output_filename(file_snippets)
                individual_path = os.path.join(OUTPUT_DIR, individual_filename)
                individual_df.to_csv(individual_path, index=False, encoding='utf-8')
                print(f"Individual file saved: {individual_path} ({len(file_snippets)} snippets)")
                
                # CONTENT VALIDATION: Ensure content preservation for this file
                print(f"Validating content preservation for {os.path.basename(file_path)}...")
                validation_result = validate_content_preservation([file_path], file_snippets)
                
                if len(validation_result) == 3:
                    is_valid, validation_report, debug_info = validation_result
                else:
                    is_valid, validation_report = validation_result
                    debug_info = None
                
                # Update file stats with validation results
                file_stats['Content_Validation_Passed'] = is_valid
                file_stats['Content_Validation_Issues'] = len(validation_report.get('speaker_content_mismatches', []))
                file_stats['Content_Validation_Missing_Sections'] = len(validation_report.get('missing_sections', []))
                file_stats['Content_Validation_Extra_Sections'] = len(validation_report.get('extra_sections', []))
                
                if not is_valid:
                    print(f"WARNING: Content preservation issues detected in {os.path.basename(file_path)}!")
                    print(f"  Content mismatches: {file_stats['Content_Validation_Issues']}")
                    print(f"  Missing sections: {file_stats['Content_Validation_Missing_Sections']}")
                    print(f"  Extra sections: {file_stats['Content_Validation_Extra_Sections']}")
                    
                    # Save individual validation report
                    file_base = os.path.splitext(os.path.basename(file_path))[0]
                    validation_report_path = os.path.join(METADATA_DIR, f"validation_report_{file_base}.json")
                    with open(validation_report_path, 'w', encoding='utf-8') as f:
                        json.dump(validation_report, f, indent=2, default=str)
                    print(f"  Validation report saved: {validation_report_path}")
                    
                    if debug_info:
                        debug_report_path = os.path.join(METADATA_DIR, f"content_debug_{file_base}.json")
                        with open(debug_report_path, 'w', encoding='utf-8') as f:
                            json.dump(debug_info, f, indent=2, ensure_ascii=False)
                        print(f"  Debug report saved: {debug_report_path}")
                else:
                    print(f"Content validation PASSED for {os.path.basename(file_path)}")
            
            # Add to combined results
            all_snippets.extend(file_snippets)
            all_processing_stats.append(file_stats)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_snippets:
        print("No snippets were processed successfully")
        return
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE - GENERATING SUMMARY")
    print(f"{'='*60}")
    
    # Calculate overall statistics from individual files
    total_snippets = sum([stats.get('Total_Snippets_Created', 0) for stats in all_processing_stats])
    total_paragraphs = sum([stats.get('Total_Paragraphs_Processed', 0) for stats in all_processing_stats])
    total_companies = len(set([stats.get('File_Name', '').split('_')[0] for stats in all_processing_stats]))
    total_processing_time = sum([stats.get('Processing_Time_Minutes', 0) for stats in all_processing_stats])
    
    # Calculate validation statistics
    files_passed_validation = sum([1 for stats in all_processing_stats if stats.get('Content_Validation_Passed', False)])
    total_validation_issues = sum([stats.get('Content_Validation_Issues', 0) for stats in all_processing_stats])
    
    print(f"Found {len(transcript_files)} transcript files total")
    print(f"Skipped {len(files_skipped)} already processed files") 
    print(f"Processed {len(files_to_process)} transcript files")
    print(f"Total snippets created: {total_snippets}")
    print(f"Total paragraphs processed: {total_paragraphs}")
    print(f"Companies covered: {total_companies}")
    print(f"Total processing time: {total_processing_time:.2f} minutes")
    if len(all_processing_stats) > 0:
        print(f"Average time per file: {total_processing_time/len(all_processing_stats):.2f} minutes")
    print(f"Content validation: {files_passed_validation}/{len(files_to_process)} files passed")
    if total_validation_issues > 0:
        print(f"WARNING: {total_validation_issues} content validation issues detected across all files")
    else:
        print(f"Content validation: All files passed successfully")
    
    # Create processing summary CSV (overview across all files)
    create_processing_summary_csv(all_processing_stats)
    
    print(f"\nIndividual CSV files saved in: {OUTPUT_DIR}")
    print(f"Processing summary saved in: {METADATA_DIR}")
    if total_validation_issues == 0:
        print(f"\nAll files processed successfully! Each file contains optimized, complete data.")
    else:
        print(f"\nProcessing completed with {total_validation_issues} validation issues. Check metadata files for details.")

def validate_content_preservation(original_transcript_files: List[str], final_snippets: List[dict]) -> tuple:
    """
    Validate that all original content from JSON files is preserved in the final snippets
    NEW APPROACH: Compare by speaker dialogue blocks instead of individual paragraphs
    
    Args:
        original_transcript_files: List of original JSON file paths
        final_snippets: List of final processed snippets
        
    Returns:
        Tuple of (is_valid: bool, validation_report: dict)
    """
    print("INFO: Validating content preservation using speaker-level dialogue comparison...")
    
    # Extract original speaker dialogues with order
    original_speakers = {}  # {(company, quarter, date, section): [(speaker_name, order, dialogue_text)]}
    
    for file_path in original_transcript_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle both single transcript and list formats
            transcripts = data if isinstance(data, list) else [data]
            
            for transcript in transcripts:
                company = transcript["symbol"]
                quarter = transcript["quarter"] 
                year = transcript["year"]
                
                # Process prepared remarks
                if "parsed_content" in transcript and "prepared_remarks" in transcript["parsed_content"]:
                    section_key = (company, quarter, year, "prepared_remarks")
                    original_speakers[section_key] = []
                    
                    for pr in transcript["parsed_content"]["prepared_remarks"]:
                        speaker_name = pr["speaker"]
                        speaker_order = pr.get("order", 0)
                        dialogue_text = pr.get("dialogue", "")
                        
                        if dialogue_text and dialogue_text.strip():
                            # Decode and clean the dialogue the same way processing does
                            cleaned_dialogue = decode_json_text(dialogue_text.strip())
                            original_speakers[section_key].append((speaker_name, speaker_order, cleaned_dialogue))
                
                # Process Q&A section
                if "parsed_content" in transcript and "qna_section" in transcript["parsed_content"]:
                    section_key = (company, quarter, year, "qna_section")
                    original_speakers[section_key] = []
                    
                    for qa in transcript["parsed_content"]["qna_section"]:
                        speaker_name = qa["speaker"]
                        speaker_order = qa.get("order", 0)
                        dialogue_text = qa.get("dialogue", "")
                        
                        if dialogue_text and dialogue_text.strip():
                            # Decode and clean the dialogue the same way processing does
                            cleaned_dialogue = decode_json_text(dialogue_text.strip())
                            original_speakers[section_key].append((speaker_name, speaker_order, cleaned_dialogue))
                    
                    # Sort by order to ensure proper sequence
                    original_speakers[section_key].sort(key=lambda x: x[1])  # Sort by order
                                
        except Exception as e:
            print(f"ERROR: Error reading {file_path}: {e}")
            continue
    
    # Reconstruct speaker dialogues from final snippets
    reconstructed_speakers = {}  # {(company, quarter, date, section): [(speaker_name, order, merged_text)]}
    
    # Group snippets by section and speaker order
    for section_key in original_speakers.keys():
        # Get all snippets for this section
        section_snippets = [
            snippet for snippet in final_snippets 
            if (snippet['Company'], snippet['Quarter'], snippet['Date'], snippet['Section']) == section_key
        ]
        
        # Group by speaker and speaker order
        speaker_groups = {}  # {(speaker_name, speaker_order): [snippets]}
        for snippet in section_snippets:
            speaker_name = snippet['Speaker_Name']
            speaker_order = snippet.get('Speaker_Order', 0)
            speaker_key = (speaker_name, speaker_order)
            
            if speaker_key not in speaker_groups:
                speaker_groups[speaker_key] = []
            speaker_groups[speaker_key].append(snippet)
        
        # Reconstruct text for each speaker
        reconstructed_speakers[section_key] = []
        for (speaker_name, speaker_order), speaker_snippets in speaker_groups.items():
            # Sort snippets by their original paragraph order to maintain sequence
            speaker_snippets.sort(key=lambda x: x.get('Original_Paragraph_Order', 0))
            
            # DEBUG: Track what we're merging for this speaker
            debug_snippet_info = []
            
            # Merge all snippet text for this speaker
            merged_text_parts = []
            for snippet in speaker_snippets:
                # Get raw snippet text without CSV formatting
                snippet_text = snippet['Snippet'].strip()
                
                # Remove quotes that were added by CSV formatting
                if snippet_text.startswith('"') and snippet_text.endswith('"'):
                    snippet_text = snippet_text[1:-1]
                
                if snippet_text:
                    merged_text_parts.append(snippet_text)
                    # DEBUG: Track each snippet being merged
                    debug_snippet_info.append({
                        'paragraph_order': snippet.get('Original_Paragraph_Order', 0),
                        'snippet_order': snippet.get('Snippet_Order', 0),
                        'snippet_level': snippet.get('Snippet_Level', 'unknown'),
                        'has_sentence_attribution': snippet.get('Has_Sentence_Attribution', False),
                        'snippet_text_preview': snippet_text[:100] + "..." if len(snippet_text) > 100 else snippet_text
                    })
            
            merged_text = " ".join(merged_text_parts)
            # Clean and normalize the reconstructed text
            merged_text = decode_json_text(merged_text.strip())
            
            reconstructed_speakers[section_key].append((speaker_name, speaker_order, merged_text, debug_snippet_info))
        
        # Sort by speaker order to match original sequence
        reconstructed_speakers[section_key].sort(key=lambda x: x[1])
    
    # Compare original vs reconstructed speaker dialogues
    validation_report = {
        'total_sections': len(original_speakers),
        'matching_sections': 0,
        'missing_sections': [],
        'extra_sections': [],
        'speaker_order_mismatches': [],
        'speaker_content_mismatches': [],
        'speaker_count_mismatches': [],
        'total_original_speakers': 0,
        'total_reconstructed_speakers': 0,
        'is_valid': True
    }
    
    # Check for missing or extra sections
    original_sections = set(original_speakers.keys())
    reconstructed_sections = set(reconstructed_speakers.keys())
    
    validation_report['missing_sections'] = list(original_sections - reconstructed_sections)
    validation_report['extra_sections'] = list(reconstructed_sections - original_sections)
    
    if validation_report['missing_sections'] or validation_report['extra_sections']:
        validation_report['is_valid'] = False
    
    # Compare content for matching sections
    for section_key in original_sections & reconstructed_sections:
        original_speaker_list = original_speakers[section_key]
        reconstructed_speaker_list = reconstructed_speakers.get(section_key, [])
        
        validation_report['total_original_speakers'] += len(original_speaker_list)
        validation_report['total_reconstructed_speakers'] += len(reconstructed_speaker_list)
        
        # Check speaker counts match
        if len(original_speaker_list) != len(reconstructed_speaker_list):
            validation_report['speaker_count_mismatches'].append({
                'section': section_key,
                'original_count': len(original_speaker_list),
                'reconstructed_count': len(reconstructed_speaker_list),
                'original_speakers': [f"{name} (order: {order})" for name, order, _ in original_speaker_list],
                'reconstructed_speakers': [f"{name} (order: {order})" for name, order, _ in reconstructed_speaker_list]
            })
            validation_report['is_valid'] = False
        
        # Compare speaker order and content
        for i, (orig_speaker, recon_speaker) in enumerate(zip(original_speaker_list, reconstructed_speaker_list)):
            orig_name, orig_order, orig_dialogue = orig_speaker
            # Handle both old format (3 items) and new format (4 items with debug info)
            if len(recon_speaker) == 4:
                recon_name, recon_order, recon_dialogue, debug_snippets = recon_speaker
            else:
                recon_name, recon_order, recon_dialogue = recon_speaker
                debug_snippets = []
            
            # Check speaker order matches
            if orig_order != recon_order or orig_name != recon_name:
                validation_report['speaker_order_mismatches'].append({
                    'section': section_key,
                    'position': i,
                    'original_speaker': f"{orig_name} (order: {orig_order})",
                    'reconstructed_speaker': f"{recon_name} (order: {recon_order})"
                })
                validation_report['is_valid'] = False
            
            # Check dialogue content matches
            # Normalize whitespace for comparison
            orig_normalized = " ".join(orig_dialogue.split())
            recon_normalized = " ".join(recon_dialogue.split())
            
            if orig_normalized != recon_normalized:
                validation_report['speaker_content_mismatches'].append({
                    'section': section_key,
                    'speaker': f"{orig_name} (order: {orig_order})",
                    'original_length': len(orig_normalized),
                    'reconstructed_length': len(recon_normalized),
                    'original_preview': orig_normalized[:300] + "..." if len(orig_normalized) > 300 else orig_normalized,
                    'reconstructed_preview': recon_normalized[:300] + "..." if len(recon_normalized) > 300 else recon_normalized
                })
                validation_report['is_valid'] = False
        
        # Count matching sections
        if (len(original_speaker_list) == len(reconstructed_speaker_list) and 
            not any(mismatch['section'] == section_key for mismatch in validation_report['speaker_order_mismatches']) and
            not any(mismatch['section'] == section_key for mismatch in validation_report['speaker_content_mismatches'])):
            validation_report['matching_sections'] += 1
    
    # Print validation results
    print(f"STATS: Speaker-Level Content Validation Results:")
    print(f"   Total sections: {validation_report['total_sections']}")
    print(f"   Matching sections: {validation_report['matching_sections']}")
    print(f"   Total speakers: {validation_report['total_original_speakers']} -> {validation_report['total_reconstructed_speakers']}")
    
    if validation_report['is_valid']:
        print("SUCCESS: Content preservation PASSED - All original speaker content preserved!")
    else:
        print("FAILED: Content preservation FAILED - Issues detected:")
        
        if validation_report['missing_sections']:
            print(f"   Missing sections: {len(validation_report['missing_sections'])}")
        if validation_report['extra_sections']:
            print(f"   Extra sections: {len(validation_report['extra_sections'])}")
        if validation_report['speaker_count_mismatches']:
            print(f"   Speaker count mismatches: {len(validation_report['speaker_count_mismatches'])}")
            for mismatch in validation_report['speaker_count_mismatches'][:2]:
                print(f"     Section {mismatch['section']}: {mismatch['original_count']} -> {mismatch['reconstructed_count']} speakers")
        if validation_report['speaker_order_mismatches']:
            print(f"   Speaker order mismatches: {len(validation_report['speaker_order_mismatches'])}")
            for mismatch in validation_report['speaker_order_mismatches'][:3]:
                print(f"     Position {mismatch['position']}: {mismatch['original_speaker']} -> {mismatch['reconstructed_speaker']}")
        if validation_report['speaker_content_mismatches']:
            print(f"   Speaker content mismatches: {len(validation_report['speaker_content_mismatches'])}")
            for i, mismatch in enumerate(validation_report['speaker_content_mismatches'][:3]):
                print(f"   Mismatch {i+1}: {mismatch['speaker']}")
                print(f"     Original ({mismatch['original_length']} chars):     {mismatch['original_preview']}")
                print(f"     Reconstructed ({mismatch['reconstructed_length']} chars): {mismatch['reconstructed_preview']}")
    
    # SAVE DETAILED DEBUG INFORMATION FOR ANALYSIS
    if not validation_report['is_valid']:
        debug_info = {
            'validation_summary': {
                'total_sections': validation_report['total_sections'],
                'matching_sections': validation_report['matching_sections'],
                'total_original_speakers': validation_report['total_original_speakers'],
                'total_reconstructed_speakers': validation_report['total_reconstructed_speakers']
            },
            'original_speakers': {},
            'reconstructed_speakers': {},
            'detailed_mismatches': validation_report['speaker_content_mismatches']
        }
        
        # Add full original and reconstructed content for analysis
        for section_key, speaker_list in original_speakers.items():
            section_name = f"{section_key[0]}_{section_key[1]}_{section_key[2]}_{section_key[3]}"
            debug_info['original_speakers'][section_name] = [
                {
                    'speaker_name': name,
                    'speaker_order': order,
                    'dialogue_length': len(dialogue),
                    'dialogue_text': dialogue
                }
                for name, order, dialogue in speaker_list
            ]
        
        for section_key, speaker_list in reconstructed_speakers.items():
            section_name = f"{section_key[0]}_{section_key[1]}_{section_key[2]}_{section_key[3]}"
            debug_info['reconstructed_speakers'][section_name] = []
            for speaker_data in speaker_list:
                if len(speaker_data) == 4:
                    name, order, merged_text, debug_snippets = speaker_data
                    speaker_debug = {
                        'speaker_name': name,
                        'speaker_order': order,
                        'merged_text_length': len(merged_text),
                        'merged_text': merged_text,
                        'snippets_merged': debug_snippets,
                        'total_snippets_count': len(debug_snippets)
                    }
                else:
                    name, order, merged_text = speaker_data
                    speaker_debug = {
                        'speaker_name': name,
                        'speaker_order': order,
                        'merged_text_length': len(merged_text),
                        'merged_text': merged_text,
                        'snippets_merged': [],
                        'total_snippets_count': 0
                    }
                debug_info['reconstructed_speakers'][section_name].append(speaker_debug)
        
        return validation_report['is_valid'], validation_report, debug_info
    
    return validation_report['is_valid'], validation_report

if __name__ == "__main__":
    main()



# python topic_classification.py

