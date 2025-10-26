# -*- coding: utf-8 -*-
"""
Attribution Classification for Earnings Call Transcripts
========================================================

This script processes CSV files from topic classification and adds attribution analysis
to identify when speakers attribute outcomes to specific causes, whether those outcomes
are positive/negative, and whether the attribution has an internal/external locus of control.

Input: CSV files from parallel_cpu/04/ with "topic_class" in filename
Output: Enhanced CSV files with attribution columns in parallel_cpu/05/

main()
 For each topic_class CSV file:
    process_attribution_classification() # Complete file processing
       Load CSV file
       Apply intelligent filtering (Meeting Logistics >85%, external_members)
       Process each row with LLM attribution classification
       Save enhanced CSV immediately after processing
       validate_data_integrity() # Comprehensive validation per file
    Collect file stats
 create_attribution_processing_summary_csv() # Combined summary only

Output Structure:
- parallel_cpu/05/
   *_attribution_class.csv (enhanced data files)
   metadata/
       attribution_processing_summary.csv (processing statistics)
       *_validation_report.json (data integrity reports, if issues found)

Attribution Classifications:
1. Attribution Present (Y/N) - Does the text contain causal attribution?
2. Attribution Outcome (Positive/Negative/Neither) - What is the polarity of the attributed outcome?
3. Attribution Locus (Internal/External/Neither) - Is the cause internal or external to the company?
4. Attribution Effect (Revenue/Costs/Demand/Operations/Supply/Other/Neither) - What business area is affected by the attribution?
5. Attribution Cause (1-2 words) - What is the specific cause mentioned in the attribution?

Each classification includes a confidence score (0-100%).

Features:
- Intelligent filtering (Meeting Logistics >85% confidence, external_members)
- Comprehensive validation ensuring data integrity and narrative structure preservation
- Detailed processing statistics and performance metrics
- LLM best practices with structured outputs and error handling

Future Enhancement Opportunities (from causal extraction analysis):
- Extract specific cause_span and effect_span phrases
- Analyze certainty_modality (certain vs hedged language)
- Identify evidence_type (quantitative vs narrative)
- Extract exact attribution_quote substrings
- Classify OutcomePolarity with more nuanced understanding
- Add temporal analysis of attribution patterns
"""

import pandas as pd
import openai
import os
import glob
import json
import re
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pydantic import BaseModel, Field
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set OpenAI API key from config file
from config import get_openai_api_key
openai.api_key = get_openai_api_key()

# Previous hardcoded API key (commented out):
# openai.api_key = ""
#openai.api_key = "sk-"
# # Set OpenAI API key (updated to use environment variable)
# openai.api_key = os.getenv('OPENAI_API_KEY')

# GPT Model configuration
#GPT_MODEL = "gpt-4o-mini-2024-07-18"  # Cost-effective model for classification tasks
GPT_MODEL = "gpt-4.1-mini"

# Directory paths (updated for parallel_cpu structure)
INPUT_DIR = "04"  # Points to parallel_cpu/04/ where previous script outputs
OUTPUT_DIR = "05"  # Points to parallel_cpu/05/ for step 5 output
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

class AttributionClassification(BaseModel):
    attribution_present: str = Field(
        description="Whether attribution is present in the text",
        pattern="^[YN]$"
    )
    attribution_present_confidence: int = Field(
        ge=0, le=100,
        description="Confidence score for attribution presence detection (0-100%)"
    )
    attribution_outcome: str = Field(
        description="Polarity of the attributed outcome",
        pattern="^(Positive|Negative|Neither)$"
    )
    attribution_outcome_confidence: int = Field(
        ge=0, le=100,
        description="Confidence score for outcome polarity classification (0-100%)"
    )
    attribution_locus: str = Field(
        description="Locus of control for the attribution",
        pattern="^(Internal|External|Both|Neither)$"
    )
    attribution_locus_confidence: int = Field(
        ge=0, le=100,
        description="Confidence score for locus of control classification (0-100%)"
    )
    attribution_effect: str = Field(
        description="Business area/domain affected by the attribution",
        pattern="^(Revenue|Costs|Demand|Operations|Supply|Other|Neither)$"
    )
    attribution_effect_confidence: int = Field(
        ge=0, le=100,
        description="Confidence score for attribution effect classification (0-100%)"
    )
    attribution_cause: str = Field(
        description="The specific cause of the attribution in 1-2 words",
        max_length=50
    )
    attribution_cause_confidence: int = Field(
        ge=0, le=100,
        description="Confidence score for attribution cause identification (0-100%)"
    )

def create_attribution_prompt(text: str, speaker: str, company: str) -> str:
    """
    Create an optimized attribution classification prompt using LLM best practices.
    
    Best Practices Applied:
    - Clear, specific instructions with examples
    - Structured output format requirements
    - Confidence scoring for reliability assessment
    - Focus on classification only (no text generation)
    - Explicit field definitions to reduce ambiguity
    """
    
    # Future enhancement note: The causal extraction approach provides more granular analysis:
    # - cause_span: Extract exact phrases naming the reason
    # - effect_span: Extract exact phrases naming the result  
    # - certainty_modality: Analyze hedged vs certain language
    # - evidence_type: Distinguish quantitative vs narrative evidence
    # - attribution_quote: Extract specific attribution statements
    
    prompt = f"""Analyze the following text from an earnings call transcript for attribution characteristics.

TEXT: "{text}"
SPEAKER: {speaker} from {company}

Classify this text across five attribution dimensions:

1. ATTRIBUTION PRESENT:
   - Y: Text contains causal attribution in either form:
     a) Attribution statements: Someone explains a result (cause → effect relationships)
     b) Attribution questions: Someone requests an explanation for a result (seeking attribution)
   - N: Text lacks both attribution statements and attribution questions
   
   Attribution statement indicators: "due to", "because of", "driven by", "resulted from", "thanks to", "attributed to", "caused by", "as a result of", "led to"
   Attribution question indicators: "what drove", "why did", "what caused", "what led to", "can you explain", "what's behind", "what factors", "how do you account for", "what would you attribute"

2. ATTRIBUTION OUTCOME (if attribution present):
   - Positive: Attributed outcome is favorable/beneficial (growth, success, improvement)
   - Negative: Attributed outcome is unfavorable/detrimental (decline, loss, challenge)
   - Neither: Outcome is neutral, factual, or mixed

3. ATTRIBUTION LOCUS (if attribution present):
   - Internal: Cause attributed to company's own actions, decisions, capabilities, or controllable factors
   - External: Cause attributed to market conditions, competitors, customers, regulations, or uncontrollable external factors
   - Both: Use ONLY when both internal and external factors are clearly present AND neither factor is clearly the primary/dominant cause
   - Neither: No clear attribution or ambiguous locus

4. ATTRIBUTION EFFECT (if attribution present):
   - Revenue: Impact on sales, revenue generation, top-line growth
   - Costs: Impact on expenses, cost structure, margins, operational costs
   - Demand: Impact on customer demand, market demand, product/service uptake
   - Operations: Impact on operational efficiency, processes, execution, productivity
   - Supply: Impact on supply chain, inventory, sourcing, production capacity
   - Other: Impact on areas not covered above (regulatory, competitive position, etc.)
   - Neither: No clear business effect or ambiguous impact

5. ATTRIBUTION CAUSE (if attribution present):
   - Identify the specific cause mentioned in 1-2 words
   - Focus on the root factor driving the outcome
   - Examples: "strategy execution", "market conditions", "supply disruptions", "new product", "cost reduction", "economic uncertainty", "competitive pressure", "operational efficiency", "customer behavior", "regulatory changes"
   - Use "Unknown" if cause is not specified or unclear

EXAMPLES:

Attribution Statements (explaining results):
"Revenue increased due to strong execution of our strategy" → Y, Positive, Internal, Revenue, "strategy execution"
"Sales declined because of challenging market conditions" → Y, Negative, External, Revenue, "market conditions"
"Growth was driven by our new product launch" → Y, Positive, Internal, Revenue, "new product"
"Margins compressed due to supply chain disruptions" → Y, Negative, External, Supply, "supply disruptions"
"Cost savings resulted from operational improvements" → Y, Positive, Internal, Costs, "operational efficiency"
"Customer demand increased thanks to our marketing campaign" → Y, Positive, Internal, Demand, "marketing campaign"
"Production delays were caused by equipment failures" → Y, Negative, Internal, Operations, "equipment failures"
"Profits declined due to economic uncertainty affecting customer spending" → Y, Negative, External, Revenue, "economic uncertainty"
"Our growth was driven by both our new product innovations and favorable market conditions" → Y, Positive, Both, Revenue, "product and market"

Attribution Questions (seeking explanations):
"What drove the strong performance in Q3?" → Y, Positive, Neither, Revenue, "Unknown"
"Can you explain why margins declined this quarter?" → Y, Negative, Neither, Costs, "Unknown"
"What factors led to the revenue beat?" → Y, Positive, Neither, Revenue, "Unknown"
"What's behind the increase in operating expenses?" → Y, Negative, Neither, Costs, "Unknown"
"How do you account for the supply chain improvements?" → Y, Positive, Neither, Supply, "Unknown"
"What would you attribute the demand weakness to?" → Y, Negative, Neither, Demand, "Unknown"

Non-attribution statements:
"We reported quarterly earnings of $2.5 billion" → N, Neither, Neither, Neither, "Unknown"
"Our revenue was $500 million this quarter" → N, Neither, Neither, Neither, "Unknown"

RESPONSE FORMAT:
Provide your classification as valid JSON with confidence scores (0-100%) for each dimension:

{{
    "attribution_present": "Y" or "N",
    "attribution_present_confidence": confidence_score,
    "attribution_outcome": "Positive" or "Negative" or "Neither",
    "attribution_outcome_confidence": confidence_score,
    "attribution_locus": "Internal" or "External" or "Both" or "Neither", 
    "attribution_locus_confidence": confidence_score,
    "attribution_effect": "Revenue" or "Costs" or "Demand" or "Operations" or "Supply" or "Other" or "Neither",
    "attribution_effect_confidence": confidence_score,
    "attribution_cause": "1-2 word description of the cause or Unknown",
    "attribution_cause_confidence": confidence_score
}}

IMPORTANT: Respond ONLY with the JSON object. Do not include explanations or additional text."""

    return prompt

def query_llm_structured(prompt: str, retries: int = 3, stats_tracker: dict = None) -> Optional[AttributionClassification]:
    """
    Query LLM with structured output and comprehensive error handling.
    
    Args:
        prompt: The formatted attribution classification prompt
        retries: Number of retry attempts for failed requests
        
    Returns:
        AttributionClassification object or None if all attempts fail
    """
    
    for attempt in range(retries):
        try:
            # Track LLM API calls
            if stats_tracker:
                stats_tracker['llm_api_calls'] = stats_tracker.get('llm_api_calls', 0) + 1
            
            # Use lower temperature for more consistent classification results
            response = openai.ChatCompletion.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent classification
                top_p=0.8,        # Focused sampling for structured output
                max_tokens=200    # Limit tokens since we only need classification JSON
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Parse and validate structured response
            try:
                # Extract JSON if wrapped in code blocks or extra text
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = llm_response
                
                # Parse JSON response
                response_data = json.loads(json_str)
                
                # Validate using Pydantic model
                attribution_result = AttributionClassification(**response_data)
                return attribution_result
                
            except (json.JSONDecodeError, ValueError) as parse_error:
                logger.warning(f"Attempt {attempt + 1}: JSON parsing failed - {parse_error}")
                logger.warning(f"Raw response: {llm_response[:200]}...")
                
                if attempt == retries - 1:
                    logger.error(f"All parsing attempts failed for response: {llm_response}")
                    return None
                continue
                
        except Exception as api_error:
            logger.warning(f"Attempt {attempt + 1}: API call failed - {api_error}")
            if attempt == retries - 1:
                logger.error(f"All API attempts failed: {api_error}")
                return None
            
            # Exponential backoff for retries
            import time
            time.sleep(2 ** attempt)
    
    return None

def classify_attribution(text: str, speaker: str, company: str, stats_tracker: dict = None) -> Dict[str, any]:
    """
    Classify a single text snippet for attribution characteristics.
    
    Args:
        text: The text snippet to classify
        speaker: Speaker name/role
        company: Company symbol/name
        
    Returns:
        Dictionary with attribution classifications and metadata
    """
    
    # Input validation
    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("Empty or invalid text provided for attribution classification")
        return {
            "attribution_present": "N",
            "attribution_present_confidence": 0,
            "attribution_outcome": "Neither", 
            "attribution_outcome_confidence": 0,
            "attribution_locus": "Neither",
            "attribution_locus_confidence": 0,
            "attribution_effect": "Neither",
            "attribution_effect_confidence": 0,
            "attribution_cause": "Unknown",
            "attribution_cause_confidence": 0,
            "classification_status": "empty_text",
            "error_message": "Empty or invalid text input"
        }
    
    # Clean input text
    text = text.strip()
    
    # Skip very short texts that are unlikely to contain meaningful attribution
    if len(text.split()) < 5:
        logger.info(f"Skipping very short text: {text[:50]}...")
        return {
            "attribution_present": "N",
            "attribution_present_confidence": 95,
            "attribution_outcome": "Neither",
            "attribution_outcome_confidence": 95,
            "attribution_locus": "Neither", 
            "attribution_locus_confidence": 95,
            "attribution_effect": "Neither",
            "attribution_effect_confidence": 95,
            "attribution_cause": "Unknown",
            "attribution_cause_confidence": 95,
            "classification_status": "too_short",
            "error_message": "Text too short for attribution analysis"
        }
    
    # Create prompt
    prompt = create_attribution_prompt(text, speaker, company)
    
    # Query LLM with structured output
    attribution_result = query_llm_structured(prompt, stats_tracker=stats_tracker)
    
    if attribution_result:
        result_dict = attribution_result.dict()
        result_dict["classification_status"] = "success"
        result_dict["error_message"] = None
        return result_dict
    else:
        # Fallback response for LLM failures
        logger.error(f"LLM classification failed for text: {text[:100]}...")
        return {
            "attribution_present": "N",
            "attribution_present_confidence": 0,
            "attribution_outcome": "Neither",
            "attribution_outcome_confidence": 0,
            "attribution_locus": "Neither",
            "attribution_locus_confidence": 0,
            "attribution_effect": "Neither",
            "attribution_effect_confidence": 0,
            "attribution_cause": "Unknown",
            "attribution_cause_confidence": 0,
            "classification_status": "llm_failed",
            "error_message": "LLM classification failed after all retries"
        }

def generate_attribution_output_filename(df: pd.DataFrame, input_filename: str) -> str:
    """
    Generate output filename based on company, quarter, and year information from processed data
    Each file relates to a specific quarter/year for a single company.
    
    Args:
        df: DataFrame with processed data
        input_filename: Original input filename as fallback
        
    Returns:
        Formatted filename string
    """
    if df.empty:
        # Use input filename as fallback
        base_name = os.path.splitext(input_filename)[0]
        return f"{base_name}_attribution_class.csv"
    
    # Get company, quarter, year from first row (all should be the same)
    first_row = df.iloc[0]
    
    # Try different possible column names for company
    company = 'UNKNOWN'
    for col in ['Company', 'Symbol', 'company', 'symbol']:
        if col in df.columns and pd.notna(first_row.get(col)):
            company = str(first_row[col])
            break
    
    # Try different possible column names for quarter
    quarter = 'UNKNOWN'
    for col in ['Quarter', 'quarter']:
        if col in df.columns and pd.notna(first_row.get(col)):
            quarter = str(first_row[col])
            break
    
    # Try different possible column names for year
    year = 'UNKNOWN'
    for col in ['Date', 'Year', 'date', 'year']:
        if col in df.columns and pd.notna(first_row.get(col)):
            year = str(first_row[col])
            break
    
    return f"{company}_{quarter}_{year}_attribution_class.csv"

def calculate_attribution_statistics(df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate comprehensive attribution statistics from processed DataFrame.
    
    Args:
        df: DataFrame with attribution classifications
        
    Returns:
        Dictionary with detailed attribution statistics
    """
    
    stats = {}
    
    # Filter to successful classifications only (exclude filtered rows)
    successful_df = df[df['classification_status'] == 'success']
    total_successful = len(successful_df)
    
    # Count filtered rows
    filtered_df = df[df['classification_status'] == 'filtered']
    total_filtered = len(filtered_df)
    
    if total_successful == 0:
        # Return empty stats if no successful classifications
        return {
            'total_successful_classifications': 0,
            'attribution_present_count': 0,
            'attribution_absent_count': 0,
            'attribution_present_rate': 0.0
        }
    
    # Attribution presence statistics
    present_count = len(successful_df[successful_df['attribution_present'] == 'Y'])
    absent_count = len(successful_df[successful_df['attribution_present'] == 'N'])
    
    stats.update({
        'total_successful_classifications': total_successful,
        'total_filtered_rows': total_filtered,
        'attribution_present_count': present_count,
        'attribution_absent_count': absent_count,
        'attribution_present_rate': (present_count / total_successful) * 100 if total_successful > 0 else 0.0
    })
    
    # Detailed statistics for present attributions only
    present_df = successful_df[successful_df['attribution_present'] == 'Y']
    
    if len(present_df) > 0:
        # Outcome statistics
        outcome_counts = present_df['attribution_outcome'].value_counts()
        stats.update({
            'positive_outcome_count': outcome_counts.get('Positive', 0),
            'negative_outcome_count': outcome_counts.get('Negative', 0),
            'neither_outcome_count': outcome_counts.get('Neither', 0)
        })
        
        # Locus statistics
        locus_counts = present_df['attribution_locus'].value_counts()
        stats.update({
            'internal_locus_count': locus_counts.get('Internal', 0),
            'external_locus_count': locus_counts.get('External', 0),
            'neither_locus_count': locus_counts.get('Neither', 0)
        })
        
        # Combination statistics (Outcome + Locus)
        combo_counts = present_df.groupby(['attribution_outcome', 'attribution_locus']).size()
        
        # All possible combinations
        combinations = [
            ('Positive', 'Internal'), ('Positive', 'External'), ('Positive', 'Neither'),
            ('Negative', 'Internal'), ('Negative', 'External'), ('Negative', 'Neither'),
            ('Neither', 'Internal'), ('Neither', 'External'), ('Neither', 'Neither')
        ]
        
        for outcome, locus in combinations:
            key = f'{outcome.lower()}_{locus.lower()}_count'
            stats[key] = combo_counts.get((outcome, locus), 0)
        
        # Effect statistics
        effect_counts = present_df['attribution_effect'].value_counts()
        stats.update({
            'revenue_effect_count': effect_counts.get('Revenue', 0),
            'costs_effect_count': effect_counts.get('Costs', 0),
            'demand_effect_count': effect_counts.get('Demand', 0),
            'operations_effect_count': effect_counts.get('Operations', 0),
            'supply_effect_count': effect_counts.get('Supply', 0),
            'other_effect_count': effect_counts.get('Other', 0),
            'neither_effect_count': effect_counts.get('Neither', 0)
        })
        
        # Confidence score statistics
        if 'attribution_present_confidence' in present_df.columns:
            stats.update({
                'avg_present_confidence': present_df['attribution_present_confidence'].mean(),
                'avg_outcome_confidence': present_df['attribution_outcome_confidence'].mean(),
                'avg_locus_confidence': present_df['attribution_locus_confidence'].mean(),
                'avg_effect_confidence': present_df['attribution_effect_confidence'].mean()
            })
    else:
        # No present attributions found
        stats.update({
            'positive_outcome_count': 0, 'negative_outcome_count': 0, 'neither_outcome_count': 0,
            'internal_locus_count': 0, 'external_locus_count': 0, 'neither_locus_count': 0,
            'positive_internal_count': 0, 'positive_external_count': 0, 'positive_neither_count': 0,
            'negative_internal_count': 0, 'negative_external_count': 0, 'negative_neither_count': 0,
            'neither_internal_count': 0, 'neither_external_count': 0, 'neither_neither_count': 0,
            'revenue_effect_count': 0, 'costs_effect_count': 0, 'demand_effect_count': 0,
            'operations_effect_count': 0, 'supply_effect_count': 0, 'other_effect_count': 0, 'neither_effect_count': 0,
            'avg_present_confidence': 0.0, 'avg_outcome_confidence': 0.0, 'avg_locus_confidence': 0.0, 'avg_effect_confidence': 0.0
        })
    
    return stats

def process_attribution_classification(input_file_path: str) -> Tuple[bool, Dict[str, any]]:
    """
    Process a single topic classification CSV file and add attribution classifications.
    
    Args:
        input_file_path: Path to the input CSV file
        
    Returns:
        Tuple of (success_boolean, processing_statistics_dict)
    """
    
    # Initialize processing statistics
    file_stats = {
        'filename': os.path.basename(input_file_path),
        'start_time': datetime.now(),
        'llm_api_calls': 0,
        'processing_duration_seconds': 0.0,
        'total_rows': 0,
        'successful_classifications': 0,
        'empty_text_skipped': 0,
        'too_short_skipped': 0,
        'llm_failures': 0,
        'success_rate': 0.0
    }
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing file: {os.path.basename(input_file_path)}")
        
        # Read CSV file
        df = pd.read_csv(input_file_path)
        file_stats['total_rows'] = len(df)
        logger.info(f"Loaded {len(df)} rows from {os.path.basename(input_file_path)}")
        
        # Validate required columns
        required_columns = ['Snippet']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            file_stats['processing_duration_seconds'] = time.time() - start_time
            return False, file_stats
        
        # Get optional columns with fallbacks
        speaker_col = 'speaker' if 'speaker' in df.columns else None
        company_col = 'symbol' if 'symbol' in df.columns else 'company' if 'company' in df.columns else None
        
        # Check for filtering columns
        primary_topic_col = 'Primary_Topic' if 'Primary_Topic' in df.columns else None
        confidence_col = 'Primary_Topic_Confidence' if 'Primary_Topic_Confidence' in df.columns else None
        team_col = 'Team' if 'Team' in df.columns else None
        
        # Initialize new columns
        attribution_columns = [
            'attribution_present', 'attribution_present_confidence',
            'attribution_outcome', 'attribution_outcome_confidence', 
            'attribution_locus', 'attribution_locus_confidence',
            'attribution_effect', 'attribution_effect_confidence',
            'attribution_cause', 'attribution_cause_confidence',
            'classification_status', 'error_message'
        ]
        
        for col in attribution_columns:
            df[col] = None
        
        # Add filtering statistics to file_stats
        file_stats.update({
            'meeting_logistics_filtered': 0,
            'external_members_filtered': 0,
            'total_filtered': 0,
            'processed_for_classification': 0
        })
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying attributions"):
            text = row['Snippet']
            speaker = row[speaker_col] if speaker_col else "Unknown Speaker"
            company = row[company_col] if company_col else "Unknown Company"
            
            # Check filtering criteria
            should_filter = False
            filter_reason = None
            
            # Filter 1: Meeting Logistics with high confidence
            if (primary_topic_col and confidence_col and 
                primary_topic_col in row and confidence_col in row):
                primary_topic = str(row[primary_topic_col]) if pd.notna(row[primary_topic_col]) else ""
                confidence = row[confidence_col] if pd.notna(row[confidence_col]) else 0
                
                if ("Meeting Logistics" in primary_topic and confidence > 85):
                    should_filter = True
                    filter_reason = "meeting_logistics_high_confidence"
                    file_stats['meeting_logistics_filtered'] += 1
            
            # Filter 2: External members
            if not should_filter and team_col and team_col in row:
                team = str(row[team_col]) if pd.notna(row[team_col]) else ""
                if team == "external_members":
                    should_filter = True
                    filter_reason = "external_members"
                    file_stats['external_members_filtered'] += 1
            
            if should_filter:
                # Mark as filtered but keep in dataset
                file_stats['total_filtered'] += 1
                df.at[idx, 'attribution_present'] = "filtered_out"
                df.at[idx, 'attribution_present_confidence'] = None
                df.at[idx, 'attribution_outcome'] = "filtered_out"
                df.at[idx, 'attribution_outcome_confidence'] = None
                df.at[idx, 'attribution_locus'] = "filtered_out"
                df.at[idx, 'attribution_locus_confidence'] = None
                df.at[idx, 'attribution_effect'] = "filtered_out"
                df.at[idx, 'attribution_effect_confidence'] = None
                df.at[idx, 'attribution_cause'] = "filtered_out"
                df.at[idx, 'attribution_cause_confidence'] = None
                df.at[idx, 'classification_status'] = "filtered"
                df.at[idx, 'error_message'] = f"Filtered: {filter_reason}"
            else:
                # Process normally with LLM classification
                file_stats['processed_for_classification'] += 1
                attribution_result = classify_attribution(text, speaker, company, stats_tracker=file_stats)
                
                # Update DataFrame
                for col in attribution_columns:
                    df.at[idx, col] = attribution_result.get(col)
                
                # Update statistics
                status = attribution_result.get('classification_status', 'unknown')
                if status == 'success':
                    file_stats['successful_classifications'] += 1
                elif status == 'empty_text':
                    file_stats['empty_text_skipped'] += 1
                elif status == 'too_short':
                    file_stats['too_short_skipped'] += 1
                elif status == 'llm_failed':
                    file_stats['llm_failures'] += 1
        
        # Create output filename using the new naming convention
        base_filename = os.path.basename(input_file_path)
        output_filename = generate_attribution_output_filename(df, base_filename)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Calculate processing duration
        file_stats['processing_duration_seconds'] = time.time() - start_time
        
        # Calculate success rate based on processed items only (exclude filtered rows)
        processed_count = file_stats['processed_for_classification']
        file_stats['success_rate'] = (file_stats['successful_classifications'] / processed_count) * 100 if processed_count > 0 else 0
        
        # Calculate attribution statistics
        attribution_stats = calculate_attribution_statistics(df)
        file_stats.update(attribution_stats)
        
        # Save enhanced CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved attribution-enhanced file: {output_filename}")
        
        # VALIDATION: Check data integrity and narrative structure preservation
        logger.info(f"\n{'='*60}")
        logger.info("DATA INTEGRITY & NARRATIVE STRUCTURE VALIDATION")
        logger.info(f"{'='*60}")
        
        is_valid, validation_report = validate_data_integrity(input_file_path, df)
        
        # Add validation results to file statistics
        file_stats.update({
            'validation_passed': is_valid,
            'validation_errors_count': len(validation_report.get('validation_errors', [])),
            'validation_warnings_count': len(validation_report.get('warnings', [])),
            'snippet_order_preserved': validation_report.get('snippet_order_preserved', True),
            'speaker_order_preserved': validation_report.get('speaker_order_preserved', True),
            'content_preserved': validation_report.get('content_preserved', True),
            'missing_snippet_orders_count': len(validation_report.get('missing_snippet_orders', [])),
            'duplicate_snippet_orders_count': len(validation_report.get('duplicate_snippet_orders', [])),
            'content_mismatches_count': len(validation_report.get('content_mismatches', [])),
            'speaker_order_mismatches_count': len(validation_report.get('speaker_order_mismatches', []))
        })
        
        # Save detailed validation report if there are issues
        if not is_valid or validation_report.get('warnings'):
            validation_filename = base_filename.replace('.csv', '_validation_report.json')
            validation_path = os.path.join(METADATA_DIR, validation_filename)
            
            import json
            with open(validation_path, 'w', encoding='utf-8') as f:
                json.dump(validation_report, f, indent=2, default=str)
            
            logger.info(f"Detailed validation report saved to: {validation_filename}")
            
            if not is_valid:
                logger.warning("CRITICAL: Data integrity issues detected! Review validation report.")
        
        # Log processing statistics
        logger.info(f"Processing Statistics for {base_filename}:")
        logger.info(f"  Total rows: {file_stats['total_rows']}")
        logger.info(f"  Filtered rows: {file_stats['total_filtered']}")
        logger.info(f"    - Meeting Logistics (>85% conf): {file_stats['meeting_logistics_filtered']}")
        logger.info(f"    - External members: {file_stats['external_members_filtered']}")
        logger.info(f"  Processed for classification: {file_stats['processed_for_classification']}")
        logger.info(f"  Successful classifications: {file_stats['successful_classifications']}")
        logger.info(f"  Empty text skipped: {file_stats['empty_text_skipped']}")
        logger.info(f"  Too short skipped: {file_stats['too_short_skipped']}")
        logger.info(f"  LLM failures: {file_stats['llm_failures']}")
        logger.info(f"  LLM API calls: {file_stats['llm_api_calls']}")
        logger.info(f"  Success rate: {file_stats['success_rate']:.2f}%")
        logger.info(f"  Processing duration: {file_stats['processing_duration_seconds']:.2f} seconds")
        logger.info(f"  Attribution present: {file_stats.get('attribution_present_count', 0)} ({file_stats.get('attribution_present_rate', 0):.1f}%)")
        logger.info(f"  Data validation: {'PASSED' if file_stats['validation_passed'] else 'FAILED'}")
        
        # Calculate filtering efficiency
        if file_stats['total_rows'] > 0:
            filter_rate = (file_stats['total_filtered'] / file_stats['total_rows']) * 100
            api_savings = file_stats['total_filtered']  # Number of API calls saved
            logger.info(f"  Filtering efficiency: {filter_rate:.1f}% filtered, saved {api_savings} API calls")
        
        return True, file_stats
        
    except Exception as e:
        file_stats['processing_duration_seconds'] = time.time() - start_time
        logger.error(f"Error processing {input_file_path}: {e}")
        return False, file_stats

def create_attribution_processing_summary_csv(all_processing_stats: List[dict], output_dir: str):
    if not all_processing_stats:
        logger.warning("No processing statistics to summarize")
        return
    
    # Create DataFrame from processing statistics
    summary_df = pd.DataFrame(all_processing_stats)
    
    # Format datetime columns
    if 'start_time' in summary_df.columns:
        summary_df['start_time'] = summary_df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Round floating point columns
    float_columns = ['processing_duration_seconds', 'success_rate', 'attribution_present_rate', 
                    'avg_present_confidence', 'avg_outcome_confidence', 'avg_locus_confidence']
    for col in float_columns:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(2)
    
    # Reorder columns for better readability
    column_order = [
        # File info
        'filename', 'start_time', 'processing_duration_seconds',
        
        # Processing statistics and filtering
        'total_rows', 'total_filtered', 'meeting_logistics_filtered', 'external_members_filtered',
        'processed_for_classification', 'successful_classifications', 'success_rate', 'llm_api_calls',
        'empty_text_skipped', 'too_short_skipped', 'llm_failures',
        
        # Data validation results
        'validation_passed', 'validation_errors_count', 'validation_warnings_count',
        'snippet_order_preserved', 'speaker_order_preserved', 'content_preserved',
        'missing_snippet_orders_count', 'duplicate_snippet_orders_count', 
        'content_mismatches_count', 'speaker_order_mismatches_count',
        
        # Attribution presence
        'total_successful_classifications', 'total_filtered_rows', 'attribution_present_count', 
        'attribution_absent_count', 'attribution_present_rate',
        
        # Outcome statistics
        'positive_outcome_count', 'negative_outcome_count', 'neither_outcome_count',
        
        # Locus statistics
        'internal_locus_count', 'external_locus_count', 'neither_locus_count',
        
        # Effect statistics
        'revenue_effect_count', 'costs_effect_count', 'demand_effect_count',
        'operations_effect_count', 'supply_effect_count', 'other_effect_count', 'neither_effect_count',
        
        # Combination statistics
        'positive_internal_count', 'positive_external_count', 'positive_neither_count',
        'negative_internal_count', 'negative_external_count', 'negative_neither_count',
        'neither_internal_count', 'neither_external_count', 'neither_neither_count',
        
        # Confidence statistics
        'avg_present_confidence', 'avg_outcome_confidence', 'avg_locus_confidence', 'avg_effect_confidence'
    ]
    
    # Reorder columns, keeping any extra columns at the end
    existing_columns = [col for col in column_order if col in summary_df.columns]
    extra_columns = [col for col in summary_df.columns if col not in column_order]
    final_column_order = existing_columns + extra_columns
    summary_df = summary_df[final_column_order]
    
    # Save summary CSV
    summary_path = os.path.join(output_dir, "attribution_processing_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"Saved processing summary to: {summary_path}")
    
    # Log aggregate statistics
    logger.info("\n" + "="*80)
    logger.info("ATTRIBUTION PROCESSING SUMMARY")
    logger.info("="*80)
    
    total_files = len(summary_df)
    total_rows = summary_df['total_rows'].sum()
    total_successful = summary_df['successful_classifications'].sum()
    total_processed = summary_df['processed_for_classification'].sum()
    total_api_calls = summary_df['llm_api_calls'].sum()
    total_duration = summary_df['processing_duration_seconds'].sum()
    
    logger.info(f"Files processed: {total_files}")
    logger.info(f"Total text snippets: {total_rows}")
    logger.info(f"Successful classifications: {total_successful}")
    logger.info(f"Overall success rate: {(total_successful/total_processed)*100:.2f}% (of processed items)")
    logger.info(f"Total LLM API calls: {total_api_calls}")
    logger.info(f"Total processing time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    # Validation summary reporting
    if 'validation_passed' in summary_df.columns:
        total_files_processed = len(summary_df) - 1  # Exclude the summary row
        validation_passed_count = summary_df['validation_passed'].sum() if 'validation_passed' in summary_df.columns else 0
        validation_failed_count = total_files_processed - validation_passed_count
        
        logger.info(f"\nData Validation Summary:")
        logger.info(f"Files with validation PASSED: {validation_passed_count}/{total_files_processed}")
        if validation_failed_count > 0:
            logger.info(f"Files with validation FAILED: {validation_failed_count}")
            logger.warning("Some files failed validation - check individual validation reports")
        
        # Detailed validation statistics
        if 'validation_errors_count' in summary_df.columns:
            total_errors = summary_df['validation_errors_count'].sum()
            total_warnings = summary_df['validation_warnings_count'].sum() if 'validation_warnings_count' in summary_df.columns else 0
            
            if total_errors > 0:
                logger.info(f"Total validation errors: {total_errors}")
            if total_warnings > 0:
                logger.info(f"Total validation warnings: {total_warnings}")
        
        # Data integrity summary
        if all(col in summary_df.columns for col in ['snippet_order_preserved', 'speaker_order_preserved', 'content_preserved']):
            snippet_order_issues = len(summary_df) - summary_df['snippet_order_preserved'].sum() - 1  # Exclude summary row
            speaker_order_issues = len(summary_df) - summary_df['speaker_order_preserved'].sum() - 1
            content_issues = len(summary_df) - summary_df['content_preserved'].sum() - 1
            
            logger.info(f"Data integrity issues:")
            logger.info(f"  Snippet order issues: {snippet_order_issues} files")
            logger.info(f"  Speaker order issues: {speaker_order_issues} files") 
            logger.info(f"  Content preservation issues: {content_issues} files")
    
    # Filtering statistics aggregates
    if 'total_filtered' in summary_df.columns:
        total_filtered = summary_df['total_filtered'].sum()
        total_logistics_filtered = summary_df['meeting_logistics_filtered'].sum()
        total_external_filtered = summary_df['external_members_filtered'].sum()
        filter_rate = (total_filtered / total_rows) * 100 if total_rows > 0 else 0
        
        logger.info(f"\nFiltering Efficiency:")
        logger.info(f"Total rows filtered: {total_filtered} ({filter_rate:.1f}%)")
        logger.info(f"  Meeting Logistics (>85% conf): {total_logistics_filtered}")
        logger.info(f"  External members: {total_external_filtered}")
        logger.info(f"API calls saved by filtering: {total_filtered}")
        if total_duration > 0 and total_api_calls > 0:
            time_per_call = total_duration / total_api_calls
            estimated_time_saved = total_filtered * time_per_call
            logger.info(f"Estimated time saved: {estimated_time_saved:.1f} seconds ({estimated_time_saved/60:.1f} minutes)")
    
    # Attribution statistics aggregates
    if 'attribution_present_count' in summary_df.columns:
        total_present = summary_df['attribution_present_count'].sum()
        present_rate = (total_present / total_successful) * 100 if total_successful > 0 else 0
        
        logger.info(f"\nAttribution Analysis:")
        logger.info(f"Total attributions present: {total_present} ({present_rate:.1f}%)")
        
        # Outcome breakdown
        positive_total = summary_df['positive_outcome_count'].sum()
        negative_total = summary_df['negative_outcome_count'].sum() 
        neither_outcome_total = summary_df['neither_outcome_count'].sum()
        
        logger.info(f"  Positive outcomes: {positive_total}")
        logger.info(f"  Negative outcomes: {negative_total}")
        logger.info(f"  Neither outcomes: {neither_outcome_total}")
        
        # Locus breakdown
        internal_total = summary_df['internal_locus_count'].sum()
        external_total = summary_df['external_locus_count'].sum()
        neither_locus_total = summary_df['neither_locus_count'].sum()
        
        logger.info(f"  Internal locus: {internal_total}")
        logger.info(f"  External locus: {external_total}")
        logger.info(f"  Neither locus: {neither_locus_total}")
        
        # Effect breakdown
        revenue_total = summary_df['revenue_effect_count'].sum()
        costs_total = summary_df['costs_effect_count'].sum()
        demand_total = summary_df['demand_effect_count'].sum()
        operations_total = summary_df['operations_effect_count'].sum()
        supply_total = summary_df['supply_effect_count'].sum()
        other_effect_total = summary_df['other_effect_count'].sum()
        
        logger.info(f"  Revenue effects: {revenue_total}")
        logger.info(f"  Cost effects: {costs_total}")
        logger.info(f"  Demand effects: {demand_total}")
        logger.info(f"  Operations effects: {operations_total}")
        logger.info(f"  Supply effects: {supply_total}")
        logger.info(f"  Other effects: {other_effect_total}")
        
        # Key combinations
        pos_internal = summary_df['positive_internal_count'].sum()
        pos_external = summary_df['positive_external_count'].sum()
        neg_internal = summary_df['negative_internal_count'].sum()
        neg_external = summary_df['negative_external_count'].sum()
        
        logger.info(f"\nKey Attribution Patterns:")
        logger.info(f"  Positive + Internal: {pos_internal}")
        logger.info(f"  Positive + External: {pos_external}")
        logger.info(f"  Negative + Internal: {neg_internal}")
        logger.info(f"  Negative + External: {neg_external}")
    
    logger.info("="*80)

def validate_data_integrity(input_file_path: str, output_df: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
    """
    Validate that the output DataFrame preserves all data and maintains narrative structure
    compared to the original input CSV file.
    
    Args:
        input_file_path: Path to the original input CSV file  
        output_df: The processed DataFrame with attribution classifications
        
    Returns:
        Tuple of (is_valid: bool, validation_report: dict)
    """
    
    logger.info("Validating data integrity and narrative structure preservation...")
    
    validation_report = {
        'input_file': os.path.basename(input_file_path),
        'is_valid': True,
        'validation_errors': [],
        'warnings': [],
        'original_row_count': 0,
        'output_row_count': 0,
        'snippet_order_preserved': True,
        'speaker_order_preserved': True,
        'content_preserved': True,
        'missing_snippet_orders': [],
        'duplicate_snippet_orders': [],
        'speaker_order_mismatches': [],
        'content_mismatches': [],
        'missing_required_columns': [],
        'unexpected_changes': []
    }
    
    try:
        # Read original CSV file
        original_df = pd.read_csv(input_file_path)
        validation_report['original_row_count'] = len(original_df)
        validation_report['output_row_count'] = len(output_df)
        
        logger.info(f"Original CSV: {len(original_df)} rows")
        logger.info(f"Output DataFrame: {len(output_df)} rows")
        
        # Check 1: Row count preservation
        if len(original_df) != len(output_df):
            validation_report['is_valid'] = False
            validation_report['validation_errors'].append(
                f"Row count mismatch: original={len(original_df)}, output={len(output_df)}"
            )
        
        # Check 2: Required columns presence
        required_original_columns = ['Snippet', 'Snippet_Order']
        if 'Speaker_Name' in original_df.columns:
            required_original_columns.append('Speaker_Name')
        if 'Speaker_Order' in original_df.columns:
            required_original_columns.append('Speaker_Order')
        
        missing_columns = [col for col in required_original_columns if col not in output_df.columns]
        if missing_columns:
            validation_report['is_valid'] = False
            validation_report['missing_required_columns'] = missing_columns
            validation_report['validation_errors'].append(
                f"Missing required columns in output: {missing_columns}"
            )
        
        # Check 3: Snippet Order preservation and sequence
        if 'Snippet_Order' in original_df.columns and 'Snippet_Order' in output_df.columns:
            original_snippet_orders = original_df['Snippet_Order'].tolist()
            output_snippet_orders = output_df['Snippet_Order'].tolist()
            
            # Check if all original snippet orders are present
            missing_orders = set(original_snippet_orders) - set(output_snippet_orders)
            if missing_orders:
                validation_report['is_valid'] = False
                validation_report['missing_snippet_orders'] = sorted(list(missing_orders))
                validation_report['validation_errors'].append(
                    f"Missing snippet orders: {len(missing_orders)} orders missing"
                )
            
            # Check for duplicates
            output_order_counts = pd.Series(output_snippet_orders).value_counts()
            duplicates = output_order_counts[output_order_counts > 1].index.tolist()
            if duplicates:
                validation_report['is_valid'] = False
                validation_report['duplicate_snippet_orders'] = sorted(duplicates)
                validation_report['validation_errors'].append(
                    f"Duplicate snippet orders found: {duplicates}"
                )
            
            # Check if sequence is preserved (should be consecutive if no gaps)
            if original_snippet_orders == output_snippet_orders:
                logger.info("Snippet order sequence perfectly preserved")
            else:
                validation_report['snippet_order_preserved'] = False
                validation_report['warnings'].append(
                    "Snippet order sequence differs from original (may be acceptable if filtered)"
                )
        
        # Check 4: Speaker Order preservation (if available)
        if ('Speaker_Order' in original_df.columns and 'Speaker_Order' in output_df.columns and
            'Speaker_Name' in original_df.columns and 'Speaker_Name' in output_df.columns):
            
            # Group by speaker and check order preservation within speakers
            speaker_order_issues = []
            
            for speaker in original_df['Speaker_Name'].unique():
                if pd.isna(speaker):
                    continue
                    
                orig_speaker_df = original_df[original_df['Speaker_Name'] == speaker].sort_values('Snippet_Order')
                out_speaker_df = output_df[output_df['Speaker_Name'] == speaker].sort_values('Snippet_Order')
                
                if len(orig_speaker_df) != len(out_speaker_df):
                    speaker_order_issues.append({
                        'speaker': speaker,
                        'issue': 'different_snippet_count',
                        'original_count': len(orig_speaker_df),
                        'output_count': len(out_speaker_df)
                    })
                
                # Check if speaker order values are preserved
                orig_speaker_orders = orig_speaker_df['Speaker_Order'].tolist()
                out_speaker_orders = out_speaker_df['Speaker_Order'].tolist()
                
                if orig_speaker_orders != out_speaker_orders:
                    speaker_order_issues.append({
                        'speaker': speaker,
                        'issue': 'speaker_order_mismatch',
                        'original_orders': orig_speaker_orders[:5],  # First 5 for brevity
                        'output_orders': out_speaker_orders[:5]
                    })
            
            if speaker_order_issues:
                validation_report['speaker_order_preserved'] = False
                validation_report['speaker_order_mismatches'] = speaker_order_issues
                if len(speaker_order_issues) > 3:  # Only escalate to error if many issues
                    validation_report['is_valid'] = False
                    validation_report['validation_errors'].append(
                        f"Significant speaker order mismatches: {len(speaker_order_issues)} speakers affected"
                    )
                else:
                    validation_report['warnings'].append(
                        f"Minor speaker order differences: {len(speaker_order_issues)} speakers affected"
                    )
        
        # Check 5: Content preservation for shared rows (by Snippet_Order)
        if 'Snippet_Order' in original_df.columns and 'Snippet_Order' in output_df.columns:
            # Create lookup dictionaries by Snippet_Order
            orig_by_order = original_df.set_index('Snippet_Order')['Snippet'].to_dict()
            out_by_order = output_df.set_index('Snippet_Order')['Snippet'].to_dict()
            
            content_mismatches = []
            
            # Check content for matching snippet orders
            common_orders = set(orig_by_order.keys()) & set(out_by_order.keys())
            for snippet_order in sorted(list(common_orders)[:20]):  # Check first 20 for performance
                original_text = str(orig_by_order[snippet_order]).strip()
                output_text = str(out_by_order[snippet_order]).strip()
                
                # Normalize for comparison (remove extra quotes/whitespace that CSV processing might add)
                original_normalized = original_text.replace('""', '"').strip('"').strip()
                output_normalized = output_text.replace('""', '"').strip('"').strip()
                
                if original_normalized != output_normalized:
                    content_mismatches.append({
                        'snippet_order': snippet_order,
                        'original_length': len(original_normalized),
                        'output_length': len(output_normalized),
                        'original_preview': original_normalized[:100] + "..." if len(original_normalized) > 100 else original_normalized,
                        'output_preview': output_normalized[:100] + "..." if len(output_normalized) > 100 else output_normalized
                    })
            
            if content_mismatches:
                validation_report['content_preserved'] = False
                validation_report['content_mismatches'] = content_mismatches
                if len(content_mismatches) > 5:  # Only error if many mismatches
                    validation_report['is_valid'] = False
                    validation_report['validation_errors'].append(
                        f"Significant content changes detected: {len(content_mismatches)} snippets modified"
                    )
                else:
                    validation_report['warnings'].append(
                        f"Minor content differences: {len(content_mismatches)} snippets (may be CSV formatting)"
                    )
        
        # Check 6: Attribution columns added correctly
        expected_attribution_columns = [
            'attribution_present', 'attribution_present_confidence',
            'attribution_outcome', 'attribution_outcome_confidence', 
            'attribution_locus', 'attribution_locus_confidence',
            'attribution_effect', 'attribution_effect_confidence',
            'attribution_cause', 'attribution_cause_confidence',
            'classification_status', 'error_message'
        ]
        
        missing_attribution_cols = [col for col in expected_attribution_columns if col not in output_df.columns]
        if missing_attribution_cols:
            validation_report['is_valid'] = False
            validation_report['validation_errors'].append(
                f"Missing attribution columns: {missing_attribution_cols}"
            )
        
        # Check 7: Filtering status consistency
        if 'classification_status' in output_df.columns:
            status_counts = output_df['classification_status'].value_counts()
            logger.info(f"Classification status distribution: {dict(status_counts)}")
            
            # Ensure filtered rows have appropriate attribution values
            filtered_rows = output_df[output_df['classification_status'] == 'filtered']
            if len(filtered_rows) > 0:
                # Check that filtered rows have 'filtered_out' attribution values
                filtered_present_values = filtered_rows['attribution_present'].unique()
                if not all(val == 'filtered_out' for val in filtered_present_values if pd.notna(val)):
                    validation_report['warnings'].append(
                        f"Filtered rows don't all have 'filtered_out' attribution_present value"
                    )
        
        # Log validation results
        if validation_report['is_valid']:
            logger.info("Data integrity validation PASSED")
            if validation_report['warnings']:
                logger.info(f"   With {len(validation_report['warnings'])} warnings (acceptable)")
                for warning in validation_report['warnings'][:3]:
                    logger.info(f"   WARNING: {warning}")
        else:
            logger.error("Data integrity validation FAILED")
            logger.error(f"   {len(validation_report['validation_errors'])} critical errors detected:")
            for error in validation_report['validation_errors']:
                logger.error(f"   ERROR: {error}")
        
        # Additional summary info
        if validation_report['missing_snippet_orders']:
            logger.warning(f"   Missing snippet orders: {len(validation_report['missing_snippet_orders'])}")
        if validation_report['duplicate_snippet_orders']:
            logger.warning(f"   Duplicate snippet orders: {len(validation_report['duplicate_snippet_orders'])}")
        if validation_report['content_mismatches']:
            logger.info(f"   Content differences: {len(validation_report['content_mismatches'])} snippets")
        if validation_report['speaker_order_mismatches']:
            logger.info(f"   Speaker order issues: {len(validation_report['speaker_order_mismatches'])} speakers")
        
    except Exception as e:
        validation_report['is_valid'] = False
        validation_report['validation_errors'].append(f"Validation process failed: {str(e)}")
        logger.error(f"Error during validation: {e}")
    
    return validation_report['is_valid'], validation_report

def main():
    logger.info("Starting Attribution Classification Process")
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Find all topic classification CSV files
    pattern = os.path.join(INPUT_DIR, "*topic_class*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        logger.warning(f"No CSV files found matching pattern: {pattern}")
        return
    
    logger.info(f"Found {len(csv_files)} topic classification CSV files to process")
    
    # Process each file and collect statistics
    successful_files = 0
    failed_files = 0
    all_processing_stats = []
    
    for file_path in csv_files:
        logger.info(f"\n{'='*60}")
        success, file_stats = process_attribution_classification(file_path)
        all_processing_stats.append(file_stats)
        
        if success:
            successful_files += 1
        else:
            failed_files += 1
    
    # Create processing summary CSV
    if all_processing_stats:
        create_attribution_processing_summary_csv(all_processing_stats, METADATA_DIR)
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("Attribution Classification Process Complete")
    logger.info(f"Successfully processed: {successful_files} files")
    logger.info(f"Failed to process: {failed_files} files")
    logger.info(f"Enhanced CSV files saved to: {OUTPUT_DIR}")
    logger.info(f"Summary reports and metadata saved to: {METADATA_DIR}")
    logger.info(f"Processing summary CSV: {os.path.join(METADATA_DIR, 'attribution_processing_summary.csv')}")
    
    if successful_files > 0:
        logger.info("\nOutput Structure:")
        logger.info(f"  {OUTPUT_DIR}/")
        logger.info(f"     *_attribution_class.csv (enhanced data files)")
        logger.info(f"     metadata/")
        logger.info(f"        attribution_processing_summary.csv")
        logger.info(f"        *_validation_report.json (if validation issues)")
        
        logger.info("\nNext Steps:")
        logger.info("1. Review the enhanced CSV files for attribution patterns")
        logger.info("2. Analyze attribution locus distribution (Internal vs External)")
        logger.info("3. Examine attribution outcome patterns (Positive vs Negative)")
        logger.info("4. Consider confidence score distributions for quality assessment")
        logger.info("5. Review processing summary for performance insights")
        logger.info("6. Check validation reports if any data integrity issues were flagged")
        logger.info("\nFuture Enhancement Opportunities:")
        logger.info("- Implement causal extraction for more granular cause/effect analysis")
        logger.info("- Add temporal attribution analysis")
        logger.info("- Extract specific attribution quotes for qualitative analysis")

if __name__ == "__main__":
    main()


# python attribution_classification.py