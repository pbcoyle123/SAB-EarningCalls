# -*- coding: utf-8 -*-
"""

The script uses a sophisticated 6-step hybrid approach that combines semantic analysis with optional LLM refinement.
Step 1: Sentence Tokenization - spaCy i.e. sentences = sentence_tokenize(text)
Step 2: Sentence-BERT embeddings. Uses SentenceBERT (all-MiniLM-L6-v2) to convert each sentence into a 384-dimensional vector. 
        These embeddings capture the semantic meaning of each sentence
        Similar sentences will have similar embeddings (high cosine similarity)
Step 3: Core Boundary Detection Algorithm
        3A. Calculates cosine similarity between every consecutive pair of sentences - Low similarity = potential paragraph boundary
        3B. Identifies sentences starting with conjunctions (conjunction_starters). Applies a conjunction_factor = 0.85 to make boundaries harder to create
        3C: Contextual Window Analysis - Backward Context (3 sentences) & Forward Context (3 sentences):
        3D. Topic Shift Detection - The algorithm identifies true topic shifts using multiple criteria:
            Low immediate similarity: Current sentence differs from previous (sim < 0.455 for non-conjunctions)
            Low backward context: Sentence doesn't relate to recent history (avg_backward < 0.4875)
            Higher forward similarity: Sentence relates more to upcoming content
            Substantial sentences: Both sentences have >5 words (avoids fragmenting short sentences)
Step 4: Post-Processing Filters
        4A: Short Paragraph Prevention - Prevents creating paragraphs with only 1-2 sentences. Exception: Allows short paragraphs if there's a dramatic topic shift (>50% dissimilarity)
Step 5: LLM Refinement (Optional)
        If an LLM provider is available, the script:
        Creates refinement prompt with initial paragraphs
        Asks LLM to:
        Merge very short paragraphs with related content
        Split paragraphs where topics change significantly
        Ensure natural transition points
        Parses LLM response and uses refined paragraphs if successful
        Falls back to embedding-only results if LLM fails
Step 6: Visualization
        Creates a graph showing:
        Sentence similarity scores over time
        Identified paragraph boundaries as vertical lines
        Similarity threshold as horizontal lin

"""

import spacy
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import time

# Load spaCy model for sentence tokenization
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# Load SentenceBERT model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def sentence_tokenize(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

class LLMProvider:
    def predict_paragraph_boundaries(self, text):
        raise NotImplementedError("Subclasses must implement this")

    def refine_paragraphs(self, prompt):
        """Refine paragraph segmentation based on initial embedding-based segmentation"""
        raise NotImplementedError("Subclasses must implement this")

class DeepSeekLocalProvider(LLMProvider):
    def __init__(self):
        self.model = pipeline(
            "text-generation",
            model="deepseek-ai/deepseek-llm-7b-instruct",
            tokenizer="deepseek-ai/deepseek-llm-7b-instruct",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    
    def predict_paragraph_boundaries(self, text):
        prompt = f"""
        Given the text below, identify sentence indices (starting from 1) where a new paragraph begins. 
        Provide output in JSON as follows:

        {{
          "paragraph_boundaries": [indices]
        }}

        Text:
        {text}
        """
        
        try:
            output = self.model(prompt, max_new_tokens=512)[0]['generated_text']
            # Extract the JSON part from the response
            json_str = output.split("```json")[-1].split("```")[0] if "```json" in output else output
            json_str = json_str.split("{")[1].split("}")[0]
            json_str = "{" + json_str + "}"
            boundaries = json.loads(json_str).get("paragraph_boundaries", [])
            return boundaries
        except Exception as e:
            print(f"Error processing DeepSeek output: {e}")
            return []

    def refine_paragraphs(self, prompt):
        raise NotImplementedError("Subclasses must implement this")

class HuggingFaceAPIProvider(LLMProvider):
    def __init__(self, model_name, api_token):
        from huggingface_hub.inference_api import InferenceApi
        self.inference = InferenceApi(
            repo_id=model_name,
            token=api_token
        )
    
    def predict_paragraph_boundaries(self, text):
        prompt = f"""
        Given the text below, identify sentence indices (starting from 1) where a new paragraph begins. 
        Provide output in JSON as follows:
        {{
          "paragraph_boundaries": [indices]
        }}
        Text:
        {text}
        """
        
        try:
            response = self.inference(prompt)
            # Extract the JSON part from the response
            # The response format may vary based on the model
            if isinstance(response, dict) and "generated_text" in response:
                output = response["generated_text"]
            else:
                output = response
                
            # Extract JSON content as before
            json_str = output.split("```json")[-1].split("```")[0] if "```json" in output else output
            json_str = json_str.split("{")[1].split("}")[0]
            json_str = "{" + json_str + "}"
            boundaries = json.loads(json_str).get("paragraph_boundaries", [])
            return boundaries
        except Exception as e:
            print(f"Error processing API response: {e}")
            print(f"Response received: {str(response)[:200]}...")  # Print first 200 chars of response
            return []

    def refine_paragraphs(self, prompt):
        raise NotImplementedError("Subclasses must implement this")

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key, model="gpt-4o-mini"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            print("OpenAI package not installed. Run: pip install openai")
            raise
    
    def predict_paragraph_boundaries(self, text):
        prompt = f"""
        Given the text below, identify sentence indices (starting from 1) where a new paragraph begins.
        
        Guidelines for paragraph segmentation:
        1. Each paragraph should improve readability and contain well-contained topics
        2. Paragraphs should group related ideas or topics and maintain coherent flow
        3. Short paragraphs (1-2 sentences) are acceptable if they contain enough information for readers to understand the topic. When creating these please check the previous paragraphs do not contain the same topic.
        4. Introductory statements, thank you notes, and farewells can be separate short paragraphs
        5. Look for natural topic shifts, speaker changes, and transitions in the narrative
        
        Provide output in JSON format ONLY, as follows:
        {{
          "paragraph_boundaries": [indices]
        }}
        
        Text:
        {text}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that identifies paragraph boundaries in text. Respond only with JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # The response is already formatted as JSON
            boundaries = json.loads(response.choices[0].message.content).get("paragraph_boundaries", [])
            return boundaries
        except Exception as e:
            print(f"Error processing OpenAI response: {e}")
            return []

    def refine_paragraphs(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that refines text segmentation into cohesive paragraphs."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            refined_json = json.loads(response.choices[0].message.content)
            return refined_json
        except Exception as e:
            print(f"Error processing OpenAI refinement: {e}")
            return None

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key, model="claude-3-7-sonnet-20240229"):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            print("Anthropic package not installed. Run: pip install anthropic")
            raise
    
    def predict_paragraph_boundaries(self, text):
        prompt = f"""
        Given the text below, identify sentence indices (starting from 1) where a new paragraph begins.
        
        Guidelines for paragraph segmentation:
        1. Each paragraph should improve readability and contain well-contained topics
        2. Paragraphs should group related ideas and maintain coherent flow
        3. Short paragraphs (1-2 sentences) are acceptable if they contain enough information for readers to understand the topic
        4. Introductory statements, thank you notes, and farewells can be separate short paragraphs
        5. Look for natural topic shifts, speaker changes, and transitions in the narrative
        
        Provide output in JSON format ONLY, as follows:
        {{
          "paragraph_boundaries": [indices]
        }}
        
        Text:
        {text}
        """
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1
            )
            
            # Extract JSON from response
            response_text = response.content[0].text
            
            # Try to find JSON in the response
            try:
                # First try to parse the entire response as JSON
                boundaries = json.loads(response_text).get("paragraph_boundaries", [])
            except:
                # If that fails, try to extract JSON from markdown code blocks
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    boundaries = json.loads(json_str).get("paragraph_boundaries", [])
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                    boundaries = json.loads(json_str).get("paragraph_boundaries", [])
                else:
                    # Last resort: try to find brackets
                    json_pattern = r'\{.*"paragraph_boundaries"\s*:\s*\[.*\].*\}'
                    import re
                    match = re.search(json_pattern, response_text, re.DOTALL)
                    if match:
                        boundaries = json.loads(match.group(0)).get("paragraph_boundaries", [])
                    else:
                        print("Couldn't extract JSON from Claude response")
                        return []
                        
            return boundaries
        except Exception as e:
            print(f"Error processing Claude response: {e}")
            return []

    def refine_paragraphs(self, prompt):
        raise NotImplementedError("Subclasses must implement this")

def validate_boundaries_with_embeddings(sentences, embeddings, predicted_boundaries, threshold=0.65):
    # compute cosine similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = util.cos_sim(embeddings[i], embeddings[i+1]).item()
        similarities.append(sim)
    
    # Instead of filtering out conjunction starters, just weigh them as a factor
    conjunction_starters = ["and ", "but ", "or ", "so ", "yet ", "for ", "nor ", "because "]
    is_conjunction_starter = []
    for sentence in sentences:
        starts_with_conjunction = False
        sentence_lower = sentence.lower().strip()
        for conj in conjunction_starters:
            if sentence_lower.startswith(conj):
                starts_with_conjunction = True
                break
        is_conjunction_starter.append(starts_with_conjunction)
    
    # Validate predicted boundaries based on similarity scores and context
    validated_boundaries = []
    for idx in predicted_boundaries:
        if idx < 2 or idx > len(sentences):  # Skip invalid indices
            continue
            
        # Check similarity between this sentence and previous sentence
        if 0 <= idx-2 < len(similarities):
            sim_score = similarities[idx-2]
            
            # Enhanced context: check similarity with sentences further back
            context_window = 3  # Look at 3 sentences for context
            backward_context = []
            for j in range(1, context_window+1):
                if idx-1-j >= 0:
                    back_sim = util.cos_sim(embeddings[idx-1], embeddings[idx-1-j]).item()
                    backward_context.append(back_sim)
            
            # If this sentence is similar to recent context, don't break paragraph
            avg_backward_similarity = sum(backward_context)/len(backward_context) if backward_context else 0
            
            # For conjunction starters, require a stronger signal of topic shift
            conjunction_factor = 0.85 if is_conjunction_starter[idx-1] else 1.0
            
            # Break if similarity is low and not contextually connected, 
            # with a higher threshold for conjunction-starting sentences
            if (sim_score < threshold * 0.85 * conjunction_factor and 
                avg_backward_similarity < threshold * 0.9 * conjunction_factor):
                validated_boundaries.append(idx)
    
    # Add potential boundaries based on embeddings alone with contextual checks
    additional_boundaries = []
    for i in range(1, len(sentences)):
        if i+1 in predicted_boundaries or i+1 in validated_boundaries:
            continue  # Already handled
            
        # Get similarity to previous sentence
        if i-1 < len(similarities):
            sim = similarities[i-1]
            
            # Check if this is a substantial sentence (not too short)
            current_sent_length = len(sentences[i].split())
            prev_sent_length = len(sentences[i-1].split())
            
            # Check for topic shift using contextual similarity
            backward_sims = []
            forward_sims = []
            
            # Look backward for context
            for j in range(1, min(3, i)+1):
                backward_sims.append(util.cos_sim(embeddings[i], embeddings[i-j]).item())
                
            # Look forward for context
            for j in range(1, min(3, len(sentences)-i-1)+1):
                forward_sims.append(util.cos_sim(embeddings[i], embeddings[i+j]).item())
                
            avg_backward = sum(backward_sims)/len(backward_sims) if backward_sims else 1.0
            avg_forward = sum(forward_sims)/len(forward_sims) if forward_sims else 1.0
            
            # Adjust threshold for conjunction-starting sentences
            conjunction_factor = 0.85 if is_conjunction_starter[i] else 1.0
            
            # True topic shift: requires stronger evidence for conjunction starters
            forward_bonus = 1.2 * conjunction_factor  # Higher forward similarity requirement for conjunctions
            
            # Consider the case where a new topic starts with "And" but is completely different from prior text
            topic_shift_strength = avg_backward / avg_forward if avg_forward > 0 else 2.0
            
            # True topic shift: low similarity to previous context, higher similarity to upcoming context
            if ((sim < threshold * 0.7 * conjunction_factor) and 
                (avg_backward < threshold * 0.75 * conjunction_factor) and 
                (avg_forward > avg_backward * forward_bonus) and
                current_sent_length > 5 and 
                prev_sent_length > 5):
                    
                # For conjunction starters, allow boundary only if very strong topic shift
                if not is_conjunction_starter[i] or topic_shift_strength < 0.7:
                    additional_boundaries.append(i+1)  # +1 for 1-indexing
    
    # Final set of boundaries
    final_boundaries = sorted(set(validated_boundaries + additional_boundaries))
    
    # Post-processing to remove boundaries that would create very short paragraphs
    if final_boundaries:
        filtered_boundaries = [final_boundaries[0]]
        for i in range(1, len(final_boundaries)):
            # If this boundary would create a paragraph with just 1-2 sentences, skip it
            # unless there's a strong topic shift
            if final_boundaries[i] - filtered_boundaries[-1] > 2:
                filtered_boundaries.append(final_boundaries[i])
            else:
                # Check if there's a dramatic topic shift despite few sentences
                idx1 = filtered_boundaries[-1] - 1  # Convert to 0-indexed
                idx2 = final_boundaries[i] - 1      # Convert to 0-indexed
                if idx1 >= 0 and idx2 < len(embeddings):
                    topic_shift = 1.0 - util.cos_sim(embeddings[idx1], embeddings[idx2]).item()
                    if topic_shift > 0.5:  # Significant topic change
                        filtered_boundaries.append(final_boundaries[i])
        
        return filtered_boundaries, similarities
    else:
        return [1], similarities  # Default to just one paragraph if no boundaries found

def visualize_segmentation(similarities, boundaries, threshold=0.5, output_path='paragraph_segmentation.png'):
    plt.figure(figsize=(12, 5))
    plt.plot(similarities, marker='o', linestyle='-', label='Sentence Similarity')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    
    # Mark predicted paragraph boundaries
    for b in boundaries:
        plt.axvline(x=b-2, color='green', linestyle=':', alpha=0.7)
    
    plt.xlabel('Sentence Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Sentence Embedding Similarity for Paragraph Segmentation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()

def segment_text(text, llm_provider=None, threshold=0.65, visualize=True, output_dir=None, base_filename=None):
    # step 1: split text into sentences
    sentences = sentence_tokenize(text)
    print(f"Text split into {len(sentences)} sentences.")
    
    # Step 2: Compute sentence embeddings
    embeddings = model.encode(sentences)
    print("Computed sentence embeddings.")
    
    # Step 3: Initial segmentation based on embeddings only
    initial_boundaries, similarities = validate_boundaries_with_embeddings(
        sentences, embeddings, [], threshold  # Empty list for predicted_boundaries
    )
    print(f"Initial embedding-based boundaries: {initial_boundaries}")
    
    # Step 4: Build initial paragraphs
    initial_paragraphs = []
    start_idx = 0
    for b in initial_boundaries:
        end_idx = b - 1  # Adjust for 1-indexed boundaries
        paragraph = ' '.join(sentences[start_idx:end_idx])
        initial_paragraphs.append(paragraph)
        start_idx = end_idx
    
    # Add the last paragraph
    if start_idx < len(sentences):
        paragraph = ' '.join(sentences[start_idx:])
        initial_paragraphs.append(paragraph)
    
    # Step 5: Refine paragraphs with LLM if available
    final_paragraphs = initial_paragraphs
    if llm_provider:
        try:
            # Create a refinement prompt with the initial paragraphs
            refinement_prompt = "I have segmented this text into paragraphs. Please refine the segmentation by:\n"
            refinement_prompt += "1. Merging very short paragraphs (1-2 sentences) with related paragraphs\n"
            refinement_prompt += "2. Splitting paragraphs where the topic changes significantly\n"
            refinement_prompt += "3. Ensuring paragraph breaks occur at natural transition points\n\n"
            refinement_prompt += "Initial segmentation:\n\n"
            
            for i, para in enumerate(initial_paragraphs, 1):
                refinement_prompt += f"Paragraph {i}:\n{para}\n\n"
            
            refinement_prompt += "Please provide the refined segmentation as a JSON list of paragraphs:\n"
            refinement_prompt += '{"paragraphs": ["paragraph1", "paragraph2", ...]}'
                
            # Use the LLM to refine the paragraphs
            refined_json = llm_provider.refine_paragraphs(refinement_prompt)
            
            # Parse the refined paragraphs from the LLM response
            if refined_json and "paragraphs" in refined_json:
                final_paragraphs = refined_json["paragraphs"]
                print(f"LLM refinement: adjusted from {len(initial_paragraphs)} to {len(final_paragraphs)} paragraphs")
            else:
                print("LLM refinement failed, using embedding-only segmentation")
        except Exception as e:
            print(f"Error during LLM refinement: {e}")
            print("Falling back to embedding-only segmentation")
    
    # Step 6: Visualize (optional)
    if visualize:
        if base_filename:
            viz_name = f'paragraph_segmentation_{base_filename}.png'
        else:
            viz_name = f'paragraph_segmentation_{int(time.time())}.png'
            
        viz_path = viz_name
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            viz_path = os.path.join(output_dir, viz_name)
            
        visualize_segmentation(similarities, initial_boundaries, threshold, output_path=viz_path)
        print(f"Visualization saved as '{viz_path}'")
    
    return final_paragraphs

def process_transcript_json(input_file, output_dir, llm_provider=None, threshold=0.65, visualize=False):
    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Determine if the JSON contains a single transcript or multiple transcripts
    if isinstance(data, list):
        transcripts = data
    else:
        transcripts = [data]
    
    # Process each transcript
    for transcript in transcripts:
        # Process prepared remarks
        if "parsed_content" in transcript and "prepared_remarks" in transcript["parsed_content"]:
            for remark in transcript["parsed_content"]["prepared_remarks"]:
                if "dialogue" in remark:
                    # Extract the base filename for use in visualization naming
                    base_name = os.path.basename(input_file).split('.')[0]
                    
                    # Segment the dialogue text into paragraphs
                    paragraphs = segment_text(
                        remark["dialogue"],
                        llm_provider=llm_provider,
                        threshold=threshold,
                        visualize=visualize,
                        output_dir=output_dir,
                        base_filename=base_name
                    )
                    
                    # Add numbered paragraphs to the remark
                    for i, paragraph in enumerate(paragraphs, 1):
                        remark[f"paragraph_{i}"] = paragraph
        
        # Process Q&A section
        if "parsed_content" in transcript and "qna_section" in transcript["parsed_content"]:
            for qa in transcript["parsed_content"]["qna_section"]:
                if "dialogue" in qa:
                    # Extract the base filename for use in visualization naming
                    base_name = os.path.basename(input_file).split('.')[0]
                    
                    # Segment the dialogue text into paragraphs
                    paragraphs = segment_text(
                        qa["dialogue"],
                        llm_provider=llm_provider,
                        threshold=threshold,
                        visualize=visualize,
                        output_dir=output_dir,
                        base_filename=base_name
                    )
                    
                    # Add numbered paragraphs to the qa
                    for i, paragraph in enumerate(paragraphs, 1):
                        qa[f"paragraph_{i}"] = paragraph
    
    # Create output filename
    base_name = os.path.basename(input_file)
    name_parts = base_name.split('.')
    if len(name_parts) > 1:
        output_name = '.'.join(name_parts[:-1]) + '_paragraphs.' + name_parts[-1]
    else:
        output_name = base_name + '_paragraphs'
    
    output_path = os.path.join(output_dir, output_name)
    
    # Save the modified JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    print(f"Processed {input_file} and saved to {output_path}")
    return output_path

# Main execution
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Segment dialogues in transcript JSON files into paragraphs')
    parser.add_argument('--provider', choices=['none', 'deepseek', 'huggingface', 'openai', 'claude'], 
                        default='none', help='LLM provider to use')
    parser.add_argument('--api_key', help='API key for the selected provider')
    parser.add_argument('--model', help='Model name for the selected provider')
    parser.add_argument('--output_dir', 
                        default="03",  # Points to parallel_cpu/03/ for step 3 output
                        help='Output directory for segmented JSON files')
    parser.add_argument('--threshold', type=float, default=0.65, help='Similarity threshold for paragraph boundaries')
    parser.add_argument('--process_all', action='store_true', help='Process all matching JSON files (default: only first file)')
    parser.add_argument('--no_viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    # Set up LLM provider based on arguments
    llm_provider = None
    if args.provider == 'deepseek':
        llm_provider = DeepSeekLocalProvider()
    elif args.provider == 'huggingface':
        if not args.api_key:
            parser.error("--api_key is required for Hugging Face provider")
        model_name = args.model or "deepseek-ai/deepseek-llm-7b-instruct"
        llm_provider = HuggingFaceAPIProvider(model_name=model_name, api_token=args.api_key)
    elif args.provider == 'openai':
        if not args.api_key:
            parser.error("--api_key is required for OpenAI provider")
        model_name = args.model or "gpt-4o-mini"
        llm_provider = OpenAIProvider(api_key=args.api_key, model=model_name)
    elif args.provider == 'claude':
        if not args.api_key:
            parser.error("--api_key is required for Claude provider")
        model_name = args.model or "claude-3-7-sonnet-20240229"
        llm_provider = AnthropicProvider(api_key=args.api_key, model=model_name)
    
    # Find all transcript JSON files (updated for parallel_cpu structure)
    data_dir = "02"  # Points to parallel_cpu/02/ where previous script outputs
    transcript_files = glob.glob(os.path.join(data_dir, "*transcript_with_speakers*.json"))
    
    if not transcript_files:
        print(f"No transcript files found in {data_dir}")
        exit(1)
    
    print(f"Found {len(transcript_files)} transcript files")
    
    # Process either the first file or all files
    if args.process_all:
        for file_path in transcript_files:
            process_transcript_json(
                file_path,
                args.output_dir,
                llm_provider=llm_provider,
                threshold=args.threshold,
                visualize=not args.no_viz
            )
        print(f"Processed {len(transcript_files)} files and saved to {args.output_dir}")
    else:
        # Process only the first file
        output_path = process_transcript_json(
            transcript_files[0],
            args.output_dir,
            llm_provider=llm_provider, 
            threshold=args.threshold, 
            visualize=not args.no_viz
        )
        print(f"Processed {transcript_files[0]} and saved to {output_path}")
    
    print("Done.")

# Warning for invalid API key format
# if args.provider != 'none' and (not args.api_key or args.api_key == "sk-"):
#     print("Warning: Invalid API key format detected. Using embedding-only approach.")
#     args.provider = 'none'

if args.provider != 'none' and (not args.api_key or args.api_key == "x"):
    print("Warning: Invalid API key format detected. Using embedding-only approach.")
    args.provider = 'none'




# # Process one file
# python paragraph_breaks_pipeline_03_v2trainingBatch.py

# # Process all files
# python paragraph_breaks_pipeline_03_v2trainingBatch.py --process_all

# # Process without visualization
# python paragraph_breaks_pipeline_03_v2trainingBatch.py --process_all --no_viz

# # Use custom output directory (default is "03" within parallel_cpu/)
# python paragraph_breaks_pipeline_03_v2trainingBatch.py --output_dir "custom_output_folder"