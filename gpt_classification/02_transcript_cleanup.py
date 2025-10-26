# -*- coding: utf-8 -*-
"""

### **Purpose** 
The script processes raw transcript files to:
- Parse transcripts to identify speakers and their dialogues
- Classify sections into "Prepared Remarks" and "Q&A" sections
- Identify management team members vs external participants (analysts)
- Classify Q&A dialogues as questions or answers
- Track Q&A pairs (question-answer interactions)


### **Key Components and Methods**
1. **Speaker Pattern Recognition** (Line 25) - Regex to identify speaker names with company affiliations
2. **Metadata Extraction** (`extract_metadata_sections`, Lines 27-84) - Extracts executive/analyst info from headers
3. **Name Parsing** (`parse_name_role_pairs` Lines 86-124, 
- `parse_name_role_pairs` (Lines 86-124): Parses executive name-role pairs
- `parse_name_company_pairs` Lines 126-161,
-  `names_are_similar` Lines 227-253) - Parses name-role/company pairs with fuzzy matching
4. **Transcript Parsing** (`parse_transcript`, Lines 275-333) - Main function to extract speakers and dialogues into structured data
5. **Section Classification** (`classify_sections`, Lines 441-469) - Separates prepared remarks from Q&A sections
6. **Q&A Detection** (`detect_qna_start`, Lines 334-386) - Uses transition phrases, speaker patterns, and question density to detect Q&A start
7. **Q&A Classification** (`classify_qna_dialogues`, Lines 471-499) - Labels dialogues as questions/answers and tracks Q&A pairs
8. **Unicode Cleaning** (`clean_unicode_for_csv`, Lines 501-531) - Converts Unicode to ASCII for CSV compatibility

cmd line
- Test Mode: `python transcript_cleanup.py --test`
- Normal Mode: `python transcript_cleanup.py`


"""

import os
import json
import re
import pandas as pd
import spacy
from tqdm import tqdm
import argparse

# Regex pattern to match speakers - improved to handle company names with &, periods, etc.
# Examples it handles:
# - Ken Usdin:
# - Ken Usdin :
# - Ken Usdin - Jefferies & Company:
# - Ken Usdin - Jefferies & Company :
# - Matthew O'Connor - Deutsche Bank AG:
speaker_pattern = r'([A-Z][a-zA-Z\'\.\s]+(?:\s*[-–—]\s*[A-Z][a-zA-Z\'\.\s&,]+(?:\s+(?:LLC|Inc|Corp|Company|Group|AG|LP))?)*)\s*:'

def extract_metadata_sections(lines, management_team=None, external_members=None):
    metadata_sections = {
        "executives": [],
        "analysts": [],
        "additional_metadata": []
    }
    content_lines = []
    in_metadata = False
    
    for line in lines:
        line = line.strip()
        
        # Check for metadata section starts
        if line.startswith('Executives:'):
            in_metadata = True
            # Get the executives line without the header
            execs_line = line.replace('Executives:', '').strip()
            
            # Parse executives using improved logic
            exec_entries = parse_name_role_pairs(execs_line)
            
            for name, role in exec_entries:
                metadata_sections["executives"].append({
                    "name": name.strip(),
                    "role": role.strip()
                })
            continue
            
        elif line.startswith('Analysts:'):
            in_metadata = True
            # Get the analysts line without the header
            analysts_line = line.replace('Analysts:', '').strip()
            
            # Parse analysts using improved logic
            analyst_entries = parse_name_company_pairs(analysts_line)
            
            for name, company in analyst_entries:
                metadata_sections["analysts"].append({
                    "name": name.strip(),
                    "company": company.strip()
                })
            continue
            
        elif re.match(speaker_pattern, line):  # ANY speaker pattern starts content
            in_metadata = False
            content_lines.append(line)
            continue
        
        # If we're not in metadata and not a speaker line, add to content
        if not in_metadata:
            if line and not (line.startswith('Executives:') or line.startswith('Analysts:')):
                content_lines.append(line)
        # If we're in metadata but it's additional lines, add to additional_metadata
        elif in_metadata and line:
            metadata_sections["additional_metadata"].append(line)
    
    return metadata_sections, '\n'.join(content_lines)

def parse_name_role_pairs(text):
    pairs = []
    
    # Handle different separators: - (hyphen), – (en-dash), — (em-dash)
    # Split by common patterns and clean up
    
    # First, try to split by multiple spaces (3 or more) which often separate entries
    potential_entries = re.split(r'\s{3,}', text)
    
    # If that doesn't work well, try other approaches
    if len(potential_entries) <= 1:
        # Try splitting by patterns like "Name - Role Name - Role"
        # Look for capital letter followed by lowercase, then dash
        potential_entries = re.split(r'(?=[A-Z][a-z].*?[–\-])', text)
        potential_entries = [entry.strip() for entry in potential_entries if entry.strip()]
    
    for entry in potential_entries:
        entry = entry.strip()
        if not entry:
            continue
            
        # Look for name-role pattern with various dash types
        match = re.search(r'^([A-Z][a-zA-ZÀ-ÿ\'\.\s]+?)\s*[–\-—]\s*(.+)$', entry)
        
        if match:
            name = match.group(1).strip()
            role = match.group(2).strip()
            
            # Clean up name - remove common role words that might have leaked in
            name = re.sub(r'\b(Chief|Executive|Vice|President|Officer|Director|Senior|EVP|CFO|CEO|IR)\b.*$', '', name, flags=re.IGNORECASE).strip()
            
            # Clean up role - remove trailing incomplete words
            role = re.sub(r'\s+[A-Z][a-z]*$', '', role).strip()
            
            if name and len(name.split()) >= 2:  # Ensure we have at least first and last name
                pairs.append((name, role))
    
    return pairs

def parse_name_company_pairs(text):
    pairs = []
    
    # Handle different separators and formats
    # Split by multiple spaces (3 or more) which often separate entries
    potential_entries = re.split(r'\s{3,}', text)
    
    # If that doesn't work well, try other approaches
    if len(potential_entries) <= 1:
        # Try splitting by patterns that indicate new entries
        potential_entries = re.split(r'(?=[A-Z][a-z].*?[–\-])', text)
        potential_entries = [entry.strip() for entry in potential_entries if entry.strip()]
    
    for entry in potential_entries:
        entry = entry.strip()
        if not entry:
            continue
            
        # Look for name-company pattern with various dash types
        match = re.search(r'^([A-Z][a-zA-ZÀ-ÿ\'\.\s]+?)\s*[–\-—]\s*(.+)$', entry)
        
        if match:
            name = match.group(1).strip()
            company = match.group(2).strip()
            
            # Clean up name - remove company names that might have leaked in
            name = re.sub(r'\b(Bank|Capital|Securities|Research|LLC|Inc|Corp|Company|Group|Markets|Chase|Morgan|Deutsche|Goldman|Credit|UBS|RBC)\b.*$', '', name, flags=re.IGNORECASE).strip()
            
            # Clean up company - remove trailing incomplete words or names
            company = re.sub(r'\s+[A-Z][a-z]*\s*$', '', company).strip()
            
            if name and len(name.split()) >= 2:  # Ensure we have at least first and last name
                pairs.append((name, company))
    
    return pairs

def enhance_metadata(metadata_sections, management_team, external_members, qna_section):
    # 1. First verify our preliminary assignments against metadata sections
    if metadata_sections["executives"]:
        # Add any missing executives to management team
        for exec_entry in metadata_sections["executives"]:
            exec_name = exec_entry["name"]
            # Check if this name (or a similar one) is already in management team
            if not any(names_are_similar(exec_name, existing) for existing in management_team):
                management_team.add(exec_name)
    
    if metadata_sections["analysts"]:
        # Add any missing analysts to external members
        for analyst_entry in metadata_sections["analysts"]:
            analyst_name = analyst_entry["name"]
            # Check if this name (or a similar one) is already in external members
            if not any(names_are_similar(analyst_name, existing) for existing in external_members):
                external_members.add(analyst_name)
    
    # 2. Create enhanced metadata with roles and companies
    enhanced_metadata = {
        "management_team": [],
        "external_members": []
    }
    
    # 3. Add roles from metadata sections - avoid duplicates
    processed_names = set()
    for name in management_team:
        # Skip if we've already processed a similar name
        if any(names_are_similar(name, processed) for processed in processed_names):
            continue
            
        role = ""
        for exec_entry in metadata_sections["executives"]:
            if names_are_similar(name, exec_entry["name"]):
                role = exec_entry["role"]
                break
        
        enhanced_metadata["management_team"].append({
            "name": name,
            "role": role
        })
        processed_names.add(name)
    
    # 4. Add companies from metadata sections - avoid duplicates
    processed_names = set()
    for name in external_members:
        # Skip if we've already processed a similar name
        if any(names_are_similar(name, processed) for processed in processed_names):
            continue
            
        company = ""
        for analyst_entry in metadata_sections["analysts"]:
            if names_are_similar(name, analyst_entry["name"]):
                company = analyst_entry["company"]
                break
        
        enhanced_metadata["external_members"].append({
            "name": name,
            "company": company
        })
        processed_names.add(name)
    
    return enhanced_metadata

def names_are_similar(name1, name2, threshold=0.8):
    if not name1 or not name2:
        return False
    
    # Normalize names for comparison
    name1_clean = normalize_name(name1)
    name2_clean = normalize_name(name2)
    
    # Exact match after normalization
    if name1_clean == name2_clean:
        return True
    
    # Check if one name is contained in the other (for cases like "John Smith" vs "John R. Smith")
    name1_parts = set(name1_clean.split())
    name2_parts = set(name2_clean.split())
    
    # If one set of name parts is a subset of the other, consider them similar
    if name1_parts.issubset(name2_parts) or name2_parts.issubset(name1_parts):
        return True
    
    # Check for significant overlap in name parts
    common_parts = name1_parts.intersection(name2_parts)
    if len(common_parts) >= 2:  # At least first and last name match
        return True
    
    return False

def normalize_name(name):
    if not name:
        return ""
    
    # Remove common title prefixes and role suffixes
    name = re.sub(r'^(Mr\.?|Ms\.?|Mrs\.?|Dr\.?)\s+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(Jr\.?|Sr\.?|III|II|IV)$', '', name, flags=re.IGNORECASE)
    
    # Remove middle initials for comparison
    name = re.sub(r'\s+[A-Z]\.?\s+', ' ', name)
    
    # Normalize whitespace and case
    name = ' '.join(name.split()).strip().lower()
    
    return name

def parse_transcript(content):
    parsed_transcript = []
    current_speaker = None
    dialogue = []
    order = 1  # For ordering the dialogues

    # Split content by lines and normalize newlines
    lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Check if the line matches a speaker
        match = re.match(speaker_pattern, line)
        if match:
            # If there is a previous speaker, append the dialogue to the transcript
            if current_speaker and dialogue:
                parsed_transcript.append({
                    "speaker": current_speaker,
                    "dialogue": ' '.join(dialogue),
                    "order": order,
                    "classification": None  # Initialize classification
                })
                order += 1
            
            # Update the current speaker and reset the dialogue
            current_speaker = match.group(1).strip()
            # Get the dialogue part after the speaker name and any separator
            dialogue_text = line[match.end():].strip()
            dialogue = [dialogue_text] if dialogue_text else []
        else:
            # Append to the current speaker's dialogue
            if line:  # Only append non-empty lines
                dialogue.append(line)
    
    # Append the last speaker's dialogue
    if current_speaker and dialogue:
        parsed_transcript.append({
            "speaker": current_speaker,
            "dialogue": ' '.join(dialogue),
            "order": order,
            "classification": None  # Initialize classification
        })

    if not parsed_transcript:
        print("Warning: No dialogue found. Debug information:")
        print("First few lines of content:")
        for i, line in enumerate(content.split('\n')[:5]):
            print(f"Line {i+1}: {line}")
        print("\nSpeaker pattern matches:")
        for line in content.split('\n'):
            match = re.match(speaker_pattern, line)
            if match:
                print(f"Matched: {line}")

    return parsed_transcript

def detect_qna_start(parsed_transcript, min_prepared_remarks=2):
    if len(parsed_transcript) < min_prepared_remarks + 2:
        return None  # Not enough content for Q&A
    
    # Q&A transition phrases to look for (after first few speakers)
    qna_phrases = [
        "open up the call for questions",
        "open the call for questions", 
        "first question comes from",
        "next question comes from",
        "begin the question",
        "start the question",
        "take questions",
        "q&a session",
        "question and answer"
    ]
    
    # Start checking after minimum prepared remarks
    for i in range(min_prepared_remarks, len(parsed_transcript)):
        entry = parsed_transcript[i]
        dialogue = entry['dialogue'].lower()
        
        # Check for transition phrases
        for phrase in qna_phrases:
            if phrase in dialogue:
                print(f"Q&A detected at index {i} due to phrase: '{phrase}'")
                return i
        
        # Check for back-and-forth pattern (starting from this point)
        if i >= min_prepared_remarks + 2:  # Need some entries to analyze pattern
            back_forth_score = calculate_speaker_alternation(parsed_transcript, i-3, i+3)
            if back_forth_score > 0.7:  # High alternation threshold
                print(f"Q&A detected at index {i} due to high speaker alternation")
                return i
        
        # Check for question density
        if i >= min_prepared_remarks + 1:
            question_density = calculate_question_density(parsed_transcript, i-2, i+2)
            if question_density > 0.3:  # High question mark density
                print(f"Q&A detected at index {i} due to high question density")
                return i
        
        # Check for short exchanges pattern
        if i >= min_prepared_remarks + 1:
            if detect_short_exchanges(parsed_transcript, i-1, i+2):
                print(f"Q&A detected at index {i} due to short exchange pattern")
                return i
    
    return None  # No Q&A section found

def calculate_speaker_alternation(transcript, start_idx, end_idx):
    if start_idx < 0 or end_idx >= len(transcript) or end_idx - start_idx < 2:
        return 0
    
    alternations = 0
    total_transitions = 0
    
    for i in range(start_idx, end_idx - 1):
        current_speaker = transcript[i]['speaker'].split(' - ')[0].strip()
        next_speaker = transcript[i + 1]['speaker'].split(' - ')[0].strip()
        
        if current_speaker.lower() != 'operator' and next_speaker.lower() != 'operator':
            total_transitions += 1
            if current_speaker != next_speaker:
                alternations += 1
    
    return alternations / total_transitions if total_transitions > 0 else 0

def calculate_question_density(transcript, start_idx, end_idx):
    if start_idx < 0 or end_idx >= len(transcript):
        return 0
    
    total_chars = 0
    question_marks = 0
    
    for i in range(start_idx, end_idx + 1):
        dialogue = transcript[i]['dialogue']
        total_chars += len(dialogue)
        question_marks += dialogue.count('?')
    
    return question_marks / total_chars if total_chars > 0 else 0

def detect_short_exchanges(transcript, start_idx, end_idx):
    if start_idx < 0 or end_idx >= len(transcript) or end_idx - start_idx < 2:
        return False
    
    short_count = 0
    total_count = 0
    
    for i in range(start_idx, end_idx + 1):
        dialogue = transcript[i]['dialogue']
        word_count = len(dialogue.split())
        total_count += 1
        
        if word_count < 20:  # Short dialogue threshold
            short_count += 1
    
    return (short_count / total_count) > 0.6 if total_count > 0 else False

def classify_sections(parsed_transcript):
    management_team = set()
    external_members = set()
    prepared_remarks = []
    qna_section = []
    
    # Detect Q&A start using enhanced logic
    qna_start_idx = detect_qna_start(parsed_transcript)

    for i, entry in enumerate(parsed_transcript):
        speaker = entry['speaker']
        
        if qna_start_idx is None or i < qna_start_idx:
            # Add to prepared remarks
            prepared_remarks.append(entry)
            # If speaker is not operator, add to management team
            if speaker.lower() != 'operator':
                base_name = speaker.split(' - ')[0].strip()
                management_team.add(base_name)
        else:
            # Add to Q&A section
            qna_section.append(entry)
            # If speaker is not in management team, add to external members
            if speaker.lower() != 'operator':
                base_name = speaker.split(' - ')[0].strip()
                if base_name not in management_team:
                    external_members.add(base_name)
    
    return prepared_remarks, qna_section, management_team, external_members

def classify_qna_dialogues(qna_section, management_team, external_members, confidence_threshold):
    classified_qna = []
    current_qa_pair = 0
    last_questioner = None
    
    for entry in qna_section:
        speaker = entry['speaker']
        dialogue = entry['dialogue']
        
        if speaker.lower() == 'operator':
            classification = "operator"  # Explicitly classify operators
        else:
            # First pass assumption based on team membership
            classification = "answer" if speaker in management_team else "question"
            
            # Start a new Q&A pair when we encounter a new questioner
            if classification == "question":
                if speaker != last_questioner:
                    current_qa_pair += 1
                    last_questioner = speaker
            
        # Add classification result and Q&A pair number
        entry['classification'] = classification
        if classification in ["question", "answer"]:
            entry['qa_pair'] = current_qa_pair
        
        classified_qna.append(entry)
    
    return classified_qna

def clean_unicode_for_csv(text):
    if not isinstance(text, str):
        return text
    
    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        '\u2019': "'",  # Right single quotation mark
        '\u2018': "'",  # Left single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...', # Horizontal ellipsis
        '\u00a0': ' ',  # Non-breaking space
        '\u2022': '*',  # Bullet point
        '\u00e9': 'e',  # é
        '\u00e8': 'e',  # è
        '\u00ea': 'e',  # ê
        '\u00e1': 'a',  # á
        '\u00e0': 'a',  # à
        '\u00f1': 'n',  # ñ
        '\u00fc': 'u',  # ü
        '\u00f6': 'o',  # ö
        '\u00e4': 'a',  # ä
    }
    
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    
    return text

def process_transcript_file(raw_file_path, output_dir):
    try:
        # Read and parse JSON
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Extract metadata
        symbol = transcript_data[0]['symbol']
        quarter = transcript_data[0]['quarter']
        year = transcript_data[0]['year']
        date = transcript_data[0]['date']
        
        # Get content with proper newlines
        content = transcript_data[0]['content']
        
        # Split content into lines using \n
        lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        
        # First, extract metadata sections and get cleaned content
        metadata_sections, cleaned_content = extract_metadata_sections(lines)
        
        # Now parse the cleaned content (without metadata sections)
        parsed_transcript = parse_transcript(cleaned_content)
        
        if not parsed_transcript:
            print(f"Warning: No dialogue found in transcript {raw_file_path}")
            return False
        
        # Classify sections to get management team and external members
        prepared_remarks, qna_section, management_team, external_members = classify_sections(parsed_transcript)
        
        # Enhance metadata with roles and companies
        enhanced_metadata = enhance_metadata(metadata_sections, management_team, external_members, qna_section)
        
        # Add transcript metadata
        enhanced_metadata.update({
            "symbol": symbol,
            "quarter": quarter,
            "year": year,
            "date": date
        })
        
        # Classify Q&A dialogues if Q&A section exists
        if qna_section:
            classified_qna = classify_qna_dialogues(qna_section, management_team, external_members, 0.9)
            # Combine prepared remarks and classified Q&A sections
            full_transcript = prepared_remarks + classified_qna
        else:
            # If no Q&A section, just use prepared remarks
            full_transcript = prepared_remarks
        
        # Ensure proper order numbering
        for i, entry in enumerate(full_transcript, start=1):
            entry['order'] = i
        
        # Create DataFrame
        df = pd.DataFrame(full_transcript)
        
        # Add word count column
        df['word_count'] = df['dialogue'].apply(lambda x: len(str(x).split()))
        
        # Reorder columns
        columns = ['order', 'speaker', 'dialogue', 'classification', 'word_count']
        if 'qa_pair' in df.columns:
            columns.append('qa_pair')
        df = df[columns]
        
        # Prepare structured data
        structured_data = {
            "symbol": symbol,
            "quarter": quarter,
            "year": year,
            "date": date,
            "content": content,  # Add original content here
            "metadata": {
                "management_team": [member["name"] for member in enhanced_metadata["management_team"]],
                "external_members": [member["name"] for member in enhanced_metadata["external_members"]]
            },
            "parsed_content": {
                "prepared_remarks": prepared_remarks,
                "qna_section": classified_qna if qna_section else []
            }
        }
        
        # Save structured data to JSON
        json_file_path = os.path.join(output_dir, f"{os.path.basename(raw_file_path).replace('_raw_api_response.json', '_transcript_with_speakers.json')}")
        with open(json_file_path, 'w') as f:
            json.dump(structured_data, f, indent=4)
        
        # Clean Unicode characters in dialogue for CSV compatibility
        df['dialogue'] = df['dialogue'].apply(clean_unicode_for_csv)
        df['speaker'] = df['speaker'].apply(clean_unicode_for_csv)
        
        # Save DataFrame to CSV with proper encoding
        csv_file_path = os.path.join(output_dir, f"{os.path.basename(raw_file_path).replace('_raw_api_response.json', '_transcript_structured.csv')}")
        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        
        print(f"Successfully processed {raw_file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {raw_file_path}: {str(e)}")
        return False

def process_all_transcripts(base_dir, output_dir=None):
    success_count = 0
    failed_count = 0
    total_files_found = 0
    
    print("\nSearching for transcript files...")
    
    # Walk through all subdirectories
    for root, dirs, files in tqdm(os.walk(base_dir), desc="Processing transcripts"):
        # Find raw transcript files
        raw_files = [f for f in files if f.endswith('_raw_api_response.json')]
        total_files_found += len(raw_files)
        
        if raw_files:
            print(f"\nFound {len(raw_files)} transcript files in {root}:")
            for f in raw_files:
                print(f"  {f}")
        
        for raw_file in raw_files:
            raw_file_path = os.path.join(root, raw_file)
            
            # Determine output directory
            if output_dir:
                # In test mode, use the specified output directory
                current_output_dir = output_dir
            else:
                # In normal mode, use "02" directory for sequential pipeline
                current_output_dir = "02"
            
            try:
                if process_transcript_file(raw_file_path, current_output_dir):
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Error processing {raw_file}: {str(e)}")
                failed_count += 1
    
    print(f"\nProcessing complete:")
    print(f"Total files found: {total_files_found}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {failed_count}")
    
    if total_files_found == 0:
        print("\nNo transcript files found. Please check:")
        print("1. The input directory path is correct")
        print("2. The files end with '_raw_transcript.txt'")
        print("3. You have read permissions for the directory")

def run_test_mode():
    #input_dir = r"C:\Users\pbcoy\OneDrive\Desktop\Jan 2025\Dissertation Ideas\code\dissertation\cleaned_up_classification\src\training_batch_run\training_batch\NWT.DE - Wells Fargo"
    #input_dir = r"C:\Users\pbcoy\OneDrive\Desktop\Jan 2025\Dissertation Ideas\code\dissertation\cleaned_up_classification\src\training_batch_run\training_batch\SNAP"
    #output_dir = r"C:\Users\pbcoy\OneDrive\Desktop\Jan 2025\Dissertation Ideas\code\dissertation\cleaned_up_classification\src\training_batch_run\test_output\test_cleanup_output"
    # Updated to use relative paths within parallel_cpu/ folder structure
    input_dir = "raw_files/SNAP"  # Points to parallel_cpu/raw_files/SNAP/
    output_dir = "02"  # Points to parallel_cpu/02/ for step 2 output
    

    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # List contents of input directory
    print("\nContents of input directory:")
    try:
        files = os.listdir(input_dir)
        if not files:
            print("  Directory is empty")
        else:
            for file in files:
                print(f"  {file}")
    except Exception as e:
        print(f"Error listing directory contents: {str(e)}")
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory created/verified: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        return
    
    print(f"\nRunning in test mode:")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    process_all_transcripts(input_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process transcript files.')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    
    if args.test:
        run_test_mode()
    else:
        # Set the base directory containing the transcript folders (updated for parallel_cpu structure)
        base_dir = "raw_files"
        # Create output directory for normal mode
        os.makedirs("02", exist_ok=True)
        # Process all transcripts
        process_all_transcripts(base_dir)


# python transcript_cleanup.py --test   ... to run in test mode
# python transcript_cleanup.py   ... to run in normal mode


# This code will still keep the following formatting which we can change later before giving it to chatgpt
#  it\u2019s tough to say. Now we\u2019re a, bout \u2013, bout \u2013 , "I\u2019d say very little, "Any sizing or \u2013",