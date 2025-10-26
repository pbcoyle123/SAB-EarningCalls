# -*- coding: utf-8 -*-
"""
Main Processing Pipeline for Earnings Call Transcript Analysis
============================================================

This script orchestrates the complete processing pipeline for earnings call transcripts:

1. Transcript Cleanup (02_transcript_cleanup.py)
   - Processes raw transcript files
   - Identifies speakers and sections
   - Classifies prepared remarks vs Q&A
   - Output: parallel_cpu/02/

2. Text Segmentation (03_paragraph_breaks_pipeline_03_v2trainingBatch.py)
   - Segments dialogues into semantic paragraphs
   - Uses sentence embeddings for coherent breaks
   - Output: parallel_cpu/03/

3. Topic Classification (04_topic_classification.py)
   - Classifies paragraphs into financial topics
   - Performs sentence-level attribution for multi-topic paragraphs
   - Output: parallel_cpu/04/

4. Attribution Classification (05_attribution_classification.py)
   - Analyzes causal attributions in the text
   - Identifies attribution presence, outcome, and locus of control
   - Output: parallel_cpu/05/

"""

import os
import sys
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path
import json
import glob
import argparse
from typing import Dict, List, Tuple, Optional
import platform
import importlib.util

# Environment Setup Functions
def check_and_install_dependencies() -> bool:
    setup_logger = logging.getLogger("EnvironmentSetup")
    setup_logger.setLevel(logging.INFO)
    
    # Console handler for environment setup (temporary)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        ' %(asctime)s - SETUP - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    setup_logger.addHandler(console_handler)
    
    setup_logger.info("="*80)
    setup_logger.info(" AWS EC2 ENVIRONMENT SETUP")
    setup_logger.info("="*80)
    
    # Detect system architecture
    arch = platform.machine().lower()
    system = platform.system()
    setup_logger.info(f"System: {system} {arch}")
    
    if 'x86_64' in arch or 'amd64' in arch:
        setup_logger.info(" x86_64 architecture detected - compatible with AWS EC2")
    else:
        setup_logger.warning(f"  Non-x86_64 architecture detected: {arch}")
        setup_logger.info("   Pipeline should still work but may not be optimized")
    
    # Required packages with installation commands
    # CRITICAL: openai==0.28.0 for legacy API compatibility
    required_packages = {
        'openai': 'openai==0.28.0',  # MUST be v0.28.x for legacy API
        'pandas': 'pandas>=2.0.0,<2.1.0',
        'tqdm': 'tqdm>=4.65.0,<4.66.0',
        'spacy': 'spacy>=3.6.0,<3.7.0',
        'transformers': 'transformers>=4.30.0,<4.40.0',
        'sentence-transformers': 'sentence-transformers>=2.2.0,<2.3.0',
        'sentencepiece': 'sentencepiece>=0.1.99,<0.2.0',  # Pre-built wheel
        'anthropic': 'anthropic>=0.3.0,<0.4.0',
        'pydantic': 'pydantic==2.11.7',
        'numpy': 'numpy>=1.24.0,<2.1.0',
        'matplotlib': 'matplotlib>=3.7.0,<3.8.0',
        'torch': 'torch>=2.0.0,<2.1.0',
        'nltk': 'nltk>=3.8.0,<3.9.0',
        'sklearn': 'scikit-learn>=1.3.0,<1.4.0',  # Often required by sentence-transformers
        'huggingface-hub': 'huggingface-hub>=0.15.0,<0.20.0'  # Compatible with sentence-transformers 2.2.2
    }
    
    setup_logger.info(f"\n Force reinstalling {len(required_packages)} required packages...")
    setup_logger.info("    This will overwrite any existing versions to ensure compatibility")
    
    # Force reinstall all packages (no checking for existing installations)
    all_packages = list(required_packages.values())
    setup_logger.info(f"   Packages: {', '.join(all_packages)}")
    
    try:
        # Use pip with --force-reinstall to overwrite existing versions
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall"] + all_packages
        setup_logger.info(f"   Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for full reinstall
        )
        
        if result.returncode == 0:
            setup_logger.info("    Package force reinstallation successful")
        else:
            setup_logger.error(f"    Package force reinstallation failed:")
            setup_logger.error(f"     stdout: {result.stdout}")
            setup_logger.error(f"     stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        setup_logger.error("    Package installation timed out")
        return False
    except Exception as e:
        setup_logger.error(f"    Package installation error: {e}")
        return False
    
    # Download spaCy language model
    setup_logger.info(f"\n Setting up spaCy language model...")
    try:
        import spacy
        try:
            # Try to load the model first
            nlp = spacy.load("en_core_web_sm")
            setup_logger.info("    en_core_web_sm model already available")
        except OSError:
            # Model not found, download it
            setup_logger.info("    Downloading en_core_web_sm model...")
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                setup_logger.info("    spaCy model download successful")
                # Verify model can be loaded
                nlp = spacy.load("en_core_web_sm")
                setup_logger.info("    spaCy model verified")
            else:
                setup_logger.error(f"    spaCy model download failed:")
                setup_logger.error(f"     stderr: {result.stderr}")
                return False
    except Exception as e:
        setup_logger.error(f"    spaCy setup error: {e}")
        return False
    
    # Download NLTK data
    setup_logger.info(f"\n Setting up NLTK data...")
    try:
        import nltk
        # Download required NLTK data silently
        try:
            nltk.download('punkt_tab', quiet=True)
            setup_logger.info("    punkt_tab data downloaded")
        except:
            # Fallback for older NLTK versions
            try:
                nltk.download('punkt', quiet=True)
                setup_logger.info("    punkt data downloaded (fallback)")
            except Exception as e:
                setup_logger.warning(f"     NLTK punkt download issue: {e}")
    except Exception as e:
        setup_logger.error(f"    NLTK setup error: {e}")
        return False
    
    # Verify sentence transformers model will download automatically
    setup_logger.info(f"\n Verifying sentence transformer model...")
    try:
        from sentence_transformers import SentenceTransformer
        # This will download the model if not already present
        setup_logger.info("    Downloading all-MiniLM-L6-v2 model (if needed)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        setup_logger.info("    Sentence transformer model ready")
    except Exception as e:
        setup_logger.error(f"    Sentence transformer setup error: {e}")
        return False
    
    # Test core functionality
    setup_logger.info(f"\n Running dependency verification tests...")
    
    try:
        # Test imports
        import openai
        import pandas as pd
        import spacy
        import transformers
        from sentence_transformers import SentenceTransformer, util
        import torch
        import nltk
        from pydantic import BaseModel
        
        # Test spaCy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Test sentence.")
        assert len(list(doc.sents)) > 0
        
        # Test sentence transformers
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(["Test sentence"])
        assert embeddings.shape[0] == 1
        
        setup_logger.info("    All dependency tests passed")
        
    except Exception as e:
        setup_logger.error(f"    Dependency verification failed: {e}")
        return False
    
    setup_logger.info(f"\n Environment setup completed successfully!")
    setup_logger.info(f" Ready to run earnings call transcript processing pipeline")
    setup_logger.info("="*80)
    
    # Remove the temporary handler
    setup_logger.removeHandler(console_handler)
    
    return True

# Configure logging
def setup_logging() -> Tuple[logging.Logger, str]:
    # create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pipeline_run_{timestamp}.log"
    log_path = logs_dir / log_filename
    
    # Set up logger
    logger = logging.getLogger("MainPipeline")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler for detailed logging
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler for real-time updates
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, log_filename

class PipelineStep:
    
    def __init__(self, name: str, script_name: str, description: str):
        self.name = name
        self.script_name = script_name
        self.description = description
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.success = None
        self.error_message = None
        self.input_files_count = 0
        self.output_files_count = 0
        
    def start(self):
        self.start_time = time.time()
        self.success = None
        self.error_message = None
        
    def finish(self, success: bool, error_message: str = None):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message
        
    def get_duration_str(self) -> str:
        if self.duration is None:
            return "N/A"
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
            
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "script_name": self.script_name,
            "description": self.description,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration, 2) if self.duration else None,
            "duration_formatted": self.get_duration_str(),
            "success": self.success,
            "error_message": self.error_message,
            "input_files_count": self.input_files_count,
            "output_files_count": self.output_files_count
        }

class EarningsCallPipeline:
    
    def __init__(self, test_mode: bool = False, target_companies: List[str] = None, vm_id: str = None):
        self.test_mode = test_mode
        self.target_companies = target_companies
        self.vm_id = vm_id
        self.logger, self.log_filename = setup_logging()
        self.start_time = time.time()
        self.steps = []
        self.overall_success = True
        
        # Define pipeline steps
        self.pipeline_steps = [
            PipelineStep(
                "Transcript Cleanup",
                "02_transcript_cleanup.py",
                f"Process raw transcript files, identify speakers and sections ({'test mode' if test_mode else 'full mode'})"
            ),
            PipelineStep(
                "Text Segmentation", 
                "03_paragraph_breaks_pipeline_03_v2trainingBatch.py",
                "Segment dialogues into semantic paragraphs using embeddings"
            ),
            PipelineStep(
                "Topic Classification",
                "04_topic_classification.py", 
                "Classify paragraphs into financial topics with structured outputs"
            ),
            PipelineStep(
                "Attribution Classification",
                "05_attribution_classification.py",
                "Analyze causal attributions for outcome polarity and locus of control"
            )
        ]
        
    def validate_prerequisites(self) -> bool:
        mode_text = "test mode" if self.test_mode else "normal mode"
        self.logger.info(f"Validating pipeline prerequisites ({mode_text})...")
        
        # Check if required scripts exist
        required_scripts = [step.script_name for step in self.pipeline_steps]
        missing_scripts = []
        
        for script in required_scripts:
            if not os.path.exists(script):
                missing_scripts.append(script)
                
        if missing_scripts:
            self.logger.error(f"Missing required scripts: {missing_scripts}")
            return False
            
        # Check if input data exists
        if self.test_mode:
            # In test mode, check for SNAP company specific directory
            input_dirs = [
                "raw_files/SNAP",  # For 02_transcript_cleanup.py test mode
            ]
            self.logger.info("Test mode: validating SNAP company data directory")
        else:
            # In normal mode, check for general raw_files directory
            input_dirs = [
                "raw_files",  # For 02_transcript_cleanup.py normal mode
            ]
            self.logger.info("Normal mode: validating full raw_files directory")
        
        for input_dir in input_dirs:
            if not os.path.exists(input_dir):
                self.logger.error(f"Missing input directory: {input_dir}")
                if self.test_mode:
                    self.logger.error("Test mode requires raw_files/SNAP/ directory with transcript files")
                else:
                    self.logger.error("Normal mode requires raw_files/ directory with company subdirectories")
                return False
                
        # Create output directories
        if self.test_mode:
            # Test mode uses centralized output directories with new structure
            output_dirs = [
                "02",  # Step 2: Transcript cleanup
                "03",  # Step 3: Text segmentation
                "04",  # Step 4: Topic classification
                "05",  # Step 5: Attribution classification
                "logs"  # Logging
            ]
            self.logger.info("Test mode: creating centralized output directories (02/03/04/05)")
        else:
            # Normal mode creates minimal output directories (files saved alongside inputs)
            output_dirs = [
                "02",  # Step 2: Transcript cleanup
                "03",  # Step 3: Text segmentation
                "04",  # Step 4: Topic classification
                "05",  # Step 5: Attribution classification
                "logs"  # Logging
            ]
            self.logger.info("Normal mode: creating sequential output directories (02/03/04/05)")
        
        for output_dir in output_dirs:
            os.makedirs(output_dir, exist_ok=True)
            
        self.logger.info(" All prerequisites validated successfully")
        return True
        
    def count_files_in_directory(self, directory: str, pattern: str = "*") -> int:
        if not os.path.exists(directory):
            return 0
        try:
            if self.target_companies and "raw_files" in directory:
                # Parallel mode: only count files from target companies
                files = []
                for company in self.target_companies:
                    # Search recursively in company subdirectories
                    search_pattern = os.path.join("raw_files", company, "**", pattern)
                    company_files = glob.glob(search_pattern, recursive=True)
                    files.extend(company_files)
                return len(files)
            else:
                # Search recursively in all subdirectories
                search_pattern = os.path.join(directory, "**", pattern)
                files = glob.glob(search_pattern, recursive=True)
                return len(files)
        except Exception as e:
            self.logger.warning(f"Error counting files in {directory}: {e}")
            return 0
            
    def run_step_1_transcript_cleanup(self) -> bool:
        step = self.pipeline_steps[0]
        step.start()
        
        self.logger.info(f" Starting Step 1: {step.name}")
        self.logger.info(f"   Description: {step.description}")
        
        # Count input files based on mode
        if self.test_mode:
            # Test mode: count files in SNAP directory only
            step.input_files_count = self.count_files_in_directory("raw_files/SNAP", "*_raw_api_response.json")
            self.logger.info(f"   Input files: {step.input_files_count} raw transcript files (SNAP company only)")
        elif self.target_companies:
            # Parallel mode: count files from target companies only
            step.input_files_count = self.count_files_in_directory("raw_files", "*_raw_api_response.json")
            companies_str = ", ".join(self.target_companies)
            self.logger.info(f"   Input files: {step.input_files_count} raw transcript files ({companies_str})")
        else:
            # Normal mode: count files recursively in all raw_files subdirectories
            step.input_files_count = self.count_files_in_directory("raw_files", "*_raw_api_response.json")
            self.logger.info(f"   Input files: {step.input_files_count} raw transcript files (all companies)")
        
        try:
            # Run transcript cleanup with conditional test flag
            if self.test_mode:
                cmd = [sys.executable, "02_transcript_cleanup.py", "--test"]
                self.logger.info("   Running 02_transcript_cleanup.py in TEST MODE")
            elif self.target_companies:
                # For parallel mode, we'll use normal mode but process only target companies
                # The files have already been distributed to this VM by the oracle setup script
                cmd = [sys.executable, "02_transcript_cleanup.py"]
                self.logger.info(f"   Running 02_transcript_cleanup.py in PARALLEL MODE for {self.vm_id}")
            else:
                cmd = [sys.executable, "02_transcript_cleanup.py"]
                self.logger.info("   Running 02_transcript_cleanup.py in NORMAL MODE")
            
            self.logger.info(f"   Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
                # No timeout - allow unlimited processing time
            )
            
            # Log the output
            if result.stdout:
                self.logger.info("   Script output:")
                for line in result.stdout.strip().split('\n'):
                    self.logger.info(f"     {line}")
                    
            if result.stderr:
                self.logger.warning("   Script warnings/errors:")
                for line in result.stderr.strip().split('\n'):
                    self.logger.warning(f"     {line}")
            
            success = result.returncode == 0
            
            if success:
                # Count output files in the new structure
                step.output_files_count = self.count_files_in_directory(
                    "02", "*transcript_with_speakers*.json"
                )
                self.logger.info(f"    Completed successfully")
                self.logger.info(f"   Output files: {step.output_files_count} processed transcripts in 02/")
            else:
                self.logger.error(f"    Failed with return code: {result.returncode}")
                
            step.finish(success, result.stderr if not success else None)
            return success
            
        except subprocess.TimeoutExpired:
            error_msg = "Script execution timed out (no timeout limit set)"
            self.logger.error(f"    {error_msg}")
            step.finish(False, error_msg)
            return False
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"    {error_msg}")
            step.finish(False, error_msg)
            return False
            
    def run_step_2_text_segmentation(self) -> bool:
        step = self.pipeline_steps[1]
        step.start()
        
        self.logger.info(f" Starting Step 2: {step.name}")
        self.logger.info(f"   Description: {step.description}")
        
        # Count input files from step 2 output directory
        step.input_files_count = self.count_files_in_directory(
            "02", "*transcript_with_speakers*.json"
        )
        self.logger.info(f"   Input files: {step.input_files_count} cleaned transcripts (from 02/)")
        
        if step.input_files_count == 0:
            error_msg = "No input files found from previous step"
            self.logger.error(f"    {error_msg}")
            step.finish(False, error_msg)
            return False
        
        try:
            # Run text segmentation with process_all flag and no visualization
            cmd = [sys.executable, "03_paragraph_breaks_pipeline_03_v2trainingBatch.py", "--process_all", "--no_viz"]
            self.logger.info(f"   Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
                # No timeout - allow unlimited processing time
            )
            
            # Log the output
            if result.stdout:
                self.logger.info("   Script output:")
                for line in result.stdout.strip().split('\n'):
                    self.logger.info(f"     {line}")
                    
            if result.stderr:
                self.logger.warning("   Script warnings/errors:")
                for line in result.stderr.strip().split('\n'):
                    self.logger.warning(f"     {line}")
            
            success = result.returncode == 0
            
            if success:
                # Count output files
                step.output_files_count = self.count_files_in_directory(
                    "03", "*_paragraphs.json"
                )
                self.logger.info(f"    Completed successfully")
                self.logger.info(f"   Output files: {step.output_files_count} segmented transcripts")
            else:
                self.logger.error(f"    Failed with return code: {result.returncode}")
                
            step.finish(success, result.stderr if not success else None)
            return success
            
        except subprocess.TimeoutExpired:
            error_msg = "Script execution timed out (no timeout limit set)"
            self.logger.error(f"    {error_msg}")
            step.finish(False, error_msg)
            return False
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"    {error_msg}")
            step.finish(False, error_msg)
            return False
            
    def run_step_3_topic_classification(self) -> bool:
        step = self.pipeline_steps[2]
        step.start()
        
        self.logger.info(f" Starting Step 3: {step.name}")
        self.logger.info(f"   Description: {step.description}")
        
        # Count input files
        step.input_files_count = self.count_files_in_directory(
            "03", "*_paragraphs.json"
        )
        self.logger.info(f"   Input files: {step.input_files_count} segmented transcripts")
        
        if step.input_files_count == 0:
            error_msg = "No input files found from previous step"
            self.logger.error(f"    {error_msg}")
            step.finish(False, error_msg)
            return False
        
        try:
            # Run topic classification
            cmd = [sys.executable, "04_topic_classification.py"]
            self.logger.info(f"   Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
                # No timeout - allow unlimited processing time for LLM processing
            )
            
            # Log the output
            if result.stdout:
                self.logger.info("   Script output:")
                for line in result.stdout.strip().split('\n'):
                    self.logger.info(f"     {line}")
                    
            if result.stderr:
                self.logger.warning("   Script warnings/errors:")
                for line in result.stderr.strip().split('\n'):
                    self.logger.warning(f"     {line}")
            
            success = result.returncode == 0
            
            if success:
                # Count output files
                step.output_files_count = self.count_files_in_directory(
                    "04", "*_topic_class.csv"
                )
                self.logger.info(f"    Completed successfully")
                self.logger.info(f"   Output files: {step.output_files_count} topic-classified files")
            else:
                self.logger.error(f"    Failed with return code: {result.returncode}")
                
            step.finish(success, result.stderr if not success else None)
            return success
            
        except subprocess.TimeoutExpired:
            error_msg = "Script execution timed out (no timeout limit set)"
            self.logger.error(f"    {error_msg}")
            step.finish(False, error_msg)
            return False
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"    {error_msg}")
            step.finish(False, error_msg)
            return False
            
    def run_step_4_attribution_classification(self) -> bool:
        step = self.pipeline_steps[3]
        step.start()
        
        self.logger.info(f" Starting Step 4: {step.name}")
        self.logger.info(f"   Description: {step.description}")
        
        # Count input files
        step.input_files_count = self.count_files_in_directory(
            "04", "*_topic_class.csv"
        )
        self.logger.info(f"   Input files: {step.input_files_count} topic-classified files")
        
        if step.input_files_count == 0:
            error_msg = "No input files found from previous step"
            self.logger.error(f"    {error_msg}")
            step.finish(False, error_msg)
            return False
        
        try:
            # Run attribution classification
            cmd = [sys.executable, "05_attribution_classification.py"]
            self.logger.info(f"   Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
                # No timeout - allow unlimited processing time for LLM processing
            )
            
            # Log the output
            if result.stdout:
                self.logger.info("   Script output:")
                for line in result.stdout.strip().split('\n'):
                    self.logger.info(f"     {line}")
                    
            if result.stderr:
                self.logger.warning("   Script warnings/errors:")
                for line in result.stderr.strip().split('\n'):
                    self.logger.warning(f"     {line}")
            
            success = result.returncode == 0
            
            if success:
                # Count output files
                step.output_files_count = self.count_files_in_directory(
                    "05", "*_attribution_class.csv"
                )
                self.logger.info(f"    Completed successfully")
                self.logger.info(f"   Output files: {step.output_files_count} attribution-classified files")
            else:
                self.logger.error(f"    Failed with return code: {result.returncode}")
                
            step.finish(success, result.stderr if not success else None)
            return success
            
        except subprocess.TimeoutExpired:
            error_msg = "Script execution timed out (no timeout limit set)"
            self.logger.error(f"    {error_msg}")
            step.finish(False, error_msg)
            return False
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"    {error_msg}")
            step.finish(False, error_msg)
            return False
            
    def run_pipeline(self) -> bool:
        self.logger.info("="*80)
        self.logger.info(" STARTING EARNINGS CALL TRANSCRIPT PROCESSING PIPELINE")
        self.logger.info("="*80)
        
        # Log execution mode
        mode_text = "TEST MODE" if self.test_mode else "NORMAL MODE"
        self.logger.info(f"Execution Mode: {mode_text}")
        
        if self.test_mode:
            self.logger.info("• Input: raw_files/SNAP/*_raw_api_response.json")
            self.logger.info("• Output: Sequential processing 02/ → 03/ → 04/ → 05/")
            self.logger.info("• Scope: SNAP company only (fast testing)")
        else:
            self.logger.info("• Input: raw_files/**/*_raw_api_response.json")
            self.logger.info("• Output: Sequential processing 02/ → 03/ → 04/ → 05/")
            self.logger.info("• Scope: All companies (full production run)")
        self.logger.info("")
        
        if not self.validate_prerequisites():
            self.overall_success = False
            return False
            
        # Define step execution functions
        step_functions = [
            self.run_step_1_transcript_cleanup,
            self.run_step_2_text_segmentation,
            self.run_step_3_topic_classification,
            self.run_step_4_attribution_classification
        ]
        
        # Execute each step
        for i, step_func in enumerate(step_functions, 1):
            self.logger.info(f"\n{'-'*60}")
            success = step_func()
            
            if not success:
                self.logger.error(f" Pipeline failed at Step {i}")
                self.overall_success = False
                break
            else:
                self.logger.info(f" Step {i} completed successfully")
                
        return self.overall_success
        
    def generate_summary_report(self):
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        self.logger.info("\n" + "="*80)
        self.logger.info(" PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*80)
        
        # Overall status
        status = " SUCCESS" if self.overall_success else " FAILED"
        self.logger.info(f"Overall Status: {status}")
        
        # Total timing
        total_minutes = int(total_duration // 60)
        total_seconds = int(total_duration % 60)
        self.logger.info(f"Total Execution Time: {total_minutes}m {total_seconds}s")
        
        # Step-by-step results
        self.logger.info("\nStep-by-Step Results:")
        for i, step in enumerate(self.pipeline_steps, 1):
            status_icon = "" if step.success else "" if step.success is False else "⏸"
            duration_str = step.get_duration_str()
            self.logger.info(f"  {i}. {step.name}: {status_icon} ({duration_str})")
            
            if step.input_files_count > 0:
                self.logger.info(f"     Input: {step.input_files_count} files")
            if step.output_files_count > 0:
                self.logger.info(f"     Output: {step.output_files_count} files")
            if step.error_message:
                self.logger.error(f"     Error: {step.error_message}")
                
        # Final output summary
        if self.overall_success:
            final_files = self.count_files_in_directory(
                "05", "*_attribution_class.csv"
            )
            self.logger.info(f"\n Pipeline completed successfully!")
            self.logger.info(f" Final output: {final_files} fully processed transcript files")
            self.logger.info(f" Location: parallel_cpu/05/")
        else:
            self.logger.info(f"\n Pipeline failed - check individual step errors above")
            
        self.logger.info(f"\n Detailed log saved to: logs/{self.log_filename}")
        
    def save_run_summary(self):
        summary = {
            "pipeline_run": {
                "timestamp": datetime.now().isoformat(),
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration_seconds": round(time.time() - self.start_time, 2),
                "overall_success": self.overall_success,
                "log_filename": self.log_filename
            },
            "steps": [step.to_dict() for step in self.pipeline_steps]
        }
        
        # Save to logs directory
        summary_filename = f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_path = Path("logs") / summary_filename
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f" Pipeline summary saved to: {summary_path}")
        
        # Also update the latest run summary
        latest_path = Path("logs") / "latest_pipeline_run.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description='Earnings Call Transcript Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXECUTION MODES:

Normal Mode (python main.py):
- Processes ALL transcript files in raw_files/ directory
- Input: parallel_cpu/raw_files/**/*_raw_api_response.json files
- Output: Sequential processing through 02/ → 03/ → 04/ → 05/ directories

Test Mode (python main.py --test):
- Processes ONLY SNAP company transcript files for quick testing
- Input: parallel_cpu/raw_files/SNAP/*_raw_api_response.json files  
- Output: Sequential processing through 02/ → 03/ → 04/ → 05/ directories

Setup Mode (python main.py --setup-only):
- Only runs environment setup without processing any data
- Installs all required dependencies and models for AWS EC2
- Verifies all components are working correctly
        """
    )
    parser.add_argument(
        '--test', 
        action='store_true', 
        help='Run in test mode (processes only SNAP company data)'
    )
    parser.add_argument(
        '--setup-only',
        action='store_true',
        help='Only run environment setup without processing any data'
    )
    parser.add_argument(
        '--skip-setup',
        action='store_true',
        help='Skip automatic environment setup (assumes dependencies already installed)'
    )
    parser.add_argument(
        '--companies',
        type=str,
        help='Comma-separated list of company directories to process (e.g., "AAPL,SNAP,TSLA")'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key for this VM instance'
    )
    parser.add_argument(
        '--parallel-vm',
        type=str,
        help='VM identifier for parallel processing (e.g., vm1, vm2, vm3, vm4)'
    )
    args = parser.parse_args()
    
    # Handle setup-only mode
    if args.setup_only:
        print(" Running environment setup only...")
        success = check_and_install_dependencies()
        if success:
            print(" Environment setup completed successfully!")
            print(" You can now run the pipeline with: python main.py")
            sys.exit(0)
        else:
            print(" Environment setup failed!")
            sys.exit(1)
    
    # Run environment setup unless explicitly skipped
    if not args.skip_setup:
        print(" Checking environment setup...")
        setup_success = check_and_install_dependencies()
        if not setup_success:
            print(" Environment setup failed! Use --skip-setup to bypass.")
            sys.exit(1)
        print(" Environment ready!")
    else:
        print("Skipping environment setup as requested")
    
    # Handle API key setup
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
        print(f" API key set from command line for {args.parallel_vm or 'this instance'}")
    else:
        # Try to set from config file
        try:
            from config import get_openai_api_key
            api_key = get_openai_api_key()
            os.environ['OPENAI_API_KEY'] = api_key
            print(f" API key loaded from config file for {args.parallel_vm or 'this instance'}")
        except (ImportError, ValueError) as e:
            print(f"  API key setup: {e}")
            print("   Please either use --api_key argument or set up config.json")
    
    # Handle company filtering for parallel processing
    target_companies = None
    if args.companies:
        target_companies = [c.strip() for c in args.companies.split(',')]
        print(f" Target companies: {target_companies}")
    
    # Display startup information
    if args.parallel_vm:
        mode_text = f"PARALLEL MODE ({args.parallel_vm})"
    elif args.test:
        mode_text = "TEST MODE"
    else:
        mode_text = "NORMAL MODE"
    print(f" Earnings Call Transcript Processing Pipeline - {mode_text}")
    print("=" * 60)
    
    if args.test:
        print(" Test Mode Configuration:")
        print("   • Input: raw_files/SNAP/*_raw_api_response.json")
        print("   • Output: Sequential processing 02/ → 03/ → 04/ → 05/")
        print("   • Scope: SNAP company only (fast testing)")
    else:
        print(" Normal Mode Configuration:")
        print("   • Input: raw_files/**/*_raw_api_response.json")
        print("   • Output: Sequential processing 02/ → 03/ → 04/ → 05/")
        print("   • Scope: All companies (full production run)")
    print()
    
    # Initialize and run pipeline
    pipeline = EarningsCallPipeline(
        test_mode=args.test, 
        target_companies=target_companies,
        vm_id=args.parallel_vm
    )
    
    try:
        success = pipeline.run_pipeline()
        pipeline.generate_summary_report()
        pipeline.save_run_summary()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        pipeline.logger.error("\n  Pipeline interrupted by user")
        pipeline.overall_success = False
        pipeline.generate_summary_report()
        pipeline.save_run_summary()
        sys.exit(1)
        
    except Exception as e:
        pipeline.logger.error(f"\n Unexpected pipeline error: {str(e)}")
        pipeline.overall_success = False
        pipeline.generate_summary_report()
        pipeline.save_run_summary()
        sys.exit(1)

if __name__ == "__main__":
    main()

# Usage:
# python main.py                    # Normal mode - processes all companies (with auto-setup)
# python main.py --test             # Test mode - processes SNAP company only
# python main.py --setup-only       # Only run environment setup, no data processing
# python main.py --skip-setup       # Skip environment setup (assume already installed)
# python main.py --api-key YOUR_KEY # Provide API key via command line
#
# AWS EC2 Examples:
# python main.py --setup-only                          # Initial VM setup
# python main.py --test --api-key sk-...               # Test run with API key
# python main.py --parallel-vm vm1 --companies "AAPL"  # Parallel processing
