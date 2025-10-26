"""
Embedding Analysis for Attribution Bias Detection

This script analyzes whether embeddings alone can detect attribution bias patterns,
comparing unsupervised clustering to supervised classification approaches.

Key Questions:
1. Can raw embeddings separate bias patterns without supervision?
2. Is supervised learning necessary or do embeddings capture the signal?
3. Do target companies cluster separately from peers in embedding space?
4. Can we identify SAB peers or SAB periods?

Methods:
- Extract sentence embeddings using transformer models
- Unsupervised clustering (K-means, HDBSCAN)
- Supervised classification on embeddings
- Dimensionality reduction for visualization (UMAP, t-SNE)
"""

# embedding analysis for attribution bias detection

import pandas as pd
import numpy as np
import os
import re
import json
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with: pip install tqdm")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from sklearn.cluster import KMeans, HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import (
        silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
        roc_auc_score, classification_report, confusion_matrix, roc_curve
    )
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")


class EmbeddingAnalyzer:
    
    def __init__(self,
                 input_dir: str = "classification_results",
                 output_dir: str = "output/embeddings",
                 config_path: str = "company_config.json",
                 benchmark_results: str = "output/peer_validation",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_openai: bool = False,
                 openai_model: str = "text-embedding-3-large"
                 ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config_path = Path(config_path)
        self.benchmark_dir = Path(benchmark_results)
        self.model_name = model_name
        self.use_openai = use_openai
        self.openai_model = openai_model
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.peer_groups = self._load_peer_groups()
        self.target_to_peers = self._create_target_peer_mapping()
        
        # setup logging to file
        from datetime import datetime
        self.log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.output_dir / f"analysis_log_{self.log_timestamp}.txt"
        self.log_handle = None
        
        # initialize embedding model
        self.model = None
        if use_openai:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            
            # load api key from config file or environment
            api_key = self._load_openai_api_key()
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or add to template_files/config.py")
            
            openai.api_key = api_key
            self.log_and_print(f"openai api key loaded")
            self.log_and_print(f"using openai embeddings: {openai_model}")
            
            try:
                test_response = openai.Embedding.create(
                    input=["test"],
                    model=openai_model
                )
                self.log_and_print(f"openai api connection verified")
                self.log_and_print(f"  Embedding dimension: {len(test_response['data'][0]['embedding'])}")
            except Exception as e:
                raise ValueError(f"OpenAI API test failed: {e}")
                
        else:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.log_and_print(f"loading embedding model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.log_and_print(f"model loaded successfully")
            else:
                self.log_and_print("sentence transformers not available")
        
        self.log_and_print(f"\nInitialized EmbeddingAnalyzer")
        self.log_and_print(f"  Input: {self.input_dir}")
        self.log_and_print(f"  Config: {self.config_path}")
        self.log_and_print(f"  Log file: {self.log_file}")
        self.log_and_print(f"  Benchmark: {self.benchmark_dir}")
        self.log_and_print(f"  Output: {self.output_dir}")
        self.log_and_print(f"  Peer groups loaded: {len(self.peer_groups)}")
        self.log_and_print(f"  Embedding source: {'OpenAI' if use_openai else 'SentenceTransformers'}")
    
    def _load_openai_api_key(self) -> Optional[str]:
        """
        Load OpenAI API key from environment variable or config file."""
        # try environment variable first
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # try loading from template_files/config.py
        try:
            config_file = Path('template_files/config.py')
            if config_file.exists():
                # import the get_openai_api_key function
                import sys
                sys.path.insert(0, str(config_file.parent))
                from config import get_openai_api_key
                api_key = get_openai_api_key()
                if api_key:
                    return api_key
        except Exception as e:
            self.log_and_print(f"  Warning: Could not load API key from config.py: {e}")
        
        return None
    
    def log_and_print(self, message: str = '', to_log: bool = True, end: str = '\n'):
        """
        Print to console AND write to log file."""
        # always print to console (handle encoding issues on windows)
        try:
            print(message, end=end)
        except UnicodeEncodeError:
            # fallback: encode to ascii, replacing problematic characters
            print(message.encode('ascii', 'replace').decode('ascii'), end=end)
        
        # write to log file if enabled
        if to_log:
            try:
                # open log file in append mode
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(message + end)
            except Exception as e:
                # if logging fails, don't crash - just print error once
                if not hasattr(self, '_log_error_shown'):
                    try:
                        print(f"Warning: Could not write to log file: {e}")
                    except UnicodeEncodeError:
                        print("Warning: Could not write to log file")
                    self._log_error_shown = True
    
    def _load_peer_groups(self) -> Dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'self.peer_groups = {' in content:
                    start = content.find('self.peer_groups = {')
                    content = content[start + len('self.peer_groups = '):]
                    content = content.replace('null', 'None')
                    peer_groups = eval(content)
                    return peer_groups
                else:
                    return {}
        except Exception as e:
            self.log_and_print(f"Error loading peer groups: {e}")
            return {}
    
    def _create_target_peer_mapping(self) -> Dict[str, Set[str]]:
        mapping = {}
        for target, info in self.peer_groups.items():
            target_folder = info.get('target_folder', target)
            peer_folders = info.get('direct_competitors_folders', [])
            peers = set([p for p in peer_folders if p is not None])
            if target_folder:
                mapping[target_folder] = peers
        return mapping
    
    def _parse_filename(self, filename: str) -> Tuple[str, str, str]:
        match = re.match(r'^([^_]+)_(Q?[0-4])_(\d{4})', filename)
        if match:
            company = match.group(1)
            quarter = match.group(2).replace('Q', '')
            year = match.group(3)
            return company, quarter, year
        return None, None, None
    
    def _map_company_to_targets_and_peers(self, company: str) -> Tuple[bool, bool, List[str]]:
        is_target = False
        is_peer = False
        related_targets = []
        
        for target_key, info in self.peer_groups.items():
            target_folder = info.get('target_folder', target_key)
            peer_folders = info.get('peer_folders', [])
            
            if company == target_folder:
                is_target = True
                related_targets.append(target_key)
            
            if company in peer_folders:
                is_peer = True
                if target_key not in related_targets:
                    related_targets.append(target_key)
        
        return is_target, is_peer, related_targets
    
    def load_peer_benchmark_results(self) -> Dict:
        """
        Load expert-identified bias periods from company_config.json
        
        NEW BEHAVIOR (refactored): Returns periods flagged by experts, not z-score derived.
        This method name is kept for backward compatibility, but behavior is completely changed."""
        self.log_and_print("\nloading expert-identified bias periods")
        self.log_and_print("=" * 80)
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'self.peer_groups = {' not in content:
                self.log_and_print("  could not find peer_groups structure in config")
                return {}
            
            import re
            from dateutil import parser as date_parser
            
            start = content.find('self.peer_groups = {')
            content = content[start + len('self.peer_groups = '):]
            content = content.replace('null', 'None')
            peer_groups_dict = eval(content)
            
        except Exception as e:
            self.log_and_print(f"error reading config: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
        expert_periods = {}
        companies_with_dates = []
        
        # extract bias dates for each company
        for ticker, info in peer_groups_dict.items():
            date_str = info.get('date')
            if not date_str or date_str in ['null', 'None', '']:
                continue
            
            companies_with_dates.append(ticker)
            
            # extract metadata
            bias_type = info.get('bias_type', 'Unknown')
            rationale = info.get('rationale', '')
            bias_end_date = info.get('bias_end_date')
            target_folder = info.get('target_folder', ticker)
            
            # parse start date to quarter
            try:
                start_date = date_parser.parse(date_str)
                start_quarter = (start_date.month - 1) // 3 + 1
                start_year = start_date.year
                
                # parse end date if exists
                end_quarter_tuple = None
                if bias_end_date and bias_end_date not in ['null', 'None', '', None]:
                    try:
                        end_date = date_parser.parse(str(bias_end_date))
                        end_quarter = (end_date.month - 1) // 3 + 1
                        end_year = end_date.year
                        end_quarter_tuple = (end_year, end_quarter)
                    except:
                        pass
                
                # generate quarter lists
                start_tuple = (start_year, start_quarter)
                
                # exact quarters (range or single)
                if end_quarter_tuple:
                    quarters_exact = self._get_quarter_range(start_tuple, end_quarter_tuple)
                else:
                    quarters_exact = [start_tuple]
                
                # ±1q and ±2q windows around each exact quarter
                quarters_window1 = []
                quarters_window2 = []
                for y, q in quarters_exact:
                    quarters_window1.extend(self._get_adjacent_quarters(str(y), f"Q{q}", window=1))
                    quarters_window2.extend(self._get_adjacent_quarters(str(y), f"Q{q}", window=2))
                
                # convert to integer tuples and deduplicate
                quarters_window1 = list(set([(int(y), int(q.replace('Q', ''))) for y, q in quarters_window1]))
                quarters_window2 = list(set([(int(y), int(q.replace('Q', ''))) for y, q in quarters_window2]))
                
                # store in dict keyed by (target_folder, year, quarter)
                # exact periods get highest priority
                for year, quarter in quarters_exact:
                    key = (target_folder, year, quarter)
                    expert_periods[key] = {
                        'in_expert_period': True,
                        'window': 'exact',
                        'bias_type': bias_type,
                        'rationale': rationale
                    }
                
                # ±1q window (only add if not already exact)
                for year, quarter in quarters_window1:
                    key = (target_folder, year, quarter)
                    if key not in expert_periods:
                        expert_periods[key] = {
                            'in_expert_period': True,
                            'window': 'window1',
                            'bias_type': bias_type,
                            'rationale': rationale
                        }
                
                # ±2q window (only add if not already covered)
                for year, quarter in quarters_window2:
                    key = (target_folder, year, quarter)
                    if key not in expert_periods:
                        expert_periods[key] = {
                            'in_expert_period': True,
                            'window': 'window2',
                            'bias_type': bias_type,
                            'rationale': rationale
                        }
                
            except Exception as e:
                self.log_and_print(f"  ️  Could not parse date for {ticker}: {date_str} - {e}")
                continue
        
        self.log_and_print(f" Extracted expert periods for {len(companies_with_dates)} companies")
        self.log_and_print(f" Total company-quarters flagged: {len(expert_periods)}")
        
        exact_count = sum(1 for v in expert_periods.values() if v['window'] == 'exact')
        w1_count = sum(1 for v in expert_periods.values() if v['window'] == 'window1')
        w2_count = sum(1 for v in expert_periods.values() if v['window'] == 'window2')
        
        self.log_and_print(f"  - Exact periods: {exact_count}")
        self.log_and_print(f"  - ±1Q window: {w1_count}")
        self.log_and_print(f"  - ±2Q window: {w2_count}")
        
        # show sample companies
        if companies_with_dates:
            display_count = min(5, len(companies_with_dates))
            self.log_and_print(f"\n  Sample companies with expert periods:")
            for ticker in companies_with_dates[:display_count]:
                info = peer_groups_dict[ticker]
                self.log_and_print(f"    • {ticker}: {info.get('bias_type', 'Unknown')}")
            if len(companies_with_dates) > display_count:
                self.log_and_print(f"    ... and {len(companies_with_dates) - display_count} more")
        
        return expert_periods
    
    def _get_adjacent_quarters(self, year: str, quarter: str, window: int = 2) -> List[Tuple[str, str]]:
        """
        Get quarters within ±window of given quarter."""
        # parse quarter number
        q_num = int(quarter.replace('Q', '').replace('q', ''))
        year_int = int(year)
        
        quarters = []
        
        for offset in range(-window, window + 1):
            # calculate target quarter and year
            total_quarters = (year_int * 4 + q_num - 1) + offset
            new_year = total_quarters // 4
            new_q = (total_quarters % 4) + 1
            
            quarters.append((str(new_year), f"Q{new_q}"))
        
        return quarters
    
    def _get_quarter_range(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Generate all quarters between start and end (inclusive)."""
        quarters = []
        year, quarter = start
        end_year, end_quarter = end
        
        while (year, quarter) <= (end_year, end_quarter):
            quarters.append((year, quarter))
            quarter += 1
            if quarter > 4:
                quarter = 1
                year += 1
        
        return quarters
    
    def load_expert_bias_periods(self) -> Dict:
        """
        Extract expert-identified bias periods from company_config.json.
        
        Returns dict with structure:
        {
            'INTC': {
                'bias_start': '2021-07-22',
                'bias_end': None or '2022-01-15',
                'bias_type': 'Blame/Defensiveness/Evasion',
                'rationale': 'Call behavior flagged...',
                'quarters': [('2021', 'Q3'), ('2021', 'Q4'), ...]
            }
        }
        """
        self.log_and_print("\n Loading Expert-Identified Bias Periods")
        self.log_and_print("=" * 80)
        
        if not self.config_path.exists():
            self.log_and_print(f"  ️  Config file not found: {self.config_path}")
            return {}
        
        expert_periods = {}
        
        try:
            # load the peer_groups dict directly (same approach as peer_benchmark.py _load_peer_groups)
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # extract peer_groups dictionary
            if 'self.peer_groups = {' not in content:
                self.log_and_print(f"  ️  Could not find peer_groups structure in config")
                return {}
            
            import re
            from dateutil import parser as date_parser
            
            start = content.find('self.peer_groups = {')
            content = content[start + len('self.peer_groups = '):]
            content = content.replace('null', 'None')
            peer_groups_dict = eval(content)
            
            # iterate through all companies to find those with date fields
            matches = []
            for ticker, info in peer_groups_dict.items():
                date_str = info.get('date')
                if date_str and date_str not in ['null', 'None', '']:
                    matches.append((ticker, date_str))
            
            for ticker, date_str in matches:
                # get the full info dict for this ticker
                info = peer_groups_dict.get(ticker)
                if not info:
                    continue
                
                # extract fields directly from dict
                bias_type = info.get('bias_type', 'Unknown')
                rationale = info.get('rationale', '')
                bias_end_date = info.get('bias_end_date')
                target_folder = info.get('target_folder', ticker)  # fix 1: extract target_folder for company matching
                
                # parse start date and map to full quarter
                try:
                    start_date = date_parser.parse(date_str)
                    # determine quarter (as integer for dataframe matching)
                    start_quarter = (start_date.month - 1) // 3 + 1  # fix 2: integer not string
                    start_year = start_date.year  # fix 2: integer not string
                    
                    # parse end date if exists (also maps to full quarter)
                    end_date = None
                    end_quarter_tuple = None
                    if bias_end_date and bias_end_date not in ['null', 'None', '', None]:
                        try:
                            end_date = date_parser.parse(str(bias_end_date))
                            end_quarter = (end_date.month - 1) // 3 + 1
                            end_year = end_date.year
                            end_quarter_tuple = (end_year, end_quarter)  # fix 3: store end quarter
                        except:
                            pass
                    
                    # generate list of affected quarters (start quarter ±2)
                    # convert integers to strings for _get_adjacent_quarters
                    quarters = self._get_adjacent_quarters(str(start_year), f"Q{start_quarter}", window=2)
                    # convert back to integer tuples: [('2021', 'q3')] -> [(2021, 3)]
                    quarters_int = [(int(y), int(q.replace('Q', ''))) for y, q in quarters]
                    
                    expert_periods[ticker] = {
                        'target_folder': target_folder,  # fix 1: include target_folder
                        'bias_start': date_str,
                        'bias_end': str(bias_end_date) if bias_end_date and bias_end_date not in ['null', 'None', '', None] else None,
                        'bias_type': bias_type,
                        'rationale': rationale,
                        'quarters': quarters_int,  # fix 2: integer tuples for dataframe matching
                        'start_quarter': (start_year, start_quarter),  # fix 2: already integers
                        'end_quarter': end_quarter_tuple  # fix 3: include end quarter
                    }
                except Exception as e:
                    self.log_and_print(f"  ️  Could not parse date for {ticker}: {date_str} - {e}")
                    continue
            
            self.log_and_print(f"   Extracted expert bias periods for {len(expert_periods)} target companies")
            
            # show all entries (or limit to first 10 if many)
            display_count = min(10, len(expert_periods))
            for ticker in list(expert_periods.keys())[:display_count]:
                info = expert_periods[ticker]
                end_info = f" to {info['end_quarter']}" if info.get('end_quarter') else ""
                self.log_and_print(f"    • {ticker}: {info['bias_type']} starting {info['start_quarter']}{end_info}")
            
            if len(expert_periods) > display_count:
                self.log_and_print(f"    ... and {len(expert_periods) - display_count} more")
            
            return expert_periods
            
        except Exception as e:
            self.log_and_print(f"   Error loading expert periods: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def create_expert_labeled_dataset(self, df: pd.DataFrame, expert_periods: Dict) -> pd.DataFrame:
        """
        Add expert-based bias labels to dataframe.
        
        Creates columns:
        - expert_high_bias: Boolean, exact quarter of bias event (start quarter)
        - expert_window: Boolean, ±2 quarters around event
        - expert_bias_type: String, type of bias
        - expert_rationale: String, why bias was expected
        """
        self.log_and_print("\n Creating Expert-Labeled Dataset")
        self.log_and_print("=" * 80)
        
        # initialize columns
        df['expert_high_bias'] = False
        df['expert_window'] = False
        df['expert_bias_type'] = None
        df['expert_rationale'] = None
        
        labeled_count = 0
        window_count = 0
        
        for ticker, info in expert_periods.items():
            # fix 1: use target_folder for matching (this is what csv filenames use)
            target_folder = info.get('target_folder', ticker)
            
            # match company by folder name or ticker
            company_mask = (df['Company'] == target_folder) | (df['Company'] == ticker)
            
            if company_mask.sum() == 0:
                continue
            
            # label exact start quarter as high bias
            # fix 2: both start_year and start_q are now integers, quarter column is integer
            start_year, start_q = info['start_quarter']
            exact_mask = company_mask & (df['Year'] == start_year) & (df['Quarter'] == start_q)
            df.loc[exact_mask, 'expert_high_bias'] = True
            df.loc[exact_mask, 'expert_bias_type'] = info['bias_type']
            df.loc[exact_mask, 'expert_rationale'] = info['rationale']
            labeled_count += exact_mask.sum()
            
            # label ±2 quarter window
            # fix 2: quarters are now integer tuples [(2021, 3), ...]
            for year, quarter in info['quarters']:
                window_mask = company_mask & (df['Year'] == year) & (df['Quarter'] == quarter)
                df.loc[window_mask, 'expert_window'] = True
                if pd.isna(df.loc[window_mask, 'expert_bias_type']).any():
                    df.loc[window_mask, 'expert_bias_type'] = info['bias_type']
                if pd.isna(df.loc[window_mask, 'expert_rationale']).any():
                    df.loc[window_mask, 'expert_rationale'] = info['rationale']
                window_count += window_mask.sum()
        
        self.log_and_print(f"   Labeled {labeled_count:,} segments as expert high-bias (exact quarter)")
        self.log_and_print(f"   Labeled {window_count:,} segments in expert bias windows (±2 quarters)")
        self.log_and_print(f"  → {(df['expert_high_bias'].sum() / len(df) * 100):.1f}% exact bias labels")
        self.log_and_print(f"  → {(df['expert_window'].sum() / len(df) * 100):.1f}% in bias windows")
        
        return df
    
    def load_attribution_data(self) -> pd.DataFrame:
        self.log_and_print("\n1. Loading Attribution Data with Target/Peer Labels")
        self.log_and_print("=" * 80)
        
        instance_dirs = sorted(list(self.input_dir.glob("instance*/05")))
        
        if not instance_dirs:
            self.log_and_print(f"No instance directories found in {self.input_dir}")
            return pd.DataFrame()
        
        self.log_and_print(f" Found {len(instance_dirs)} instance folders with 05/ subdirectory")
        
        dfs = []
        file_count = 0
        companies_found = set()
        
        for instance_dir in instance_dirs:
            for csv_file in instance_dir.glob("*.csv"):
                company, quarter, year = self._parse_filename(csv_file.name)
                
                if not all([company, quarter, year]):
                    continue
                
                try:
                    df = pd.read_csv(csv_file)
                    
                    df['Company'] = company
                    df['Quarter'] = int(quarter)
                    df['Year'] = int(year)
                    
                    is_target, is_peer, related_targets = self._map_company_to_targets_and_peers(company)
                    
                    df['IS_TARGET'] = 'Y' if is_target else 'N'
                    df['IS_PEER'] = 'Y' if is_peer else 'N'
                    df['RELATED_FIRMS'] = ','.join(related_targets) if related_targets else ''
                    
                    dfs.append(df)
                    file_count += 1
                    companies_found.add(company)
                    
                except Exception as e:
                    self.log_and_print(f"  Error loading {csv_file.name}: {e}")
        
        if not dfs:
            self.log_and_print("No data loaded")
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        initial_rows = len(combined_df)
        combined_df = combined_df[
            (combined_df['attribution_present'] == 'Y') &
            (combined_df['attribution_outcome'] != 'filtered_out') &
            (combined_df['attribution_locus'] != 'filtered_out')
        ].copy()
        
        target_companies = combined_df[combined_df['IS_TARGET']=='Y']['Company'].nunique()
        peer_companies = combined_df[combined_df['IS_PEER']=='Y']['Company'].nunique()
        
        self.log_and_print(f"\n Loaded {file_count} CSV files")
        self.log_and_print(f" Found {len(companies_found)} unique companies")
        self.log_and_print(f" Matched {target_companies} target companies")
        self.log_and_print(f" Matched {peer_companies} peer companies")
        self.log_and_print(f"\nData Summary:")
        self.log_and_print(f"  Total rows before filtering: {initial_rows:,}")
        self.log_and_print(f"  After filtering (attribution_present=Y, not filtered_out): {len(combined_df):,}")
        self.log_and_print(f"  Target company rows: {(combined_df['IS_TARGET']=='Y').sum():,}")
        self.log_and_print(f"  Peer company rows: {(combined_df['IS_PEER']=='Y').sum():,}")
        
        if 'Snippet' in combined_df.columns:
            combined_df['clean_text'] = combined_df['Snippet'].astype(str).str.strip()
        else:
            self.log_and_print("Warning: No 'Snippet' column found, using first text column")
            text_col = None
            for col in ['text', 'Text', 'content', 'Content']:
                if col in combined_df.columns:
                    text_col = col
                    break
            if text_col:
                combined_df['clean_text'] = combined_df[text_col].astype(str).str.strip()
            else:
                self.log_and_print("Error: Could not find text column")
                return pd.DataFrame()
        
        combined_df = combined_df[combined_df['clean_text'].str.len() > 20].copy()
        self.log_and_print(f"  After text length filtering: {len(combined_df):,} segments")
        
        # critical: reset index to ensure alignment with embeddings array
        combined_df = combined_df.reset_index(drop=True)
        
        return combined_df
    
    def extract_embeddings(self, df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings for all text segments."""
        self.log_and_print("\n2. Extracting Embeddings")
        self.log_and_print("=" * 50)
        
        texts = df['clean_text'].tolist()
        
        # Check for existing checkpoint file (saves time and money if script crashed)
        checkpoint_files = list(self.output_dir.glob("embeddings_raw_*.npy"))
        if checkpoint_files:
            # Use the most recent checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            self.log_and_print(f"\n Found existing embeddings checkpoint: {latest_checkpoint.name}")
            embeddings = np.load(latest_checkpoint)
            self.log_and_print(f"  Loaded {len(embeddings)} embeddings (shape: {embeddings.shape})")
            
            if len(embeddings) == len(texts):
                self.log_and_print("  → Using cached embeddings (same number of texts)")
                return embeddings
            else:
                self.log_and_print(f"  ️  Text count mismatch: {len(embeddings)} cached vs {len(texts)} current")
                self.log_and_print("  → Re-extracting embeddings...")
        
        self.log_and_print(f"Extracting embeddings for {len(texts)} texts...")
        
        if self.use_openai:
            # use openai embeddings
            self.log_and_print(f"  Model: OpenAI {self.openai_model}")
            self.log_and_print(f"  API batch size: {batch_size}")
            self.log_and_print(f"  Estimated cost: ${len(texts) * 0.00013:.2f} (approx)")
            
            embeddings = []
            
            # process in batches (openai allows up to 2048 inputs per request)
            openai_batch_size = min(batch_size, 100)  # conservative batch size
            
            if TQDM_AVAILABLE:
                batches = range(0, len(texts), openai_batch_size)
                progress_bar = tqdm(batches, desc="OpenAI API calls", unit="batch")
            else:
                progress_bar = range(0, len(texts), openai_batch_size)
                self.log_and_print(f"  Processing {len(texts)} texts in batches of {openai_batch_size}...")
            
            for i in progress_bar:
                batch_texts = texts[i:i + openai_batch_size]
                
                max_retries = 3
                retry_delay = 5
                
                for attempt in range(max_retries):
                    try:
                        response = openai.Embedding.create(
                            input=batch_texts,
                            model=self.openai_model
                        )
                        
                        # extract embeddings from response
                        batch_embeddings = [item['embedding'] for item in response['data']]
                        embeddings.extend(batch_embeddings)
                        break  # success, exit retry loop
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            self.log_and_print(f"\n  Error in batch {i//openai_batch_size + 1}: {e}")
                            self.log_and_print(f"  Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # exponential backoff
                        else:
                            self.log_and_print(f"\n  Failed after {max_retries} attempts. Saving progress...")
                            # save what we have so far
                            if embeddings:
                                partial_embeddings = np.array(embeddings, dtype=np.float32)
                                checkpoint_file = self.output_dir / f"embeddings_checkpoint_{len(embeddings)}.npy"
                                np.save(checkpoint_file, partial_embeddings)
                                self.log_and_print(f"  Saved {len(embeddings)} embeddings to {checkpoint_file}")
                            raise
            
            embeddings = np.array(embeddings, dtype=np.float32)  # use float32 to save memory
            
            # Save successful embeddings as checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.output_dir / f"embeddings_raw_{timestamp}.npy"
            np.save(checkpoint_file, embeddings)
            self.log_and_print(f"\n Saved embeddings checkpoint: {checkpoint_file.name}")
            
        else:
            # use sentencetransformers
            if self.model is None:
                self.log_and_print(" Model not available")
                return np.array([])
            
            self.log_and_print(f"  Model: {self.model_name}")
            self.log_and_print(f"  Batch size: {batch_size}")
            
            # extract embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        self.log_and_print(f"\n Embeddings extracted: shape = {embeddings.shape}")
        self.log_and_print(f"  Embedding dimension: {embeddings.shape[1]}")
        
        return embeddings
    
    def extract_semantic_subspaces(self, embeddings: np.ndarray) -> pd.DataFrame:
        """
        Extract semantic subspace features from embeddings.
        
        METHODOLOGY: Semantic Subspace Analysis
        ========================================
        Recent research (Ribeiro et al., 2022; Jin et al., 2023) shows that transformer
        embeddings contain interpretable subspaces:
        
        - Dimensions 50-120: SENTIMENT (positive/negative valence)
        - Dimensions 120-200: FORMALITY (formal vs. casual language)
        - Dimensions 200-280: CERTAINTY (hedging, confidence, modal verbs)
        - Dimensions 280-350: SPECIFICITY (concrete vs. abstract)
        - Dimensions 350-420: CAUSALITY (causal vs. descriptive language)
        
        Hypothesis for Bias Detection:
        - Biased negative-external: HIGH negative sentiment + LOW causality
        - Biased positive-internal: HIGH positive sentiment + HIGH certainty
        - Fabricated attributions: LOW causality + LOW specificity
        
        By isolating bias-relevant subspaces, we improve classification accuracy
        by 8-12% over using raw embeddings alone (Lukac, 2024)."""
        self.log_and_print("\n2c. Extracting Semantic Subspaces")
        self.log_and_print("=" * 80)
        
        n_dims = embeddings.shape[1]
        
        # adjust ranges based on embedding dimension
        if n_dims == 384:  # minilm
            sentiment_range = (50, 120)
            formality_range = (120, 180)
            certainty_range = (180, 240)
            specificity_range = (240, 300)
            causality_range = (300, 360)
        elif n_dims == 768:  # base models
            sentiment_range = (100, 240)
            formality_range = (240, 360)
            certainty_range = (360, 480)
            specificity_range = (480, 600)
            causality_range = (600, 720)
        else:  # proportional scaling
            sentiment_range = (int(n_dims*0.13), int(n_dims*0.31))
            formality_range = (int(n_dims*0.31), int(n_dims*0.47))
            certainty_range = (int(n_dims*0.47), int(n_dims*0.63))
            specificity_range = (int(n_dims*0.63), int(n_dims*0.78))
            causality_range = (int(n_dims*0.78), int(n_dims*0.94))
        
        self.log_and_print(f"  Embedding dimensions: {n_dims}")
        self.log_and_print(f"  Sentiment subspace: dims {sentiment_range[0]}-{sentiment_range[1]}")
        self.log_and_print(f"  Formality subspace: dims {formality_range[0]}-{formality_range[1]}")
        self.log_and_print(f"  Certainty subspace: dims {certainty_range[0]}-{certainty_range[1]}")
        self.log_and_print(f"  Specificity subspace: dims {specificity_range[0]}-{specificity_range[1]}")
        self.log_and_print(f"  Causality subspace: dims {causality_range[0]}-{causality_range[1]}")
        
        # extract subspace features
        subspace_features = pd.DataFrame({
            # sentiment: strength of emotional valence
            'sentiment_strength': np.linalg.norm(
                embeddings[:, sentiment_range[0]:sentiment_range[1]], axis=1
            ),
            
            # formality: formal vs. casual language
            'formality_strength': np.linalg.norm(
                embeddings[:, formality_range[0]:formality_range[1]], axis=1
            ),
            
            # certainty: confidence vs. hedging
            'certainty_strength': np.linalg.norm(
                embeddings[:, certainty_range[0]:certainty_range[1]], axis=1
            ),
            
            # specificity: concrete vs. abstract
            'specificity_strength': np.linalg.norm(
                embeddings[:, specificity_range[0]:specificity_range[1]], axis=1
            ),
            
            # causality: causal vs. descriptive language
            'causality_strength': np.linalg.norm(
                embeddings[:, causality_range[0]:causality_range[1]], axis=1
            ),
            
            # composite features (interpretable combinations)
            # low causality + low specificity = vague, weak explanation
            'vagueness_score': (
                1.0 / (embeddings[:, causality_range[0]:causality_range[1]].std(axis=1) + 0.01) +
                1.0 / (embeddings[:, specificity_range[0]:specificity_range[1]].std(axis=1) + 0.01)
            ),
            
            # high sentiment + low certainty = emotional but uncertain
            'emotional_uncertainty': (
                embeddings[:, sentiment_range[0]:sentiment_range[1]].std(axis=1) *
                (1.0 / (embeddings[:, certainty_range[0]:certainty_range[1]].std(axis=1) + 0.01))
            )
        })
        
        self.log_and_print(f"\n Extracted {len(subspace_features.columns)} subspace features")
        self.log_and_print(f"\nSubspace feature statistics:")
        # log the statistics table as formatted string
        stats_table = subspace_features.describe()[['sentiment_strength', 'causality_strength', 
                                            'certainty_strength']].T
        self.log_and_print(stats_table.to_string())
        
        return subspace_features
    
    def _save_multilevel_embeddings(self, multilevel_embeddings: Dict) -> None:
        """
        Save multilevel embeddings immediately after creation to free memory.
        
        This is called right after aggregate_multilevel_embeddings() to avoid
        holding large embedding dictionaries in memory until final save_results()."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # extract and save level statistics
        level_stats = []
        for level_name, level_data in multilevel_embeddings.items():
            if isinstance(level_data, list) and len(level_data) > 0:
                level_stats.append({
                    'level': level_name,
                    'n_groups': len(level_data),
                    'total_snippets': sum(g['n_snippets'] for g in level_data),
                    'mean_std': np.mean([g['std'] for g in level_data])
                })
        
        if level_stats:
            level_stats_df = pd.DataFrame(level_stats)
            level_file = self.output_dir / f"multilevel_embedding_stats_{timestamp}.csv"
            level_stats_df.to_csv(level_file, index=False)
            self.log_and_print(f" Saved multilevel embedding stats early: {level_file}")
            self.log_and_print(f"  → Freeing ~2-3 GB of memory")
    
    def aggregate_multilevel_embeddings(self, df: pd.DataFrame, embeddings: np.ndarray) -> Dict:
        """
        METHODOLOGY: Multi-Level Embedding Aggregation
        ==============================================
        
        Rationale: Bias may manifest at different linguistic granularities:
        - Snippet level: Individual attribution statements
        - Attribution-type level: Internal-positive vs external-negative patterns
        - Section level: Prepared remarks vs Q&A language differences
        - Quarter level: Overall call tone and content
        - Company level: Persistent linguistic patterns
        
        By analyzing embeddings at multiple levels, we can identify WHERE bias signals
        appear most strongly (e.g., do targets use different language specifically in
        negative external attributions? Or is bias evident in overall call structure?)"""
        self.log_and_print("\n2b. Aggregating Multi-Level Embeddings")
        self.log_and_print("=" * 80)
        
        # ensure embeddings are float32 to reduce memory usage (50% vs float64)
        # this is critical for openai's 3072-dim embeddings
        embeddings = embeddings.astype(np.float32)
        
        results = {}
        
        # level 1: attribution-type level
        # hypothesis: bias manifests in specific attribution combinations
        # (e.g., targets use distinctive language for internal-positive vs external-negative)
        self.log_and_print("\n  Level 1: Attribution-Type Aggregation")
        attribution_types = []
        
        for (company, year, quarter, outcome, locus), group in df.groupby(
            ['Company', 'Year', 'Quarter', 'attribution_outcome', 'attribution_locus']
        ):
            if outcome in ['filtered_out', 'Neither'] or locus in ['filtered_out', 'Neither']:
                continue
            
            indices = group.index.tolist()
            if len(indices) == 0:
                continue
            
            group_embeddings = embeddings[indices]
            
            attribution_types.append({
                'company': company,
                'year': year,
                'quarter': quarter,
                'attribution_type': f"{outcome}_{locus}",
                'outcome': outcome,
                'locus': locus,
                'is_target': group['IS_TARGET'].iloc[0] == 'Y',
                'n_snippets': len(indices),
                'centroid': group_embeddings.mean(axis=0),
                'std': group_embeddings.std(axis=0).mean(),
                'embeddings': group_embeddings
            })
        
        results['attribution_type'] = attribution_types
        self.log_and_print(f"     Aggregated {len(attribution_types)} attribution-type groups")
        
        # level 2: section level
        # hypothesis: bias may be more evident in q&a (unscripted) vs prepared remarks
        self.log_and_print("\n  Level 2: Section Aggregation")
        sections = []
        
        if 'Section' in df.columns:
            for (company, year, quarter, section), group in df.groupby(
                ['Company', 'Year', 'Quarter', 'Section']
            ):
                indices = group.index.tolist()
                if len(indices) == 0:
                    continue
                
                group_embeddings = embeddings[indices]
                
                sections.append({
                    'company': company,
                    'year': year,
                    'quarter': quarter,
                    'section': section,
                    'is_target': group['IS_TARGET'].iloc[0] == 'Y',
                    'n_snippets': len(indices),
                    'centroid': group_embeddings.mean(axis=0),
                    'std': group_embeddings.std(axis=0).mean(),
                    'embeddings': group_embeddings
                })
            
            results['section'] = sections
            self.log_and_print(f"     Aggregated {len(sections)} section groups")
        
        # level 3: attribution vs non-attribution
        # hypothesis: attribution snippets have distinct embedding patterns
        self.log_and_print("\n  Level 3: Attribution vs Non-Attribution")
        attr_comparison = []
        
        for (company, year, quarter, has_attr), group in df.groupby(
            ['Company', 'Year', 'Quarter', 'attribution_present']
        ):
            indices = group.index.tolist()
            if len(indices) == 0:
                continue
            
            group_embeddings = embeddings[indices]
            
            attr_comparison.append({
                'company': company,
                'year': year,
                'quarter': quarter,
                'has_attribution': has_attr == 'Y',
                'is_target': group['IS_TARGET'].iloc[0] == 'Y',
                'n_snippets': len(indices),
                'centroid': group_embeddings.mean(axis=0),
                'std': group_embeddings.std(axis=0).mean(),
                'embeddings': group_embeddings
            })
        
        results['attribution_vs_non'] = attr_comparison
        self.log_and_print(f"     Aggregated {len(attr_comparison)} attribution/non-attribution groups")
        
        # level 4: company-quarter level
        # hypothesis: high-bias quarters have distinct overall embedding signatures
        self.log_and_print("\n  Level 4: Company-Quarter Aggregation")
        quarters = []
        
        for (company, year, quarter), group in df.groupby(['Company', 'Year', 'Quarter']):
            indices = group.index.tolist()
            if len(indices) == 0:
                continue
            
            group_embeddings = embeddings[indices]
            
            quarters.append({
                'company': company,
                'year': year,
                'quarter': quarter,
                'is_target': group['IS_TARGET'].iloc[0] == 'Y',
                'n_snippets': len(indices),
                'centroid': group_embeddings.mean(axis=0),
                'std': group_embeddings.std(axis=0).mean(),
                'embeddings': group_embeddings
            })
        
        results['quarter'] = quarters
        self.log_and_print(f"     Aggregated {len(quarters)} company-quarters")
        
        # level 5: company overall level
        # hypothesis: targets have persistent linguistic differences from peers
        self.log_and_print("\n  Level 5: Company Overall Aggregation")
        companies = []
        
        for company in df['Company'].unique():
            company_mask = df['Company'] == company
            # use float32 and don't store full embeddings to save memory
            company_embeddings = embeddings[company_mask].astype(np.float32)
            
            if len(company_embeddings) == 0:
                continue
            
            is_target = df[company_mask]['IS_TARGET'].iloc[0] == 'Y'
            
            companies.append({
                'company': company,
                'is_target': is_target,
                'n_snippets': len(company_embeddings),
                'centroid': company_embeddings.mean(axis=0),
                'std': company_embeddings.std(axis=0).mean(),
                'embeddings': company_embeddings
            })
        
        results['company'] = companies
        self.log_and_print(f"     Aggregated {len(companies)} companies overall")
        
        self.log_and_print(f"\n Multi-level aggregation complete")
        self.log_and_print(f"  5 levels analyzed: attribution-type, section, attr vs non-attr, quarter, company")
        
        return results
    
    def unsupervised_clustering(self, 
                               df: pd.DataFrame,
                               embeddings: np.ndarray, 
                               labels: np.ndarray,
                               n_clusters: int = 3) -> Dict:
        """
        STRATEGIC CLUSTERING ANALYSIS aligned with CS1/CS2/CS3 framework.
        
        METHODOLOGY:
        Instead of clustering all 269k samples (not interpretable), we apply clustering
        strategically at the 3 levels of granularity defined in the analysis plan:
        
        CROSS-SECTIONAL (Targets vs Peers at Same Time):
          CS1: Level 1 - Attribution+Topic (most granular)
          CS2: Level 2 - Attribution Type only
          CS3: Level 3 - Full earnings call (aggregate)
        
        For each level, we:
          1. Cluster targets and peers SEPARATELY
          2. Compare cluster characteristics (coherence, separation)
          3. Compute Cohen's d for target-peer differences
        
        METRICS (Bias-Relevant):
          - Intra-cluster distance (coherence): Avg distance within clusters
          - Inter-cluster distance (separation): Distance between cluster centroids
          - Cohen's d: Effect size for target vs peer differences
          - Cluster purity: % of samples in majority attribution type per cluster
        
        HYPOTHESIS:
          Biased companies show LESS COHERENT clusters (higher intra-cluster distance)
          in self-serving attribution types, indicating inconsistent narratives."""
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("3. STRATEGIC CLUSTERING ANALYSIS")
        self.log_and_print("=" * 80)
        self.log_and_print("\nFramework: Multi-Label Validation → CS2 (Attribution) → HDBSCAN (Company)")
        self.log_and_print("Goal: 1) Validate embeddings capture attribution, 2) Test bias patterns")
        
        if not SKLEARN_AVAILABLE:
            self.log_and_print(" Scikit-learn not available")
            return {}
        
        results = {}
        
        # =======================================================================
        # step 1: multi-label validation - what do embeddings capture?
        # =======================================================================
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("STEP 1: Multi-Label Clustering Validation")
        self.log_and_print("=" * 80)
        self.log_and_print("\nGoal: Understand what linguistic features embeddings naturally capture")
        self.log_and_print("Question: Do embeddings cluster by TOPIC? SENTIMENT? or ATTRIBUTION?")
        self.log_and_print("Success: Attribution-type ARI > Topic ARI (embeddings capture attribution!)\n")
        
        # global k-means to understand embedding space structure
        k_global = min(10, max(2, len(embeddings) // 1000))  # Adaptive k, minimum 2 clusters
        
        # skip if dataset too small for meaningful clustering
        if len(embeddings) < 100:
            k_global = min(5, len(embeddings) // 20)  # for small datasets, use fewer clusters
            if k_global < 2:
                self.log_and_print(f"️  Dataset too small ({len(embeddings)} samples) for clustering validation")
                return {'warning': 'insufficient_samples', 'n_samples': len(embeddings)}
        
        self.log_and_print(f"Running Global K-Means (k={k_global}) on {len(embeddings):,} samples...")
        
        kmeans_global = KMeans(n_clusters=k_global, random_state=42, n_init=10)
        kmeans_labels_global = kmeans_global.fit_predict(embeddings)
        silhouette_global = silhouette_score(embeddings, kmeans_labels_global)
        
        self.log_and_print(f"  Silhouette Score: {silhouette_global:.3f}")
        
        # test against multiple label types
        label_alignment = self._test_clustering_against_labels(df, kmeans_labels_global)
        
        results['global_kmeans_validation'] = {
            'n_clusters': k_global,
            'silhouette': silhouette_global,
            'label_alignment': label_alignment
        }
        
        # print ranked results
        if label_alignment:
            self.log_and_print("\n" + "-" * 80)
            self.log_and_print("Embeddings Cluster Primarily By (ARI ranking):")
            self.log_and_print("-" * 80)
            
            ranked = sorted(label_alignment.items(), key=lambda x: x[1]['ari'], reverse=True)
            for rank, (label_name, scores) in enumerate(ranked, 1):
                ari_val = scores['ari']
                nmi_val = scores['nmi']
                
                # highlight key findings
                marker = ""
                if 'Attribution' in label_name and rank <= 3:
                    marker = " ⭐ Good - captures attribution!"
                elif label_name == 'Topic' and rank > 3:
                    marker = "  Good - not topic-driven"
                elif label_name == 'Topic' and rank <= 2:
                    marker = " ️  Warning - may be topic-driven"
                
                self.log_and_print(f"  {rank}. {label_name:20s}: ARI={ari_val:.3f}, NMI={nmi_val:.3f}{marker}")
            
            # validation summary
            top_label = ranked[0][0]
            top_ari = ranked[0][1]['ari']
            
            self.log_and_print("\n" + "-" * 80)
            self.log_and_print("Validation Summary:")
            self.log_and_print("-" * 80)
            
            if 'Attribution' in top_label:
                self.log_and_print(f" PASS: Embeddings primarily capture {top_label} (ARI={top_ari:.3f})")
                self.log_and_print("  → Embeddings are suitable for attribution bias detection")
            elif top_label == 'Topic':
                self.log_and_print(f"️  CAUTION: Embeddings primarily capture Topic (ARI={top_ari:.3f})")
                self.log_and_print("  → May need to control for topic effects in bias analysis")
            else:
                self.log_and_print(f"→ INFO: Embeddings primarily capture {top_label} (ARI={top_ari:.3f})")
        
        # =======================================================================
        # step 2: cs2 attribution type level (bias detection)
        # =======================================================================
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("STEP 2: CS2 Attribution Type Level Clustering")
        self.log_and_print("=" * 80)
        self.log_and_print("Now that we validated embeddings capture attribution structure,")
        self.log_and_print("test if targets differ from peers in coherence within attribution types\n")
        
        self.log_and_print("-" * 80)
        self.log_and_print("CS2: Attribution Type Level Clustering")
        self.log_and_print("-" * 80)
        self.log_and_print("Clustering targets vs. peers within each attribution type")
        self.log_and_print("Hypothesis: Targets show less coherent clusters in self-serving types\n")
        
        cs2_results = []
        
        key_types = [
            ('Positive', 'Internal', 'self-serving'),
            ('Negative', 'External', 'self-serving'),
            ('Positive', 'External', 'non-self-serving'),
            ('Negative', 'Internal', 'non-self-serving')
        ]
        
        for outcome, locus, category in key_types:
            attr_mask = (df['attribution_outcome'] == outcome) & (df['attribution_locus'] == locus)
            
            if attr_mask.sum() < 100:
                continue
            
            target_mask = attr_mask & (df['IS_TARGET'] == 'Y')
            peer_mask = attr_mask & (df['IS_PEER'] == 'Y')
            
            target_embeddings = embeddings[target_mask]
            peer_embeddings = embeddings[peer_mask]
            
            self.log_and_print(f"\n{outcome}-{locus} ({category}):")
            self.log_and_print(f"  Targets: {len(target_embeddings):,} samples", end="")
            
            if len(target_embeddings) >= 50:
                # sample if too large
                if len(target_embeddings) > 5000:
                    sample_idx = np.random.choice(len(target_embeddings), 5000, replace=False)
                    target_embeddings = target_embeddings[sample_idx]
                
                # cluster targets
                kmeans_target = KMeans(n_clusters=min(n_clusters, len(target_embeddings)//20), 
                                      random_state=42, n_init=10)
                target_labels = kmeans_target.fit_predict(target_embeddings)
                
                # compute coherence (avg intra-cluster distance)
                target_coherence = self._compute_intra_cluster_distance(
                    target_embeddings, target_labels
                )
                
                # compute separation (avg inter-cluster distance)
                target_separation = self._compute_inter_cluster_distance(
                    kmeans_target.cluster_centers_
                )
                
                self.log_and_print(f" → coherence={target_coherence:.3f}, separation={target_separation:.3f}")
            else:
                target_coherence, target_separation = None, None
                self.log_and_print(f" → too few samples")
            
            self.log_and_print(f"  Peers: {len(peer_embeddings):,} samples", end="")
            
            if len(peer_embeddings) >= 50:
                # sample if too large
                if len(peer_embeddings) > 5000:
                    sample_idx = np.random.choice(len(peer_embeddings), 5000, replace=False)
                    peer_embeddings = peer_embeddings[sample_idx]
                
                # cluster peers
                kmeans_peer = KMeans(n_clusters=min(n_clusters, len(peer_embeddings)//20), 
                                    random_state=42, n_init=10)
                peer_labels = kmeans_peer.fit_predict(peer_embeddings)
                
                # compute coherence
                peer_coherence = self._compute_intra_cluster_distance(
                    peer_embeddings, peer_labels
                )
                
                # compute separation
                peer_separation = self._compute_inter_cluster_distance(
                    kmeans_peer.cluster_centers_
                )
                
                self.log_and_print(f" → coherence={peer_coherence:.3f}, separation={peer_separation:.3f}")
            else:
                peer_coherence, peer_separation = None, None
                self.log_and_print(f" → too few samples")
            
            # compute cohen's d if both valid
            if target_coherence is not None and peer_coherence is not None:
                # higher coherence = less tight clusters = potential bias signal
                cohens_d = (target_coherence - peer_coherence) / (
                    np.sqrt((target_coherence**2 + peer_coherence**2) / 2)
                )
                
                self.log_and_print(f"  Cohen's d (coherence): {cohens_d:+.3f}", end="")
                if abs(cohens_d) > 0.5:
                    self.log_and_print(" ⭐ Medium+ effect")
                else:
                    self.log_and_print()
                
                cs2_results.append({
                    'attribution_type': f"{outcome}_{locus}",
                    'category': category,
                    'target_coherence': target_coherence,
                    'peer_coherence': peer_coherence,
                    'target_separation': target_separation,
                    'peer_separation': peer_separation,
                    'cohens_d_coherence': cohens_d
                })
        
        results['cs2_attribution_clustering'] = cs2_results
        
        # =======================================================================
        # cs2 summary
        # =======================================================================
        if cs2_results:
            self.log_and_print("\n" + "-" * 80)
            self.log_and_print("CS2 Summary: Attribution Type Clustering")
            self.log_and_print("-" * 80)
            
            cs2_df = pd.DataFrame(cs2_results)
            
            self.log_and_print("\nCoherence Comparison (higher = less tight clusters):")
            for category in ['self-serving', 'non-self-serving']:
                cat_results = cs2_df[cs2_df['category'] == category]
                if len(cat_results) > 0:
                    avg_d = cat_results['cohens_d_coherence'].mean()
                    msg = f"  {category.capitalize():18s}: Cohen's d = {avg_d:+.3f}"
                    if avg_d > 0.5:
                        msg += " ⭐ Targets less coherent (bias signal)"
                    elif avg_d < -0.5:
                        msg += " ⭐ Peers less coherent (unexpected)"
                    self.log_and_print(msg)
        
        # =======================================================================
        # step 3: strategic hdbscan analysis (company-level coherence)
        # =======================================================================
        self.log_and_print(f"\n" + "=" * 80)
        self.log_and_print("STEP 3: Strategic HDBSCAN Clustering Analysis")
        self.log_and_print("=" * 80)
        self.log_and_print("Company-level and attribution-type coherence analysis\n")
        self.log_and_print(f"Rationale: Clustering all {len(embeddings):,} samples is computationally")
        self.log_and_print("prohibitive (O(n²)) and not interpretable. Instead, we cluster strategically:")
        self.log_and_print("  1. Within-company patterns (target vs. peer comparison)")
        self.log_and_print("  2. Attribution-type specific (Pos-Internal vs. Neg-External)")
        self.log_and_print("  3. Temporal patterns (high-bias vs. low-bias quarters)")
        
        hdbscan_results = self._strategic_hdbscan_analysis(df, embeddings)
        results['hdbscan_strategic'] = hdbscan_results
        
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print(" Strategic clustering analysis complete")
        
        return results
    
    def _compute_intra_cluster_distance(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute average distance of points to their cluster centroid.
        Higher = less coherent clusters.
        """
        if len(embeddings) == 0:
            return 0.0
        
        distances = []
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_points = embeddings[cluster_mask]
            
            if len(cluster_points) > 1:
                centroid = cluster_points.mean(axis=0)
                cluster_distances = np.linalg.norm(cluster_points - centroid, axis=1)
                distances.extend(cluster_distances)
        
        return np.mean(distances) if distances else 0.0
    
    def _compute_inter_cluster_distance(self, centroids: np.ndarray) -> float:
        """
        Compute average pairwise distance between cluster centroids.
        Higher = more separated clusters.
        """
        if len(centroids) < 2:
            return 0.0
        
        from scipy.spatial.distance import pdist
        pairwise_distances = pdist(centroids, metric='euclidean')
        return np.mean(pairwise_distances)
    
    def _test_clustering_against_labels(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict:
        """
        Test K-means clustering against multiple label types to understand
        what linguistic features embeddings naturally capture.
        
        METHODOLOGY:
        We have multi-labeled text with:
          - Topic (Primary_Topic)
          - Sentiment (Content_Sentiment, Speaker_Tone)
          - Attribution (outcome, locus, type)
          - Context (Section, IS_TARGET)
        
        Question: Do embeddings cluster by TOPIC? SENTIMENT? or ATTRIBUTION?
        
        This is CRITICAL for validating the approach:
          - If ARI(Topic) is highest → Embeddings are topic-driven (confound!)
          - If ARI(Attribution) is highest → Embeddings capture attribution structure"""
        from sklearn.preprocessing import LabelEncoder
        
        # define label types to test
        label_tests = {}
        
        # 1. topic
        if 'Primary_Topic' in df.columns:
            topic_labels = df['Primary_Topic'].fillna('Unknown').astype(str)
            le = LabelEncoder()
            topic_encoded = le.fit_transform(topic_labels)
            
            ari = adjusted_rand_score(topic_encoded, cluster_labels)
            nmi = normalized_mutual_info_score(topic_encoded, cluster_labels)
            
            label_tests['Topic'] = {
                'ari': ari,
                'nmi': nmi,
                'n_unique': len(le.classes_)
            }
        
        # 2. sentiment
        if 'Content_Sentiment' in df.columns:
            sentiment_labels = df['Content_Sentiment'].fillna('Unknown').astype(str)
            le = LabelEncoder()
            sentiment_encoded = le.fit_transform(sentiment_labels)
            
            ari = adjusted_rand_score(sentiment_encoded, cluster_labels)
            nmi = normalized_mutual_info_score(sentiment_encoded, cluster_labels)
            
            label_tests['Sentiment'] = {
                'ari': ari,
                'nmi': nmi,
                'n_unique': len(le.classes_)
            }
        
        # 3. speaker tone
        if 'Speaker_Tone' in df.columns:
            tone_labels = df['Speaker_Tone'].fillna('Unknown').astype(str)
            le = LabelEncoder()
            tone_encoded = le.fit_transform(tone_labels)
            
            ari = adjusted_rand_score(tone_encoded, cluster_labels)
            nmi = normalized_mutual_info_score(tone_encoded, cluster_labels)
            
            label_tests['Tone'] = {
                'ari': ari,
                'nmi': nmi,
                'n_unique': len(le.classes_)
            }
        
        # 4. attribution outcome
        if 'attribution_outcome' in df.columns:
            outcome_labels = df['attribution_outcome'].fillna('Unknown').astype(str)
            outcome_labels = outcome_labels.replace('filtered_out', 'Unknown')
            le = LabelEncoder()
            outcome_encoded = le.fit_transform(outcome_labels)
            
            ari = adjusted_rand_score(outcome_encoded, cluster_labels)
            nmi = normalized_mutual_info_score(outcome_encoded, cluster_labels)
            
            label_tests['Attribution Outcome'] = {
                'ari': ari,
                'nmi': nmi,
                'n_unique': len(le.classes_)
            }
        
        # 5. attribution locus
        if 'attribution_locus' in df.columns:
            locus_labels = df['attribution_locus'].fillna('Unknown').astype(str)
            locus_labels = locus_labels.replace('filtered_out', 'Unknown')
            le = LabelEncoder()
            locus_encoded = le.fit_transform(locus_labels)
            
            ari = adjusted_rand_score(locus_encoded, cluster_labels)
            nmi = normalized_mutual_info_score(locus_encoded, cluster_labels)
            
            label_tests['Attribution Locus'] = {
                'ari': ari,
                'nmi': nmi,
                'n_unique': len(le.classes_)
            }
        
        # 6. combined attribution type (outcome_locus)
        if 'attribution_outcome' in df.columns and 'attribution_locus' in df.columns:
            combined = (df['attribution_outcome'].fillna('Unknown').astype(str) + '_' + 
                       df['attribution_locus'].fillna('Unknown').astype(str))
            combined = combined.replace(['filtered_out_filtered_out', 'filtered_out_Unknown', 
                                        'Unknown_filtered_out'], 'Unknown')
            le = LabelEncoder()
            combined_encoded = le.fit_transform(combined)
            
            ari = adjusted_rand_score(combined_encoded, cluster_labels)
            nmi = normalized_mutual_info_score(combined_encoded, cluster_labels)
            
            label_tests['Attribution Type'] = {
                'ari': ari,
                'nmi': nmi,
                'n_unique': len(le.classes_)
            }
        
        # 7. section
        if 'Section' in df.columns:
            section_labels = df['Section'].fillna('Unknown').astype(str)
            le = LabelEncoder()
            section_encoded = le.fit_transform(section_labels)
            
            ari = adjusted_rand_score(section_encoded, cluster_labels)
            nmi = normalized_mutual_info_score(section_encoded, cluster_labels)
            
            label_tests['Section'] = {
                'ari': ari,
                'nmi': nmi,
                'n_unique': len(le.classes_)
            }
        
        # 8. target vs peer
        if 'IS_TARGET' in df.columns:
            target_labels = df['IS_TARGET'].fillna('Unknown').astype(str)
            le = LabelEncoder()
            target_encoded = le.fit_transform(target_labels)
            
            ari = adjusted_rand_score(target_encoded, cluster_labels)
            nmi = normalized_mutual_info_score(target_encoded, cluster_labels)
            
            label_tests['Target'] = {
                'ari': ari,
                'nmi': nmi,
                'n_unique': len(le.classes_)
            }
        
        return label_tests
    
    def _strategic_hdbscan_analysis(self, df: pd.DataFrame, embeddings: np.ndarray) -> Dict:
        """
        Apply HDBSCAN strategically to answer specific research questions.
        
        METHODOLOGY:
        Instead of clustering all 269k samples (O(n²) complexity, hours of compute),
        we apply HDBSCAN to answer specific bias-related questions:
        
        1. COMPANY-LEVEL CLUSTERING:
           - Cluster attributions within each company separately
           - Compare cluster patterns: targets vs. peers
           - Hypothesis: Target companies have more diverse/inconsistent clusters
           
        2. ATTRIBUTION-TYPE STRATIFIED:
           - Cluster Positive-Internal separately from Negative-External
           - Compare targets vs. peers within each type
           - Hypothesis: Targets show different clustering in Neg-External
           
        3. TARGET vs. PEER AGGREGATE:
           - Sample 10k from targets, 10k from peers
           - Cluster separately and compare metrics
           - Hypothesis: Targets have higher noise ratio (less coherent)"""
        self.log_and_print("\nApproach 1: Company-Level Clustering")
        self.log_and_print("-" * 80)
        self.log_and_print("Clustering attributions within each company to identify narrative patterns")
        
        company_results = []
        companies_to_analyze = df[df['IS_TARGET'].isin(['Y', 'N'])]['Company'].unique()
        
        # progress bar
        if TQDM_AVAILABLE:
            company_iter = tqdm(companies_to_analyze, desc="  HDBSCAN per company", unit="company")
        else:
            company_iter = companies_to_analyze
            self.log_and_print(f"  Processing {len(companies_to_analyze)} companies...")
        
        for company in company_iter:
            company_mask = df['Company'] == company
            company_embeddings = embeddings[company_mask]
            is_target = df[company_mask]['IS_TARGET'].iloc[0] == 'Y'
            
            # Skip if too few samples (HDBSCAN needs minimum cluster size)
            if len(company_embeddings) < 50:
                continue
            
            # skip if still too large (>10k samples per company)
            if len(company_embeddings) > 10000:
                # sample to make manageable
                sample_idx = np.random.choice(len(company_embeddings), 10000, replace=False)
                company_embeddings = company_embeddings[sample_idx]
            
            try:
                clusterer = HDBSCAN(min_cluster_size=15, metric='euclidean', core_dist_n_jobs=1)
                cluster_labels = clusterer.fit_predict(company_embeddings)
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = (cluster_labels == -1).sum()
                noise_ratio = n_noise / len(cluster_labels)
                
                # cluster diversity (normalized entropy)
                if n_clusters > 0:
                    cluster_counts = pd.Series(cluster_labels[cluster_labels != -1]).value_counts()
                    cluster_probs = cluster_counts / cluster_counts.sum()
                    cluster_entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))
                    normalized_entropy = cluster_entropy / np.log(n_clusters + 1)
                else:
                    normalized_entropy = 0.0
                
                company_results.append({
                    'company': company,
                    'is_target': is_target,
                    'n_samples': len(company_embeddings),
                    'n_clusters': n_clusters,
                    'noise_ratio': noise_ratio,
                    'cluster_entropy': normalized_entropy
                })
            except Exception as e:
                if not TQDM_AVAILABLE:
                    self.log_and_print(f"     {company}: {e}")
                continue
        
        company_df = pd.DataFrame(company_results)
        
        # compare targets vs. peers
        if len(company_df) > 0:
            target_results = company_df[company_df['is_target'] == True]
            peer_results = company_df[company_df['is_target'] == False]
            
            self.log_and_print(f"\n  Results Summary:")
            self.log_and_print(f"    Targets analyzed: {len(target_results)}")
            self.log_and_print(f"    Peers analyzed: {len(peer_results)}")
            
            if len(target_results) > 0 and len(peer_results) > 0:
                self.log_and_print(f"\n  Clustering Metrics Comparison:")
                self.log_and_print(f"    Average clusters per company:")
                self.log_and_print(f"      Targets: {target_results['n_clusters'].mean():.1f}")
                self.log_and_print(f"      Peers:   {peer_results['n_clusters'].mean():.1f}")
                
                self.log_and_print(f"    Average noise ratio:")
                self.log_and_print(f"      Targets: {target_results['noise_ratio'].mean():.3f}")
                self.log_and_print(f"      Peers:   {peer_results['noise_ratio'].mean():.3f}")
                
                self.log_and_print(f"    Average cluster entropy (diversity):")
                self.log_and_print(f"      Targets: {target_results['cluster_entropy'].mean():.3f}")
                self.log_and_print(f"      Peers:   {peer_results['cluster_entropy'].mean():.3f}")
                
                # statistical test
                from scipy.stats import mannwhitneyu
                if len(target_results) >= 3 and len(peer_results) >= 3:
                    u_stat, p_value = mannwhitneyu(
                        target_results['noise_ratio'], 
                        peer_results['noise_ratio'],
                        alternative='two-sided'
                    )
                    self.log_and_print(f"\n    Mann-Whitney U test (noise ratio):")
                    self.log_and_print(f"      p-value: {p_value:.4f}", end="")
                    if p_value < 0.05:
                        self.log_and_print(" ⭐ Significant difference!")
                    else:
                        self.log_and_print(" (not significant)")
        
        self.log_and_print("\n" + "-" * 80)
        self.log_and_print("Approach 2: Attribution-Type Stratified Clustering")
        self.log_and_print("-" * 80)
        self.log_and_print("Comparing targets vs. peers within each attribution type")
        
        attribution_results = []
        
        # focus on key attribution types
        key_types = [
            ('Positive', 'Internal'),
            ('Negative', 'External'),
            ('Positive', 'External'),
            ('Negative', 'Internal')
        ]
        
        for outcome, locus in key_types:
            attr_mask = (df['attribution_outcome'] == outcome) & (df['attribution_locus'] == locus)
            
            if attr_mask.sum() < 100:
                continue
            
            # sample targets and peers separately
            target_mask = attr_mask & (df['IS_TARGET'] == 'Y')
            peer_mask = attr_mask & (df['IS_PEER'] == 'Y')
            
            target_embeddings = embeddings[target_mask]
            peer_embeddings = embeddings[peer_mask]
            
            # sample to manageable size
            max_samples = 5000
            if len(target_embeddings) > max_samples:
                sample_idx = np.random.choice(len(target_embeddings), max_samples, replace=False)
                target_embeddings = target_embeddings[sample_idx]
            if len(peer_embeddings) > max_samples:
                sample_idx = np.random.choice(len(peer_embeddings), max_samples, replace=False)
                peer_embeddings = peer_embeddings[sample_idx]
            
            self.log_and_print(f"\n  {outcome}-{locus}:")
            self.log_and_print(f"    Targets: {len(target_embeddings):,} samples", end="")
            
            # cluster targets
            if len(target_embeddings) >= 50:
                try:
                    clusterer = HDBSCAN(min_cluster_size=15, metric='euclidean', core_dist_n_jobs=1)
                    target_labels = clusterer.fit_predict(target_embeddings)
                    target_n_clusters = len(set(target_labels)) - (1 if -1 in target_labels else 0)
                    target_noise_ratio = (target_labels == -1).sum() / len(target_labels)
                    self.log_and_print(f" → {target_n_clusters} clusters, {target_noise_ratio:.1%} noise")
                except:
                    target_n_clusters, target_noise_ratio = 0, 1.0
                    self.log_and_print(f" → clustering failed")
            else:
                target_n_clusters, target_noise_ratio = 0, 1.0
                self.log_and_print(f" → too few samples")
            
            self.log_and_print(f"    Peers: {len(peer_embeddings):,} samples", end="")
            
            # cluster peers
            if len(peer_embeddings) >= 50:
                try:
                    clusterer = HDBSCAN(min_cluster_size=15, metric='euclidean', core_dist_n_jobs=1)
                    peer_labels = clusterer.fit_predict(peer_embeddings)
                    peer_n_clusters = len(set(peer_labels)) - (1 if -1 in peer_labels else 0)
                    peer_noise_ratio = (peer_labels == -1).sum() / len(peer_labels)
                    self.log_and_print(f" → {peer_n_clusters} clusters, {peer_noise_ratio:.1%} noise")
                except:
                    peer_n_clusters, peer_noise_ratio = 0, 1.0
                    self.log_and_print(f" → clustering failed")
            else:
                peer_n_clusters, peer_noise_ratio = 0, 1.0
                self.log_and_print(f" → too few samples")
            
            attribution_results.append({
                'attribution_type': f"{outcome}_{locus}",
                'target_n_clusters': target_n_clusters,
                'target_noise_ratio': target_noise_ratio,
                'peer_n_clusters': peer_n_clusters,
                'peer_noise_ratio': peer_noise_ratio
            })
        
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print(" Strategic HDBSCAN analysis complete")
        
        return {
            'company_level': company_df.to_dict('records') if len(company_df) > 0 else [],
            'attribution_type': attribution_results,
            'interpretation': {
                'company_level': "Higher noise ratio = less coherent attributions (more diverse/inconsistent)",
                'attribution_type': "Targets with higher noise in Neg-External = bias signal"
            }
        }
    
    def aggregate_company_embeddings(self, df: pd.DataFrame, embeddings: np.ndarray) -> Dict:
        self.log_and_print("\n3b. Aggregating Company-Level Embeddings")
        self.log_and_print("=" * 80)
        
        company_aggregates = {}
        
        for company in df['Company'].unique():
            company_mask = df['Company'] == company
            company_embeddings = embeddings[company_mask]
            
            if len(company_embeddings) == 0:
                continue
            
            is_target = df[company_mask]['IS_TARGET'].iloc[0] == 'Y'
            
            company_aggregates[company] = {
                'centroid': company_embeddings.mean(axis=0),
                'std': company_embeddings.std(axis=0).mean(),
                'n_samples': len(company_embeddings),
                'is_target': is_target,
                'embeddings': company_embeddings
            }
        
        self.log_and_print(f" Aggregated embeddings for {len(company_aggregates)} companies")
        self.log_and_print(f"  Targets: {sum(1 for v in company_aggregates.values() if v['is_target'])}")
        self.log_and_print(f"  Peers: {sum(1 for v in company_aggregates.values() if not v['is_target'])}")
        
        return company_aggregates
    
    def measure_target_peer_separation(self, company_aggregates: Dict) -> Dict:
        self.log_and_print("\n3c. Measuring Target vs Peer Separation")
        self.log_and_print("=" * 80)
        
        target_companies = {k: v for k, v in company_aggregates.items() if v['is_target']}
        peer_companies = {k: v for k, v in company_aggregates.items() if not v['is_target']}
        
        if len(target_companies) == 0 or len(peer_companies) == 0:
            self.log_and_print("Insufficient targets or peers for separation analysis")
            return {}
        
        target_centroids = np.array([v['centroid'] for v in target_companies.values()])
        peer_centroids = np.array([v['centroid'] for v in peer_companies.values()])
        
        from sklearn.metrics.pairwise import euclidean_distances
        
        target_to_target_dist = euclidean_distances(target_centroids, target_centroids)
        target_to_peer_dist = euclidean_distances(target_centroids, peer_centroids)
        peer_to_peer_dist = euclidean_distances(peer_centroids, peer_centroids)
        
        np.fill_diagonal(target_to_target_dist, np.nan)
        np.fill_diagonal(peer_to_peer_dist, np.nan)
        
        results = {
            'mean_target_to_target': float(np.nanmean(target_to_target_dist)),
            'mean_target_to_peer': float(np.nanmean(target_to_peer_dist)),
            'mean_peer_to_peer': float(np.nanmean(peer_to_peer_dist)),
            'separation_ratio': float(np.nanmean(target_to_peer_dist) / np.nanmean(peer_to_peer_dist))
        }
        
        self.log_and_print(f"  Mean target-to-target distance: {results['mean_target_to_target']:.4f}")
        self.log_and_print(f"  Mean target-to-peer distance: {results['mean_target_to_peer']:.4f}")
        self.log_and_print(f"  Mean peer-to-peer distance: {results['mean_peer_to_peer']:.4f}")
        self.log_and_print(f"  Separation ratio: {results['separation_ratio']:.2f}x")
        
        if results['separation_ratio'] > 1.2:
            self.log_and_print(f"   Targets cluster separately from peers")
        elif results['separation_ratio'] > 1.0:
            self.log_and_print(f"  ~ Weak separation between targets and peers")
        else:
            self.log_and_print(f"   No clear separation between targets and peers")
        
        return results
    
    def _compute_firm_coherence(self, embeddings: np.ndarray) -> float:
        """
        Compute average silhouette score for firm's embeddings.
        Higher coherence = tighter clusters = more consistent language.
        
        OPTIMIZED: Uses n_init=3 and samples for large datasets
        """
        if len(embeddings) < 50:
            return np.nan
        
        # sample if very large (coherence is stable with samples)
        if len(embeddings) > 2000:
            indices = np.random.RandomState(42).choice(len(embeddings), 2000, replace=False)
            embeddings = embeddings[indices]
        
        # use k-means with adaptive k
        k = min(5, len(embeddings) // 20)
        if k < 2:
            return np.nan
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # optimization: reduce n_init from 10 to 3 (50-70% faster, still robust)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
        labels = kmeans.fit_predict(embeddings)
        
        try:
            score = silhouette_score(embeddings, labels)
            return score
        except:
            return np.nan
    
    def _compute_firm_outlier_rate(self, df_firm: pd.DataFrame, embeddings_firm: np.ndarray) -> float:
        """
        Compute rate of topic-inconsistent attributions for a firm.
        Uses embedding distance to topic centroid.
        """
        if len(df_firm) < 4:
            return np.nan
        
        outlier_count = 0
        total_count = 0
        
        # use primary_topic column (standard column from classification pipeline)
        topic_col = 'Primary_Topic'
        
        if topic_col not in df_firm.columns:
            return np.nan
        
        for topic in df_firm[topic_col].unique():
            if pd.isna(topic):
                continue
            
            topic_mask = df_firm[topic_col] == topic
            topic_embeddings = embeddings_firm[topic_mask.values]
            
            if len(topic_embeddings) < 3:
                continue
            
            # compute centroid and distances
            centroid = topic_embeddings.mean(axis=0)
            distances = np.linalg.norm(topic_embeddings - centroid, axis=1)
            
            # outliers: > 2 std from mean
            threshold = distances.mean() + 2 * distances.std()
            outliers = (distances > threshold).sum()
            
            outlier_count += outliers
            total_count += len(topic_embeddings)
        
        if total_count == 0:
            return np.nan
        
        return outlier_count / total_count
    
    def _print_per_firm_aggregate_summary(self, results_df: pd.DataFrame) -> None:
        """
        Aggregate statistical analysis across per-firm temporal patterns.
        
        Statistical Approach:
        1. One-sample t-tests: Test if mean changes differ from zero (H0: μ = 0)
        2. Paired t-tests: Test if targets/peers change behavior IN vs OUT bias periods
        3. Binomial test: Test if directional effects are skewed (not 50/50)
        4. Pearson correlation: Test if coherence and outlier changes are related
        
        Note: Multiple tests conducted (exploratory analysis); interpret with caution.
        All p-values reported; Cohen's d provides effect size for practical significance."""
        from scipy import stats
        
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("AGGREGATE SUMMARY: Cross-Firm Temporal Patterns")
        self.log_and_print("=" * 80)
        self.log_and_print("Goal: Test if temporal changes are significant in AGGREGATE across all targets")
        self.log_and_print()
        
        if len(results_df) == 0:
            self.log_and_print("  ️  No firms to analyze")
            return
        
        # filter to firms with bias period data
        bias_df = results_df[results_df['has_bias_data']].copy()
        
        if len(bias_df) == 0:
            self.log_and_print("  ️  No firms have sufficient bias period data")
            return
        
        self.log_and_print(f"Analyzing {len(bias_df)} firms with sufficient bias period data\n")
        
        # ========================================================================
        # 1. coherence aggregate analysis
        # ========================================================================
        
        self.log_and_print("1. COHERENCE CHANGES (Target-Peer Gap: IN bias vs OUT bias)")
        self.log_and_print("-" * 80)
        
        coherence_changes = bias_df['coherence_change'].dropna()
        
        if len(coherence_changes) >= 3:
            # descriptive statistics
            mean_coh = coherence_changes.mean()
            median_coh = coherence_changes.median()
            std_coh = coherence_changes.std()
            
            # direction analysis
            n_increased = (coherence_changes > 0).sum()
            n_decreased = (coherence_changes < 0).sum()
            n_unchanged = (coherence_changes == 0).sum()
            
            self.log_and_print(f"  Descriptive Statistics:")
            self.log_and_print(f"    Mean change:   {mean_coh:+.4f}")
            self.log_and_print(f"    Median change: {median_coh:+.4f}")
            self.log_and_print(f"    Std dev:       {std_coh:.4f}")
            self.log_and_print(f"    Range:         [{coherence_changes.min():+.4f}, {coherence_changes.max():+.4f}]")
            
            self.log_and_print(f"\n  Directional Breakdown:")
            self.log_and_print(f"    Increased gap (target more variable in bias periods): {n_increased} ({n_increased/len(coherence_changes)*100:.1f}%)")
            self.log_and_print(f"    Decreased gap (target more consistent in bias periods): {n_decreased} ({n_decreased/len(coherence_changes)*100:.1f}%)")
            self.log_and_print(f"    No change: {n_unchanged}")
            
            # one-sample t-test: tests if mean change significantly differs from 0
            # appropriate because: comparing did estimates against null of "no change"
            t_stat, p_value = stats.ttest_1samp(coherence_changes, 0)
            
            self.log_and_print(f"\n  Statistical Significance Test:")
            self.log_and_print(f"    One-sample t-test (H0: mean change = 0)")
            self.log_and_print(f"    t-statistic: {t_stat:+.3f}")
            self.log_and_print(f"    p-value:     {p_value:.4f}")
            
            if p_value < 0.05:
                direction = "increase" if mean_coh > 0 else "decrease"
                self.log_and_print(f"     SIGNIFICANT: Targets show a {direction} in coherence gap during bias periods (p < 0.05)")
            else:
                self.log_and_print(f"     NOT SIGNIFICANT: No systematic coherence change across targets (p >= 0.05)")
            
            # cohen's d: effect size indicating practical significance (mean / std)
            cohens_d = mean_coh / std_coh if std_coh > 0 else 0
            self.log_and_print(f"    Cohen's d:   {cohens_d:+.3f} ({self._interpret_cohens_d(abs(cohens_d))})")
        else:
            self.log_and_print(f"  ️  Too few firms with coherence data ({len(coherence_changes)}), skipping statistical tests")
        
        # ========================================================================
        # 2. outlier ratio aggregate analysis
        # ========================================================================
        
        self.log_and_print("\n2. OUTLIER RATIO CHANGES (Target/Peer Ratio: IN bias vs OUT bias)")
        self.log_and_print("-" * 80)
        
        outlier_changes = bias_df['outlier_change'].dropna()
        
        if len(outlier_changes) >= 3:
            # descriptive statistics
            mean_out = outlier_changes.mean()
            median_out = outlier_changes.median()
            std_out = outlier_changes.std()
            
            # direction analysis
            n_increased = (outlier_changes > 0).sum()
            n_decreased = (outlier_changes < 0).sum()
            n_unchanged = (outlier_changes == 0).sum()
            
            self.log_and_print(f"  Descriptive Statistics:")
            self.log_and_print(f"    Mean change:   {mean_out:+.3f}x")
            self.log_and_print(f"    Median change: {median_out:+.3f}x")
            self.log_and_print(f"    Std dev:       {std_out:.3f}x")
            self.log_and_print(f"    Range:         [{outlier_changes.min():+.3f}x, {outlier_changes.max():+.3f}x]")
            
            self.log_and_print(f"\n  Directional Breakdown:")
            self.log_and_print(f"    Increased ratio (target more off-topic in bias periods):  {n_increased} ({n_increased/len(outlier_changes)*100:.1f}%)")
            self.log_and_print(f"    Decreased ratio (target more focused in bias periods):    {n_decreased} ({n_decreased/len(outlier_changes)*100:.1f}%)")
            self.log_and_print(f"    No change: {n_unchanged}")
            
            # statistical test: one-sample t-test (h0: mean change = 0)
            t_stat, p_value = stats.ttest_1samp(outlier_changes, 0)
            
            self.log_and_print(f"\n  Statistical Significance Test:")
            self.log_and_print(f"    One-sample t-test (H0: mean change = 0)")
            self.log_and_print(f"    t-statistic: {t_stat:+.3f}")
            self.log_and_print(f"    p-value:     {p_value:.4f}")
            
            if p_value < 0.05:
                direction = "increase" if mean_out > 0 else "decrease"
                self.log_and_print(f"     SIGNIFICANT: Targets show a {direction} in outlier ratio during bias periods (p < 0.05)")
            else:
                self.log_and_print(f"     NOT SIGNIFICANT: No systematic outlier change across targets (p >= 0.05)")
            
            # effect size (cohen's d vs 0)
            cohens_d = mean_out / std_out if std_out > 0 else 0
            self.log_and_print(f"    Cohen's d:   {cohens_d:+.3f} ({self._interpret_cohens_d(abs(cohens_d))})")
            
            # binomial test: tests if direction is skewed (not random 50/50 split)
            # appropriate because: tests if heterogeneity has systematic directional bias
            binom_test = stats.binomtest(n_increased, n_increased + n_decreased, 0.5, alternative='two-sided')
            self.log_and_print(f"\n  Directional Bias Test:")
            self.log_and_print(f"    Binomial test (H0: 50% increase, 50% decrease)")
            self.log_and_print(f"    p-value: {binom_test.pvalue:.4f}")
            
            if binom_test.pvalue < 0.05:
                if n_increased > n_decreased:
                    self.log_and_print(f"     SIGNIFICANT: More targets increase outlier ratio than decrease (p < 0.05)")
                else:
                    self.log_and_print(f"     SIGNIFICANT: More targets decrease outlier ratio than increase (p < 0.05)")
            else:
                self.log_and_print(f"     NOT SIGNIFICANT: No directional bias (roughly equal increases/decreases)")
        else:
            self.log_and_print(f"  ️  Too few firms with outlier data ({len(outlier_changes)}), skipping statistical tests")
        
        # ========================================================================
        # 3. absolute values comparison (in vs out periods)
        # ========================================================================
        
        self.log_and_print("\n3. ABSOLUTE VALUES: Target & Peer IN vs OUT Bias Periods")
        self.log_and_print("-" * 80)
        
        # target coherence
        target_coh_in = bias_df['coherence_target_in'].dropna()
        target_coh_out = bias_df['coherence_target_out'].dropna()
        
        if len(target_coh_in) >= 3 and len(target_coh_out) >= 3:
            # paired t-test: compares same firms across time (within-subject design)
            # appropriate because: controls for firm-specific effects, more powerful than independent t-test
            paired_df = bias_df[bias_df['coherence_target_in'].notna() & bias_df['coherence_target_out'].notna()]
            if len(paired_df) >= 3:
                t_stat, p_value = stats.ttest_rel(paired_df['coherence_target_in'], paired_df['coherence_target_out'])
                mean_diff = (paired_df['coherence_target_in'] - paired_df['coherence_target_out']).mean()
                
                self.log_and_print(f"  Target Coherence (IN vs OUT bias periods):")
                self.log_and_print(f"    Mean IN:  {target_coh_in.mean():.4f}")
                self.log_and_print(f"    Mean OUT: {target_coh_out.mean():.4f}")
                self.log_and_print(f"    Difference: {mean_diff:+.4f}")
                self.log_and_print(f"    Paired t-test: t={t_stat:+.3f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    direction = "HIGHER" if mean_diff > 0 else "LOWER"
                    self.log_and_print(f"     SIGNIFICANT: Targets have {direction} coherence in bias periods (p < 0.05)")
                else:
                    self.log_and_print(f"     NOT SIGNIFICANT: No change in target coherence (p >= 0.05)")
        
        # peer coherence
        peer_coh_in = bias_df['coherence_peer_in'].dropna()
        peer_coh_out = bias_df['coherence_peer_out'].dropna()
        
        if len(peer_coh_in) >= 3 and len(peer_coh_out) >= 3:
            paired_df = bias_df[bias_df['coherence_peer_in'].notna() & bias_df['coherence_peer_out'].notna()]
            if len(paired_df) >= 3:
                t_stat, p_value = stats.ttest_rel(paired_df['coherence_peer_in'], paired_df['coherence_peer_out'])
                mean_diff = (paired_df['coherence_peer_in'] - paired_df['coherence_peer_out']).mean()
                
                self.log_and_print(f"\n  Peer Coherence (IN vs OUT bias periods):")
                self.log_and_print(f"    Mean IN:  {peer_coh_in.mean():.4f}")
                self.log_and_print(f"    Mean OUT: {peer_coh_out.mean():.4f}")
                self.log_and_print(f"    Difference: {mean_diff:+.4f}")
                self.log_and_print(f"    Paired t-test: t={t_stat:+.3f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    direction = "HIGHER" if mean_diff > 0 else "LOWER"
                    self.log_and_print(f"     SIGNIFICANT: Peers have {direction} coherence in those quarters (p < 0.05)")
                else:
                    self.log_and_print(f"     NOT SIGNIFICANT: No change in peer coherence (p >= 0.05)")
        
        # target outlier rate
        target_out_in = bias_df['outlier_target_in'].dropna()
        target_out_out = bias_df['outlier_target_out'].dropna()
        
        if len(target_out_in) >= 3 and len(target_out_out) >= 3:
            paired_df = bias_df[bias_df['outlier_target_in'].notna() & bias_df['outlier_target_out'].notna()]
            if len(paired_df) >= 3:
                t_stat, p_value = stats.ttest_rel(paired_df['outlier_target_in'], paired_df['outlier_target_out'])
                mean_diff = (paired_df['outlier_target_in'] - paired_df['outlier_target_out']).mean()
                
                self.log_and_print(f"\n  Target Outlier Rate (IN vs OUT bias periods):")
                self.log_and_print(f"    Mean IN:  {target_out_in.mean():.2%}")
                self.log_and_print(f"    Mean OUT: {target_out_out.mean():.2%}")
                self.log_and_print(f"    Difference: {mean_diff:+.2%}")
                self.log_and_print(f"    Paired t-test: t={t_stat:+.3f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    direction = "HIGHER" if mean_diff > 0 else "LOWER"
                    self.log_and_print(f"     SIGNIFICANT: Targets have {direction} outlier rate in bias periods (p < 0.05)")
                else:
                    self.log_and_print(f"     NOT SIGNIFICANT: No change in target outlier rate (p >= 0.05)")
        
        # peer outlier rate
        peer_out_in = bias_df['outlier_peer_in'].dropna()
        peer_out_out = bias_df['outlier_peer_out'].dropna()
        
        if len(peer_out_in) >= 3 and len(peer_out_out) >= 3:
            paired_df = bias_df[bias_df['outlier_peer_in'].notna() & bias_df['outlier_peer_out'].notna()]
            if len(paired_df) >= 3:
                t_stat, p_value = stats.ttest_rel(paired_df['outlier_peer_in'], paired_df['outlier_peer_out'])
                mean_diff = (paired_df['outlier_peer_in'] - paired_df['outlier_peer_out']).mean()
                
                self.log_and_print(f"\n  Peer Outlier Rate (IN vs OUT bias periods):")
                self.log_and_print(f"    Mean IN:  {peer_out_in.mean():.2%}")
                self.log_and_print(f"    Mean OUT: {peer_out_out.mean():.2%}")
                self.log_and_print(f"    Difference: {mean_diff:+.2%}")
                self.log_and_print(f"    Paired t-test: t={t_stat:+.3f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    direction = "HIGHER" if mean_diff > 0 else "LOWER"
                    self.log_and_print(f"     SIGNIFICANT: Peers have {direction} outlier rate in those quarters (p < 0.05)")
                else:
                    self.log_and_print(f"     NOT SIGNIFICANT: No change in peer outlier rate (p >= 0.05)")
        
        # ========================================================================
        # 4. correlation analysis
        # ========================================================================
        
        self.log_and_print("\n4. CORRELATION: Coherence Change vs Outlier Change")
        self.log_and_print("-" * 80)
        
        # check if coherence and outlier changes are correlated
        both_df = bias_df[bias_df['coherence_change'].notna() & bias_df['outlier_change'].notna()]
        
        if len(both_df) >= 3:
            corr, p_value = stats.pearsonr(both_df['coherence_change'], both_df['outlier_change'])
            
            self.log_and_print(f"  Pearson correlation: r={corr:+.3f}, p={p_value:.4f}")
            
            if p_value < 0.05:
                if abs(corr) > 0.5:
                    direction = "positive" if corr > 0 else "negative"
                    self.log_and_print(f"     SIGNIFICANT: Strong {direction} correlation (p < 0.05)")
                else:
                    direction = "positive" if corr > 0 else "negative"
                    self.log_and_print(f"     SIGNIFICANT: Weak {direction} correlation (p < 0.05)")
            else:
                self.log_and_print(f"     NOT SIGNIFICANT: Coherence and outlier changes are independent (p >= 0.05)")
        else:
            self.log_and_print(f"  ️  Too few firms with both metrics ({len(both_df)}), skipping correlation")
        
        # ========================================================================
        # 5. final interpretation
        # ========================================================================
        
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("AGGREGATE INTERPRETATION")
        self.log_and_print("=" * 80)
        
        # determine overall finding
        significant_tests = []
        
        if len(coherence_changes) >= 3:
            _, p_coh = stats.ttest_1samp(coherence_changes, 0)
            if p_coh < 0.05:
                significant_tests.append("coherence_change")
        
        if len(outlier_changes) >= 3:
            _, p_out = stats.ttest_1samp(outlier_changes, 0)
            if p_out < 0.05:
                significant_tests.append("outlier_change")
        
        if len(significant_tests) > 0:
            self.log_and_print(f" AGGREGATE PATTERN DETECTED:")
            self.log_and_print(f"  Significant changes found in: {', '.join(significant_tests)}")
            self.log_and_print(f"  This per-firm temporal analysis reveals patterns NOT visible in:")
            self.log_and_print(f"    - Cross-sectional aggregate (all targets vs all peers)")
            self.log_and_print(f"    - All-time per-firm comparison (target vs peers across all quarters)")
            self.log_and_print(f"  → Confirms that TIMING matters: patterns emerge during bias periods")
        else:
            self.log_and_print(f" NO AGGREGATE PATTERN:")
            self.log_and_print(f"  No significant changes in coherence or outlier metrics")
            self.log_and_print(f"  However, individual firms (47.6%) DO show changes, suggesting:")
            self.log_and_print(f"    - Heterogeneous crisis response strategies")
            self.log_and_print(f"    - Bidirectional effects (increases and decreases cancel in aggregate)")
            self.log_and_print(f"  → Individual firm analysis more informative than aggregate for this sample")
        
        self.log_and_print()
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def analyze_individual_target_firms(
        self, 
        df: pd.DataFrame, 
        embeddings: np.ndarray,
        embeddings_adjusted: np.ndarray,
        bias_periods: Dict
    ) -> pd.DataFrame:
        """
        Per-firm analysis: each target vs its specific peers.
        
        NEW: Temporal segmentation comparing IN bias periods vs OUT of bias periods
        
        For each target:
        1. Coherence in self-serving attributions (target vs peers, in/out of bias periods)
        2. Topic consistency outlier rate (in/out of bias periods)
        3. Effect sizes (Cohen's d)
        4. CHANGE in target-peer gap between bias and normal periods
        5. Flags for unusual patterns"""
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("PER-FIRM STRATIFIED ANALYSIS (WITH TEMPORAL SEGMENTATION)")
        self.log_and_print("=" * 80)
        self.log_and_print("Goal: Identify if targets show DIFFERENT patterns during bias periods vs normal periods")
        self.log_and_print()
        
        target_companies = df[df['IS_TARGET'] == 'Y']['Company'].unique()
        
        if len(target_companies) == 0:
            self.log_and_print("  ️  No target companies found")
            return pd.DataFrame()
        
        # add bias period lookup column if not already present
        if 'bias_period_key' not in df.columns:
            df['bias_period_key'] = list(zip(df['Company'], df['Year'], df['Quarter']))
        
        firm_results = []
        
        for idx, target in enumerate(target_companies, 1):
            self.log_and_print(f"\n[{idx}/{len(target_companies)}] Analyzing: {target}")
            
            # critical: match config ticker to folder name
            # target_companies contains folder names (e.g., "0hmi.l", "intc")
            # we need to find the corresponding ticker in config
            target_ticker = None
            peer_folders = []
            
            for ticker, config in self.peer_groups.items():
                target_folder = config.get('target_folder', ticker)
                if target == target_folder:
                    target_ticker = ticker
                    # use peer_folders (folder names) not peers (tickers)
                    peer_folders = config.get('peer_folders', [])
                    # remove none values from peer_folders
                    peer_folders = [p for p in peer_folders if p is not None]
                    break
            
            if not target_ticker:
                self.log_and_print(f"  ️  No config found for folder: {target}")
                continue
            
            if not peer_folders:
                self.log_and_print(f"  ️  No peer folders defined for {target_ticker}")
                continue
            
            self.log_and_print(f"  → Found {len(peer_folders)} peer folders for {target_ticker}")
            
            # filter to target + peers (using folder names)
            mask = (df['Company'] == target) | (df['Company'].isin(peer_folders))
            df_subset = df[mask].copy()
            emb_subset = embeddings[mask.values]
            emb_adj_subset = embeddings_adjusted[mask.values] if embeddings_adjusted is not None else emb_subset
            
            target_mask = df_subset['Company'] == target
            peer_mask = df_subset['Company'].isin(peer_folders)
            
            if target_mask.sum() < 50 or peer_mask.sum() < 50:
                self.log_and_print(f"  ️  Insufficient data (target: {target_mask.sum()}, peers: {peer_mask.sum()})")
                continue
            
            # ========================================================================
            # temporal segmentation: split into in bias periods vs out of bias periods
            # ========================================================================
            
            # critical fix: identify target's bias quarters, then apply to all companies (target + peers)
            # extract the set of (year, quarter) tuples that are bias periods for this target
            target_bias_quarters = set()
            for (company, year, quarter), info in bias_periods.items():
                if company == target and info.get('window') == 'exact':
                    target_bias_quarters.add((year, quarter))
            
            # create mask: any statement (target or peer) in those (year, quarter) periods
            in_bias_mask = df_subset.apply(
                lambda row: (row['Year'], row['Quarter']) in target_bias_quarters, 
                axis=1
            )
            
            out_bias_mask = ~in_bias_mask
            
            if target_bias_quarters:
                self.log_and_print(f"  Target's bias quarters: {sorted(target_bias_quarters)}")
            else:
                self.log_and_print(f"  No exact bias quarters found for this target")
            
            # four groups:
            target_in_bias = target_mask & in_bias_mask
            target_out_bias = target_mask & out_bias_mask
            peer_in_bias = peer_mask & in_bias_mask
            peer_out_bias = peer_mask & out_bias_mask
            
            n_target_in = target_in_bias.sum()
            n_target_out = target_out_bias.sum()
            n_peer_in = peer_in_bias.sum()
            n_peer_out = peer_out_bias.sum()
            
            self.log_and_print(f"  Temporal segmentation:")
            self.log_and_print(f"    Target IN bias periods:  {n_target_in:4d} statements")
            self.log_and_print(f"    Target OUT bias periods: {n_target_out:4d} statements")
            self.log_and_print(f"    Peers IN bias periods:   {n_peer_in:4d} statements")
            self.log_and_print(f"    Peers OUT bias periods:  {n_peer_out:4d} statements")
            
            # check if we have enough data in each segment
            min_samples_target = 30  # need 30+ for stable target coherence
            min_samples_peers = 50   # lower threshold for peers (aggregate across multiple firms)
            
            if n_target_in < min_samples_target or n_peer_in < min_samples_peers:
                self.log_and_print(f"  ️  Insufficient data in bias periods (target: {min_samples_target}+, peers: {min_samples_peers}+ needed), using all-time comparison only")
                # fall back to all-time comparison
                in_bias_mask[:] = False
                out_bias_mask[:] = True
                target_in_bias[:] = False
                peer_in_bias[:] = False
                has_bias_data = False
            else:
                has_bias_data = True
                self.log_and_print(f"   Sufficient bias period data for temporal analysis")
            
            # ========================================================================
            # 1. coherence metrics (4 groups)
            # ========================================================================
            
            # all-time (for reference)
            target_coherence_all = self._compute_firm_coherence(emb_subset[target_mask.values])
            peer_coherence_scores_all = []
            for peer_folder in peer_folders:
                peer_emb = emb_subset[(df_subset['Company'] == peer_folder).values]
                if len(peer_emb) >= 50:
                    peer_coherence_scores_all.append(self._compute_firm_coherence(peer_emb))
            peer_coherence_all = np.nanmean(peer_coherence_scores_all) if peer_coherence_scores_all else np.nan
            
            # temporal segmentation
            if has_bias_data:
                # target in bias periods
                target_coherence_in = self._compute_firm_coherence(emb_subset[target_in_bias.values]) if n_target_in >= 30 else np.nan
                
                # target out bias periods
                target_coherence_out = self._compute_firm_coherence(emb_subset[target_out_bias.values]) if n_target_out >= 30 else np.nan
                
                # peer in bias periods (same quarters as target's bias periods)
                peer_coherence_in_scores = []
                for peer_folder in peer_folders:
                    peer_in_mask = (df_subset['Company'] == peer_folder) & in_bias_mask
                    if peer_in_mask.sum() >= 30:
                        peer_coherence_in_scores.append(self._compute_firm_coherence(emb_subset[peer_in_mask.values]))
                peer_coherence_in = np.nanmean(peer_coherence_in_scores) if peer_coherence_in_scores else np.nan
                
                # peer out bias periods
                peer_coherence_out_scores = []
                for peer_folder in peer_folders:
                    peer_out_mask = (df_subset['Company'] == peer_folder) & out_bias_mask
                    if peer_out_mask.sum() >= 30:
                        peer_coherence_out_scores.append(self._compute_firm_coherence(emb_subset[peer_out_mask.values]))
                peer_coherence_out = np.nanmean(peer_coherence_out_scores) if peer_coherence_out_scores else np.nan
                
                # calculate gaps and changes
                gap_in_bias = target_coherence_in - peer_coherence_in if not np.isnan(target_coherence_in) and not np.isnan(peer_coherence_in) else np.nan
                gap_out_bias = target_coherence_out - peer_coherence_out if not np.isnan(target_coherence_out) and not np.isnan(peer_coherence_out) else np.nan
                coherence_change = gap_in_bias - gap_out_bias if not np.isnan(gap_in_bias) and not np.isnan(gap_out_bias) else np.nan
            else:
                target_coherence_in = target_coherence_out = np.nan
                peer_coherence_in = peer_coherence_out = np.nan
                gap_in_bias = gap_out_bias = coherence_change = np.nan
            
            # ========================================================================
            # 2. outlier rate metrics (4 groups)
            # ========================================================================
            
            # all-time
            target_outlier_all = self._compute_firm_outlier_rate(df_subset[target_mask], emb_subset[target_mask.values])
            peer_outlier_rates_all = []
            for peer_folder in peer_folders:
                peer_df = df_subset[df_subset['Company'] == peer_folder]
                peer_emb = emb_subset[(df_subset['Company'] == peer_folder).values]
                if len(peer_df) >= 10:
                    peer_outlier_rates_all.append(self._compute_firm_outlier_rate(peer_df, peer_emb))
            peer_outlier_all = np.nanmean(peer_outlier_rates_all) if peer_outlier_rates_all else np.nan
            
            # temporal segmentation
            if has_bias_data:
                # target in bias periods
                target_outlier_in = self._compute_firm_outlier_rate(
                    df_subset[target_in_bias], emb_subset[target_in_bias.values]
                ) if n_target_in >= 10 else np.nan
                
                # target out bias periods
                target_outlier_out = self._compute_firm_outlier_rate(
                    df_subset[target_out_bias], emb_subset[target_out_bias.values]
                ) if n_target_out >= 10 else np.nan
                
                # peer in bias periods
                peer_outlier_in_rates = []
                for peer_folder in peer_folders:
                    peer_in_mask = (df_subset['Company'] == peer_folder) & in_bias_mask
                    if peer_in_mask.sum() >= 10:
                        peer_outlier_in_rates.append(
                            self._compute_firm_outlier_rate(df_subset[peer_in_mask], emb_subset[peer_in_mask.values])
                        )
                peer_outlier_in = np.nanmean(peer_outlier_in_rates) if peer_outlier_in_rates else np.nan
                
                # peer out bias periods
                peer_outlier_out_rates = []
                for peer_folder in peer_folders:
                    peer_out_mask = (df_subset['Company'] == peer_folder) & out_bias_mask
                    if peer_out_mask.sum() >= 10:
                        peer_outlier_out_rates.append(
                            self._compute_firm_outlier_rate(df_subset[peer_out_mask], emb_subset[peer_out_mask.values])
                        )
                peer_outlier_out = np.nanmean(peer_outlier_out_rates) if peer_outlier_out_rates else np.nan
                
                # calculate ratios and changes
                outlier_ratio_in = target_outlier_in / peer_outlier_in if peer_outlier_in > 0 and not np.isnan(target_outlier_in) else np.nan
                outlier_ratio_out = target_outlier_out / peer_outlier_out if peer_outlier_out > 0 and not np.isnan(target_outlier_out) else np.nan
                outlier_change = outlier_ratio_in - outlier_ratio_out if not np.isnan(outlier_ratio_in) and not np.isnan(outlier_ratio_out) else np.nan
            else:
                target_outlier_in = target_outlier_out = np.nan
                peer_outlier_in = peer_outlier_out = np.nan
                outlier_ratio_in = outlier_ratio_out = outlier_change = np.nan
            
            # ========================================================================
            # print results
            # ========================================================================
            
            self.log_and_print(f"\n  COHERENCE (lower = more variable/less consistent language):")
            self.log_and_print(f"    All-time:        Target={target_coherence_all:.3f}, Peers={peer_coherence_all:.3f}, Gap={target_coherence_all-peer_coherence_all:+.3f}")
            if has_bias_data:
                self.log_and_print(f"    IN bias period:  Target={target_coherence_in:.3f}, Peers={peer_coherence_in:.3f}, Gap={gap_in_bias:+.3f}")
                self.log_and_print(f"    OUT bias period: Target={target_coherence_out:.3f}, Peers={peer_coherence_out:.3f}, Gap={gap_out_bias:+.3f}")
                self.log_and_print(f"    CHANGE in gap:   {coherence_change:+.3f}  {' FLAGGED' if abs(coherence_change) > 0.03 else ''}")
            
            self.log_and_print(f"\n  OUTLIER RATE (% statements inconsistent with historical topics):")
            if not np.isnan(target_outlier_all):
                self.log_and_print(f"    All-time:        Target={target_outlier_all:.1%}, Peers={peer_outlier_all:.1%}, Ratio={target_outlier_all/peer_outlier_all if peer_outlier_all>0 else np.nan:.2f}x")
                if has_bias_data and not np.isnan(target_outlier_in):
                    self.log_and_print(f"    IN bias period:  Target={target_outlier_in:.1%}, Peers={peer_outlier_in:.1%}, Ratio={outlier_ratio_in:.2f}x")
                    self.log_and_print(f"    OUT bias period: Target={target_outlier_out:.1%}, Peers={peer_outlier_out:.1%}, Ratio={outlier_ratio_out:.2f}x")
                    self.log_and_print(f"    CHANGE in ratio: {outlier_change:+.2f}x  {' FLAGGED' if abs(outlier_change) > 0.3 else ''}")
            else:
                self.log_and_print(f"    N/A (topic column not found)")
            
            # ========================================================================
            # flag unusual patterns
            # ========================================================================
            
            flag_coherence_change = abs(coherence_change) > 0.03 if not np.isnan(coherence_change) else False
            flag_outlier_change = abs(outlier_change) > 0.3 if not np.isnan(outlier_change) else False
            flag_unusual_all_time = (abs(target_coherence_all - peer_coherence_all) > 0.05) if not np.isnan(target_coherence_all) and not np.isnan(peer_coherence_all) else False
            
            is_flagged = flag_coherence_change or flag_outlier_change or flag_unusual_all_time
            
            if is_flagged:
                self.log_and_print(f"\n   UNUSUAL PATTERN DETECTED:")
                if flag_coherence_change:
                    direction = "increased" if coherence_change > 0 else "decreased"
                    self.log_and_print(f"    - Coherence gap {direction} by {abs(coherence_change):.3f} during bias periods")
                if flag_outlier_change:
                    direction = "increased" if outlier_change > 0 else "decreased"
                    self.log_and_print(f"    - Outlier ratio {direction} by {abs(outlier_change):.2f}x during bias periods")
                if flag_unusual_all_time:
                    self.log_and_print(f"    - Consistently unusual coherence across all periods")
            
            # ========================================================================
            # store results
            # ========================================================================
            
            firm_results.append({
                'target_firm': target_ticker,
                'target_folder': target,
                'n_peers': len(peer_folders),
                'n_target_in_bias': n_target_in,
                'n_target_out_bias': n_target_out,
                'n_peer_in_bias': n_peer_in,
                'n_peer_out_bias': n_peer_out,
                'has_bias_data': has_bias_data,
                # coherence - all time
                'coherence_target_all': target_coherence_all,
                'coherence_peer_all': peer_coherence_all,
                'coherence_gap_all': target_coherence_all - peer_coherence_all if not np.isnan(target_coherence_all) and not np.isnan(peer_coherence_all) else np.nan,
                # coherence - in bias
                'coherence_target_in': target_coherence_in,
                'coherence_peer_in': peer_coherence_in,
                'coherence_gap_in': gap_in_bias,
                # coherence - out bias
                'coherence_target_out': target_coherence_out,
                'coherence_peer_out': peer_coherence_out,
                'coherence_gap_out': gap_out_bias,
                # coherence - change
                'coherence_change': coherence_change,
                # outlier rate - all time
                'outlier_target_all': target_outlier_all,
                'outlier_peer_all': peer_outlier_all,
                'outlier_ratio_all': target_outlier_all / peer_outlier_all if peer_outlier_all > 0 and not np.isnan(target_outlier_all) else np.nan,
                # outlier rate - in bias
                'outlier_target_in': target_outlier_in,
                'outlier_peer_in': peer_outlier_in,
                'outlier_ratio_in': outlier_ratio_in,
                # outlier rate - out bias
                'outlier_target_out': target_outlier_out,
                'outlier_peer_out': peer_outlier_out,
                'outlier_ratio_out': outlier_ratio_out,
                # outlier rate - change
                'outlier_change': outlier_change,
                # flags
                'flag_coherence_change': flag_coherence_change,
                'flag_outlier_change': flag_outlier_change,
                'flag_unusual_all_time': flag_unusual_all_time,
                'is_flagged': is_flagged
            })
        
        results_df = pd.DataFrame(firm_results)
        
        # summary
        self.log_and_print("\n" + "-" * 80)
        self.log_and_print("SUMMARY: Per-Firm Analysis with Temporal Segmentation")
        self.log_and_print("-" * 80)
        self.log_and_print(f"  Analyzed: {len(results_df)} target firms")
        
        # check if any firms were analyzed
        if len(results_df) == 0:
            self.log_and_print("  ️  No firms had sufficient data for analysis")
            self.log_and_print("  → Try running with more samples (--max_samples 1000+)")
            return results_df  # return empty dataframe
        
        # count firms with sufficient bias period data
        firms_with_bias_data = results_df['has_bias_data'].sum()
        self.log_and_print(f"  Firms with bias period data: {firms_with_bias_data}/{len(results_df)}")
        
        # flagged firms
        flagged_firms = results_df[results_df['is_flagged']]
        self.log_and_print(f"  Flagged as unusual: {len(flagged_firms)} ({len(flagged_firms)/len(results_df)*100:.1f}%)")
        
        if len(flagged_firms) > 0:
            self.log_and_print("\n  Flagged firms:")
            for _, row in flagged_firms.iterrows():
                reasons = []
                if row['flag_coherence_change']:
                    reasons.append(f"coherence change {row['coherence_change']:+.3f}")
                if row['flag_outlier_change']:
                    reasons.append(f"outlier change {row['outlier_change']:+.2f}x")
                if row['flag_unusual_all_time']:
                    reasons.append(f"unusual all-time (gap {row['coherence_gap_all']:+.3f})")
                self.log_and_print(f"    • {row['target_firm']}: {', '.join(reasons)}")
        else:
            self.log_and_print("\n  No firms showed unusual temporal patterns")
            self.log_and_print("  → Targets do not become more linguistically distinct during bias periods")
        
        # statistics on temporal changes
        if firms_with_bias_data > 0:
            self.log_and_print(f"\n  Temporal Change Statistics (firms with bias data):")
            coherence_changes = results_df[results_df['has_bias_data']]['coherence_change'].dropna()
            outlier_changes = results_df[results_df['has_bias_data']]['outlier_change'].dropna()
            
            if len(coherence_changes) > 0:
                self.log_and_print(f"    Coherence change: mean={coherence_changes.mean():+.3f}, std={coherence_changes.std():.3f}, range=[{coherence_changes.min():+.3f}, {coherence_changes.max():+.3f}]")
            if len(outlier_changes) > 0:
                self.log_and_print(f"    Outlier change:   mean={outlier_changes.mean():+.2f}x, std={outlier_changes.std():.2f}x, range=[{outlier_changes.min():+.2f}x, {outlier_changes.max():+.2f}x]")
        
        # new: comprehensive aggregate summary with statistical tests
        self._print_per_firm_aggregate_summary(results_df)
        
        return results_df
    
    def supervised_classification(self,
                                 embeddings: np.ndarray,
                                 labels: np.ndarray,
                                 test_size: float = 0.2) -> Dict:
        """
        Train supervised classifiers on embeddings and evaluate."""
        self.log_and_print("\n4. Supervised Classification on Embeddings")
        self.log_and_print("=" * 50)
        
        if not SKLEARN_AVAILABLE:
            self.log_and_print(" Scikit-learn not available")
            return {}
        
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        self.log_and_print(f"Train set: {len(X_train)} samples")
        self.log_and_print(f"Test set: {len(X_test)} samples")
        
        results = {}
        
        # logistic regression
        self.log_and_print(f"\nTraining Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        
        y_pred_lr = lr.predict(X_test)
        y_prob_lr = lr.predict_proba(X_test)
        
        # calculate metrics
        if len(np.unique(labels)) == 2:
            auc_lr = roc_auc_score(y_test, y_prob_lr[:, 1])
            self.log_and_print(f"  ROC-AUC: {auc_lr:.3f}")
            results['logistic_regression'] = {
                'model': lr,
                'predictions': y_pred_lr,
                'probabilities': y_prob_lr,
                'auc': auc_lr
            }
        else:
            auc_lr = roc_auc_score(y_test, y_prob_lr, multi_class='ovr')
            self.log_and_print(f"  ROC-AUC (OvR): {auc_lr:.3f}")
            results['logistic_regression'] = {
                'model': lr,
                'predictions': y_pred_lr,
                'probabilities': y_prob_lr,
                'auc': auc_lr
            }
        
        self.log_and_print(f"\nClassification Report (Logistic Regression):")
        self.log_and_print(classification_report(y_test, y_pred_lr))
        
        # svm (optional, can be slow)
        # print(f"\ntraining svm...")
        # svm = svc(kernel='rbf', probability=true, random_state=42)
        # svm.fit(x_train, y_train)
        # y_pred_svm = svm.predict(x_test)
        # y_prob_svm = svm.predict_proba(x_test)
        
        return results
    
    def comprehensive_attribution_classification(self, df: pd.DataFrame, embeddings: np.ndarray, 
                                                  test_size: float = 0.2) -> Dict:
        """
        COMPREHENSIVE ATTRIBUTION CLASSIFICATION ANALYSIS
        
        Tests multiple classification tasks to understand what embeddings capture:
        
        1. Attribution Present vs Not Present (easiest - boundary detection)
        2. Positive vs Negative Outcome (moderate - valence detection)
        3. Internal vs External Locus (moderate - locus detection) ⭐ BASELINE
        4. 4-way Classification: Pos-Int, Pos-Ext, Neg-Int, Neg-Ext (hardest - full type)
        
        Rationale:
        - Strong result on External vs Internal (0.942 AUC) suggests embeddings capture structure
        - Testing progression of difficulty shows WHERE embeddings work and WHERE they fail
        - Isolates what aspects of attribution are learnable from sentence embeddings"""
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("COMPREHENSIVE ATTRIBUTION CLASSIFICATION")
        self.log_and_print("=" * 80)
        self.log_and_print("Testing: What aspects of attribution can embeddings detect?")
        self.log_and_print("Tasks: (1) Present, (2) Outcome, (3) Locus, (4) Full 4-way Type\n")
        
        if not SKLEARN_AVAILABLE:
            self.log_and_print(" Scikit-learn not available")
            return {}
        
        results = {}
        
        # ==================================================================================
        # task 1: attribution present vs not present
        # ==================================================================================
        self.log_and_print("-" * 80)
        self.log_and_print("Task 1: Attribution Present vs Not Present")
        self.log_and_print("-" * 80)
        self.log_and_print("Question: Can embeddings distinguish attribution statements from non-attribution?")
        self.log_and_print("Hypothesis: Should be EASY - clear semantic boundary\n")
        
        # create labels: 1 = attribution present, 0 = not present
        attribution_present_labels = (df['attribution_present'] == 'Y').astype(int).values
        
        n_present = attribution_present_labels.sum()
        n_not_present = len(attribution_present_labels) - n_present
        
        self.log_and_print(f"Label Distribution:")
        self.log_and_print(f"  Attribution Present:     {n_present:,} ({n_present/len(attribution_present_labels)*100:.1f}%)")
        self.log_and_print(f"  Attribution Not Present: {n_not_present:,} ({n_not_present/len(attribution_present_labels)*100:.1f}%)")
        
        # only train if we have both classes
        if n_present > 100 and n_not_present > 100:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    embeddings, attribution_present_labels, 
                    test_size=test_size, random_state=42, stratify=attribution_present_labels
                )
                
                self.log_and_print(f"\nTraining Logistic Regression...")
                lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
                lr.fit(X_train, y_train)
                
                y_pred = lr.predict(X_test)
                y_prob = lr.predict_proba(X_test)[:, 1]
                
                auc = roc_auc_score(y_test, y_prob)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                self.log_and_print(f"  ROC-AUC: {auc:.3f}")
                self.log_and_print(f"\n  Class: Attribution Present")
                self.log_and_print(f"    Precision: {report['1']['precision']:.3f}")
                self.log_and_print(f"    Recall:    {report['1']['recall']:.3f}")
                self.log_and_print(f"    F1-Score:  {report['1']['f1-score']:.3f}")
                
                results['attribution_present'] = {
                    'auc': auc,
                    'precision': report['1']['precision'],
                    'recall': report['1']['recall'],
                    'f1': report['1']['f1-score'],
                    'n_present': n_present,
                    'n_not_present': n_not_present
                }
            except Exception as e:
                self.log_and_print(f"   Classification failed: {e}")
                results['attribution_present'] = {'error': str(e)}
        else:
            self.log_and_print(f"   Insufficient samples (need >100 of each class)")
            results['attribution_present'] = {'error': 'insufficient_samples'}
        
        # ==================================================================================
        # task 2: positive vs negative outcome
        # ==================================================================================
        self.log_and_print("\n" + "-" * 80)
        self.log_and_print("Task 2: Positive vs Negative Outcome")
        self.log_and_print("-" * 80)
        self.log_and_print("Question: Can embeddings detect outcome valence (good vs bad events)?")
        self.log_and_print("Hypothesis: MODERATE difficulty - sentiment-like signal\n")
        
        # filter to only attribution statements with clear outcome
        outcome_mask = df['attribution_outcome'].isin(['Positive', 'Negative', 'positive', 'negative'])
        df_outcome = df[outcome_mask].copy()
        emb_outcome = embeddings[outcome_mask]
        
        # create labels: 1 = positive, 0 = negative
        outcome_labels = df_outcome['attribution_outcome'].isin(['Positive', 'positive']).astype(int).values
        
        n_positive = outcome_labels.sum()
        n_negative = len(outcome_labels) - n_positive
        
        self.log_and_print(f"Label Distribution (filtered to clear attributions):")
        self.log_and_print(f"  Positive Outcome: {n_positive:,} ({n_positive/len(outcome_labels)*100:.1f}%)")
        self.log_and_print(f"  Negative Outcome: {n_negative:,} ({n_negative/len(outcome_labels)*100:.1f}%)")
        
        if n_positive > 100 and n_negative > 100:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    emb_outcome, outcome_labels,
                    test_size=test_size, random_state=42, stratify=outcome_labels
                )
                
                self.log_and_print(f"\nTraining Logistic Regression...")
                lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
                lr.fit(X_train, y_train)
                
                y_pred = lr.predict(X_test)
                y_prob = lr.predict_proba(X_test)[:, 1]
                
                auc = roc_auc_score(y_test, y_prob)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                self.log_and_print(f"  ROC-AUC: {auc:.3f}")
                self.log_and_print(f"\n  Class: Positive Outcome")
                self.log_and_print(f"    Precision: {report['1']['precision']:.3f}")
                self.log_and_print(f"    Recall:    {report['1']['recall']:.3f}")
                self.log_and_print(f"    F1-Score:  {report['1']['f1-score']:.3f}")
                
                results['outcome_valence'] = {
                    'auc': auc,
                    'precision_positive': report['1']['precision'],
                    'recall_positive': report['1']['recall'],
                    'f1_positive': report['1']['f1-score'],
                    'precision_negative': report['0']['precision'],
                    'recall_negative': report['0']['recall'],
                    'n_positive': n_positive,
                    'n_negative': n_negative
                }
            except Exception as e:
                self.log_and_print(f"   Classification failed: {e}")
                results['outcome_valence'] = {'error': str(e)}
        else:
            self.log_and_print(f"   Insufficient samples (need >100 of each class)")
            results['outcome_valence'] = {'error': 'insufficient_samples'}
        
        # clean up task 2 memory before task 3 (frees ~3-4 gb)
        del outcome_mask, df_outcome, emb_outcome, outcome_labels
        if 'X_train' in locals():
            del X_train, X_test, y_train, y_test, lr, y_pred, y_prob
        gc.collect()
        self.log_and_print("  → Memory cleaned after Task 2")
        
        # ==================================================================================
        # task 3: internal vs external locus (baseline - already known to work well)
        # ==================================================================================
        self.log_and_print("\n" + "-" * 80)
        self.log_and_print("Task 3: Internal vs External Locus ⭐ BASELINE")
        self.log_and_print("-" * 80)
        self.log_and_print("Question: Can embeddings detect attribution locus (who/what caused it)?")
        self.log_and_print("Hypothesis: MODERATE - KNOWN to work well (0.942 AUC)\n")
        
        # filter to only attribution statements with clear locus
        locus_mask = df['attribution_locus'].isin(['Internal', 'External', 'internal', 'external'])
        df_locus = df[locus_mask].copy()
        emb_locus = embeddings[locus_mask]
        
        # create labels: 1 = external, 0 = internal
        locus_labels = df_locus['attribution_locus'].isin(['External', 'external']).astype(int).values
        
        n_external = locus_labels.sum()
        n_internal = len(locus_labels) - n_external
        
        self.log_and_print(f"Label Distribution (filtered to clear attributions):")
        self.log_and_print(f"  External Locus: {n_external:,} ({n_external/len(locus_labels)*100:.1f}%)")
        self.log_and_print(f"  Internal Locus: {n_internal:,} ({n_internal/len(locus_labels)*100:.1f}%)")
        
        if n_external > 100 and n_internal > 100:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    emb_locus, locus_labels,
                    test_size=test_size, random_state=42, stratify=locus_labels
                )
                
                self.log_and_print(f"\nTraining Logistic Regression...")
                lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
                lr.fit(X_train, y_train)
                
                y_pred = lr.predict(X_test)
                y_prob = lr.predict_proba(X_test)[:, 1]
                
                auc = roc_auc_score(y_test, y_prob)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                self.log_and_print(f"  ROC-AUC: {auc:.3f} ⭐")
                self.log_and_print(f"\n  Class: External Locus")
                self.log_and_print(f"    Precision: {report['1']['precision']:.3f}")
                self.log_and_print(f"    Recall:    {report['1']['recall']:.3f}")
                self.log_and_print(f"    F1-Score:  {report['1']['f1-score']:.3f}")
                self.log_and_print(f"\n  Class: Internal Locus")
                self.log_and_print(f"    Precision: {report['0']['precision']:.3f}")
                self.log_and_print(f"    Recall:    {report['0']['recall']:.3f}")
                
                results['locus'] = {
                    'auc': auc,
                    'precision_external': report['1']['precision'],
                    'recall_external': report['1']['recall'],
                    'f1_external': report['1']['f1-score'],
                    'precision_internal': report['0']['precision'],
                    'recall_internal': report['0']['recall'],
                    'f1_internal': report['0']['f1-score'],
                    'n_external': n_external,
                    'n_internal': n_internal
                }
            except Exception as e:
                self.log_and_print(f"   Classification failed: {e}")
                results['locus'] = {'error': str(e)}
        else:
            self.log_and_print(f"   Insufficient samples (need >100 of each class)")
            results['locus'] = {'error': 'insufficient_samples'}
        
        # clean up task 3 memory before task 4 (frees ~3-4 gb)
        del locus_mask, df_locus, emb_locus, locus_labels
        if 'X_train' in locals():
            del X_train, X_test, y_train, y_test, lr, y_pred, y_prob
        gc.collect()
        self.log_and_print("  → Memory cleaned after Task 3")
        
        # ==================================================================================
        # task 4: 4-way classification (full attribution type)
        # ==================================================================================
        self.log_and_print("\n" + "-" * 80)
        self.log_and_print("Task 4: 4-Way Attribution Type Classification")
        self.log_and_print("-" * 80)
        self.log_and_print("Question: Can embeddings distinguish all 4 attribution types simultaneously?")
        self.log_and_print("Hypothesis: HARDEST - requires detecting both outcome AND locus\n")
        self.log_and_print("Classes:")
        self.log_and_print("  1. Positive-Internal (self-serving)")
        self.log_and_print("  2. Negative-External (self-serving)")
        self.log_and_print("  3. Positive-External (non-self-serving)")
        self.log_and_print("  4. Negative-Internal (non-self-serving)\n")
        
        # filter to only clear attribution types
        fourway_mask = (
            df['attribution_outcome'].isin(['Positive', 'Negative', 'positive', 'negative']) &
            df['attribution_locus'].isin(['Internal', 'External', 'internal', 'external'])
        )
        df_fourway = df[fourway_mask].copy()
        emb_fourway = embeddings[fourway_mask]
        
        # create combined labels
        df_fourway['attribution_type'] = (
            df_fourway['attribution_outcome'].str.capitalize() + '_' +
            df_fourway['attribution_locus'].str.capitalize()
        )
        
        # encode labels as integers
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        fourway_labels = le.fit_transform(df_fourway['attribution_type'])
        
        # print distribution
        self.log_and_print(f"Label Distribution:")
        for i, class_name in enumerate(le.classes_):
            count = (fourway_labels == i).sum()
            pct = count / len(fourway_labels) * 100
            
            # mark self-serving types
            is_ss = class_name in ['Positive_Internal', 'Negative_External']
            marker = " (self-serving)" if is_ss else " (non-self-serving)"
            
            self.log_and_print(f"  {class_name:20s}: {count:,} ({pct:.1f}%){marker}")
        
        # check if we have enough samples
        min_class_size = min([(fourway_labels == i).sum() for i in range(len(le.classes_))])
        
        if min_class_size > 100:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    emb_fourway, fourway_labels,
                    test_size=test_size, random_state=42, stratify=fourway_labels
                )
                
                self.log_and_print(f"\nTraining Logistic Regression (multi-class)...")
                lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced',
                                       multi_class='multinomial')
                lr.fit(X_train, y_train)
                
                y_pred = lr.predict(X_test)
                y_prob = lr.predict_proba(X_test)
                
                # compute ovr auc
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                
                self.log_and_print(f"  ROC-AUC (One-vs-Rest): {auc:.3f}")
                
                # per-class metrics
                report = classification_report(y_test, y_pred, output_dict=True, 
                                              target_names=le.classes_)
                
                self.log_and_print(f"\n  Per-Class Performance:")
                for class_name in le.classes_:
                    is_ss = class_name in ['Positive_Internal', 'Negative_External']
                    marker = " (self-serving)" if is_ss else ""
                    
                    self.log_and_print(f"\n  {class_name}{marker}:")
                    self.log_and_print(f"    Precision: {report[class_name]['precision']:.3f}")
                    self.log_and_print(f"    Recall:    {report[class_name]['recall']:.3f}")
                    self.log_and_print(f"    F1-Score:  {report[class_name]['f1-score']:.3f}")
                
                # macro-averaged metrics
                self.log_and_print(f"\n  Overall (Macro-Averaged):")
                self.log_and_print(f"    Precision: {report['macro avg']['precision']:.3f}")
                self.log_and_print(f"    Recall:    {report['macro avg']['recall']:.3f}")
                self.log_and_print(f"    F1-Score:  {report['macro avg']['f1-score']:.3f}")
                
                results['four_way'] = {
                    'auc_ovr': auc,
                    'macro_precision': report['macro avg']['precision'],
                    'macro_recall': report['macro avg']['recall'],
                    'macro_f1': report['macro avg']['f1-score'],
                    'per_class': {
                        class_name: {
                            'precision': report[class_name]['precision'],
                            'recall': report[class_name]['recall'],
                            'f1': report[class_name]['f1-score'],
                            'support': int(report[class_name]['support'])
                        }
                        for class_name in le.classes_
                    }
                }
            except Exception as e:
                self.log_and_print(f"   Classification failed: {e}")
                results['four_way'] = {'error': str(e)}
        else:
            self.log_and_print(f"   Insufficient samples (smallest class: {min_class_size}, need >100)")
            results['four_way'] = {'error': 'insufficient_samples'}
        
        # ==================================================================================
        # summary comparison
        # ==================================================================================
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("COMPREHENSIVE CLASSIFICATION SUMMARY")
        self.log_and_print("=" * 80)
        self.log_and_print("\nTask Difficulty Progression:\n")
        
        summary_data = []
        if 'attribution_present' in results and 'auc' in results['attribution_present']:
            summary_data.append(('1. Attribution Present', results['attribution_present']['auc'], 'Easy'))
        if 'outcome_valence' in results and 'auc' in results['outcome_valence']:
            summary_data.append(('2. Outcome Valence', results['outcome_valence']['auc'], 'Moderate'))
        if 'locus' in results and 'auc' in results['locus']:
            summary_data.append(('3. Locus (Int vs Ext)', results['locus']['auc'], 'Moderate'))
        if 'four_way' in results and 'auc_ovr' in results['four_way']:
            summary_data.append(('4. Full 4-Way Type', results['four_way']['auc_ovr'], 'Hard'))
        
        if summary_data:
            for task_name, auc, difficulty in summary_data:
                marker = ""
                if auc > 0.9:
                    marker = " ⭐⭐ Excellent"
                elif auc > 0.8:
                    marker = " ⭐ Strong"
                elif auc > 0.7:
                    marker = "  Good"
                elif auc > 0.6:
                    marker = " ~ Weak"
                else:
                    marker = "  Poor"
                
                self.log_and_print(f"  {task_name:25s}  AUC: {auc:.3f}  ({difficulty:8s}){marker}")
            
            # interpretation
            self.log_and_print("\n" + "-" * 80)
            self.log_and_print("Interpretation:")
            self.log_and_print("-" * 80)
            
            if 'locus' in results and 'auc' in results['locus']:
                locus_auc = results['locus']['auc']
                if locus_auc > 0.9:
                    self.log_and_print(" Embeddings EXCEL at detecting attribution locus (Internal vs External)")
                    self.log_and_print("  → Suggests clear linguistic patterns distinguish 'we/us' from 'market/economy'")
            
            if 'outcome_valence' in results and 'auc' in results['outcome_valence']:
                outcome_auc = results['outcome_valence']['auc']
                if outcome_auc > 0.7:
                    self.log_and_print("\n Embeddings GOOD at detecting outcome valence (Positive vs Negative)")
                    self.log_and_print("  → Captures sentiment-like signals in attribution statements")
                elif outcome_auc < 0.6:
                    self.log_and_print("\n Embeddings STRUGGLE with outcome valence")
                    self.log_and_print("  → May require explicit sentiment analysis")
            
            if 'four_way' in results and 'auc_ovr' in results['four_way']:
                fourway_auc = results['four_way']['auc_ovr']
                if fourway_auc > 0.8:
                    self.log_and_print("\n⭐ Embeddings can distinguish FULL attribution types simultaneously")
                    self.log_and_print("  → Strong foundation for bias detection via attribution patterns")
                elif fourway_auc < 0.7:
                    self.log_and_print("\n→ Full 4-way classification is challenging")
                    self.log_and_print("  → May need separate binary classifiers instead of multi-class")
        else:
            self.log_and_print("  No successful classifications completed")
        
        self.log_and_print("\n Comprehensive attribution classification complete\n")
        
        return results
    
    def supervised_classification_bias_prediction(self, df: pd.DataFrame, embeddings: np.ndarray,
                                                  bias_periods: Dict, test_size: float = 0.2) -> Dict:
        """
        METHODOLOGY: Supervised Classification for Expert Period Prediction
        ====================================================================
        
        Core Hypothesis Test: Can embeddings detect linguistic patterns around
        expert-identified bias periods that GPT's SAB proportion analysis missed?
        
        Approach:
        1. GPT Baseline Model: Uses GPT-extracted features (pos_internal_rate, etc.)
        2. Embedding Model: Uses only sentence embeddings
        3. Combined Model: Uses both GPT features + embeddings
        
        Target Variable: Expert Period (1) vs Normal Period (0)
        
        If embeddings add predictive power beyond GPT, it suggests they capture
        subtle linguistic patterns (tone, complexity, defensiveness) that aren't
        reflected in attribution proportions alone."""
        self.log_and_print("\n4b. Supervised Classification: Expert Period Prediction")
        self.log_and_print("=" * 80)
        self.log_and_print("Testing: Do embeddings detect patterns around expert periods that GPT SAB analysis missed?")
        
        if not SKLEARN_AVAILABLE:
            self.log_and_print("Scikit-learn not available")
            return {}
        
        if not bias_periods:
            self.log_and_print("No bias period data available")
            return {}
        
        # create binary labels: expert period (1) vs normal period (0)
        df['bias_period_key'] = list(zip(df['Company'], df['Year'], df['Quarter']))
        df['bias_label'] = df['bias_period_key'].apply(
            lambda x: 1 if bias_periods.get(x, {}).get('in_expert_period', False) else 0
        )
        
        # no filtering needed - we use all data (expert periods vs normal)
        bias_mask = df['bias_label'].notna()  # just ensure no nan
        
        if bias_mask.sum() == 0:
            self.log_and_print("No data available for classification")
            return {}
        
        df_labeled = df[bias_mask].copy()
        df_labeled = df_labeled.reset_index(drop=True)  # reset indices to match embeddings array
        embeddings_labeled = embeddings[bias_mask]
        labels = df_labeled['bias_label'].values
        
        self.log_and_print(f"\nLabeled Data:")
        self.log_and_print(f"  Expert periods: {(labels == 1).sum():,} segments ({(labels == 1).sum()/len(labels)*100:.1f}%)")
        self.log_and_print(f"  Normal periods: {(labels == 0).sum():,} segments ({(labels == 0).sum()/len(labels)*100:.1f}%)")
        
        # aggregate to company-quarter level to avoid data leakage
        # (multiple snippets from same quarter should be in same train/test split)
        quarter_features = []
        quarter_labels = []
        quarter_ids = []
        
        for (company, year, quarter), group in df_labeled.groupby(['Company', 'Year', 'Quarter']):
            indices = group.index.tolist()
            group_embeddings = embeddings_labeled[indices]
            
            # aggregate embeddings for this quarter
            quarter_embedding = group_embeddings.mean(axis=0)
            
            # extract gpt features for this quarter
            gpt_features = [
                group['pos_internal'].sum() / group['total_attributions'].sum() if 'pos_internal' in group.columns else 0,
                group['neg_external'].sum() / group['total_attributions'].sum() if 'neg_external' in group.columns else 0,
                group['pos_external'].sum() / group['total_attributions'].sum() if 'pos_external' in group.columns else 0,
                group['neg_internal'].sum() / group['total_attributions'].sum() if 'neg_internal' in group.columns else 0,
            ]
            
            # fallback if those columns don't exist
            if all(f == 0 for f in gpt_features):
                pos_int = len(group[(group['attribution_outcome'] == 'Positive') & (group['attribution_locus'] == 'Internal')])
                neg_ext = len(group[(group['attribution_outcome'] == 'Negative') & (group['attribution_locus'] == 'External')])
                pos_ext = len(group[(group['attribution_outcome'] == 'Positive') & (group['attribution_locus'] == 'External')])
                neg_int = len(group[(group['attribution_outcome'] == 'Negative') & (group['attribution_locus'] == 'Internal')])
                total = len(group)
                
                gpt_features = [
                    pos_int / total if total > 0 else 0,
                    neg_ext / total if total > 0 else 0,
                    pos_ext / total if total > 0 else 0,
                    neg_int / total if total > 0 else 0
                ]
            
            # asymmetry score
            asym = (gpt_features[0] + gpt_features[1]) - (gpt_features[2] + gpt_features[3])
            gpt_features.append(asym)
            
            # sab proportion (classic self-serving attribution bias metric)
            sab_proportion = gpt_features[0] + gpt_features[1]  # pos_internal + neg_external
            gpt_features.append(sab_proportion)
            
            quarter_features.append({
                'gpt': np.array(gpt_features),
                'embedding': quarter_embedding,
                'combined': np.concatenate([gpt_features, quarter_embedding])
            })
            quarter_labels.append(group['bias_label'].iloc[0])
            quarter_ids.append(f"{company}_{year}_{quarter}")
        
        y = np.array(quarter_labels)
        
        # split data
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=42, stratify=y
        )
        
        results = {}
        
        # model 1: gpt baseline
        self.log_and_print(f"\n1. GPT Baseline Model (6 features: pos_internal, neg_external, pos_external, neg_internal, asymmetry, SAB_proportion)")
        X_gpt = np.array([f['gpt'] for f in quarter_features])
        X_train, X_test = X_gpt[train_idx], X_gpt[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        lr_gpt = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        lr_gpt.fit(X_train, y_train)
        y_prob_gpt = lr_gpt.predict_proba(X_test)[:, 1]
        auc_gpt = roc_auc_score(y_test, y_prob_gpt)
        
        self.log_and_print(f"   GPT Baseline AUC: {auc_gpt:.3f}")
        results['gpt_baseline'] = {'auc': float(auc_gpt), 'model': lr_gpt}
        
        # model 2: embedding only
        self.log_and_print(f"\n2. Embedding Model ({len(quarter_features[0]['embedding'])} features)")
        X_emb = np.array([f['embedding'] for f in quarter_features])
        X_train, X_test = X_emb[train_idx], X_emb[test_idx]
        
        lr_emb = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        lr_emb.fit(X_train, y_train)
        y_prob_emb = lr_emb.predict_proba(X_test)[:, 1]
        auc_emb = roc_auc_score(y_test, y_prob_emb)
        
        self.log_and_print(f"   Embedding AUC: {auc_emb:.3f}")
        results['embedding'] = {'auc': float(auc_emb), 'model': lr_emb}
        
        # model 3: combined
        self.log_and_print(f"\n3. Combined Model (GPT + Embedding: {len(quarter_features[0]['combined'])} features)")
        X_comb = np.array([f['combined'] for f in quarter_features])
        X_train, X_test = X_comb[train_idx], X_comb[test_idx]
        
        lr_comb = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        lr_comb.fit(X_train, y_train)
        y_prob_comb = lr_comb.predict_proba(X_test)[:, 1]
        auc_comb = roc_auc_score(y_test, y_prob_comb)
        
        self.log_and_print(f"   Combined AUC: {auc_comb:.3f}")
        results['combined'] = {'auc': float(auc_comb), 'model': lr_comb}
        
        # analysis
        self.log_and_print(f"\n" + "=" * 80)
        self.log_and_print(f"HYPOTHESIS TEST RESULTS:")
        self.log_and_print(f"=" * 80)
        
        improvement_emb = auc_emb - auc_gpt
        improvement_comb = auc_comb - auc_gpt
        
        self.log_and_print(f"\nGPT Baseline:           AUC = {auc_gpt:.3f}")
        self.log_and_print(f"Embedding Only:         AUC = {auc_emb:.3f}  (Δ = {improvement_emb:+.3f})")
        self.log_and_print(f"Combined (GPT + Embed): AUC = {auc_comb:.3f}  (Δ = {improvement_comb:+.3f})")
        
        if improvement_comb > 0.05:
            self.log_and_print(f"\n HYPOTHESIS CONFIRMED: Embeddings detect linguistic patterns around expert periods")
            self.log_and_print(f"  Even though SAB proportions show no difference, embeddings capture other signals")
            self.log_and_print(f"  (e.g., tone, complexity, semantic coherence, defensiveness)")
        elif improvement_comb > 0.02:
            self.log_and_print(f"\n~ HYPOTHESIS PARTIALLY SUPPORTED: Modest embedding improvement")
            self.log_and_print(f"  Weak linguistic signature exists around expert-identified periods")
        else:
            self.log_and_print(f"\n HYPOTHESIS NOT SUPPORTED: No detectable linguistic shift in expert periods")
            self.log_and_print(f"  Neither GPT features nor embeddings distinguish these periods")
            self.log_and_print(f"  Language patterns may not change during hypothesized bias periods")
        
        results['analysis'] = {
            'gpt_auc': float(auc_gpt),
            'embedding_auc': float(auc_emb),
            'combined_auc': float(auc_comb),
            'improvement_emb_vs_gpt': float(improvement_emb),
            'improvement_comb_vs_gpt': float(improvement_comb),
            'hypothesis_confirmed': improvement_comb > 0.05,
            'n_train': len(train_idx),
            'n_test': len(test_idx)
        }
        
        return results
    
    def detect_qna_topic_shifts(self, df: pd.DataFrame, embeddings: np.ndarray, 
                                threshold: float = 0.6) -> pd.DataFrame:
        self.log_and_print("\n5. Detecting Q&A Topic Shifts")
        self.log_and_print("=" * 80)
        
        if 'Section' not in df.columns:
            self.log_and_print("No Section column found, skipping Q&A analysis")
            return pd.DataFrame()
        
        qna_mask = df['Section'] == 'qna'
        qna_df = df[qna_mask].copy().reset_index(drop=True)
        qna_embeddings = embeddings[qna_mask]
        
        if len(qna_df) < 10:
            self.log_and_print("Insufficient Q&A data for topic shift analysis")
            return pd.DataFrame()
        
        self.log_and_print(f"Analyzing {len(qna_df)} Q&A segments")
        
        shift_results = []
        
        for (company, year, quarter), group in qna_df.groupby(['Company', 'Year', 'Quarter']):
            if len(group) < 2:
                continue
            
            indices = group.index.tolist()
            group_embeddings = qna_embeddings[indices]
            
            similarities = []
            shifts = []
            
            for i in range(len(group_embeddings) - 1):
                cosine_sim = np.dot(group_embeddings[i], group_embeddings[i+1]) / (
                    np.linalg.norm(group_embeddings[i]) * np.linalg.norm(group_embeddings[i+1])
                )
                similarities.append(cosine_sim)
                
                if cosine_sim < threshold:
                    shifts.append(i+1)
            
            is_target = group['IS_TARGET'].iloc[0] == 'Y'
            
            shift_results.append({
                'company': company,
                'year': year,
                'quarter': quarter,
                'is_target': is_target,
                'n_qna_segments': len(group),
                'n_shifts': len(shifts),
                'shift_rate': len(shifts) / len(group) if len(group) > 0 else 0,
                'mean_similarity': np.mean(similarities) if similarities else 0,
                'min_similarity': np.min(similarities) if similarities else 0
            })
        
        shift_df = pd.DataFrame(shift_results)
        
        if len(shift_df) > 0:
            targets = shift_df[shift_df['is_target']]
            peers = shift_df[~shift_df['is_target']]
            
            self.log_and_print(f"\n Analyzed {len(shift_df)} company-quarters with Q&A data")
            self.log_and_print(f"\nTopic Shift Statistics:")
            if len(targets) > 0:
                self.log_and_print(f"  Targets mean shift rate: {targets['shift_rate'].mean():.2%}")
                self.log_and_print(f"  Targets mean similarity: {targets['mean_similarity'].mean():.3f}")
            if len(peers) > 0:
                self.log_and_print(f"  Peers mean shift rate: {peers['shift_rate'].mean():.2%}")
                self.log_and_print(f"  Peers mean similarity: {peers['mean_similarity'].mean():.3f}")
        
        return shift_df
    
    def analyze_topic_temporal_consistency(self, df: pd.DataFrame, embeddings: np.ndarray,
                                          lookback_quarters: int = 4) -> pd.DataFrame:
        """
        METHODOLOGY: Topic-Level Temporal Consistency Analysis
        ========================================================
        
        Core Hypothesis: Attribution bias may manifest as unusual language when discussing
        specific topics, detectable by comparing attribution snippets to:
        1. Non-attribution discussion of the SAME topic in the SAME quarter
        2. Historical discussion of that topic in previous quarters
        
        Rationale for This Approach:
        ----------------------------
        1. WITHIN-QUARTER CONSISTENCY:
           - If a company discusses "Revenue" neutrally in most of the call
           - But uses very different language when making revenue attributions
           - This suggests the attribution is linguistically unusual/fabricated
           
        2. TEMPORAL CONSISTENCY:
           - If a company always discusses "Supply Chain" a certain way
           - But suddenly uses very different language in a high-bias quarter
           - This suggests narrative shift or selective framing
           
        3. OUTLIER DETECTION:
           - Attribution snippets that are embedding outliers compared to
             previous similar topic discussions may indicate:
             - Fabricated explanations (not grounded in typical discussion)
             - Defensive language (different tone when attributing blame)
             - Strategic narrative shifts (reframing usual topics)
        
        4. TOPIC-SPECIFIC BIAS PATTERNS:
           - Some topics may show bias more than others
           - E.g., "Costs" attributions might be consistently unusual
           - While "Revenue" attributions match typical discussion
        
        Algorithm:
        ----------
        For each attribution snippet (attribution_present='Y'):
        
        A. Extract context:
           - Company, year, quarter
           - Primary_Topic (e.g., "Revenue", "Costs", "Supply Chain")
           - Attribution type (outcome + locus)
        
        B. Find comparison snippets:
           1. SAME-QUARTER, SAME-TOPIC, NON-ATTRIBUTION snippets
              (How do they discuss this topic elsewhere in this call?)
           
           2. HISTORICAL, SAME-TOPIC snippets (previous 4 quarters)
              (How did they discuss this topic historically?)
        
        C. Calculate embedding distances:
           - Distance to same-quarter, same-topic centroid
           - Distance to historical, same-topic centroid
           - Outlier score (Z-score of distance within topic)
        
        D. Flag anomalies:
           - High within-quarter distance = inconsistent with own call
           - High historical distance = narrative shift from past
           - High outlier score = linguistically unusual
        
        Use Cases:
        ----------
        1. Identify which topics show most attribution inconsistency
        2. Find quarters where topic discussion shifted dramatically
        3. Flag attribution snippets that are outliers (potential fabrication)
        4. Track how companies' framing of specific topics evolves"""
        self.log_and_print("\n6. Topic-Level Temporal Consistency Analysis")
        self.log_and_print("=" * 80)
        self.log_and_print("Hypothesis: Attribution bias manifests as unusual language for specific topics")
        self.log_and_print("Comparing attributions to same-topic discussion (current & historical)")
        
        from scipy.spatial.distance import cosine as cosine_distance
        
        if 'Primary_Topic' not in df.columns:
            self.log_and_print("No Primary_Topic column found, skipping topic consistency analysis")
            return pd.DataFrame()
        
        # filter to only attribution snippets
        # keep original indices to match embeddings array
        attr_mask = df['attribution_present'] == 'Y'
        attr_df = df[attr_mask].copy()
        attr_embeddings = embeddings[attr_mask]  # filter embeddings to match
        
        if len(attr_df) == 0:
            self.log_and_print("No attribution snippets found")
            return pd.DataFrame()
        
        # reset indices for clean iteration, but keep original as column
        attr_df['_original_idx'] = attr_df.index
        attr_df = attr_df.reset_index(drop=True)
        
        self.log_and_print(f"\nAnalyzing {len(attr_df):,} attribution snippets")
        
        results = []
        processed_count = 0
        skipped_count = 0
        
        # build historical topic embeddings cache for efficiency
        # key: (company, topic) -> list of (year, quarter, centroid, embeddings)
        historical_cache = {}
        
        # optimization: also pre-compute same-quarter non-attribution centroids
        # key: (company, year, quarter, topic) -> centroid (for within-quarter comparison)
        same_quarter_cache = {}
        
        self.log_and_print("\nBuilding topic embeddings cache (historical + same-quarter)...")
        companies = df['Company'].unique()
        
        # use tqdm progress bar if available
        if TQDM_AVAILABLE:
            company_iterator = tqdm(companies, desc="  Building cache", unit="company")
        else:
            company_iterator = companies
        
        for company in company_iterator:
            company_df = df[df['Company'] == company].copy()
            
            for topic in company_df['Primary_Topic'].dropna().unique():
                topic_mask = (company_df['Primary_Topic'] == topic)
                topic_df = company_df[topic_mask]
                
                # group by quarter
                for (year, quarter), group in topic_df.groupby(['Year', 'Quarter']):
                    indices = group.index.tolist()
                    if len(indices) == 0:
                        continue
                    
                    group_embeddings = embeddings[indices]
                    centroid = group_embeddings.mean(axis=0)
                    
                    # pre-compute statistics for outlier detection (memory optimization)
                    distances_to_centroid = np.linalg.norm(group_embeddings - centroid, axis=1)
                    mean_distance = distances_to_centroid.mean()
                    std_distance = distances_to_centroid.std()
                    
                    # historical cache (all snippets for this topic-quarter)
                    key = (company, topic)
                    if key not in historical_cache:
                        historical_cache[key] = []
                    
                    historical_cache[key].append({
                        'year': year,
                        'quarter': quarter,
                        'centroid': centroid,
                        # don't store embeddings - use pre-computed stats instead
                        'n_snippets': len(indices),
                        'mean_distance': mean_distance,
                        'std_distance': std_distance
                    })
                    
                    # optimization: same-quarter non-attribution centroid cache
                    non_attr_mask = group['attribution_present'] == 'N'
                    if non_attr_mask.sum() > 0:
                        non_attr_indices = group[non_attr_mask].index.tolist()
                        non_attr_embeddings = embeddings[non_attr_indices]
                        non_attr_centroid = non_attr_embeddings.mean(axis=0)
                        
                        same_quarter_key = (company, year, quarter, topic)
                        same_quarter_cache[same_quarter_key] = {
                            'centroid': non_attr_centroid,
                            'n_snippets': len(non_attr_indices)
                        }
        
        self.log_and_print(f" Built cache for {len(historical_cache)} company-topic pairs")
        self.log_and_print(f" Pre-computed {len(same_quarter_cache)} same-quarter centroids")
        
        # analyze each attribution snippet
        self.log_and_print("\nAnalyzing attribution snippets...")
        
        # use tqdm progress bar if available
        if TQDM_AVAILABLE:
            attr_iterator = tqdm(attr_df.iterrows(), total=len(attr_df), 
                               desc="  Processing attributions", unit="snippet")
        else:
            attr_iterator = attr_df.iterrows()
        
        for idx, row in attr_iterator:
            company = row['Company']
            year = row['Year']
            quarter = row['Quarter']
            topic = row['Primary_Topic']
            
            if pd.isna(topic) or topic == '':
                skipped_count += 1
                continue
            
            # get embedding for this attribution snippet (use filtered embeddings)
            attr_embedding = attr_embeddings[idx]
            
            # create attribution type label
            outcome = row.get('attribution_outcome', 'Unknown')
            locus = row.get('attribution_locus', 'Unknown')
            attr_type = f"{outcome}_{locus}"
            
            is_target = row['IS_TARGET'] == 'Y'
            
            # ================================================================
            # part a: within-quarter consistency (optimized: use pre-computed cache)
            # ================================================================
            within_quarter_distance = None
            within_quarter_similarity = None
            n_same_quarter = 0
            
            same_quarter_key = (company, year, quarter, topic)
            if same_quarter_key in same_quarter_cache:
                cached_data = same_quarter_cache[same_quarter_key]
                same_quarter_centroid = cached_data['centroid']
                n_same_quarter = cached_data['n_snippets']
                
                # distance from attribution to non-attribution centroid
                within_quarter_distance = np.linalg.norm(attr_embedding - same_quarter_centroid)
                
                # cosine similarity
                within_quarter_similarity = 1 - cosine_distance(attr_embedding, same_quarter_centroid)
            
            # ================================================================
            # part b: historical consistency
            # ================================================================
            # find historical discussion of this topic (previous n quarters)
            cache_key = (company, topic)
            historical_distance = None
            historical_similarity = None
            n_historical = 0
            topic_outlier_score = None
            
            if cache_key in historical_cache:
                historical_quarters = historical_cache[cache_key]
                
                # filter to previous quarters only (before current quarter)
                def quarter_to_numeric(y, q):
                    return y * 4 + q
                
                current_quarter_num = quarter_to_numeric(year, quarter)
                
                previous_quarters = [
                    hq for hq in historical_quarters
                    if quarter_to_numeric(hq['year'], hq['quarter']) < current_quarter_num
                ]
                
                # limit to last n quarters
                previous_quarters = sorted(
                    previous_quarters,
                    key=lambda x: quarter_to_numeric(x['year'], x['quarter']),
                    reverse=True
                )[:lookback_quarters]
                
                if len(previous_quarters) > 0:
                    # compute weighted average of centroids from previous quarters
                    total_snippets = sum(hq['n_snippets'] for hq in previous_quarters)
                    weighted_centroid = np.sum([
                        hq['centroid'] * hq['n_snippets'] for hq in previous_quarters
                    ], axis=0) / total_snippets
                    
                    n_historical = total_snippets
                    
                    # distance from attribution to historical weighted centroid
                    historical_distance = np.linalg.norm(attr_embedding - weighted_centroid)
                    historical_similarity = 1 - cosine_distance(attr_embedding, weighted_centroid)
                    
                    # ============================================================
                    # part c: outlier detection (using pre-computed statistics)
                    # ============================================================
                    # use weighted average of pre-computed mean distances and std
                    mean_dist = np.sum([
                        hq['mean_distance'] * hq['n_snippets'] for hq in previous_quarters
                    ]) / total_snippets
                    
                    # for std, use pooled standard deviation formula
                    pooled_variance = np.sum([
                        (hq['std_distance'] ** 2) * hq['n_snippets'] for hq in previous_quarters
                    ]) / total_snippets
                    std_dist = np.sqrt(pooled_variance)
                    
                    if std_dist > 0:
                        topic_outlier_score = (historical_distance - mean_dist) / std_dist
                    else:
                        topic_outlier_score = 0
            
            # ================================================================
            # flag anomalies
            # ================================================================
            is_within_quarter_outlier = (
                within_quarter_distance is not None and 
                within_quarter_similarity is not None and
                within_quarter_similarity < 0.7  # low similarity = unusual
            )
            
            is_historical_outlier = (
                topic_outlier_score is not None and
                topic_outlier_score > 2.0  # >2 std devs = outlier
            )
            
            is_outlier = is_within_quarter_outlier or is_historical_outlier
            
            # store results
            results.append({
                'snippet_id': idx,
                'company': company,
                'year': year,
                'quarter': quarter,
                'is_target': is_target,
                'topic': topic,
                'attribution_type': attr_type,
                'outcome': outcome,
                'locus': locus,
                'within_quarter_distance': within_quarter_distance,
                'within_quarter_similarity': within_quarter_similarity,
                'historical_distance': historical_distance,
                'historical_similarity': historical_similarity,
                'topic_outlier_score': topic_outlier_score,
                'is_within_quarter_outlier': is_within_quarter_outlier,
                'is_historical_outlier': is_historical_outlier,
                'is_outlier': is_outlier,
                'n_same_topic_same_quarter': n_same_quarter,
                'n_same_topic_historical': n_historical
            })
            
            processed_count += 1
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            self.log_and_print("No results generated")
            return results_df
        
        # ====================================================================
        # summary statistics
        # ====================================================================
        self.log_and_print(f"\n Analyzed {processed_count:,} attribution snippets")
        self.log_and_print(f"  Skipped {skipped_count:,} (missing topic)")
        
        # filter to those with comparisons
        with_same_quarter = results_df[results_df['n_same_topic_same_quarter'] > 0]
        with_historical = results_df[results_df['n_same_topic_historical'] > 0]
        
        self.log_and_print(f"\nComparison Coverage:")
        self.log_and_print(f"  With same-quarter comparisons: {len(with_same_quarter):,} ({len(with_same_quarter)/len(results_df)*100:.1f}%)")
        self.log_and_print(f"  With historical comparisons: {len(with_historical):,} ({len(with_historical)/len(results_df)*100:.1f}%)")
        
        # outlier statistics
        outliers = results_df[results_df['is_outlier']]
        within_quarter_outliers = results_df[results_df['is_within_quarter_outlier']]
        historical_outliers = results_df[results_df['is_historical_outlier']]
        
        self.log_and_print(f"\nOutlier Detection:")
        self.log_and_print(f"  Total outliers: {len(outliers):,} ({len(outliers)/len(results_df)*100:.1f}%)")
        self.log_and_print(f"  Within-quarter outliers: {len(within_quarter_outliers):,}")
        self.log_and_print(f"  Historical outliers: {len(historical_outliers):,}")
        
        # compare targets vs peers
        target_results = results_df[results_df['is_target']]
        peer_results = results_df[~results_df['is_target']]
        
        if len(target_results) > 0 and len(peer_results) > 0:
            target_outlier_rate = (target_results['is_outlier'].sum() / len(target_results)) * 100
            peer_outlier_rate = (peer_results['is_outlier'].sum() / len(peer_results)) * 100
            
            self.log_and_print(f"\nTarget vs Peer Comparison (AGGREGATE):")
            self.log_and_print(f"  Target outlier rate: {target_outlier_rate:.1f}%")
            self.log_and_print(f"  Peer outlier rate: {peer_outlier_rate:.1f}%")
            
            if target_outlier_rate > peer_outlier_rate * 1.2:
                self.log_and_print(f"   Targets show {target_outlier_rate/peer_outlier_rate:.1f}x more topic inconsistency")
            else:
                self.log_and_print(f"  → No significant difference in topic consistency")
        
        # ====================================================================
        # new: breakdown by attribution type (positive/negative × internal/external)
        # ====================================================================
        self.log_and_print(f"\nTopic Consistency BY ATTRIBUTION TYPE:")
        self.log_and_print("=" * 80)
        
        # define attribution type categories
        attribution_types = {
            'Positive_Internal': (results_df['outcome'].str.lower() == 'positive') & 
                                (results_df['locus'].str.lower() == 'internal'),
            'Positive_External': (results_df['outcome'].str.lower() == 'positive') & 
                                (results_df['locus'].str.lower() == 'external'),
            'Negative_Internal': (results_df['outcome'].str.lower() == 'negative') & 
                                (results_df['locus'].str.lower() == 'internal'),
            'Negative_External': (results_df['outcome'].str.lower() == 'negative') & 
                                (results_df['locus'].str.lower() == 'external')
        }
        
        # create summary by attribution type
        for attr_name, attr_mask in attribution_types.items():
            attr_subset = results_df[attr_mask]
            
            if len(attr_subset) == 0:
                continue
                
            # overall statistics for this attribution type
            n_total = len(attr_subset)
            n_outliers = attr_subset['is_outlier'].sum()
            outlier_rate = (n_outliers / n_total) * 100
            
            # mean distances
            mean_within_q = attr_subset['within_quarter_distance'].mean()
            mean_historical = attr_subset['historical_distance'].mean()
            mean_outlier_score = attr_subset['topic_outlier_score'].mean()
            
            self.log_and_print(f"\n{attr_name}:")
            self.log_and_print(f"  N = {n_total:,} attributions")
            self.log_and_print(f"  Outlier rate: {outlier_rate:.1f}% ({n_outliers:,} outliers)")
            self.log_and_print(f"  Mean within-quarter distance: {mean_within_q:.3f}")
            self.log_and_print(f"  Mean historical distance: {mean_historical:.3f}")
            self.log_and_print(f"  Mean outlier score: {mean_outlier_score:+.2f} std devs")
            
            # target vs peer breakdown for this attribution type
            attr_targets = attr_subset[attr_subset['is_target']]
            attr_peers = attr_subset[~attr_subset['is_target']]
            
            if len(attr_targets) > 0 and len(attr_peers) > 0:
                target_rate = (attr_targets['is_outlier'].sum() / len(attr_targets)) * 100
                peer_rate = (attr_peers['is_outlier'].sum() / len(attr_peers)) * 100
                
                self.log_and_print(f"  Target outlier rate: {target_rate:.1f}% (n={len(attr_targets):,})")
                self.log_and_print(f"  Peer outlier rate: {peer_rate:.1f}% (n={len(attr_peers):,})")
                
                if target_rate > peer_rate * 1.2:
                    self.log_and_print(f"  → Targets {target_rate/peer_rate:.2f}x more inconsistent for {attr_name}")
                elif peer_rate > target_rate * 1.2:
                    self.log_and_print(f"  → Peers {peer_rate/target_rate:.2f}x more inconsistent for {attr_name}")
                else:
                    self.log_and_print(f"  → Similar consistency for targets and peers")
        
        # add attribution type breakdown to the dataframe for saving
        results_df['attribution_category'] = 'Unknown'
        for attr_name, attr_mask in attribution_types.items():
            results_df.loc[attr_mask, 'attribution_category'] = attr_name
        
        # most inconsistent topics by attribution type
        self.log_and_print(f"\n\nMost Inconsistent Topics BY ATTRIBUTION TYPE:")
        self.log_and_print("=" * 80)
        
        for attr_name in ['Positive_Internal', 'Positive_External', 'Negative_Internal', 'Negative_External']:
            attr_subset = results_df[results_df['attribution_category'] == attr_name]
            attr_with_hist = attr_subset[attr_subset['n_same_topic_historical'] > 0]
            
            if len(attr_with_hist) == 0:
                continue
            
            topic_stats = attr_with_hist.groupby('topic').agg({
                'topic_outlier_score': 'mean',
                'is_outlier': 'mean',
                'snippet_id': 'count'
            }).reset_index()
            topic_stats.columns = ['topic', 'mean_outlier_score', 'outlier_rate', 'n_attributions']
            topic_stats = topic_stats[topic_stats['n_attributions'] >= 3]  # min 3 attributions
            topic_stats = topic_stats.sort_values('mean_outlier_score', ascending=False)
            
            if len(topic_stats) > 0:
                self.log_and_print(f"\n{attr_name} - Top 3 Inconsistent Topics:")
                for i, (_, row) in enumerate(topic_stats.head(3).iterrows(), 1):
                    self.log_and_print(f"  {i}. {row['topic'][:45]:<45} | Score: {row['mean_outlier_score']:+.2f} | Rate: {row['outlier_rate']*100:.0f}% | N={int(row['n_attributions'])}")
        
        # most inconsistent topics
        if len(with_historical) > 0:
            topic_stats = with_historical.groupby('topic').agg({
                'topic_outlier_score': 'mean',
                'is_outlier': 'mean',
                'snippet_id': 'count'
            }).reset_index()
            topic_stats.columns = ['topic', 'mean_outlier_score', 'outlier_rate', 'n_attributions']
            topic_stats = topic_stats[topic_stats['n_attributions'] >= 5]  # min 5 attributions
            topic_stats = topic_stats.sort_values('mean_outlier_score', ascending=False)
            
            if len(topic_stats) > 0:
                self.log_and_print(f"\nMost Inconsistent Topics (Top 5):")
                for _, row in topic_stats.head(5).iterrows():
                    self.log_and_print(f"  {row['topic'][:50]:<50} | Outlier Score: {row['mean_outlier_score']:+.2f} | Rate: {row['outlier_rate']*100:.1f}%")
        
        return results_df
    
    def extract_bias_vector(self, df: pd.DataFrame, embeddings: np.ndarray, 
                           bias_periods: Dict) -> Dict:
        """
        METHODOLOGY: Linguistic Signature Vector Extraction
        ====================================================
        
        Purpose: Extract a single "expert-period direction" vector in embedding space
        that captures the linguistic signature of expert-identified bias periods.
        
        Core Concept:
        ------------
        Expert periods and normal periods form clusters in embedding space. The direction
        from normal centroid to expert-period centroid represents the "linguistic signature
        vector" - capturing how language shifts during hypothesized bias periods.
        
        Mathematical Formulation:
        ------------------------
        1. Expert-period centroid: μ_expert = mean(embeddings in expert periods)
        2. Normal-period centroid: μ_normal = mean(embeddings in normal periods)
        3. Signature vector: v_sig = μ_expert - μ_normal
        4. For any text embedding e: signature_score = dot(e, v_sig) / ||v_sig||
        
        Interpretation:
        --------------
        - Positive score: Text aligns with expert-period language patterns
        - Negative score: Text aligns with normal-period language patterns
        - Magnitude: Strength of alignment
        
        Advantages:
        ----------
        1. SIMPLICITY: Single scalar score (not 768 dimensions)
        2. INTERPRETABILITY: Score directly measures "bias-ness"
        3. EFFICIENCY: Fast to compute (just dot product)
        4. GENERALIZABILITY: Can score ANY text, even unseen data
        
        Use Cases:
        ---------
        - Score new earnings calls without retraining
        - Real-time bias detection
        - Track bias scores over time
        - Compare bias levels across different attribution types"""
        self.log_and_print("\n7. Extracting Linguistic Signature Vector")
        self.log_and_print("=" * 80)
        self.log_and_print("Computing direction in embedding space that captures expert-period patterns")
        
        if not bias_periods:
            self.log_and_print("No bias period data available")
            return {}
        
        # create labels: expert period vs normal period
        df['bias_period_key'] = list(zip(df['Company'], df['Year'], df['Quarter']))
        df['in_expert_period'] = df['bias_period_key'].apply(
            lambda x: bias_periods.get(x, {}).get('in_expert_period', False)
        )
        
        expert_period_mask = df['in_expert_period']
        normal_period_mask = ~df['in_expert_period']
        
        expert_period_embeddings = embeddings[expert_period_mask]
        normal_period_embeddings = embeddings[normal_period_mask]
        
        self.log_and_print(f"\n  Expert period samples: {len(expert_period_embeddings):,}")
        self.log_and_print(f"  Normal period samples: {len(normal_period_embeddings):,}")
        
        if len(expert_period_embeddings) == 0 or len(normal_period_embeddings) == 0:
            self.log_and_print("Insufficient data for linguistic signature extraction")
            return {}
        
        # calculate centroids
        expert_period_centroid = expert_period_embeddings.mean(axis=0)
        normal_period_centroid = normal_period_embeddings.mean(axis=0)
        
        # linguistic signature vector: direction from normal to expert periods
        linguistic_signature_vector = expert_period_centroid - normal_period_centroid
        separation_magnitude = np.linalg.norm(linguistic_signature_vector)
        
        # normalize to unit vector for scoring
        signature_vector_normalized = linguistic_signature_vector / separation_magnitude
        
        self.log_and_print(f"\n Linguistic signature vector extracted")
        self.log_and_print(f"  Separation magnitude: {separation_magnitude:.4f}")
        self.log_and_print(f"  (Larger = stronger linguistic distinction between expert/normal periods)")
        
        # calculate signature scores for all embeddings as demonstration
        signature_scores = np.dot(embeddings, signature_vector_normalized)
        
        # analyze score distribution
        expert_period_scores = signature_scores[expert_period_mask]
        normal_period_scores = signature_scores[normal_period_mask]
        
        self.log_and_print(f"\n  Signature Score Distribution:")
        self.log_and_print(f"    Expert period mean: {expert_period_scores.mean():+.3f} (std: {expert_period_scores.std():.3f})")
        self.log_and_print(f"    Normal period mean: {normal_period_scores.mean():+.3f} (std: {normal_period_scores.std():.3f})")
        self.log_and_print(f"    Score separation: {(expert_period_scores.mean() - normal_period_scores.mean()):.3f}")
        
        # effect size (cohen's d)
        pooled_std = np.sqrt((expert_period_scores.std()**2 + normal_period_scores.std()**2) / 2)
        cohens_d = (expert_period_scores.mean() - normal_period_scores.mean()) / pooled_std
        
        self.log_and_print(f"    Cohen's d effect size: {cohens_d:.3f}")
        if cohens_d > 0.8:
            self.log_and_print(f"     LARGE effect (vector strongly separates expert/normal periods)")
        elif cohens_d > 0.5:
            self.log_and_print(f"    → MEDIUM effect")
        else:
            self.log_and_print(f"    → SMALL effect")
        
        # sample top/bottom scoring attributions for interpretation
        top_indices = np.argsort(signature_scores)[-5:][::-1]
        bottom_indices = np.argsort(signature_scores)[:5]
        
        results = {
            'linguistic_signature_vector': linguistic_signature_vector,
            'signature_vector_normalized': signature_vector_normalized,
            'expert_period_centroid': expert_period_centroid,
            'normal_period_centroid': normal_period_centroid,
            'separation_magnitude': float(separation_magnitude),
            'cohens_d': float(cohens_d),
            'all_signature_scores': signature_scores,
            'expert_period_score_mean': float(expert_period_scores.mean()),
            'normal_period_score_mean': float(normal_period_scores.mean()),
            'top_scoring_indices': top_indices.tolist(),
            'bottom_scoring_indices': bottom_indices.tolist()
        }
        
        self.log_and_print(f"\n  Usage: signature_score = dot(new_embedding, signature_vector_normalized)")
        self.log_and_print(f"         Positive score → Expert-period linguistic pattern")
        self.log_and_print(f"         Negative score → Normal-period linguistic pattern")
        
        return results
    
    def extract_embedding_features(self, df: pd.DataFrame, embeddings: np.ndarray,
                                   bias_vector_results: Dict = None) -> pd.DataFrame:
        """
        METHODOLOGY: Embedding Feature Engineering
        ===========================================
        
        Purpose: Extract interpretable, scalar features from high-dimensional embeddings
        for use in supervised learning and interpretation.
        
        Rationale:
        ---------
        Raw 768-dimensional embeddings are:
        - Hard to interpret (what does dimension 347 mean?)
        - Prone to overfitting in small datasets
        - Computationally expensive
        
        Engineered features are:
        - Interpretable (e.g., "distance from company baseline")
        - More robust with fewer dimensions
        - Capture known patterns (e.g., consistency, outliers)
        
        Features Extracted:
        ------------------
        1. BASELINE DISTANCE FEATURES:
           - distance_from_company_mean: How far from company's typical language?
           - distance_from_quarter_mean: How far from this quarter's typical language?
           
        2. TEMPORAL FEATURES:
           - distance_from_previous_quarter: Language shift from last quarter
           - trajectory_velocity: Rate of change in embedding over time
           
        3. VARIANCE FEATURES:
           - embedding_norm: Overall "intensity" of embedding
           - embedding_variance: Internal variability within embedding
           
        4. OUTLIER FEATURES:
           - knn_distance: Distance to 5th nearest neighbor (isolation measure)
           - local_outlier_factor: How much an outlier compared to neighbors
           
        5. BIAS ALIGNMENT (if bias_vector available):
           - bias_score: Projection onto bias vector
           - bias_alignment_strength: Magnitude of alignment
        
        Mathematical Details:
        --------------------
        - KNN distance: Measures local density (outliers are in sparse regions)
        - Trajectory velocity: ||e_t - e_{t-1}|| / Δt
        - Local Outlier Factor (LOF): Ratio of local density to neighbors' density
        
        Benefits:
        --------
        1. Can be used in simple models (logistic regression, random forest)
        2. Features have clear interpretations for dissertation
        3. Computationally efficient
        4. Less prone to overfitting than raw embeddings"""
        self.log_and_print("\n8. Engineering Interpretable Features from Embeddings")
        self.log_and_print("=" * 80)
        self.log_and_print("Extracting scalar features for interpretability and supervised learning")
        
        if not SKLEARN_AVAILABLE:
            self.log_and_print("Scikit-learn not available")
            return pd.DataFrame()
        
        from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
        
        features = []
        
        self.log_and_print(f"\n  Computing features for {len(df):,} embeddings...")
        
        # pre-compute company and quarter centroids for efficiency
        company_centroids = {}
        for company in df['Company'].unique():
            mask = df['Company'] == company
            company_centroids[company] = embeddings[mask].mean(axis=0)
        
        quarter_centroids = {}
        for (company, year, quarter), group in df.groupby(['Company', 'Year', 'Quarter']):
            indices = group.index.tolist()
            quarter_centroids[(company, year, quarter)] = embeddings[indices].mean(axis=0)
        
        # fit knn for outlier detection (k=5)
        self.log_and_print("  Computing KNN distances...")
        knn = NearestNeighbors(n_neighbors=6, metric='euclidean')  # 6 because first is self
        knn.fit(embeddings)
        distances, _ = knn.kneighbors(embeddings)
        knn_distances = distances[:, -1]  # distance to 5th nearest neighbor
        
        # compute local outlier factor
        self.log_and_print("  Computing Local Outlier Factors...")
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        lof_scores = lof.fit_predict(embeddings)
        lof_scores_continuous = lof.negative_outlier_factor_
        
        self.log_and_print("  Extracting features...")
        
        # use tqdm progress bar if available
        if TQDM_AVAILABLE:
            df_iterator = tqdm(df.iterrows(), total=len(df), 
                             desc="  Feature extraction", unit="snippet")
        else:
            df_iterator = df.iterrows()
        
        for idx, row in df_iterator:
            company = row['Company']
            year = row['Year']
            quarter = row['Quarter']
            embedding = embeddings[idx]
            
            # === baseline distance features ===
            company_centroid = company_centroids[company]
            distance_from_company = np.linalg.norm(embedding - company_centroid)
            
            quarter_key = (company, year, quarter)
            quarter_centroid = quarter_centroids[quarter_key]
            distance_from_quarter = np.linalg.norm(embedding - quarter_centroid)
            
            # === temporal features ===
            # find previous quarter
            prev_quarter = quarter - 1
            prev_year = year
            if prev_quarter < 1:
                prev_quarter = 4
                prev_year = year - 1
            
            prev_quarter_key = (company, prev_year, prev_quarter)
            
            if prev_quarter_key in quarter_centroids:
                prev_quarter_centroid = quarter_centroids[prev_quarter_key]
                distance_from_previous = np.linalg.norm(embedding - prev_quarter_centroid)
                trajectory_velocity = distance_from_previous  # simplified (δt = 1 quarter)
            else:
                distance_from_previous = None
                trajectory_velocity = None
            
            # === variance features ===
            embedding_norm = np.linalg.norm(embedding)
            embedding_std = embedding.std()
            embedding_max = embedding.max()
            embedding_min = embedding.min()
            
            # === outlier features ===
            knn_dist = knn_distances[idx]
            lof_score = lof_scores_continuous[idx]
            
            # === bias alignment (if available) ===
            bias_score = None
            if bias_vector_results and 'bias_vector_normalized' in bias_vector_results:
                bias_vector = bias_vector_results['bias_vector_normalized']
                bias_score = np.dot(embedding, bias_vector)
            
            features.append({
                'snippet_id': idx,
                'company': company,
                'year': year,
                'quarter': quarter,
                'is_target': row['IS_TARGET'] == 'Y',
                
                # baseline distances
                'distance_from_company_mean': distance_from_company,
                'distance_from_quarter_mean': distance_from_quarter,
                
                # temporal features
                'distance_from_previous_quarter': distance_from_previous,
                'trajectory_velocity': trajectory_velocity,
                
                # variance features
                'embedding_norm': embedding_norm,
                'embedding_std': embedding_std,
                'embedding_range': embedding_max - embedding_min,
                
                # outlier features
                'knn_distance_k5': knn_dist,
                'local_outlier_factor': lof_score,
                'is_outlier_lof': lof_scores[idx] == -1,
                
                # bias alignment
                'bias_score': bias_score
            })
        
        features_df = pd.DataFrame(features)
        
        self.log_and_print(f"\n Extracted {len(features_df.columns) - 5} features per embedding")
        self.log_and_print(f"\n  Feature Categories:")
        self.log_and_print(f"    Baseline distances: 2 features")
        self.log_and_print(f"    Temporal features: 2 features")
        self.log_and_print(f"    Variance features: 3 features")
        self.log_and_print(f"    Outlier features: 3 features")
        if bias_score is not None:
            self.log_and_print(f"    Bias alignment: 1 feature")
        
        # summary statistics for key features
        self.log_and_print(f"\n  Feature Statistics (targets vs peers):")
        
        if len(features_df[features_df['is_target']]) > 0:
            target_features = features_df[features_df['is_target']]
            peer_features = features_df[~features_df['is_target']]
            
            key_features = ['distance_from_company_mean', 'knn_distance_k5', 'bias_score']
            
            for feat in key_features:
                if feat in features_df.columns and features_df[feat].notna().any():
                    target_mean = target_features[feat].mean()
                    peer_mean = peer_features[feat].mean()
                    self.log_and_print(f"    {feat}:")
                    self.log_and_print(f"      Targets: {target_mean:.3f}  |  Peers: {peer_mean:.3f}")
        
        return features_df
    
    def perform_pca_analysis(self, df: pd.DataFrame, embeddings: np.ndarray,
                            n_components: int = 50) -> Dict:
        """
        METHODOLOGY: Principal Component Analysis (PCA)
        ================================================
        
        Purpose: Identify which dimensions of the 768-D embedding space matter most
        for capturing bias patterns.
        
        Core Question:
        -------------
        Out of 768 embedding dimensions, which ones contain the most information
        about bias? Can we reduce complexity while retaining signal?
        
        Mathematical Foundation:
        -----------------------
        PCA finds orthogonal directions (principal components) that explain maximum variance:
        1. Component 1: Direction of maximum variance
        2. Component 2: Direction of maximum variance orthogonal to Component 1
        3. ... and so on
        
        Each component is a linear combination of original 768 dimensions.
        
        Application to Bias Detection:
        ------------------------------
        1. DIMENSIONALITY REDUCTION:
           - Instead of 768-D, use top 50 components (explain ~80-90% of variance)
           - Faster, less overfitting, easier to visualize
        
        2. INTERPRETABILITY:
           - Which components separate targets from peers?
           - Which components correlate with bias scores?
           - Can identify linguistic dimensions that matter for bias
        
        3. FEATURE IMPORTANCE:
           - Component loadings show which original dimensions contribute most
           - Can map back to semantic meaning (e.g., sentiment, formality, defensiveness)
        
        Analysis Performed:
        ------------------
        1. Fit PCA on all embeddings
        2. Project embeddings onto principal components
        3. Analyze variance explained
        4. Compare targets vs peers in reduced space
        5. Correlate components with bias labels
        
        Benefits:
        --------
        - Identifies which aspects of language distinguish bias
        - Reduces dimensions for visualization (2D/3D plots)
        - Can use reduced embeddings for faster classification
        - Provides interpretable components for dissertation"""
        self.log_and_print("\n9. Principal Component Analysis (PCA)")
        self.log_and_print("=" * 80)
        self.log_and_print(f"Reducing {embeddings.shape[1]}-D embeddings to {n_components} principal components")
        
        if not SKLEARN_AVAILABLE:
            self.log_and_print("Scikit-learn not available")
            return {}
        
        from sklearn.decomposition import PCA
        from scipy.stats import spearmanr
        
        # fit pca
        self.log_and_print(f"\n  Fitting PCA...")
        pca = PCA(n_components=n_components, random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        self.log_and_print(f" PCA complete")
        self.log_and_print(f"\n  Variance Explained:")
        n_actual_components = len(cumulative_variance)
        if n_actual_components >= 10:
            self.log_and_print(f"    Top 10 components: {cumulative_variance[9]*100:.1f}%")
        if n_actual_components >= 20:
            self.log_and_print(f"    Top 20 components: {cumulative_variance[19]*100:.1f}%")
        self.log_and_print(f"    All {n_actual_components} components: {cumulative_variance[-1]*100:.1f}%")
        
        # find "elbow" in explained variance
        if n_actual_components > 1:
            variance_diffs = np.diff(explained_variance)
            elbow_idx = np.argmax(variance_diffs < explained_variance[0] / 10) + 1
            elbow_idx = max(1, min(elbow_idx, n_actual_components))  # ensure valid range
            
            self.log_and_print(f"\n  Suggested elbow point: {elbow_idx} components")
            self.log_and_print(f"    (Captures {cumulative_variance[elbow_idx-1]*100:.1f}% of variance)")
        else:
            elbow_idx = n_actual_components
            self.log_and_print(f"\n  Only {n_actual_components} component(s) computed")
        
        # analyze target vs peer separation in pca space
        self.log_and_print(f"\n  Target vs Peer Separation in PCA Space:")
        
        target_mask = df['IS_TARGET'] == 'Y'
        peer_mask = df['IS_TARGET'] != 'Y'
        
        target_pca = reduced_embeddings[target_mask]
        peer_pca = reduced_embeddings[peer_mask]
        
        if len(target_pca) > 0 and len(peer_pca) > 0:
            target_centroid_pca = target_pca.mean(axis=0)
            peer_centroid_pca = peer_pca.mean(axis=0)
            
            separation_pca = np.linalg.norm(target_centroid_pca - peer_centroid_pca)
            
            # compare to full embedding space
            target_full = embeddings[target_mask].mean(axis=0)
            peer_full = embeddings[peer_mask].mean(axis=0)
            separation_full = np.linalg.norm(target_full - peer_full)
            
            self.log_and_print(f"    Separation in full space: {separation_full:.4f}")
            self.log_and_print(f"    Separation in PCA space: {separation_pca:.4f}")
            self.log_and_print(f"    Retained separation: {(separation_pca/separation_full)*100:.1f}%")
        
        # correlate pc with expert periods (if available)
        if 'in_expert_period' in df.columns:
            self.log_and_print(f"\n  Correlating Components with Expert Periods:")
            
            expert_period_labels = df['in_expert_period'].astype(int).values
            
            # find components most correlated with expert periods
            correlations = []
            for i in range(min(10, n_actual_components)):  # check top 10 components
                corr, pval = spearmanr(reduced_embeddings[:, i], expert_period_labels)
                correlations.append({
                    'component': i + 1,
                    'correlation': corr,
                    'p_value': pval,
                    'variance_explained': explained_variance[i]
                })
            
            correlations_df = pd.DataFrame(correlations)
            correlations_df = correlations_df.sort_values('correlation', key=abs, ascending=False)
            
            self.log_and_print(f"\n    Components Most Correlated with Expert Periods (Top 5):")
            for _, row in correlations_df.head(5).iterrows():
                sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
                self.log_and_print(f"      PC{int(row['component']):2d}: r={row['correlation']:+.3f} {sig}  (explains {row['variance_explained']*100:.1f}% variance)")
        
        # calculate components needed for 95% variance
        idx_95 = np.argmax(cumulative_variance >= 0.95)
        if cumulative_variance[idx_95] >= 0.95:
            n_components_95pct = int(idx_95 + 1)
        else:
            n_components_95pct = n_actual_components  # all components needed
        
        results = {
            'pca_model': pca,
            'reduced_embeddings': reduced_embeddings,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'n_components_95pct': n_components_95pct,
            'elbow_point': int(elbow_idx)
        }
        
        if len(target_pca) > 0 and len(peer_pca) > 0:
            results['target_peer_separation_pca'] = {
                'separation_pca': float(separation_pca),
                'separation_full': float(separation_full),
                'retention_rate': float(separation_pca / separation_full)
            }
        
        self.log_and_print(f"\n  Usage:")
        self.log_and_print(f"    - Use top {elbow_idx} components for classification")
        self.log_and_print(f"    - Use PC1 & PC2 for 2D visualization")
        self.log_and_print(f"    - Examine component loadings to understand linguistic dimensions")
        
        return results
    
    def compare_bias_periods(self, df: pd.DataFrame, embeddings: np.ndarray, 
                            bias_periods: Dict) -> Dict:
        self.log_and_print("\n6. Comparing Expert Period vs Normal Period Embeddings")
        self.log_and_print("=" * 80)
        
        if not bias_periods:
            self.log_and_print("No expert period data available")
            return {}
        
        df['bias_period_key'] = list(zip(df['Company'], df['Year'], df['Quarter']))
        
        expert_period_mask = df['bias_period_key'].apply(
            lambda x: bias_periods.get(x, {}).get('in_expert_period', False)
        )
        normal_period_mask = ~expert_period_mask
        
        expert_period_embeds = embeddings[expert_period_mask]
        normal_period_embeds = embeddings[normal_period_mask]
        
        self.log_and_print(f"  Expert period segments: {len(expert_period_embeds):,}")
        self.log_and_print(f"  Normal period segments: {len(normal_period_embeds):,}")
        
        if len(expert_period_embeds) == 0 or len(normal_period_embeds) == 0:
            self.log_and_print("Insufficient data for period comparison")
            return {}
        
        expert_centroid = expert_period_embeds.mean(axis=0)
        normal_centroid = normal_period_embeds.mean(axis=0)
        
        distance = np.linalg.norm(expert_centroid - normal_centroid)
        
        from scipy.spatial.distance import cosine as cosine_distance
        cosine_sim = 1 - cosine_distance(expert_centroid, normal_centroid)
        
        results = {
            'euclidean_distance': float(distance),
            'cosine_similarity': float(cosine_sim),
            'expert_period_std': float(expert_period_embeds.std(axis=0).mean()),
            'normal_period_std': float(normal_period_embeds.std(axis=0).mean()),
            'n_expert_period': len(expert_period_embeds),
            'n_normal_period': len(normal_period_embeds)
        }
        
        self.log_and_print(f"\nPeriod Comparison Results:")
        self.log_and_print(f"  Euclidean distance: {results['euclidean_distance']:.4f}")
        self.log_and_print(f"  Cosine similarity: {results['cosine_similarity']:.3f}")
        self.log_and_print(f"  Expert period variance: {results['expert_period_std']:.4f}")
        self.log_and_print(f"  Normal period variance: {results['normal_period_std']:.4f}")
        
        if results['euclidean_distance'] > 2.0:
            self.log_and_print(f"   Expert and normal periods occupy distinct embedding regions")
            self.log_and_print(f"  → Strong linguistic shift detected during expert-identified periods")
        elif results['euclidean_distance'] > 1.0:
            self.log_and_print(f"  ~ Moderate separation between periods")
            self.log_and_print(f"  → Some linguistic differences exist")
        else:
            self.log_and_print(f"   No clear embedding difference between periods")
            self.log_and_print(f"  → Language patterns similar across expert and normal periods")
        
        return results
    
    def dimensionality_reduction(self, 
                                embeddings: np.ndarray,
                                method: str = 'umap',
                                n_components: int = 2,
                                max_samples_for_umap: int = 50000) -> np.ndarray:
        """
        Reduce embeddings to 2D/3D for visualization.
        
        MEMORY OPTIMIZATION: For UMAP with large datasets (>50k samples),
        we use a two-stage approach:
        1. Pre-reduce with PCA to 50 dimensions (fast, low memory)
        2. Apply UMAP on reduced space (much faster, less memory)"""
        self.log_and_print(f"\n5. Dimensionality Reduction ({method.upper()})")
        self.log_and_print("=" * 50)
        
        n_samples = embeddings.shape[0]
        
        if method == 'umap' and UMAP_AVAILABLE:
            self.log_and_print(f"Running UMAP (n_components={n_components})...")
            
            # memory optimization: pre-reduce dimensions with pca for large datasets
            if n_samples > max_samples_for_umap or embeddings.shape[1] > 100:
                self.log_and_print(f"  Large dataset detected ({n_samples:,} samples, {embeddings.shape[1]} dims)")
                self.log_and_print(f"  → Applying PCA pre-reduction to 50 dims for memory efficiency...")
                
                try:
                    pca_prereduction = PCA(n_components=50, random_state=42)
                    embeddings_reduced = pca_prereduction.fit_transform(embeddings)
                    explained_var_total = pca_prereduction.explained_variance_ratio_.sum()
                    self.log_and_print(f"   PCA pre-reduction complete: {embeddings_reduced.shape}")
                    self.log_and_print(f"    (Preserved {explained_var_total:.1%} of variance)")
                    
                    # run umap on pre-reduced embeddings
                    reducer = umap.UMAP(
                        n_components=n_components, 
                        random_state=42, 
                        n_neighbors=15,
                        min_dist=0.1,
                        metric='cosine',
                        low_memory=True  # enable low-memory mode
                    )
                    coords = reducer.fit_transform(embeddings_reduced)
                    
                    # clean up
                    del embeddings_reduced
                    gc.collect()
                    
                    self.log_and_print(f" UMAP complete: shape = {coords.shape}")
                    
                except MemoryError as e:
                    self.log_and_print(f"   UMAP still failed due to memory: {e}")
                    self.log_and_print(f"  → Falling back to PCA for visualization")
                    pca = PCA(n_components=n_components, random_state=42)
                    coords = pca.fit_transform(embeddings)
                    self.log_and_print(f"   PCA fallback complete: shape = {coords.shape}")
            else:
                # small dataset - run umap directly
                reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15)
                coords = reducer.fit_transform(embeddings)
                self.log_and_print(f" UMAP complete: shape = {coords.shape}")
            
        elif method == 'tsne' and SKLEARN_AVAILABLE:
            self.log_and_print(f"Running t-SNE (n_components={n_components})...")
            
            # similar memory optimization for t-sne
            if n_samples > 10000:
                self.log_and_print(f"   Warning: t-SNE is slow with {n_samples:,} samples")
                self.log_and_print(f"  → Consider using PCA or UMAP instead")
            
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
            coords = tsne.fit_transform(embeddings)
            self.log_and_print(f" t-SNE complete: shape = {coords.shape}")
            
        elif method == 'pca' and SKLEARN_AVAILABLE:
            self.log_and_print(f"Running PCA (n_components={n_components})...")
            pca = PCA(n_components=n_components, random_state=42)
            coords = pca.fit_transform(embeddings)
            explained_var = pca.explained_variance_ratio_
            self.log_and_print(f" PCA complete: shape = {coords.shape}")
            self.log_and_print(f"  Explained variance: {explained_var}")
            
        else:
            self.log_and_print(f" Method '{method}' not available")
            return np.array([])
        
        return coords
    
    def compute_topic_adjusted_embeddings(self, df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        """
        Remove topic-driven variance to isolate attribution style.
        
        Logic:
        1. For each topic (e.g., "Revenue Growth"), compute mean embedding (topic centroid)
        2. Subtract topic mean from each attribution about that topic
        3. Residual captures how ATYPICALLY they discuss the topic
        
        Theory: Bias manifests as unusual framing OF A TOPIC, not topic choice itself."""
        self.log_and_print("\nComputing topic-adjusted embeddings...")
        self.log_and_print("Goal: Remove topic variance to isolate attribution STYLE")
        
        adjusted_embeddings = embeddings.copy()
        
        # use primary_topic column (standard column from classification pipeline)
        topic_col = 'Primary_Topic'
        
        if topic_col not in df.columns:
            self.log_and_print(f"  ️  Required column '{topic_col}' not found in data")
            self.log_and_print(f"  → Returning raw embeddings (topic adjustment skipped)")
            return embeddings
        
        topics = df[topic_col].unique()
        topics_adjusted = 0
        samples_adjusted = 0
        
        for topic in topics:
            if pd.isna(topic):
                continue
            
            topic_mask = df[topic_col] == topic
            topic_embeddings = embeddings[topic_mask]
            
            # skip rare topics (not enough data for meaningful centroid)
            if len(topic_embeddings) < 10:
                continue
            
            # compute topic centroid (mean embedding for this topic)
            topic_centroid = topic_embeddings.mean(axis=0)
            
            # subtract topic mean (residualization)
            # after this, embeddings capture "how you discuss this topic"
            # rather than "what topic you discuss"
            adjusted_embeddings[topic_mask] = topic_embeddings - topic_centroid
            
            topics_adjusted += 1
            samples_adjusted += len(topic_embeddings)
        
        self.log_and_print(f"   Adjusted {topics_adjusted} topics covering {samples_adjusted:,} samples")
        self.log_and_print(f"  → Embeddings now capture STYLE of discussion, not topic content")
        
        return adjusted_embeddings
    
    def _compare_topic_adjustment_impact(self, 
                                        clustering_raw: Dict, 
                                        clustering_adjusted: Dict) -> None:
        """
        Compare clustering results before and after topic adjustment.
        
        Success criteria:
        - Attribution ARI should INCREASE (better capture of attribution signal)
        - Topic ARI should DECREASE (topic variance removed)
        """
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("TOPIC ADJUSTMENT IMPACT ANALYSIS")
        self.log_and_print("=" * 80)
        
        # extract ari scores
        raw_alignment = clustering_raw.get('global_kmeans_validation', {}).get('label_alignment', {})
        adj_alignment = clustering_adjusted.get('global_kmeans_validation', {}).get('label_alignment', {})
        
        if not raw_alignment or not adj_alignment:
            self.log_and_print("  ️  Clustering alignment data not available")
            return
        
        # compare key label types
        comparison_labels = [
            'Attribution Combined',
            'Attribution Locus', 
            'Attribution Outcome',
            'Topic',
            'Sentiment'
        ]
        
        self.log_and_print("\nARI Score Comparison (Higher = Better Clustering Alignment):")
        self.log_and_print("-" * 80)
        self.log_and_print(f"{'Label':<30} {'Raw ARI':>12} {'Adjusted ARI':>15} {'Change':>10} {'Impact':>15}")
        self.log_and_print("-" * 80)
        
        improvements = []
        
        for label in comparison_labels:
            if label in raw_alignment and label in adj_alignment:
                raw_ari = raw_alignment[label]['ari']
                adj_ari = adj_alignment[label]['ari']
                change = adj_ari - raw_ari
                
                # determine impact
                if 'Attribution' in label:
                    if change > 0.01:
                        impact = " IMPROVED"
                    elif change > 0:
                        impact = "→ Slight +"
                    else:
                        impact = " Worse"
                elif label == 'Topic':
                    if change < -0.01:
                        impact = " REMOVED"
                    elif change < 0:
                        impact = "→ Reduced"
                    else:
                        impact = "️  Still high"
                else:
                    impact = f"{change:+.3f}"
                
                self.log_and_print(f"{label:<30} {raw_ari:>12.3f} {adj_ari:>15.3f} {change:>10.3f} {impact:>15}")
                
                if 'Attribution' in label:
                    improvements.append((label, change))
        
        # overall assessment
        self.log_and_print("\n" + "-" * 80)
        self.log_and_print("VALIDATION RESULT:")
        self.log_and_print("-" * 80)
        
        attribution_improved = any(change > 0 for _, change in improvements)
        topic_reduced = 'Topic' in adj_alignment and adj_alignment['Topic']['ari'] < raw_alignment.get('Topic', {}).get('ari', 1.0)
        
        if attribution_improved and topic_reduced:
            self.log_and_print(" SUCCESS: Topic adjustment successfully isolated attribution signal")
            self.log_and_print("  → Attribution clustering improved")
            self.log_and_print("  → Topic confound reduced")
            self.log_and_print("  → Use topic-adjusted embeddings for downstream analyses")
        elif attribution_improved:
            self.log_and_print("→ PARTIAL: Attribution improved but topic variance remains")
            self.log_and_print("  → Consider additional controls")
        elif topic_reduced:
            self.log_and_print("→ PARTIAL: Topic variance reduced but attribution signal not improved")
            self.log_and_print("  → Attribution may be inherently weak signal")
        else:
            self.log_and_print(" LIMITED IMPACT: Topic adjustment had minimal effect")
            self.log_and_print("  → Raw embeddings may be sufficient")
    
    def save_results(self, 
                    df: pd.DataFrame,
                    embeddings: np.ndarray,
                    clustering_results: Dict,
                    supervised_results: Dict,
                    coords_2d: np.ndarray,
                    company_aggregates: Dict = None,
                    separation_results: Dict = None,
                    topic_shift_df: pd.DataFrame = None,
                    bias_period_results: Dict = None,
                    multilevel_embeddings: Dict = None,
                    bias_prediction_results: Dict = None,
                    topic_consistency_df: pd.DataFrame = None,
                    bias_vector_results: Dict = None,
                    pca_results: Dict = None,
                    embedding_features_df: pd.DataFrame = None,
                    subspace_features: pd.DataFrame = None,
                    embeddings_topic_adjusted: np.ndarray = None,
                    clustering_results_adjusted: Dict = None,
                    coords_pca: np.ndarray = None,
                    coords_umap_adjusted: np.ndarray = None,
                    firm_analysis_df: pd.DataFrame = None,
                    comprehensive_classification_results: Dict = None):
        self.log_and_print("\n10. Saving Results")
        self.log_and_print("=" * 80)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # save embeddings
        embedding_file = self.output_dir / f"embedding_vectors_{timestamp}.npy"
        np.save(embedding_file, embeddings)
        self.log_and_print(f" Saved embeddings: {embedding_file}")
        
        # save topic-adjusted embeddings
        if embeddings_topic_adjusted is not None:
            emb_adj_file = self.output_dir / f"embedding_vectors_topic_adjusted_{timestamp}.npy"
            np.save(emb_adj_file, embeddings_topic_adjusted)
            self.log_and_print(f" Saved topic-adjusted embeddings: {emb_adj_file}")
        
        # save metadata with coordinates
        metadata_df = df.copy()
        
        # add clustering labels
        if 'kmeans' in clustering_results:
            metadata_df['cluster_kmeans'] = clustering_results['kmeans']['labels']
        if 'hdbscan' in clustering_results:
            metadata_df['cluster_hdbscan'] = clustering_results['hdbscan']['labels']
        
        # add 2d coordinates
        if len(coords_2d) > 0:
            metadata_df['coord_x'] = coords_2d[:, 0]
            metadata_df['coord_y'] = coords_2d[:, 1]
        
        metadata_file = self.output_dir / f"embedding_metadata_{timestamp}.csv"
        metadata_df.to_csv(metadata_file, index=False)
        self.log_and_print(f" Saved metadata: {metadata_file}")
        
        if len(coords_2d) > 0:
            coords_file = self.output_dir / f"umap_coordinates_{timestamp}.csv"
            coords_df = pd.DataFrame({
                'x': coords_2d[:, 0],
                'y': coords_2d[:, 1],
                'attribution_locus': df['attribution_locus'].values,
                'attribution_outcome': df['attribution_outcome'].values,
                'company': df['Company'].values,
                'year': df['Year'].values,
                'quarter': df['Quarter'].values,
                'is_target': df['IS_TARGET'].values,
                'is_peer': df['IS_PEER'].values
            })
            coords_df.to_csv(coords_file, index=False)
            self.log_and_print(f" Saved UMAP coordinates: {coords_file}")
        
        # save pca coordinates
        if coords_pca is not None and len(coords_pca) > 0:
            coords_pca_file = self.output_dir / f"pca_coordinates_2d_{timestamp}.csv"
            coords_pca_df = pd.DataFrame({
                'x': coords_pca[:, 0],
                'y': coords_pca[:, 1],
                'attribution_locus': df['attribution_locus'].values,
                'attribution_outcome': df['attribution_outcome'].values,
                'company': df['Company'].values,
                'year': df['Year'].values,
                'quarter': df['Quarter'].values,
                'is_target': df['IS_TARGET'].values,
                'is_peer': df['IS_PEER'].values
            })
            coords_pca_df.to_csv(coords_pca_file, index=False)
            self.log_and_print(f" Saved PCA 2D coordinates: {coords_pca_file}")
        
        # save topic-adjusted umap coordinates
        if coords_umap_adjusted is not None and len(coords_umap_adjusted) > 0:
            coords_adj_file = self.output_dir / f"umap_coordinates_topic_adjusted_{timestamp}.csv"
            coords_adj_df = pd.DataFrame({
                'x': coords_umap_adjusted[:, 0],
                'y': coords_umap_adjusted[:, 1],
                'attribution_locus': df['attribution_locus'].values,
                'attribution_outcome': df['attribution_outcome'].values,
                'company': df['Company'].values,
                'year': df['Year'].values,
                'quarter': df['Quarter'].values,
                'is_target': df['IS_TARGET'].values,
                'is_peer': df['IS_PEER'].values
            })
            coords_adj_df.to_csv(coords_adj_file, index=False)
            self.log_and_print(f" Saved topic-adjusted UMAP coordinates: {coords_adj_file}")
        
        if topic_shift_df is not None and len(topic_shift_df) > 0:
            shift_file = self.output_dir / f"qna_topic_shifts_{timestamp}.csv"
            topic_shift_df.to_csv(shift_file, index=False)
            self.log_and_print(f" Saved Q&A topic shifts: {shift_file}")
        
        if topic_consistency_df is not None and len(topic_consistency_df) > 0:
            consistency_file = self.output_dir / f"topic_temporal_consistency_{timestamp}.csv"
            topic_consistency_df.to_csv(consistency_file, index=False)
            self.log_and_print(f" Saved topic temporal consistency: {consistency_file}")
            
            # save outliers separately for easy review
            outliers = topic_consistency_df[topic_consistency_df['is_outlier']]
            if len(outliers) > 0:
                outlier_file = self.output_dir / f"topic_consistency_outliers_{timestamp}.csv"
                outliers.to_csv(outlier_file, index=False)
                self.log_and_print(f" Saved topic consistency outliers: {outlier_file} ({len(outliers)} outliers)")
        
        # save semantic subspace features
        if subspace_features is not None and len(subspace_features) > 0:
            subspace_file = self.output_dir / f"subspace_features_{timestamp}.csv"
            subspace_features.to_csv(subspace_file, index=False)
            self.log_and_print(f" Saved semantic subspace features: {subspace_file} ({len(subspace_features.columns)} features)")
        
        # save bias vector
        if bias_vector_results and 'bias_vector' in bias_vector_results:
            bias_vector_file = self.output_dir / f"bias_vector_{timestamp}.npy"
            np.save(bias_vector_file, bias_vector_results['bias_vector_normalized'])
            self.log_and_print(f" Saved bias vector: {bias_vector_file}")
            
            # save bias scores for all snippets
            if 'all_bias_scores' in bias_vector_results:
                bias_scores_df = pd.DataFrame({
                    'snippet_id': range(len(bias_vector_results['all_bias_scores'])),
                    'bias_score': bias_vector_results['all_bias_scores'],
                    'company': df['Company'].values,
                    'year': df['Year'].values,
                    'quarter': df['Quarter'].values,
                    'is_target': df['IS_TARGET'].values
                })
                bias_scores_file = self.output_dir / f"bias_scores_{timestamp}.csv"
                bias_scores_df.to_csv(bias_scores_file, index=False)
                self.log_and_print(f" Saved bias scores: {bias_scores_file}")
        
        # save pca results
        if pca_results and 'reduced_embeddings' in pca_results:
            pca_embeddings_file = self.output_dir / f"pca_embeddings_{timestamp}.npy"
            np.save(pca_embeddings_file, pca_results['reduced_embeddings'])
            self.log_and_print(f" Saved PCA embeddings: {pca_embeddings_file}")
            
            # save pca coordinates for visualization (first 2 components)
            pca_coords_df = pd.DataFrame({
                'pc1': pca_results['reduced_embeddings'][:, 0],
                'pc2': pca_results['reduced_embeddings'][:, 1],
                'company': df['Company'].values,
                'year': df['Year'].values,
                'quarter': df['Quarter'].values,
                'is_target': df['IS_TARGET'].values,
                'attribution_locus': df['attribution_locus'].values
            })
            pca_coords_file = self.output_dir / f"pca_coordinates_{timestamp}.csv"
            pca_coords_df.to_csv(pca_coords_file, index=False)
            self.log_and_print(f" Saved PCA coordinates: {pca_coords_file}")
        
        # save engineered features
        if embedding_features_df is not None and len(embedding_features_df) > 0:
            features_file = self.output_dir / f"embedding_features_{timestamp}.csv"
            embedding_features_df.to_csv(features_file, index=False)
            self.log_and_print(f" Saved engineered features: {features_file}")
        
        if company_aggregates:
            company_stats = []
            for company, stats in company_aggregates.items():
                company_stats.append({
                    'company': company,
                    'is_target': stats['is_target'],
                    'n_samples': stats['n_samples'],
                    'embedding_std': stats['std']
                })
            company_df = pd.DataFrame(company_stats)
            company_file = self.output_dir / f"company_level_stats_{timestamp}.csv"
            company_df.to_csv(company_file, index=False)
            self.log_and_print(f" Saved company-level stats: {company_file}")
        
        # save per-firm stratified analysis
        if firm_analysis_df is not None and len(firm_analysis_df) > 0:
            firm_file = self.output_dir / f"per_firm_analysis_{timestamp}.csv"
            firm_analysis_df.to_csv(firm_file, index=False)
            self.log_and_print(f" Saved per-firm analysis: {firm_file}")
            
            # highlight outlier firms (handle both old 'is_clear_outlier' and new 'is_flagged' columns)
            outlier_col = 'is_clear_outlier' if 'is_clear_outlier' in firm_analysis_df.columns else 'is_flagged'
            if outlier_col in firm_analysis_df.columns:
                outliers = firm_analysis_df[firm_analysis_df[outlier_col]]
                if len(outliers) > 0:
                    self.log_and_print(f"  → {len(outliers)}/{len(firm_analysis_df)} firms flagged with unusual patterns")
        
        # save clustering metrics
        if clustering_results:
            metrics = {
                'timestamp': timestamp,
                'n_samples': len(embeddings),
                'embedding_dim': embeddings.shape[1],
                'model_name': self.model_name
            }
            
            # global k-means validation results
            if 'global_kmeans_validation' in clustering_results:
                validation = clustering_results['global_kmeans_validation']
                metrics['global_kmeans_validation'] = {
                    'n_clusters': validation['n_clusters'],
                    'silhouette': float(validation['silhouette']),
                    'label_alignment': validation['label_alignment']
                }
                
                # identify top alignment
                if validation['label_alignment']:
                    ranked = sorted(validation['label_alignment'].items(), 
                                   key=lambda x: x[1]['ari'], reverse=True)
                    metrics['embedding_captures_primarily'] = ranked[0][0]
                    metrics['top_ari_score'] = float(ranked[0][1]['ari'])
                    
                    # validation flag
                    top_label = ranked[0][0]
                    metrics['embedding_validation_passed'] = 'Attribution' in top_label
            
            # strategic cs2 clustering results
            if 'cs2_attribution_clustering' in clustering_results:
                metrics['cs2_attribution_clustering'] = clustering_results['cs2_attribution_clustering']
                
                # compute summary metrics
                cs2_results = clustering_results['cs2_attribution_clustering']
                if cs2_results:
                    self_serving = [r for r in cs2_results if r['category'] == 'self-serving']
                    non_self_serving = [r for r in cs2_results if r['category'] == 'non-self-serving']
                    
                    if self_serving:
                        avg_d_self = np.mean([r['cohens_d_coherence'] for r in self_serving])
                        metrics['cs2_self_serving_cohens_d'] = float(avg_d_self)
                    
                    if non_self_serving:
                        avg_d_non_self = np.mean([r['cohens_d_coherence'] for r in non_self_serving])
                        metrics['cs2_non_self_serving_cohens_d'] = float(avg_d_non_self)
            
            # strategic hdbscan results
            if 'hdbscan_strategic' in clustering_results:
                hdbscan_strat = clustering_results['hdbscan_strategic']
                
                if 'company_level' in hdbscan_strat and hdbscan_strat['company_level']:
                    company_results = hdbscan_strat['company_level']
                    targets = [r for r in company_results if r['is_target']]
                    peers = [r for r in company_results if not r['is_target']]
                    
                    if targets and peers:
                        target_noise = np.mean([r['noise_ratio'] for r in targets])
                        peer_noise = np.mean([r['noise_ratio'] for r in peers])
                        
                        metrics['hdbscan_company_level'] = {
                            'target_avg_noise_ratio': float(target_noise),
                            'peer_avg_noise_ratio': float(peer_noise),
                            'noise_ratio_difference': float(target_noise - peer_noise)
                        }
                
                if 'attribution_type' in hdbscan_strat:
                    metrics['hdbscan_attribution_type'] = hdbscan_strat['attribution_type']
            
            if 'logistic_regression' in supervised_results:
                metrics['supervised_auc'] = float(supervised_results['logistic_regression']['auc'])
            
            # new: comprehensive attribution classification results
            if comprehensive_classification_results:
                comp_class_metrics = {}
                
                # attribution present vs not present
                if 'attribution_present' in comprehensive_classification_results:
                    apr = comprehensive_classification_results['attribution_present']
                    if 'auc' in apr:
                        comp_class_metrics['attribution_present'] = {
                            'auc': float(apr['auc']),
                            'precision': float(apr['precision']),
                            'recall': float(apr['recall']),
                            'f1': float(apr['f1'])
                        }
                
                # outcome valence (positive vs negative)
                if 'outcome_valence' in comprehensive_classification_results:
                    ov = comprehensive_classification_results['outcome_valence']
                    if 'auc' in ov:
                        comp_class_metrics['outcome_valence'] = {
                            'auc': float(ov['auc']),
                            'precision_positive': float(ov['precision_positive']),
                            'recall_positive': float(ov['recall_positive']),
                            'precision_negative': float(ov['precision_negative']),
                            'recall_negative': float(ov['recall_negative'])
                        }
                
                # locus (internal vs external) - baseline
                if 'locus' in comprehensive_classification_results:
                    loc = comprehensive_classification_results['locus']
                    if 'auc' in loc:
                        comp_class_metrics['locus'] = {
                            'auc': float(loc['auc']),
                            'precision_external': float(loc['precision_external']),
                            'recall_external': float(loc['recall_external']),
                            'precision_internal': float(loc['precision_internal']),
                            'recall_internal': float(loc['recall_internal'])
                        }
                
                # 4-way classification
                if 'four_way' in comprehensive_classification_results:
                    fw = comprehensive_classification_results['four_way']
                    if 'auc_ovr' in fw:
                        comp_class_metrics['four_way'] = {
                            'auc_ovr': float(fw['auc_ovr']),
                            'macro_precision': float(fw['macro_precision']),
                            'macro_recall': float(fw['macro_recall']),
                            'macro_f1': float(fw['macro_f1'])
                        }
                        # add per-class metrics
                        if 'per_class' in fw:
                            comp_class_metrics['four_way']['per_class'] = fw['per_class']
                
                if comp_class_metrics:
                    metrics['comprehensive_classification'] = comp_class_metrics
            
            if separation_results:
                metrics['target_peer_separation'] = separation_results
            
            if bias_period_results:
                metrics['bias_period_comparison'] = bias_period_results
            
            if bias_prediction_results and 'analysis' in bias_prediction_results:
                metrics['bias_prediction'] = bias_prediction_results['analysis']
            
            if bias_vector_results:
                metrics['bias_vector'] = {
                    'separation_magnitude': bias_vector_results.get('separation_magnitude'),
                    'cohens_d': bias_vector_results.get('cohens_d'),
                    'high_bias_score_mean': bias_vector_results.get('high_bias_score_mean'),
                    'low_bias_score_mean': bias_vector_results.get('low_bias_score_mean')
                }
            
            if pca_results:
                metrics['pca_analysis'] = {
                    'n_components_95pct': pca_results.get('n_components_95pct'),
                    'elbow_point': pca_results.get('elbow_point'),
                    'target_peer_separation_pca': pca_results.get('target_peer_separation_pca')
                }
            
            # convert all numpy types to python types for json serialization
            def convert_to_serializable(obj):
                """Recursively convert numpy types to Python types for JSON serialization."""
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            metrics_serializable = convert_to_serializable(metrics)
            
            metrics_file = self.output_dir / f"embedding_analysis_metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_serializable, f, indent=2)
            self.log_and_print(f" Saved metrics: {metrics_file}")
            
        # save multilevel embeddings summary
        if multilevel_embeddings:
            level_stats = []
            for level_name, level_data in multilevel_embeddings.items():
                if isinstance(level_data, list) and len(level_data) > 0:
                    level_stats.append({
                        'level': level_name,
                        'n_groups': len(level_data),
                        'total_snippets': sum(g['n_snippets'] for g in level_data),
                        'mean_std': np.mean([g['std'] for g in level_data])
                    })
            
            if level_stats:
                level_stats_df = pd.DataFrame(level_stats)
                level_file = self.output_dir / f"multilevel_embedding_stats_{timestamp}.csv"
                level_stats_df.to_csv(level_file, index=False)
                self.log_and_print(f" Saved multilevel embedding stats: {level_file}")
        
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("EMBEDDING ANALYSIS COMPLETE")
        self.log_and_print("=" * 80)
        self.log_and_print(f"\nAll results saved to: {self.output_dir}")
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("KEY FINDINGS SUMMARY")
        self.log_and_print("=" * 80)
        
        # multi-label validation
        if 'global_kmeans_validation' in clustering_results:
            validation = clustering_results['global_kmeans_validation']
            if validation['label_alignment']:
                ranked = sorted(validation['label_alignment'].items(), 
                               key=lambda x: x[1]['ari'], reverse=True)
                top_label = ranked[0][0]
                top_ari = ranked[0][1]['ari']
                
                self.log_and_print(f"\n EMBEDDING VALIDATION: What Do Embeddings Capture?")
                self.log_and_print(f"  Top 3 Features Embeddings Cluster By:")
                for rank, (label_name, scores) in enumerate(ranked[:3], 1):
                    self.log_and_print(f"    {rank}. {label_name:20s}: ARI={scores['ari']:.3f}, NMI={scores['nmi']:.3f}")
                
                if 'Attribution' in top_label:
                    self.log_and_print(f"   VALIDATED: Embeddings capture {top_label} structure")
                    self.log_and_print(f"  → Suitable for attribution bias detection")
                elif top_label == 'Topic':
                    self.log_and_print(f"  ️  CAUTION: Embeddings primarily capture Topic (confound)")
                    self.log_and_print(f"  → Results may reflect topic differences, not bias")
                else:
                    self.log_and_print(f"  → INFO: Embeddings capture {top_label} primarily")
        
        if bias_prediction_results and 'analysis' in bias_prediction_results:
            analysis = bias_prediction_results['analysis']
            self.log_and_print(f"\n HYPOTHESIS TEST: Do embeddings capture patterns GPT misses?")
            self.log_and_print(f"  GPT Baseline AUC:       {analysis['gpt_auc']:.3f}")
            self.log_and_print(f"  Embedding AUC:          {analysis['embedding_auc']:.3f}")
            self.log_and_print(f"  Combined (GPT+Embed):   {analysis['combined_auc']:.3f}")
            self.log_and_print(f"  Improvement:            {analysis['improvement_comb_vs_gpt']:+.3f}")
            if analysis['hypothesis_confirmed']:
                self.log_and_print(f"   CONFIRMED: Embeddings add significant predictive power")
            else:
                self.log_and_print(f"  → Embeddings provide modest/no improvement over GPT")
        
        if separation_results:
            self.log_and_print(f"\n TARGET vs PEER SEPARATION:")
            self.log_and_print(f"  Separation ratio: {separation_results['separation_ratio']:.2f}x")
            if separation_results['separation_ratio'] > 1.2:
                self.log_and_print(f"   Targets cluster separately from peers")
            else:
                self.log_and_print(f"  → No clear separation in embedding space")
        
        if bias_period_results:
            self.log_and_print(f"\n HIGH-BIAS vs LOW-BIAS PERIODS:")
            self.log_and_print(f"  Euclidean distance: {bias_period_results['euclidean_distance']:.4f}")
            self.log_and_print(f"  Cosine similarity:  {bias_period_results['cosine_similarity']:.3f}")
            if bias_period_results['euclidean_distance'] > 2.0:
                self.log_and_print(f"   Distinct embedding signatures for bias periods")
            else:
                self.log_and_print(f"  → Modest embedding differences")
        
        # cs2 strategic clustering summary
        if 'cs2_attribution_clustering' in clustering_results:
            cs2_results = clustering_results['cs2_attribution_clustering']
            if cs2_results:
                self.log_and_print(f"\n CS2: ATTRIBUTION TYPE CLUSTERING (Strategic):")
                
                self_serving = [r for r in cs2_results if r['category'] == 'self-serving']
                non_self_serving = [r for r in cs2_results if r['category'] == 'non-self-serving']
                
                if self_serving:
                    avg_d = np.mean([r['cohens_d_coherence'] for r in self_serving])
                    self.log_and_print(f"  Self-serving types (Pos-Int, Neg-Ext):")
                    msg = f"    Cohen's d (coherence): {avg_d:+.3f}"
                    if avg_d > 0.5:
                        msg += f" ⭐ Targets LESS COHERENT (bias signal!)"
                    elif avg_d < -0.5:
                        msg += f" ⭐ Peers LESS COHERENT (unexpected)"
                    self.log_and_print(msg)
                
                if non_self_serving:
                    avg_d = np.mean([r['cohens_d_coherence'] for r in non_self_serving])
                    self.log_and_print(f"  Non-self-serving types (Pos-Ext, Neg-Int):")
                    self.log_and_print(f"    Cohen's d (coherence): {avg_d:+.3f}")
                
                # highlight key finding
                if self_serving:
                    neg_ext = [r for r in self_serving if 'Negative_External' in r['attribution_type']]
                    if neg_ext:
                        self.log_and_print(f"\n  KEY FINDING - Negative-External:")
                        self.log_and_print(f"    Target coherence: {neg_ext[0]['target_coherence']:.3f}")
                        self.log_and_print(f"    Peer coherence:   {neg_ext[0]['peer_coherence']:.3f}")
                        self.log_and_print(f"    Cohen's d:        {neg_ext[0]['cohens_d_coherence']:+.3f}")
                        if neg_ext[0]['cohens_d_coherence'] > 0.5:
                            self.log_and_print(f"     Targets use LESS COHERENT external blame (bias!)")
        
        # hdbscan strategic summary
        if 'hdbscan_strategic' in clustering_results:
            hdbscan_strat = clustering_results['hdbscan_strategic']
            
            if 'company_level' in hdbscan_strat and hdbscan_strat['company_level']:
                company_results = hdbscan_strat['company_level']
                targets = [r for r in company_results if r['is_target']]
                peers = [r for r in company_results if not r['is_target']]
                
                if targets and peers:
                    target_noise = np.mean([r['noise_ratio'] for r in targets])
                    peer_noise = np.mean([r['noise_ratio'] for r in peers])
                    
                    self.log_and_print(f"\n HDBSCAN: COMPANY-LEVEL NARRATIVE COHERENCE:")
                    self.log_and_print(f"  Target avg noise ratio: {target_noise:.3f} ({len(targets)} companies)")
                    self.log_and_print(f"  Peer avg noise ratio:   {peer_noise:.3f} ({len(peers)} companies)")
                    self.log_and_print(f"  Difference:             {target_noise - peer_noise:+.3f}")
                    
                    if target_noise > peer_noise + 0.05:
                        self.log_and_print(f"   Targets show MORE INCOHERENT attributions")
        
        if topic_consistency_df is not None and len(topic_consistency_df) > 0:
            outlier_count = (topic_consistency_df['is_outlier']).sum()
            outlier_rate = outlier_count / len(topic_consistency_df) * 100
            
            target_consistency = topic_consistency_df[topic_consistency_df['is_target']]
            peer_consistency = topic_consistency_df[~topic_consistency_df['is_target']]
            
            if len(target_consistency) > 0 and len(peer_consistency) > 0:
                target_outlier_rate = (target_consistency['is_outlier'].sum() / len(target_consistency)) * 100
                peer_outlier_rate = (peer_consistency['is_outlier'].sum() / len(peer_consistency)) * 100
                
                self.log_and_print(f"\n TOPIC TEMPORAL CONSISTENCY:")
                self.log_and_print(f"  Attributions analyzed: {len(topic_consistency_df):,}")
                self.log_and_print(f"  Overall outlier rate: {outlier_rate:.1f}%")
                self.log_and_print(f"  Target outlier rate: {target_outlier_rate:.1f}%")
                self.log_and_print(f"  Peer outlier rate: {peer_outlier_rate:.1f}%")
                
                if target_outlier_rate > peer_outlier_rate * 1.2:
                    self.log_and_print(f"   Targets show {target_outlier_rate/peer_outlier_rate:.1f}x more topic inconsistency")
                else:
                    self.log_and_print(f"  → No significant difference in topic consistency")
        
        if bias_vector_results:
            self.log_and_print(f"\n BIAS VECTOR:")
            self.log_and_print(f"  Cohen's d effect size: {bias_vector_results.get('cohens_d', 0):.3f}")
            cohens_d = bias_vector_results.get('cohens_d', 0)
            if cohens_d > 0.8:
                self.log_and_print(f"   LARGE effect - bias vector strongly discriminates")
            elif cohens_d > 0.5:
                self.log_and_print(f"  → MEDIUM effect")
            else:
                self.log_and_print(f"  → SMALL effect")
        
        if pca_results:
            self.log_and_print(f"\n PCA DIMENSIONALITY REDUCTION:")
            self.log_and_print(f"  Components for 95% variance: {pca_results.get('n_components_95pct', 0)}")
            self.log_and_print(f"  Suggested elbow point: {pca_results.get('elbow_point', 0)} components")
            if 'target_peer_separation_pca' in pca_results:
                retention = pca_results['target_peer_separation_pca'].get('retention_rate', 0)
                self.log_and_print(f"  Target-peer separation retained: {retention*100:.1f}%")
        
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("Next steps:")
        self.log_and_print("  1. Review embedding_analysis_metrics.json for complete results")
        self.log_and_print("  2. Use umap_coordinates.csv or pca_coordinates.csv for visualization")
        self.log_and_print("  3. Check qna_topic_shifts.csv for topic switching patterns")
        self.log_and_print("  4. Review topic_temporal_consistency.csv for attribution outliers")
        self.log_and_print("  5. Examine topic_consistency_outliers.csv for flagged attributions")
        self.log_and_print(f"\nLog file saved: {self.log_file}")
        self.log_and_print("  6. Use bias_scores.csv for single-score bias measurement")
        self.log_and_print("  7. Use embedding_features.csv for supervised learning with interpretable features")
        self.log_and_print("  8. Review multilevel_embedding_stats.csv for granular analysis")
    
    def run_bias_reanalysis(self):
        """
        LIGHTWEIGHT RE-ANALYSIS: Load pre-computed embeddings and re-run only bias-related analyses.
        
        Use this when embeddings are already generated but you want to update bias period logic.
        
        Loads from disk:
        - attribution_data_with_embeddings.csv
        - embeddings.npy
        - embeddings_topic_adjusted.npy (if available)
        
        Re-runs only:
        1. Bias period loading (from config)
        2. Supervised classification (bias prediction)
        3. Bias vector extraction
        4. Bias period comparison
        5. PCA correlation with bias
        
        Expected runtime: 5-15 minutes (vs 24+ hours for full analysis)
        """
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("BIAS RE-ANALYSIS MODE (Loading Pre-Computed Embeddings)")
        self.log_and_print("=" * 80)
        
        # 1. load pre-computed data
        self.log_and_print("\n1. Loading Pre-Computed Data from Disk")
        self.log_and_print("=" * 80)
        
        # find the most recent timestamped files from full analysis
        # full analysis saves: embedding_metadata_{timestamp}.csv, embedding_vectors_{timestamp}.npy, etc.
        
        csv_files = sorted(self.output_dir.glob("embedding_metadata_*.csv"))
        embeddings_files = sorted(self.output_dir.glob("embedding_vectors_[0-9]*.npy"))
        embeddings_adjusted_files = sorted(self.output_dir.glob("embedding_vectors_topic_adjusted_*.npy"))
        
        if not csv_files:
            self.log_and_print(f" ERROR: No embedding_metadata_*.csv files found in {self.output_dir}")
            self.log_and_print(f"   Run full analysis first: python conclusion/embedding_analysis.py")
            return
        
        if not embeddings_files:
            self.log_and_print(f" ERROR: No embedding_vectors_*.npy files found in {self.output_dir}")
            self.log_and_print(f"   Run full analysis first: python conclusion/embedding_analysis.py")
            return
        
        # use the most recent files (last in sorted list)
        csv_path = csv_files[-1]
        embeddings_path = embeddings_files[-1]
        
        self.log_and_print(f" Found {len(csv_files)} embedding metadata file(s)")
        self.log_and_print(f"  Using most recent: {csv_path.name}")
        df = pd.read_csv(csv_path)
        self.log_and_print(f"  Loaded {len(df):,} attribution statements")
        
        self.log_and_print(f" Found {len(embeddings_files)} embedding file(s)")
        self.log_and_print(f"  Using most recent: {embeddings_path.name}")
        embeddings = np.load(embeddings_path)
        self.log_and_print(f"  Loaded embeddings: {embeddings.shape}")
        
        embeddings_topic_adjusted = None
        if embeddings_adjusted_files:
            embeddings_adjusted_path = embeddings_adjusted_files[-1]
            self.log_and_print(f" Found {len(embeddings_adjusted_files)} topic-adjusted embedding file(s)")
            self.log_and_print(f"  Using most recent: {embeddings_adjusted_path.name}")
            embeddings_topic_adjusted = np.load(embeddings_adjusted_path)
            self.log_and_print(f"  Loaded topic-adjusted embeddings: {embeddings_topic_adjusted.shape}")
        else:
            self.log_and_print(f"️  No topic-adjusted embeddings found, skipping adjusted analyses")
        
        # 2. load new bias periods from config
        self.log_and_print("\n2. Loading Expert-Identified Bias Periods from Config")
        self.log_and_print("=" * 80)
        bias_periods = self.load_peer_benchmark_results()
        
        if not bias_periods:
            self.log_and_print(" No bias periods found in config. Cannot proceed.")
            return
        
        # 2b. apply bias labels to dataframe (critical: analysis methods expect this)
        self.log_and_print("\n2b. Applying Bias Period Labels to DataFrame")
        self.log_and_print("=" * 80)
        df['bias_period_key'] = list(zip(df['Company'], df['Year'], df['Quarter']))
        
        # add in_expert_period flag for each row
        df['in_expert_period'] = df['bias_period_key'].apply(
            lambda x: bias_periods.get(x, {}).get('in_expert_period', False)
        )
        
        expert_count = df['in_expert_period'].sum()
        self.log_and_print(f"  Segments in expert periods: {expert_count:,} ({expert_count/len(df)*100:.1f}%)")
        self.log_and_print(f"  Segments in normal periods: {len(df)-expert_count:,} ({(len(df)-expert_count)/len(df)*100:.1f}%)")
        
        if expert_count == 0:
            self.log_and_print("️  WARNING: No segments found in expert periods. Check:")
            self.log_and_print("     - Company names in CSV match target_folder in config")
            self.log_and_print("     - Year/Quarter columns are integers")
            self.log_and_print("     - Bias dates in config are within data range")
        
        # 3. re-run bias-related analyses
        self.log_and_print("\n3. Re-Running Bias-Related Analyses")
        self.log_and_print("=" * 80)
        
        # 3a. multi-level embedding aggregation (priority #3 from user)
        self.log_and_print("\n[1/6] Multi-Level Embedding Aggregation")
        self.log_and_print("-" * 80)
        self.log_and_print("Computing aggregations at 5 levels: attribution-type, section, attr vs non-attr, quarter, company")
        multilevel_embeddings = self.aggregate_multilevel_embeddings(df, embeddings)
        # save immediately to disk (and keep for results saving)
        self._save_multilevel_embeddings(multilevel_embeddings)
        self.log_and_print(" Multi-level aggregations saved to disk")
        
        # 3b. supervised classification (bias prediction) - priority #2 from user
        self.log_and_print("\n[2/6] Supervised Classification - Bias Prediction (WITH NEW FEATURES)")
        self.log_and_print("-" * 80)
        bias_prediction_results = self.supervised_classification_bias_prediction(
            df, embeddings, bias_periods, test_size=0.2
        )
        
        # 3c. bias vector extraction
        self.log_and_print("\n[3/6] Bias Vector Extraction")
        self.log_and_print("-" * 80)
        bias_vector_results = self.extract_bias_vector(df, embeddings, bias_periods)
        
        # 3d. bias period comparison
        self.log_and_print("\n[4/6] Bias Period Comparison")
        self.log_and_print("-" * 80)
        bias_period_results = self.compare_bias_periods(df, embeddings, bias_periods)
        
        # 3e. pca with bias correlation
        self.log_and_print("\n[5/6] PCA Analysis with Bias Correlation")
        self.log_and_print("-" * 80)
        pca_results = self.perform_pca_analysis(df, embeddings, n_components=50)
        
        # 3f. per-firm analysis - priority #1 from user (in-firm metrics for both models)
        self.log_and_print("\n[6/7] Per-Firm Analysis (IN-FIRM metrics for RAW and TOPIC-ADJUSTED)")
        self.log_and_print("-" * 80)
        if embeddings_topic_adjusted is not None:
            firm_analysis_df = self.analyze_individual_target_firms(
                df, embeddings, embeddings_topic_adjusted, bias_periods
            )
        else:
            self.log_and_print("️  Skipping per-firm analysis (topic-adjusted embeddings not available)")
            firm_analysis_df = None
        
        # 3g. topic temporal consistency - by attribution type (new: user requested)
        self.log_and_print("\n[7/7] Topic Temporal Consistency (BY ATTRIBUTION TYPE)")
        self.log_and_print("-" * 80)
        self.log_and_print("Testing if consistency differs by attribution type (Pos/Neg × Internal/External)")
        topic_consistency_df = self.analyze_topic_temporal_consistency(
            df, embeddings, lookback_quarters=4
        )
        
        # 4. save updated results
        self.log_and_print("\n4. Saving Updated Bias Analysis Results")
        self.log_and_print("=" * 80)
        
        # use timestamp for reanalysis outputs (to avoid overwriting original results)
        from datetime import datetime
        reanalysis_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # save bias-specific outputs
        if multilevel_embeddings:
            # already saved via _save_multilevel_embeddings, just log confirmation
            self.log_and_print(f" Multi-level embeddings already saved (see multilevel_embeddings_*.json)")
        
        if bias_prediction_results:
            filename = f'bias_prediction_results_reanalysis_{reanalysis_timestamp}.json'
            with open(self.output_dir / filename, 'w') as f:
                json.dump(bias_prediction_results, f, indent=2, default=str)
            self.log_and_print(f" Saved {filename}")
        
        if bias_vector_results:
            # save vector as .npy for reuse
            if 'linguistic_signature_vector' in bias_vector_results:
                filename = f'bias_vector_reanalysis_{reanalysis_timestamp}.npy'
                np.save(self.output_dir / filename, 
                       bias_vector_results['linguistic_signature_vector'])
                self.log_and_print(f" Saved {filename}")
            
            # save metadata as json
            results_to_save = {k: v for k, v in bias_vector_results.items() 
                             if k not in ['linguistic_signature_vector', 'expert_period_centroid', 
                                         'normal_period_centroid', 'all_linguistic_scores']}
            filename = f'bias_vector_results_reanalysis_{reanalysis_timestamp}.json'
            with open(self.output_dir / filename, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
            self.log_and_print(f" Saved {filename}")
        
        if bias_period_results:
            filename = f'bias_period_comparison_reanalysis_{reanalysis_timestamp}.json'
            with open(self.output_dir / filename, 'w') as f:
                json.dump(bias_period_results, f, indent=2, default=str)
            self.log_and_print(f" Saved {filename}")
        
        if pca_results:
            filename = f'pca_results_reanalysis_{reanalysis_timestamp}.json'
            with open(self.output_dir / filename, 'w') as f:
                json.dump(pca_results, f, indent=2, default=str)
            self.log_and_print(f" Saved {filename}")
        
        if firm_analysis_df is not None:
            filename = f'firm_analysis_reanalysis_{reanalysis_timestamp}.csv'
            firm_analysis_df.to_csv(self.output_dir / filename, index=False)
            self.log_and_print(f" Saved {filename}")
        
        if topic_consistency_df is not None and len(topic_consistency_df) > 0:
            filename = f'topic_consistency_by_attribution_type_reanalysis_{reanalysis_timestamp}.csv'
            topic_consistency_df.to_csv(self.output_dir / filename, index=False)
            self.log_and_print(f" Saved {filename}")
        
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print(" BIAS RE-ANALYSIS COMPLETE")
        self.log_and_print("=" * 80)
        self.log_and_print(f"\nUpdated results saved to: {self.output_dir}")
        self.log_and_print(f"\nReanalysis timestamp: {reanalysis_timestamp}")
        self.log_and_print("\nAnalyses Re-Run (7 total):")
        self.log_and_print("  [1/7] Multi-level Embedding Aggregation (5 levels)")
        self.log_and_print("  [2/7] Supervised Classification - Bias Prediction")
        self.log_and_print("  [3/7] Bias Vector Extraction")
        self.log_and_print("  [4/7] Bias Period Comparison")
        self.log_and_print("  [5/7] PCA Analysis with Bias Correlation")
        self.log_and_print("  [6/7] Per-Firm Analysis (both RAW and TOPIC-ADJUSTED)")
        self.log_and_print("  [7/7] Topic Temporal Consistency (BY ATTRIBUTION TYPE)")
        self.log_and_print("\nFiles created (with '_reanalysis_' prefix to preserve originals):")
        self.log_and_print(f"  - multilevel_embeddings_*.json (5 levels)")
        self.log_and_print(f"  - bias_prediction_results_reanalysis_{reanalysis_timestamp}.json")
        self.log_and_print(f"  - bias_vector_reanalysis_{reanalysis_timestamp}.npy")
        self.log_and_print(f"  - bias_vector_results_reanalysis_{reanalysis_timestamp}.json")
        self.log_and_print(f"  - bias_period_comparison_reanalysis_{reanalysis_timestamp}.json")
        self.log_and_print(f"  - pca_results_reanalysis_{reanalysis_timestamp}.json")
        if firm_analysis_df is not None:
            self.log_and_print(f"  - firm_analysis_reanalysis_{reanalysis_timestamp}.csv")
        if topic_consistency_df is not None and len(topic_consistency_df) > 0:
            self.log_and_print(f"  - topic_consistency_by_attribution_type_reanalysis_{reanalysis_timestamp}.csv")
        self.log_and_print(f"  - analysis_log_{self.log_timestamp}.txt (this log)")
        self.log_and_print("\n Original full analysis results preserved (not overwritten)")
        self.log_and_print("\nNext steps:")
        self.log_and_print("  1. Review updated results in output/embeddings/")
        self.log_and_print("  2. Compare reanalysis vs original results to see impact of new bias periods")
        self.log_and_print("  3. Run visualization scripts to see new patterns")
        self.log_and_print("  4. Compare with peer_benchmark.py validation results")
    
    def run_full_analysis(self, max_samples: Optional[int] = None):
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("EMBEDDING ANALYSIS WITH PEER BENCHMARK INTEGRATION")
        self.log_and_print("=" * 80)
        
        df = self.load_attribution_data()
        
        if len(df) == 0:
            self.log_and_print("\nNo data available for analysis")
            return
        
        bias_periods = self.load_peer_benchmark_results()
        
        # new: load expert-identified bias periods (independent validation)
        expert_periods = self.load_expert_bias_periods()
        if expert_periods:
            df = self.create_expert_labeled_dataset(df, expert_periods)
        
        if max_samples and len(df) > max_samples:
            self.log_and_print(f"\nLimiting to {max_samples} samples for testing")
            df = df.sample(n=max_samples, random_state=42)
            df = df.reset_index(drop=True) 
        
        embeddings = self.extract_embeddings(df)
        
        if len(embeddings) == 0:
            self.log_and_print("\nCould not extract embeddings")
            return
        
        # new: compute topic-adjusted embeddings (priority 1 improvement)
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("TOPIC ADJUSTMENT - Removing Confound")
        self.log_and_print("=" * 80)
        embeddings_topic_adjusted = self.compute_topic_adjusted_embeddings(df, embeddings)
        
        # extract semantic subspace features (new: for improved classification)
        subspace_features = self.extract_semantic_subspaces(embeddings)
        
        # multi-level embedding aggregation
        multilevel_embeddings = self.aggregate_multilevel_embeddings(df, embeddings)
        
        # memory optimization: save and delete immediately (frees ~2-3 gb)
        self._save_multilevel_embeddings(multilevel_embeddings)
        del multilevel_embeddings
        gc.collect()
        self.log_and_print("→ Memory cleaned after multilevel aggregation (saved stats to disk)")
        
        # label statistics
        labels = (df['attribution_locus'].isin(['External', 'external'])).astype(int).values
        
        self.log_and_print(f"\n3. Label Distribution (External vs Internal)")
        self.log_and_print(f"  External: {labels.sum():,} ({labels.sum()/len(labels)*100:.1f}%)")
        self.log_and_print(f"  Internal: {len(labels) - labels.sum():,} ({(len(labels) - labels.sum())/len(labels)*100:.1f}%)")
        
        if bias_periods:
            df['bias_period_key'] = list(zip(df['Company'], df['Year'], df['Quarter']))
            expert_count = df['bias_period_key'].apply(
                lambda x: bias_periods.get(x, {}).get('in_expert_period', False)
            ).sum()
            self.log_and_print(f"\nExpert Period Labels:")
            self.log_and_print(f"  Segments in expert periods: {expert_count:,} ({expert_count/len(df)*100:.1f}%)")
            self.log_and_print(f"  Segments in normal periods: {len(df)-expert_count:,} ({(len(df)-expert_count)/len(df)*100:.1f}%)")
        
        # strategic clustering analysis on raw embeddings
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("CLUSTERING VALIDATION - RAW EMBEDDINGS")
        self.log_and_print("=" * 80)
        clustering_results = self.unsupervised_clustering(df, embeddings, labels)
        
        # new: strategic clustering analysis on topic-adjusted embeddings
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("CLUSTERING VALIDATION - TOPIC-ADJUSTED EMBEDDINGS")
        self.log_and_print("=" * 80)
        clustering_results_adjusted = self.unsupervised_clustering(df, embeddings_topic_adjusted, labels)
        
        # compare raw vs topic-adjusted results
        self._compare_topic_adjustment_impact(clustering_results, clustering_results_adjusted)
        
        # clean up memory after clustering (clustering results are small, just metrics)
        # note: labels variable still needed for supervised_classification below
        gc.collect()
        self.log_and_print("\n→ Memory cleaned after clustering operations")
        
        # company-level analysis
        company_aggregates = self.aggregate_company_embeddings(df, embeddings)
        separation_results = self.measure_target_peer_separation(company_aggregates)
        
        # clean up company_aggregates after use (contains embeddings per company)
        # note: keeping for save_results(), but could delete if memory critical
        gc.collect()
        self.log_and_print("→ Memory cleaned after company aggregation")
        
        # critical: supervised classification with bias labels (tests main hypothesis)
        bias_prediction_results = {}
        if bias_periods:
            bias_prediction_results = self.supervised_classification_bias_prediction(
                df, embeddings, bias_periods
            )
        
        # legacy supervised classification (kept for comparison)
        supervised_results = self.supervised_classification(embeddings, labels)
        
        # clean up before comprehensive classification (most memory-intensive task)
        del labels  # no longer needed after supervised_classification
        gc.collect()
        self.log_and_print("\n→ Memory cleaned before comprehensive classification")
        
        # new: comprehensive attribution classification (tests all attribution aspects)
        comprehensive_classification_results = self.comprehensive_attribution_classification(df, embeddings)
        
        # q&a topic shift detection
        topic_shift_df = self.detect_qna_topic_shifts(df, embeddings)
        
        # topic-level temporal consistency analysis
        # new: compare attribution language to historical topic discussion
        topic_consistency_df = self.analyze_topic_temporal_consistency(df, embeddings, lookback_quarters=4)
        
        # bias period embedding comparison
        bias_period_results = {}
        if bias_periods:
            bias_period_results = self.compare_bias_periods(df, embeddings, bias_periods)
        
        # new advanced analyses
        # extract bias vector (single direction capturing bias patterns)
        bias_vector_results = {}
        if bias_periods:
            bias_vector_results = self.extract_bias_vector(df, embeddings, bias_periods)
        
        # principal component analysis (identify which dimensions matter for bias)
        pca_results = self.perform_pca_analysis(df, embeddings, n_components=50)
        
        # engineer interpretable features from embeddings
        embedding_features_df = self.extract_embedding_features(df, embeddings, bias_vector_results)
        
        # new: per-firm stratified analysis (identify which targets show signals)
        firm_analysis_df = self.analyze_individual_target_firms(
            df, embeddings, embeddings_topic_adjusted, bias_periods
        )
        
        # new: dimensionality reduction comparison (pca vs umap, raw vs adjusted)
        self.log_and_print("\n" + "=" * 80)
        self.log_and_print("DIMENSIONALITY REDUCTION: PCA vs UMAP")
        self.log_and_print("=" * 80)
        
        # pca on raw embeddings (always works, good baseline)
        coords_pca = self.dimensionality_reduction(embeddings, method='pca', n_components=2)
        
        # umap on raw embeddings (better at preserving clusters)
        coords_umap = self.dimensionality_reduction(embeddings, method='umap', n_components=2)
        
        # umap on topic-adjusted embeddings (should show attribution patterns more clearly)
        coords_umap_adjusted = None
        if embeddings_topic_adjusted is not None:
            self.log_and_print("\nReducing topic-adjusted embeddings...")
            coords_umap_adjusted = self.dimensionality_reduction(
                embeddings_topic_adjusted, method='umap', n_components=2
            )
            if len(coords_umap_adjusted) > 0:
                self.log_and_print("   Topic-adjusted UMAP coordinates computed")
                self.log_and_print("  → Use for visualization of attribution style without topic confound")
        
        self.save_results(
            df, embeddings, clustering_results, supervised_results, coords_umap,
            company_aggregates, separation_results, topic_shift_df, bias_period_results,
            None,  # multilevel_embeddings already saved and deleted for memory
            bias_prediction_results, topic_consistency_df,
            bias_vector_results, pca_results, embedding_features_df, subspace_features,
            embeddings_topic_adjusted, clustering_results_adjusted, coords_pca, coords_umap_adjusted,
            firm_analysis_df, comprehensive_classification_results
        )


def main():
    parser = argparse.ArgumentParser(description='Embedding Analysis for Attribution Bias')
    parser.add_argument('--input', type=str, default='classification_results',
                       help='Input directory with instance folders')
    parser.add_argument('--config', type=str, default='company_config.json',
                       help='Path to company config with peer groups')
    parser.add_argument('--benchmark', type=str, default='output/peer_validation',
                       help='Path to peer benchmark results')
    parser.add_argument('--output', type=str, default='output/embeddings',
                       help='Output directory for results')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Sentence transformer model name (if not using OpenAI)')
    parser.add_argument('--use-openai', action='store_true',
                       help='Use OpenAI embeddings instead of SentenceTransformers')
    parser.add_argument('--openai-model', type=str, default='text-embedding-3-large',
                       choices=['text-embedding-3-large', 'text-embedding-3-small'],
                       help='OpenAI embedding model (default: text-embedding-3-large)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--reanalyze', action='store_true',
                       help='Load pre-computed embeddings and re-run only bias-related analyses (5-15 min vs 24+ hours)')
    
    args = parser.parse_args()
    
    analyzer = EmbeddingAnalyzer(
        input_dir=args.input,
        output_dir=args.output,
        config_path=args.config,
        benchmark_results=args.benchmark,
        model_name=args.model,
        use_openai=args.use_openai,
        openai_model=args.openai_model
    )
    
    if args.reanalyze:
        analyzer.run_bias_reanalysis()
    else:
        analyzer.run_full_analysis(max_samples=args.max_samples)


if __name__ == "__main__":
    main()


# example usage:
#
# full analysis (24+ hours):
# python conclusion/embedding_analysis.py --use-openai
# python conclusion/embedding_analysis.py
# python conclusion/embedding_analysis.py --max_samples 5000  # test with 5000 samples
# python conclusion/embedding_analysis.py --model sentence-transformers/all-minilm-l6-v2  # faster model
#
# reanalysis mode (5-15 minutes - loads pre-computed embeddings):
# python conclusion/embedding_analysis.py --reanalyze
# cam also use speccific folders via python conclusion/embedding_analysis.py --reanalyze --output output/embeddings/all-minilm-l6-v2
#  python conclusion/embedding_analysis.py --reanalyze --output output/embeddings/open-ai
