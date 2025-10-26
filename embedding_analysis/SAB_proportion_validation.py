"""
GPT Classification Validation

- Do hypothesized bias periods show measurable differences?
    - Check validation tabs for z-scores > 1.5 and p-values < 0.05
- Which metrics are most informative?
    - SAB proportion, asymmetry_score, neg_ext_of_negative, pos_int_of_positive
    - See which shows strongest signal
- Do patterns differ by bias type?
    - Check bias type analysis tab
- How sensitive are results to window size?
    - Compare exact vs ±1Q vs ±2Q results
- Does your data match literature benchmarks?
    - Check distribution analysis for Pos-Int and Neg-Ext rates


Gives the following:
Load data with bias flags
1. GPT label distributions
2. Target vs Peer attribution rates
3. Calculate metrics (PRIMARY: SAB Proportion, SECONDARY: Rate Asymmetry)
    - Key Metric is SAB Proportion (PRIMARY): (pos_int + neg_ext) / total_attributions
4. Hypothesied Bias Period Validation s
5. Exact window comparison
5. ±1 Quarter window comparison
6. ±2 Quarter window comparison
7. Aggregate and analyze by bias type
8. Traditional analyses (rolling, coherence, etc.)
9. Save all results

Summary of output:
- Bar charts showing distribution vs literature (Pos-Int 69-82%, Neg-Ext 5-14%)
- Target vs Peer comparisons during bias periods
- Statistical validation (z-scores, t-tests, Cohen's d)
- Results by bias type (Blame vs Overconfidence vs Deception)

"""
# gpt classification validation for self-attribution bias detection

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import argparse
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from dateutil import parser as date_parser
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
warnings.filterwarnings('ignore')

# set matplotlib style for academic papers
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("Warning: openpyxl not available. Install with: pip install openpyxl")

class PeerBenchmark:
    
    # topic category mapping for hierarchical topic analysis
    TOPIC_CATEGORY_MAPPING = {
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
    
    def __init__(self, 
                 input_dir: str = "classification_results",
                 config_path: str = "company_config.json",
                 output_dir: str = "output/gpt_classification_validation"):
        
        self.input_dir = Path(input_dir)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # create visualizations subdirectory
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        self.peer_groups = self._load_peer_groups()
        self.target_to_peers = self._create_target_peer_mapping()
        self.expert_bias_periods = self._load_expert_bias_periods()
        
        print(f"Initialized PeerBenchmark")
        print(f"  Input: {self.input_dir}")
        print(f"  Config: {self.config_path}")
        print(f"  Output: {self.output_dir}")
        print(f"  Visualizations: {self.viz_dir}")
        print(f"  Peer groups loaded: {len(self.peer_groups)}")
        print(f"  Expert bias periods loaded: {len(self.expert_bias_periods)}")
    
    def _load_peer_groups(self) -> Dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'self.peer_groups = {' in content:
                    start = content.find('self.peer_groups = {')
                    content = content[start + len('self.peer_groups = '):]
                    # replace json null with python none for eval
                    content = content.replace('null', 'None')
                    peer_groups = eval(content)
                    print(f" Loaded {len(peer_groups)} peer groups from config")
                    
                    # count targets with peers
                    targets_with_peers = sum(1 for info in peer_groups.values() if info.get('peers'))
                    print(f" Found {targets_with_peers} target companies with defined peer groups")
                    return peer_groups
                else:
                    print(" Could not find peer_groups in config")
                    return {}
        except Exception as e:
            print(f" Error loading peer groups: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _create_target_peer_mapping(self) -> Dict[str, Set[str]]:
        """Map each target folder to its DIRECT competitor folders for comparison.
        
        NOTE: Uses direct_competitors_folders (not all peer_folders) to focus on 
        closest industry peers. Indirect competitors are excluded for cleaner signal.
        """
        mapping = {}
        for target, info in self.peer_groups.items():
            # use target_folder to get actual folder name
            target_folder = info.get('target_folder', target)
            
            # use direct_competitors_folders (not peer_folders) for focused peer analysis
            peer_folders = info.get('direct_competitors_folders', [])
            # filter out none values (missing data)
            peers = set([p for p in peer_folders if p is not None])
            
            # map using target_folder as key
            if target_folder:
                mapping[target_folder] = peers
        
        return mapping
    
    def _get_adjacent_quarters(self, year: int, quarter: int, window: int = 0) -> List[Tuple[int, int]]:
        """
        Get quarters within ±window of given quarter."""
        quarters = []
        
        for offset in range(-window, window + 1):
            # calculate total quarters from year 0
            total_quarters = (year * 4 + quarter - 1) + offset
            new_year = total_quarters // 4
            new_q = (total_quarters % 4) + 1
            
            quarters.append((new_year, new_q))
        
        return quarters
    
    def _get_pre_bias_quarters(self, year: int, quarter: int, window: int = 4) -> List[Tuple[int, int]]:
        """
        Get quarters BEFORE bias period (including the bias quarter itself).
        
        For -4Q pre-bias: returns 4 quarters before + exact quarter (total 5 quarters).
        This detects if bias emerges gradually before the expert-identified period."""
        quarters = []
        
        for offset in range(-window, 1):  # -4 to 0 (includes exact quarter)
            total_quarters = (year * 4 + quarter - 1) + offset
            new_year = total_quarters // 4
            new_q = (total_quarters % 4) + 1
            quarters.append((new_year, new_q))
        
        return quarters
    
    def _get_post_bias_quarters(self, year: int, quarter: int, window: int = 4) -> List[Tuple[int, int]]:
        """
        Get quarters AFTER bias period (including the bias quarter itself).
        
        For +4Q post-bias: returns exact quarter + 4 quarters after (total 5 quarters).
        This detects if bias persists after the expert-identified period."""
        quarters = []
        
        for offset in range(0, window + 1):  # 0 to +4 (includes exact quarter)
            total_quarters = (year * 4 + quarter - 1) + offset
            new_year = total_quarters // 4
            new_q = (total_quarters % 4) + 1
            quarters.append((new_year, new_q))
        
        return quarters
    
    def _load_expert_bias_periods(self) -> Dict:
        """
        Extract expert-identified bias periods from company_config.json.
        
        Returns dict with structure:
        {
            'INTC': {
                'bias_start': '2021-07-22',
                'bias_end': '2022-01-15' or None,
                'bias_type': 'Blame/Defensiveness/Evasion',
                'rationale': 'Call behavior flagged...',
                'start_quarter': (2021, 3),
                'end_quarter': (2022, 1) or None,
                'quarters_exact': [(2021, 3)],
                'quarters_window1': [(2021, 2), (2021, 3), (2021, 4)],
                'quarters_window2': [(2021, 1), (2021, 2), (2021, 3), (2021, 4), (2022, 1)]
            }
        }
        """
        print("\n Loading Expert-Identified Bias Periods")
        print("=" * 80)
        
        if not self.config_path.exists():
            print(f"  ️  Config file not found: {self.config_path}")
            return {}
        
        expert_periods = {}
        
        try:
            # load the peer_groups dict directly (same approach as _load_peer_groups)
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # extract peer_groups dictionary
            if 'self.peer_groups = {' not in content:
                print(f"  ️  Could not find peer_groups structure in config")
                return {}
            
            start = content.find('self.peer_groups = {')
            content = content[start + len('self.peer_groups = '):]
            content = content.replace('null', 'None')
            peer_groups_dict = eval(content)
            
            # iterate through all companies to find those with date fields
            for ticker, info in peer_groups_dict.items():
                # check if this company has a date field
                date_str = info.get('date')
                if not date_str or date_str in ['null', 'None', '']:
                    continue
                
                # extract fields from dict
                bias_type = info.get('bias_type', 'Unknown')
                rationale = info.get('rationale', '')
                bias_end_str = info.get('bias_end_date')
                target_folder = info.get('target_folder', ticker)
                
                # parse start date and map to full quarter
                # note: dates are mapped to their containing quarter to align with earnings calls
                # e.g., 2020-02-18 → 2020 q1 (entire quarter: jan 1 - mar 31)
                #       2020-11-09 → 2020 q4 (entire quarter: oct 1 - dec 31)
                try:
                    start_date = date_parser.parse(date_str)
                    start_quarter = (start_date.month - 1) // 3 + 1
                    start_year = start_date.year
                    
                    # parse end date if exists (also maps to full quarter)
                    end_date = None
                    end_quarter_tuple = None
                    if bias_end_str and bias_end_str not in ['null', 'None', '', None]:
                        try:
                            end_date = date_parser.parse(bias_end_str)
                            end_quarter = (end_date.month - 1) // 3 + 1
                            end_year = end_date.year
                            end_quarter_tuple = (end_year, end_quarter)
                        except:
                            pass
                    
                    # generate quarter lists for different windows
                    # exact: just the start quarter (or range if end exists)
                    # each date is treated as representing its entire quarter
                    if end_quarter_tuple:
                        # multiple quarters from start to end
                        quarters_exact = []
                        current_year, current_q = start_year, start_quarter
                        end_year, end_q = end_quarter_tuple
                        while (current_year, current_q) <= (end_year, end_q):
                            quarters_exact.append((current_year, current_q))
                            # increment quarter
                            if current_q == 4:
                                current_year += 1
                                current_q = 1
                            else:
                                current_q += 1
                    else:
                        # single quarter
                        quarters_exact = [(start_year, start_quarter)]
                    
                    # window 1: ±1 quarter around start quarter
                    quarters_window1 = self._get_adjacent_quarters(start_year, start_quarter, window=1)
                    
                    # window 2: ±2 quarters around start quarter
                    quarters_window2 = self._get_adjacent_quarters(start_year, start_quarter, window=2)
                    
                    # window 4: ±4 quarters around start quarter
                    quarters_window4 = self._get_adjacent_quarters(start_year, start_quarter, window=4)
                    
                    # pre-bias: -4q to exact (detect if bias emerges before)
                    quarters_pre4 = self._get_pre_bias_quarters(start_year, start_quarter, window=4)
                    
                    # post-bias: exact to +4q (detect if bias persists after)
                    quarters_post4 = self._get_post_bias_quarters(start_year, start_quarter, window=4)
                    
                    expert_periods[ticker] = {
                        'target_folder': target_folder,
                        'bias_start': date_str,
                        'bias_end': bias_end_str if bias_end_str not in ['null', 'None', '', None] else None,
                        'bias_type': bias_type,
                        'rationale': rationale,
                        'start_quarter': (start_year, start_quarter),
                        'end_quarter': end_quarter_tuple,
                        'quarters_exact': quarters_exact,
                        'quarters_window1': quarters_window1,
                        'quarters_window2': quarters_window2,
                        'quarters_window4': quarters_window4,
                        'quarters_pre4': quarters_pre4,
                        'quarters_post4': quarters_post4
                    }
                except Exception as e:
                    print(f"  ️  Could not parse date for {ticker}: {date_str} - {e}")
                    continue
            
            print(f"   Extracted expert bias periods for {len(expert_periods)} target companies")
            
            # show all entries (or limit to first 10 if many)
            display_count = min(10, len(expert_periods))
            for ticker in list(expert_periods.keys())[:display_count]:
                info = expert_periods[ticker]
                end_info = f" to {info['end_quarter']}" if info['end_quarter'] else ""
                print(f"    • {ticker}: {info['bias_type']} starting {info['start_quarter']}{end_info}")
            
            if len(expert_periods) > display_count:
                print(f"    ... and {len(expert_periods) - display_count} more")
            
            return expert_periods
            
        except Exception as e:
            print(f"   Error loading expert periods: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _add_bias_period_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add bias period flags to dataframe for exact, ±1, and ±2 quarter windows.
        
        FLAGS BOTH TARGETS AND THEIR PEERS:
        - Targets: Flagged for their own bias periods
        - Peers: Flagged when they're in the SAME quarters as their target's bias period
          (This enables "Peer (Bias Period)" comparisons showing peers during target bias periods)
        
        Adds columns:
        - is_bias_period_exact: Boolean, in exact bias quarter(s)
        - is_bias_period_window1: Boolean, within ±1 quarter of bias
        - is_bias_period_window2: Boolean, within ±2 quarters of bias
        - bias_type: String, type of bias expected
        - bias_ticker: String, ticker symbol for target
        - bias_period_label: String, descriptive label (e.g., "Pre-Bias", "Bias", "Post-Bias")
        """
        print("\n Adding Bias Period Flags to Data")
        print("=" * 80)
        print("  NOTE: Flagging BOTH targets AND their peers during bias periods")
        print("        (Peers flagged when in same quarters as their target's bias period)")
        
        # initialize columns
        df['is_bias_period_exact'] = False
        df['is_bias_period_window1'] = False
        df['is_bias_period_window2'] = False
        df['is_bias_period_window4'] = False
        df['is_bias_period_pre4'] = False
        df['is_bias_period_post4'] = False
        df['bias_type'] = None
        df['bias_ticker'] = None
        df['bias_period_label'] = 'Normal'
        
        labeled_exact = 0
        labeled_w1 = 0
        labeled_w2 = 0
        labeled_w4 = 0
        labeled_pre4 = 0
        labeled_post4 = 0
        
        for ticker, info in self.expert_bias_periods.items():
            target_folder = info['target_folder']
            
            # get peer folders for this target
            peer_folders = set()
            for peer_key, peer_info in self.peer_groups.items():
                if peer_info.get('target_folder') == target_folder:
                    peer_folders = set(peer_info.get('direct_competitors_folders', []))
                    peer_folders = {p for p in peer_folders if p is not None}
                    break
            
            # match target company by folder name or ticker
            target_mask = (df['Company'] == target_folder) | (df['Company'] == ticker)
            
            # match peer companies by folder names
            peer_mask = df['Company'].isin(peer_folders) if peer_folders else pd.Series([False] * len(df))
            
            # combined mask for targets and peers
            company_mask = target_mask | peer_mask
            
            if company_mask.sum() == 0:
                continue
            
            # label exact bias period (for both targets and peers)
            for year, quarter in info['quarters_exact']:
                exact_mask = company_mask & (df['Year'] == year) & (df['Quarter'] == quarter)
                df.loc[exact_mask, 'is_bias_period_exact'] = True
                df.loc[exact_mask, 'bias_type'] = info['bias_type']
                df.loc[exact_mask, 'bias_ticker'] = ticker
                df.loc[exact_mask, 'bias_period_label'] = 'Bias Period'
                labeled_exact += exact_mask.sum()
            
            # label ±1 quarter window
            for year, quarter in info['quarters_window1']:
                window1_mask = company_mask & (df['Year'] == year) & (df['Quarter'] == quarter)
                df.loc[window1_mask, 'is_bias_period_window1'] = True
                # only update bias_type and ticker if not already set
                if df.loc[window1_mask, 'bias_type'].isna().any():
                    df.loc[window1_mask, 'bias_type'] = info['bias_type']
                if df.loc[window1_mask, 'bias_ticker'].isna().any():
                    df.loc[window1_mask, 'bias_ticker'] = ticker
                # label as pre/post if not exact bias period
                pre_post_mask = window1_mask & ~df['is_bias_period_exact']
                if pre_post_mask.sum() > 0:
                    # determine if pre or post based on quarter
                    start_year, start_q = info['start_quarter']
                    for idx in df[pre_post_mask].index:
                        row_year, row_q = df.loc[idx, 'Year'], df.loc[idx, 'Quarter']
                        if (row_year, row_q) < (start_year, start_q):
                            df.loc[idx, 'bias_period_label'] = 'Pre-Bias'
                        else:
                            df.loc[idx, 'bias_period_label'] = 'Post-Bias'
                labeled_w1 += window1_mask.sum()
            
            # label ±2 quarter window
            for year, quarter in info['quarters_window2']:
                window2_mask = company_mask & (df['Year'] == year) & (df['Quarter'] == quarter)
                df.loc[window2_mask, 'is_bias_period_window2'] = True
                if df.loc[window2_mask, 'bias_type'].isna().any():
                    df.loc[window2_mask, 'bias_type'] = info['bias_type']
                if df.loc[window2_mask, 'bias_ticker'].isna().any():
                    df.loc[window2_mask, 'bias_ticker'] = ticker
                labeled_w2 += window2_mask.sum()
            
            # label ±4 quarter window
            for year, quarter in info['quarters_window4']:
                window4_mask = company_mask & (df['Year'] == year) & (df['Quarter'] == quarter)
                df.loc[window4_mask, 'is_bias_period_window4'] = True
                if df.loc[window4_mask, 'bias_type'].isna().any():
                    df.loc[window4_mask, 'bias_type'] = info['bias_type']
                if df.loc[window4_mask, 'bias_ticker'].isna().any():
                    df.loc[window4_mask, 'bias_ticker'] = ticker
                labeled_w4 += window4_mask.sum()
            
            # label pre-bias period (-4q to exact, including exact)
            for year, quarter in info['quarters_pre4']:
                pre4_mask = company_mask & (df['Year'] == year) & (df['Quarter'] == quarter)
                df.loc[pre4_mask, 'is_bias_period_pre4'] = True
                if df.loc[pre4_mask, 'bias_type'].isna().any():
                    df.loc[pre4_mask, 'bias_type'] = info['bias_type']
                if df.loc[pre4_mask, 'bias_ticker'].isna().any():
                    df.loc[pre4_mask, 'bias_ticker'] = ticker
                labeled_pre4 += pre4_mask.sum()
            
            # label post-bias period (exact to +4q, including exact)
            for year, quarter in info['quarters_post4']:
                post4_mask = company_mask & (df['Year'] == year) & (df['Quarter'] == quarter)
                df.loc[post4_mask, 'is_bias_period_post4'] = True
                if df.loc[post4_mask, 'bias_type'].isna().any():
                    df.loc[post4_mask, 'bias_type'] = info['bias_type']
                if df.loc[post4_mask, 'bias_ticker'].isna().any():
                    df.loc[post4_mask, 'bias_ticker'] = ticker
                labeled_post4 += post4_mask.sum()
        
        print(f"\n   Labeled {labeled_exact:,} segments in exact bias periods (targets + peers)")
        print(f"   Labeled {labeled_w1:,} segments in ±1Q bias windows (targets + peers)")
        print(f"   Labeled {labeled_w2:,} segments in ±2Q bias windows (targets + peers)")
        print(f"   Labeled {labeled_w4:,} segments in ±4Q bias windows (targets + peers)")
        print(f"   Labeled {labeled_pre4:,} segments in -4Q pre-bias windows (targets + peers)")
        print(f"   Labeled {labeled_post4:,} segments in +4Q post-bias windows (targets + peers)")
        
        # show breakdown by is_target flag
        exact_df = df[df['is_bias_period_exact'] == True]
        exact_targets = (exact_df['IS_TARGET'] == 'Y').sum() if len(exact_df) > 0 else 0
        exact_peers = (exact_df['IS_TARGET'] == 'N').sum() if len(exact_df) > 0 else 0
        
        print(f"\n   Breakdown of Exact Bias Period Labels:")
        print(f"      Target segments: {exact_targets:,}")
        print(f"      Peer segments:   {exact_peers:,}")
        print(f"      Total:           {exact_targets + exact_peers:,}")
        
        print(f"\n  → {(df['is_bias_period_exact'].sum() / len(df) * 100):.1f}% exact bias labels")
        print(f"  → {(df['is_bias_period_window1'].sum() / len(df) * 100):.1f}% in ±1Q window")
        print(f"  → {(df['is_bias_period_window2'].sum() / len(df) * 100):.1f}% in ±2Q window")
        print(f"  → {(df['is_bias_period_window4'].sum() / len(df) * 100):.1f}% in ±4Q window")
        print(f"  → {(df['is_bias_period_pre4'].sum() / len(df) * 100):.1f}% in -4Q pre-bias window")
        print(f"  → {(df['is_bias_period_post4'].sum() / len(df) * 100):.1f}% in +4Q post-bias window")
        
        return df
    
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
            # use target_folder for matching (this is what csv filenames use)
            target_folder = info.get('target_folder', target_key)
            
            # use peer_folders for peer matching (these are actual folder names)
            peer_folders = info.get('peer_folders', [])
            
            # check if this company is the target (match against target_folder)
            if company == target_folder:
                is_target = True
                related_targets.append(target_key)
            
            # check if this company is a peer (match against peer_folders list)
            if company in peer_folders:
                is_peer = True
                if target_key not in related_targets:
                    related_targets.append(target_key)
        
        return is_target, is_peer, related_targets
    
    def _map_topic_to_category(self, topic: str) -> str:
        """Map a detailed topic to its broader category."""
        if pd.isna(topic) or topic == '':
            return 'Unknown'
        
        for category, topics in self.TOPIC_CATEGORY_MAPPING.items():
            if topic in topics:
                return category
        
        return 'Others'
    
    def load_attribution_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load attribution data and return TWO DataFrames:
        1. Filtered for attribution analysis (only rows with attribution_present='Y')
        2. Unfiltered for general label analysis (includes all snippets)"""
        print("\n1. Loading Attribution Data with Mapping")
        print("=" * 80)
        
        # check all instance directories
        all_instance_dirs = sorted(list(self.input_dir.glob("instance*")))
        instance_dirs = sorted(list(self.input_dir.glob("instance*/05")))
        
        if not all_instance_dirs:
            print(f" No instance directories found in {self.input_dir}")
            return pd.DataFrame()
        
        print(f" Found {len(all_instance_dirs)} total instance folders")
        print(f" Found {len(instance_dirs)} instance folders with 05/ subdirectory")
        
        if len(instance_dirs) < len(all_instance_dirs):
            missing = [d.name for d in all_instance_dirs if not (d / "05").exists()]
            print(f"   Missing 05/ folder in: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")
        
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
                    print(f"   Error loading {csv_file.name}: {e}")
        
        if not dfs:
            print("No data loaded")
            return pd.DataFrame()
        
        combined_df_all = pd.concat(dfs, ignore_index=True)
        
        # diagnostic: check team distribution before filtering
        if 'Team' in combined_df_all.columns:
            print(f"\n Team Distribution BEFORE Attribution Filtering:")
            team_counts_before = combined_df_all['Team'].value_counts()
            for team, count in team_counts_before.items():
                print(f"    {team}: {count:,} rows ({count/len(combined_df_all)*100:.1f}%)")
        
        initial_rows = len(combined_df_all)
        
        # create filtered dataset for attribution analysis
        combined_df_attribution = combined_df_all[
            (combined_df_all['attribution_present'] == 'Y') &
            (combined_df_all['attribution_outcome'] != 'filtered_out') &
            (combined_df_all['attribution_locus'] != 'filtered_out')
        ].copy()
        
        # diagnostic: check team distribution after filtering
        if 'Team' in combined_df_attribution.columns:
            print(f"\n Team Distribution AFTER Attribution Filtering:")
            team_counts_after = combined_df_attribution['Team'].value_counts()
            for team, count in team_counts_after.items():
                print(f"    {team}: {count:,} rows ({count/len(combined_df_attribution)*100:.1f}%)")
            print(f"  ️  Note: external_members filtered out if they don't make attribution statements")
            print(f"    Solution: Use unfiltered data for non-attribution labels (Topics/Sentiment/Tone)")
        
        # count unique companies by type (using filtered data)
        unique_companies = combined_df_attribution['Company'].nunique()
        target_companies = combined_df_attribution[combined_df_attribution['IS_TARGET']=='Y']['Company'].nunique()
        peer_companies = combined_df_attribution[combined_df_attribution['IS_PEER']=='Y']['Company'].nunique()
        
        print(f"\n Loaded {file_count} CSV files")
        print(f" Found {unique_companies} unique companies in data")
        print(f" Matched {target_companies} target companies from config")
        print(f" Matched {peer_companies} peer companies from config")
        print(f"\nData Summary:")
        print(f"  Total rows (unfiltered): {initial_rows:,}")
        print(f"  After attribution filtering: {len(combined_df_attribution):,} ({len(combined_df_attribution)/initial_rows*100:.1f}%)")
        print(f"  Target company rows (filtered): {(combined_df_attribution['IS_TARGET']=='Y').sum():,}")
        print(f"  Peer company rows (filtered): {(combined_df_attribution['IS_PEER']=='Y').sum():,}")
        
        if target_companies == 0:
            print(f"\n WARNING: No target companies found! Check company name matching between:")
            print(f"    - CSV filenames (companies: {', '.join(list(companies_found)[:5])}...)")
            print(f"    - Config file targets (targets: {', '.join(list(self.peer_groups.keys())[:5])}...)")
        
        # add snippet-level attribution type and sab flag (only to filtered data)
        print(f"\n Adding Snippet-Level SAB Metrics (Attribution Data Only)")
        print("=" * 80)
        
        # create 4-way attribution type (pos-int, pos-ext, neg-int, neg-ext)
        combined_df_attribution['attribution_type_4way'] = combined_df_attribution.apply(
            lambda row: f"{row['attribution_outcome']}-{row['attribution_locus']}" 
            if row['attribution_outcome'] in ['Positive', 'Negative'] 
            and row['attribution_locus'] in ['Internal', 'External']
            else 'Other',
            axis=1
        )
        
        # flag sab statements (positive-internal or negative-external)
        combined_df_attribution['is_sab'] = (
            ((combined_df_attribution['attribution_outcome'] == 'Positive') & (combined_df_attribution['attribution_locus'] == 'Internal')) |
            ((combined_df_attribution['attribution_outcome'] == 'Negative') & (combined_df_attribution['attribution_locus'] == 'External'))
        )
        
        sab_count = combined_df_attribution['is_sab'].sum()
        sab_pct = (sab_count / len(combined_df_attribution) * 100) if len(combined_df_attribution) > 0 else 0
        
        print(f"   Created 4-way attribution types")
        print(f"   SAB statements: {sab_count:,} ({sab_pct:.1f}% of all attributions)")
        
        # value counts for attribution types
        type_counts = combined_df_attribution['attribution_type_4way'].value_counts()
        
        # calculate counts excluding 'other'
        four_way_total = sum(count for attr_type, count in type_counts.items() if attr_type != 'Other')
        other_count = type_counts.get('Other', 0)
        
        print(f"\n   Attribution Distribution (Raw):")
        for attr_type, count in type_counts.items():
            if attr_type != 'Other':
                pct = (count / len(combined_df_attribution) * 100)
                print(f"    - {attr_type}: {count:,} ({pct:.1f}% of all)")
        if other_count > 0:
            other_pct = (other_count / len(combined_df_attribution) * 100)
            print(f"    - Other/Unknown: {other_count:,} ({other_pct:.1f}% of all)")
        
        print(f"\n   Attribution Distribution (Normalized - 4-way only):")
        for attr_type, count in type_counts.items():
            if attr_type != 'Other':
                normalized_pct = (count / four_way_total * 100) if four_way_total > 0 else 0
                print(f"    - {attr_type}: {count:,} ({normalized_pct:.1f}% of 4-way)")
        
        # verify normalization
        total_normalized = sum((count / four_way_total * 100) for attr_type, count in type_counts.items() if attr_type != 'Other') if four_way_total > 0 else 0
        print(f"   Normalized total: {total_normalized:.1f}% (should be 100%)")
        
        # add topic category mapping for hierarchical analysis (to both dataframes)
        print(f"\n Adding Topic Category Mapping (Both Datasets)")
        print("=" * 80)
        
        topic_levels = ['Primary_Topic', 'Secondary_Topic', 'Tertiary_Topic', 'Quaternary_Topic', 'Quinary_Topic']
        category_levels = ['Primary_Topic_Category', 'Secondary_Topic_Category', 'Tertiary_Topic_Category', 
                          'Quaternary_Topic_Category', 'Quinary_Topic_Category']
        
        # map topics to categories in both dataframes
        for df_name, df in [('Attribution (filtered)', combined_df_attribution), ('All Labels (unfiltered)', combined_df_all)]:
            for topic_col, category_col in zip(topic_levels, category_levels):
                if topic_col in df.columns:
                    df[category_col] = df[topic_col].apply(self._map_topic_to_category)
            
            # print category distribution
            if 'Primary_Topic_Category' in df.columns:
                cat_counts = df['Primary_Topic_Category'].value_counts()
                print(f"   {df_name}: Mapped {len(df):,} topics to {len(cat_counts)} categories")
                for cat, count in cat_counts.head(5).items():
                    pct = (count / len(df) * 100)
                    print(f"    - {cat}: {count:,} ({pct:.1f}%)")
        
        # add bias period flags to both dataframes
        print(f"\n Adding Bias Period Flags (Both Datasets)")
        combined_df_attribution = self._add_bias_period_flags(combined_df_attribution)
        combined_df_all = self._add_bias_period_flags(combined_df_all)
        
        print(f"\n Returning TWO DataFrames:")
        print(f"  1. df_attribution: {len(combined_df_attribution):,} rows (for attribution analysis)")
        print(f"  2. df_all_labels: {len(combined_df_all):,} rows (for general label distributions)")
        
        return combined_df_attribution, combined_df_all
    
    def calculate_weighted_label_distributions(
        self, 
        df: pd.DataFrame,
        analysis_level: str = 'full_transcript',  # 'full_transcript' or 'section'
        section_filter: str = None,  # 'prepared_remarks', 'qna_section', or none
        team_filter: str = None  # 'management_team', 'external_members', 'unknown', or none
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate weighted distributions of GPT labels with proper accounting for:
        
        1. **Multi-Level Topic Coverage**: Topics, Temporal Context, and Sentiment can have
           Primary through Quinary levels. Each level has a coverage percentage indicating
           what proportion of the snippet it represents. We weight each level by its coverage.
           
        2. **Snippet Size Weighting**: Snippets vary in length. We weight by:
           - Snippet_Pct_Transcript: For full transcript analysis (what % of call this snippet is)
           - Snippet_Pct_Section: For section-specific analysis (what % of section this snippet is)
           
        3. **Combined Weighting Logic**:
           - For multi-level labels: weight = (coverage/100) × (snippet_pct/100)
           - For single-level labels: weight = (snippet_pct/100)
           
        4. **Optional Filtering**: Can filter by section and/or team before calculating distributions
        
        **Rationale**: Without this weighting, we'd treat a 30-second snippet covering 3 topics
        the same as a 3-minute snippet with 1 topic, and we'd count Primary topics (85% coverage)
        equally with Tertiary topics (5% coverage), leading to skewed distributions."""
        
        # define label structure
        label_types = {
            'Topic': {
                'levels': ['Primary_Topic', 'Secondary_Topic', 'Tertiary_Topic', 
                           'Quaternary_Topic', 'Quinary_Topic'],
                'coverage_cols': ['Primary_Topic_Coverage', 'Secondary_Topic_Coverage', 
                                  'Tertiary_Topic_Coverage', 'Quaternary_Topic_Coverage', 
                                  'Quinary_Topic_Coverage'],
                'multi_level': True,
                'title': 'Topic Distribution (Weighted - Detailed)'
            },
            'Topic_Category': {
                'levels': ['Primary_Topic_Category', 'Secondary_Topic_Category', 'Tertiary_Topic_Category', 
                           'Quaternary_Topic_Category', 'Quinary_Topic_Category'],
                'coverage_cols': ['Primary_Topic_Coverage', 'Secondary_Topic_Coverage', 
                                  'Tertiary_Topic_Coverage', 'Quaternary_Topic_Coverage', 
                                  'Quinary_Topic_Coverage'],
                'multi_level': True,
                'title': 'Topic Distribution (Weighted - Category Level)'
            },
            'Temporal_Context': {
                'levels': ['Primary_Temporal_Context', 'Secondary_Temporal_Context', 
                           'Tertiary_Temporal_Context', 'Quaternary_Temporal_Context', 
                           'Quinary_Temporal_Context'],
                'coverage_cols': ['Primary_Topic_Coverage', 'Secondary_Topic_Coverage', 
                                  'Tertiary_Topic_Coverage', 'Quaternary_Topic_Coverage', 
                                  'Quinary_Topic_Coverage'],  # use topic_coverage as proxy
                'multi_level': True,
                'title': 'Temporal Context Distribution (Weighted)'
            },
            'Content_Sentiment': {
                'levels': ['Primary_Content_Sentiment', 'Secondary_Content_Sentiment',
                           'Tertiary_Content_Sentiment', 'Quaternary_Content_Sentiment',
                           'Quinary_Content_Sentiment'],
                'coverage_cols': ['Primary_Topic_Coverage', 'Secondary_Topic_Coverage', 
                                  'Tertiary_Topic_Coverage', 'Quaternary_Topic_Coverage', 
                                  'Quinary_Topic_Coverage'],  # use topic_coverage as proxy
                'multi_level': True,
                'title': 'Content Sentiment Distribution (Weighted)'
            },
            'Speaker_Tone': {
                'levels': ['Speaker_Tone'],
                'coverage_cols': None,
                'multi_level': False,
                'title': 'Speaker Tone Distribution'
            },
            'attribution_present': {
                'levels': ['attribution_present'],
                'coverage_cols': None,
                'multi_level': False,
                'title': 'Attribution Presence'
            },
            'attribution_outcome': {
                'levels': ['attribution_outcome'],
                'coverage_cols': None,
                'multi_level': False,
                'title': 'Attribution Outcome'
            },
            'attribution_locus': {
                'levels': ['attribution_locus'],
                'coverage_cols': None,
                'multi_level': False,
                'title': 'Attribution Locus'
            }
        }
        
        # step 1: apply filters
        df_filtered = df.copy()
        
        if section_filter:
            df_filtered = df_filtered[df_filtered['Section'] == section_filter].copy()
            weight_col = 'Snippet_Pct_Section'
        else:
            weight_col = 'Snippet_Pct_Transcript'
        
        if team_filter:
            df_filtered = df_filtered[df_filtered['Team'] == team_filter].copy()
        
        # check if weight column exists
        if weight_col not in df_filtered.columns:
            print(f"  ️  Weight column '{weight_col}' not found, using equal weights")
            df_filtered[weight_col] = 100.0 / len(df_filtered)  # equal weight fallback
        
        results = {}
        
        # step 2: calculate distributions for each label type
        for label_name, label_info in label_types.items():
            weighted_counts = {}
            raw_counts = {}
            
            if label_info['multi_level']:
                # multi-level labels with coverage weighting
                level_cols = label_info['levels']
                coverage_cols = label_info['coverage_cols']
                
                # check if columns exist
                available_levels = [col for col in level_cols if col in df_filtered.columns]
                available_coverage = [col for col in coverage_cols if col in df_filtered.columns]
                
                if not available_levels:
                    continue
                
                for idx, row in df_filtered.iterrows():
                    snippet_weight = row[weight_col] / 100.0  # convert percentage to decimal
                    
                    # iterate through all levels
                    for level_idx, level_col in enumerate(available_levels):
                        label_value = row.get(level_col)
                        
                        # get coverage weight (default to 100 if not available)
                        if level_idx < len(available_coverage):
                            coverage = row.get(available_coverage[level_idx], 0)
                        else:
                            coverage = 100 if pd.notna(label_value) and label_value != '' else 0
                        
                        # skip if blank, zero coverage, or filtered_out
                        if pd.notna(label_value) and label_value not in ['', 'filtered_out', 'Unknown'] and coverage > 0:
                            # combined weight = (coverage/100) × snippet_weight
                            combined_weight = (coverage / 100.0) * snippet_weight
                            
                            # add to weighted counts
                            if label_value not in weighted_counts:
                                weighted_counts[label_value] = 0
                                raw_counts[label_value] = 0
                            weighted_counts[label_value] += combined_weight
                            raw_counts[label_value] += 1
            else:
                # single-level labels
                label_col = label_info['levels'][0]
                
                if label_col not in df_filtered.columns:
                    continue
                
                for idx, row in df_filtered.iterrows():
                    snippet_weight = row[weight_col] / 100.0
                    label_value = row.get(label_col)
                    
                    if pd.notna(label_value) and label_value not in ['', 'filtered_out']:
                        if label_value not in weighted_counts:
                            weighted_counts[label_value] = 0
                            raw_counts[label_value] = 0
                        weighted_counts[label_value] += snippet_weight
                        raw_counts[label_value] += 1
            
            # step 3: normalize to percentages
            if weighted_counts:
                total_weight = sum(weighted_counts.values())
                
                dist_data = []
                for label_value, weight in weighted_counts.items():
                    dist_data.append({
                        'Category': label_value,
                        'Weighted_Count': weight,
                        'Percentage': (weight / total_weight * 100) if total_weight > 0 else 0,
                        'Raw_Count': raw_counts[label_value]
                    })
                
                dist_df = pd.DataFrame(dist_data).sort_values('Percentage', ascending=False)
                results[label_name] = dist_df
        
        return results
    
    def analyze_gpt_label_distributions(self, df_attribution: pd.DataFrame, df_all_labels: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create weighted bar charts showing distribution of all GPT classifications.
        
        Uses TWO datasets:
        - df_attribution: Filtered to attribution_present='Y' (for attribution_outcome, attribution_locus, attribution_present)
        - df_all_labels: Unfiltered (for Topic, Temporal_Context, Content_Sentiment, Speaker_Tone)
        
        This ensures external_members (analysts) are included in non-attribution label distributions,
        even though they don't make attribution statements.
        
        Generates:
        1. Full transcript distributions (baseline)
        2. Section comparisons (prepared_remarks vs qna_section)
        3. Team comparisons (management vs external) - ONLY for non-attribution labels
        
        Returns dictionary of DataFrames with distribution statistics.
        """
        print("\n2. Analyzing GPT Label Distributions (Weighted)")
        print("=" * 80)
        print("  NOTE: Using unfiltered data for Topics/Sentiment/Tone (includes all speakers)")
        print("        Using filtered data for Attribution labels (only rows with attributions)")
        
        all_results = {}
        
        # define which labels use which dataset
        attribution_labels = {'attribution_present', 'attribution_outcome', 'attribution_locus'}
        non_attribution_labels = {'Topic', 'Topic_Category', 'Temporal_Context', 'Content_Sentiment', 'Speaker_Tone'}
        
        # 1. full transcript analysis (baseline)
        print("\n   Full Transcript Analysis (Baseline)")
        print("      Creating TWO versions for non-attribution labels:")
        print("        1. All Snippets (includes questions, all statements)")
        print("        2. Attribution Statements Only (subset where attribution_present='Y')")
        
        # calculate distributions for non-attribution labels on both datasets
        full_results_non_attr_all = self.calculate_weighted_label_distributions(
            df_all_labels, 
            analysis_level='full_transcript'
        )
        
        full_results_non_attr_filtered = self.calculate_weighted_label_distributions(
            df_attribution, 
            analysis_level='full_transcript'
        )
        
        # calculate distributions for attribution labels using filtered data only
        full_results_attr = self.calculate_weighted_label_distributions(
            df_attribution, 
            analysis_level='full_transcript'
        )
        
        # combine results (non-attribution labels get "all snippets" version for now)
        full_results = {}
        for label_name, dist_df in full_results_non_attr_all.items():
            if label_name in non_attribution_labels:
                full_results[label_name] = dist_df
        for label_name, dist_df in full_results_attr.items():
            if label_name in attribution_labels:
                full_results[label_name] = dist_df
        
        for label_name, dist_df in full_results.items():
            print(f"     {label_name}: {len(dist_df)} categories")
        
        # create visualizations for all versions
        print("\n   Creating Visualizations:")
        
        # for non-attribution labels: create both versions
        for label_name in non_attribution_labels:
            if label_name in full_results_non_attr_all:
                # version 1: all snippets
                dist_df_all = full_results_non_attr_all[label_name]
                if len(dist_df_all) > 0:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plot_df = dist_df_all.head(15)
                    
                    bars = ax.bar(range(len(plot_df)), plot_df['Percentage'], 
                                 color='steelblue', edgecolor='black', alpha=0.8)
                    ax.set_xticks(range(len(plot_df)))
                    # truncate long labels (over 60 characters)
                    truncated_labels = [cat[:60] + '...' if len(str(cat)) > 60 else cat 
                                       for cat in plot_df['Category']]
                    ax.set_xticklabels(truncated_labels, rotation=45, ha='right')
                    ax.set_ylabel('Weighted Percentage (%)', fontsize=12)
                    
                    title = f'{label_name} Distribution - ALL SNIPPETS\n(Includes questions, all statements)'
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
                    
                    for i, (pct, raw_count) in enumerate(zip(plot_df['Percentage'], plot_df['Raw_Count'])):
                        ax.text(i, pct + 0.5, f'{pct:.1f}%\n(n={raw_count})', 
                               ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    plt.savefig(self.viz_dir / f'distribution_{label_name}_all_snippets.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"       Saved: distribution_{label_name}_all_snippets.png")
            
            if label_name in full_results_non_attr_filtered:
                # version 2: attribution statements only
                dist_df_attr = full_results_non_attr_filtered[label_name]
                if len(dist_df_attr) > 0:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plot_df = dist_df_attr.head(15)
                    
                    bars = ax.bar(range(len(plot_df)), plot_df['Percentage'], 
                                 color='coral', edgecolor='black', alpha=0.8)
                    ax.set_xticks(range(len(plot_df)))
                    # truncate long labels (over 60 characters)
                    truncated_labels = [cat[:60] + '...' if len(str(cat)) > 60 else cat 
                                       for cat in plot_df['Category']]
                    ax.set_xticklabels(truncated_labels, rotation=45, ha='right')
                    ax.set_ylabel('Weighted Percentage (%)', fontsize=12)
                    
                    title = f'{label_name} Distribution - ATTRIBUTION STATEMENTS ONLY\n(Subset where attribution_present=Y)'
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
                    
                    for i, (pct, raw_count) in enumerate(zip(plot_df['Percentage'], plot_df['Raw_Count'])):
                        ax.text(i, pct + 0.5, f'{pct:.1f}%\n(n={raw_count})', 
                               ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    plt.savefig(self.viz_dir / f'distribution_{label_name}_attribution_only.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"       Saved: distribution_{label_name}_attribution_only.png")
        
        # for attribution labels: only create one version (they're always filtered)
        for label_name in attribution_labels:
            if label_name in full_results_attr:
                dist_df = full_results_attr[label_name]
                if len(dist_df) > 0:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plot_df = dist_df.head(15)
                    
                    bars = ax.bar(range(len(plot_df)), plot_df['Percentage'], 
                                 color='darkgreen', edgecolor='black', alpha=0.8)
                    ax.set_xticks(range(len(plot_df)))
                    # truncate long labels (over 60 characters)
                    truncated_labels = [cat[:60] + '...' if len(str(cat)) > 60 else cat 
                                       for cat in plot_df['Category']]
                    ax.set_xticklabels(truncated_labels, rotation=45, ha='right')
                    ax.set_ylabel('Weighted Percentage (%)', fontsize=12)
                    
                    title = f'{label_name} Distribution\n(Attribution statements only)'
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
                    
                    for i, (pct, raw_count) in enumerate(zip(plot_df['Percentage'], plot_df['Raw_Count'])):
                        ax.text(i, pct + 0.5, f'{pct:.1f}%\n(n={raw_count})', 
                               ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    plt.savefig(self.viz_dir / f'distribution_{label_name}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"       Saved: distribution_{label_name}.png")
        
        all_results['full_transcript'] = full_results
        
        # 2. section comparison (prepared_remarks vs qna_section)
        print("\n   Section-Level Analysis")
        print("      NOTE: Section comparisons also create TWO versions for non-attribution labels")
        
        sections = ['prepared_remarks', 'qna_section']
        section_results = {}
        
        for section in sections:
            # non-attribution labels: use unfiltered data
            section_df_all = df_all_labels[df_all_labels['Section'] == section] if 'Section' in df_all_labels.columns else pd.DataFrame()
            # attribution labels: use filtered data
            section_df_attr = df_attribution[df_attribution['Section'] == section] if 'Section' in df_attribution.columns else pd.DataFrame()
            
            if len(section_df_all) > 0 or len(section_df_attr) > 0:
                # calculate for non-attribution labels
                section_results_non_attr = self.calculate_weighted_label_distributions(
                    section_df_all,
                    analysis_level='section',
                    section_filter=section
                ) if len(section_df_all) > 0 else {}
                
                # calculate for attribution labels
                section_results_attr = self.calculate_weighted_label_distributions(
                    section_df_attr,
                    analysis_level='section',
                    section_filter=section
                ) if len(section_df_attr) > 0 else {}
                
                # combine results
                combined = {}
                for label_name, dist_df in section_results_non_attr.items():
                    if label_name in non_attribution_labels:
                        combined[label_name] = dist_df
                for label_name, dist_df in section_results_attr.items():
                    if label_name in attribution_labels:
                        combined[label_name] = dist_df
                
                section_results[section] = combined
                print(f"     {section}: {len(section_df_all):,} total snippets (non-attr labels), {len(section_df_attr):,} attribution snippets")
        
        # create comparison charts for each label type
        # note: using combined results which uses "all snippets" for non-attribution labels
        for label_name in full_results.keys():
            if all(label_name in section_results.get(s, {}) for s in sections if s in section_results):
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                for idx, section in enumerate(sections):
                    if section not in section_results:
                        continue
                    
                    dist_df = section_results[section][label_name].head(10)
                    
                    axes[idx].bar(range(len(dist_df)), dist_df['Percentage'],
                                 color='coral' if section == 'prepared_remarks' else 'skyblue',
                                 edgecolor='black', alpha=0.8)
                    axes[idx].set_xticks(range(len(dist_df)))
                    # truncate long labels (over 60 characters)
                    truncated_labels = [cat[:60] + '...' if len(str(cat)) > 60 else cat 
                                       for cat in dist_df['Category']]
                    axes[idx].set_xticklabels(truncated_labels, rotation=45, ha='right', fontsize=9)
                    axes[idx].set_ylabel('Weighted %', fontsize=11)
                    axes[idx].set_title(f'{section.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                    axes[idx].grid(axis='y', alpha=0.3)
                    
                    # add labels
                    for i, pct in enumerate(dist_df['Percentage']):
                        axes[idx].text(i, pct + 0.5, f'{pct:.1f}%', 
                                      ha='center', va='bottom', fontsize=8)
                
                # add note about data source in title
                data_note = "All Snippets" if label_name in non_attribution_labels else "Attribution Statements"
                fig.suptitle(f'{label_name} Distribution: Section Comparison ({data_note})', 
                            fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                plt.savefig(self.viz_dir / f'comparison_section_{label_name}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        all_results['by_section'] = section_results
        
        # 3. team comparison (q&a section only: management vs external members)
        print("\n   Team-Level Analysis (Q&A Section Only)")
        print("    NOTE: Team analysis focuses on Q&A where both management and analysts participate")
        print("    Prepared remarks are primarily management, so team comparison is only meaningful in Q&A")
        print("    ️  Attribution labels EXCLUDED from team comparison (external_members don't make attributions)")
        print("      Uses ALL SNIPPETS data (includes questions, all statements) - no 'attribution-only' version")
        
        # initialize team_results
        team_results = {}
        
        # filter to q&a section only - use unfiltered data (to include external_members)
        qna_df = df_all_labels[df_all_labels['Section'] == 'qna_section'] if 'Section' in df_all_labels.columns else pd.DataFrame()
        
        if len(qna_df) == 0:
            print(f"    ️ No Q&A section data found")
        else:
            # check what teams exist in q&a section
            if 'Team' in qna_df.columns:
                available_teams = qna_df['Team'].dropna().unique().tolist()
                print(f"    Available teams in Q&A: {available_teams}")
            else:
                available_teams = []
                print(f"    ️ 'Team' column not found in data")
            
            # only analyze management_team and external_members (filter out unknown)
            teams_to_analyze = ['management_team', 'external_members']
            teams = [t for t in teams_to_analyze if t in available_teams]
            
            if len(teams) == 0:
                print(f"    ️ No management_team or external_members found in Q&A section")
            else:
                for team in teams:
                    team_df = qna_df[qna_df['Team'] == team]
                    if len(team_df) > 0:
                        # calculate distributions using unfiltered q&a data
                        team_results[team] = self.calculate_weighted_label_distributions(
                            team_df,
                            analysis_level='section',
                            section_filter='qna_section',
                            team_filter=team
                        )
                        print(f"     {team}: {len(team_df):,} Q&A snippets analyzed")
                
                # create comparison charts for each label type (skip attribution labels)
                for label_name in full_results.keys():
                    # skip attribution labels in team comparison (external_members don't make attributions)
                    if label_name in attribution_labels:
                        continue
                    
                    # only create comparison if we have multiple teams with this label
                    teams_with_label = [t for t in teams if t in team_results and label_name in team_results[t]]
                    
                    if len(teams_with_label) >= 2:
                        # dynamically create subplots based on number of teams
                        n_teams = len(teams_with_label)
                        fig, axes = plt.subplots(1, n_teams, figsize=(8 * n_teams, 6))
                        
                        # handle case of only 2 teams (axes is already a list)
                        if n_teams == 2:
                            axes_list = axes
                        else:
                            axes_list = [axes] if n_teams == 1 else axes
                        
                        colors = {'management_team': 'darkgreen', 'external_members': 'purple'}
                        
                        for idx, team in enumerate(teams_with_label):
                            dist_df = team_results[team][label_name].head(10)
                            
                            ax = axes_list[idx]
                            ax.bar(range(len(dist_df)), dist_df['Percentage'],
                                         color=colors.get(team, 'steelblue'),
                                         edgecolor='black', alpha=0.7)
                            ax.set_xticks(range(len(dist_df)))
                            # truncate long labels (over 60 characters)
                            truncated_labels = [cat[:60] + '...' if len(str(cat)) > 60 else cat 
                                               for cat in dist_df['Category']]
                            ax.set_xticklabels(truncated_labels, rotation=45, ha='right', fontsize=9)
                            ax.set_ylabel('Weighted %', fontsize=11)
                            ax.set_title(f'{team.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                            ax.grid(axis='y', alpha=0.3)
                            
                            # add labels
                            for i, pct in enumerate(dist_df['Percentage']):
                                ax.text(i, pct + 0.5, f'{pct:.1f}%', 
                                              ha='center', va='bottom', fontsize=8)
                        
                        fig.suptitle(f'{label_name} Distribution: Team Comparison (Q&A Section - All Snippets)', 
                                    fontsize=14, fontweight='bold', y=1.02)
                        plt.tight_layout()
                        plt.savefig(self.viz_dir / f'comparison_team_qna_{label_name}_all_snippets.png', 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                    elif len(teams_with_label) == 1:
                        print(f"    ️ Only one team found for {label_name}, skipping comparison chart")
        
        all_results['by_team'] = team_results
        
        print(f"\n   Saved weighted distribution charts to {self.viz_dir}")
        print(f"    - Full transcript: {len(full_results)} label types")
        print(f"    - Section comparisons: {len(section_results)} sections")
        if team_results:
            print(f"    - Team comparisons (Q&A only): {len(team_results)} teams (management vs external)")
        else:
            print(f"    - Team comparisons (Q&A only): No teams found")
        
        return all_results
    
    def analyze_attribution_outcome_distributions(self, df_attribution: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze 4-way attribution outcomes and compare to literature benchmarks.
        
        NOTE: Uses FILTERED data (attribution_present='Y' only).
        
        Creates bar charts comparing Target vs Peer attribution rates.
        Literature benchmarks: Pos-Int 69-82%, Neg-Ext 5-14%
        """
        df = df_attribution  # alias for backward compatibility in method body
        print("\n3. Analyzing Attribution Outcome Distributions")
        print("=" * 80)
        
        # calculate overall distribution
        overall = df['attribution_type_4way'].value_counts()
        total = len(df)
        
        print(f"\n   Overall Attribution Distribution (n={total:,}):")
        for attr_type, count in overall.items():
            pct = (count / total * 100)
            print(f"    {attr_type:20s}: {count:6,} ({pct:5.1f}%)")
        
        # calculate normalized percentages (excluding 'other' for literature comparison)
        four_way_types = ['Positive-Internal', 'Positive-External', 'Negative-Internal', 'Negative-External']
        four_way_total = sum(overall.get(attr_type, 0) for attr_type in four_way_types)
        
        pos_int_count = overall.get('Positive-Internal', 0)
        neg_ext_count = overall.get('Negative-External', 0)
        pos_int_pct_normalized = (pos_int_count / four_way_total * 100) if four_way_total > 0 else 0
        neg_ext_pct_normalized = (neg_ext_count / four_way_total * 100) if four_way_total > 0 else 0
        
        print(f"\n   Comparison to Literature (using normalized 4-way percentages):")
        print(f"    NOTE: Normalized percentages exclude 'Other/Unknown' category to match literature")
        print(f"    Positive-Internal: {pos_int_pct_normalized:.1f}% (Literature: 69-82%)")
        if 69 <= pos_int_pct_normalized <= 82:
            print(f"       Within expected range")
        else:
            print(f"      ️  Outside expected range")
        
        print(f"    Negative-External: {neg_ext_pct_normalized:.1f}% (Literature: 5-14%)")
        if 5 <= neg_ext_pct_normalized <= 14:
            print(f"       Within expected range")
        else:
            print(f"      ️  Outside expected range")
        
        # create simple overall distribution bar chart (normalized 4-way)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        attribution_types_simple = ['Positive-Internal', 'Positive-External', 
                                    'Negative-Internal', 'Negative-External']
        simple_values = []
        for attr_type in attribution_types_simple:
            count = overall.get(attr_type, 0)
            pct = (count / four_way_total * 100) if four_way_total > 0 else 0
            simple_values.append(pct)
        
        colors_simple = ['#2ca02c', '#98df8a', '#d62728', '#ff9896']
        bars = ax.bar(range(len(attribution_types_simple)), simple_values, 
                     color=colors_simple, edgecolor='black', linewidth=1.5, alpha=0.85)
        
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Overall Attribution Type Distribution (All Companies)\nNormalized 4-Way Classification',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(range(len(attribution_types_simple)))
        ax.set_xticklabels([at.replace('-', '\n') for at in attribution_types_simple], 
                          fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # add percentage labels on bars
        for bar, val in zip(bars, simple_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # add sample sizes below
        for i, (bar, attr_type) in enumerate(zip(bars, attribution_types_simple)):
            count = overall.get(attr_type, 0)
            ax.text(bar.get_x() + bar.get_width()/2., -2,
                   f'n={count:,}', ha='center', va='top', fontsize=9, style='italic')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'attribution_overall_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n   Saved: attribution_overall_distribution.png")
        
        # target vs peer comparison (using normalized percentages)
        target_df = df[df['IS_TARGET'] == 'Y']
        peer_df = df[df['IS_TARGET'] == 'N']
        
        # calculate normalized percentages for targets and peers
        target_four_way_total = sum((target_df['attribution_type_4way'] == attr_type).sum() 
                                     for attr_type in four_way_types)
        peer_four_way_total = sum((peer_df['attribution_type_4way'] == attr_type).sum() 
                                   for attr_type in four_way_types)
        
        comparison_data = []
        
        for attr_type in ['Positive-Internal', 'Positive-External', 'Negative-Internal', 'Negative-External']:
            target_count = (target_df['attribution_type_4way'] == attr_type).sum()
            target_pct_normalized = (target_count / target_four_way_total * 100) if target_four_way_total > 0 else 0
            
            peer_count = (peer_df['attribution_type_4way'] == attr_type).sum()
            peer_pct_normalized = (peer_count / peer_four_way_total * 100) if peer_four_way_total > 0 else 0
            
            comparison_data.append({
                'Attribution_Type': attr_type,
                'Target_Count': target_count,
                'Target_Pct_Normalized': target_pct_normalized,
                'Peer_Count': peer_count,
                'Peer_Pct_Normalized': peer_pct_normalized,
                'Difference_Pct': target_pct_normalized - peer_pct_normalized
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print(f"\n   Target vs Peer Comparison (using normalized 4-way percentages):")
        print(f"    Targets: n={len(target_df):,}, Peers: n={len(peer_df):,}")
        for _, row in comparison_df.iterrows():
            diff = row['Difference_Pct']
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"    {row['Attribution_Type']:20s}: Target {row['Target_Pct_Normalized']:5.1f}% vs Peer {row['Peer_Pct_Normalized']:5.1f}% ({symbol} {abs(diff):4.1f}%)")
        
        # create grouped bar chart (using normalized percentages)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, comparison_df['Target_Pct_Normalized'], width, 
                       label='Target Firms', color='#d62728', edgecolor='black')
        bars2 = ax.bar(x + width/2, comparison_df['Peer_Pct_Normalized'], width,
                       label='Peer Firms', color='#1f77b4', edgecolor='black')
        
        ax.set_ylabel('Normalized Percentage (%)', fontsize=12)
        ax.set_title('Attribution Type Distribution: Target vs Peer Firms\n(Normalized - Excluding Other/Unknown)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Attribution_Type'], rotation=15, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # add percentage labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'attribution_target_vs_peer.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # create enhanced sab comparison charts
        # split by bias period to show more nuanced patterns
        
        # filter by bias periods (using exact bias period flag)
        target_bias = target_df[target_df['is_bias_period_exact'] == True] if 'is_bias_period_exact' in target_df.columns else pd.DataFrame()
        target_normal = target_df[target_df['is_bias_period_exact'] == False] if 'is_bias_period_exact' in target_df.columns else target_df
        peer_bias = peer_df[peer_df['is_bias_period_exact'] == True] if 'is_bias_period_exact' in peer_df.columns else pd.DataFrame()
        peer_normal = peer_df[peer_df['is_bias_period_exact'] == False] if 'is_bias_period_exact' in peer_df.columns else peer_df
        
        # calculate sab proportions for each group
        target_bias_sab = (target_bias['is_sab'].sum() / len(target_bias) * 100) if len(target_bias) > 0 else 0
        target_normal_sab = (target_normal['is_sab'].sum() / len(target_normal) * 100) if len(target_normal) > 0 else 0
        peer_bias_sab = (peer_bias['is_sab'].sum() / len(peer_bias) * 100) if len(peer_bias) > 0 else 0
        peer_normal_sab = (peer_normal['is_sab'].sum() / len(peer_normal) * 100) if len(peer_normal) > 0 else 0
        
        # chart 1: sab proportion by period
        fig, ax = plt.subplots(figsize=(10, 7))
        x_pos = np.arange(4)
        labels = ['Target\n(Bias Period)', 'Target\n(Normal)', 'Peer\n(Bias Period)', 'Peer\n(Normal)']
        values = [target_bias_sab, target_normal_sab, peer_bias_sab, peer_normal_sab]
        colors = ['#d62728', '#ff9896', '#1f77b4', '#aec7e8']
        
        bars = ax.bar(x_pos, values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('SAB Proportion (%)', fontsize=12)
        ax.set_title('SAB Proportion: Target vs Peer, Bias Period vs Normal\n(Pos-Int + Neg-Ext) / All Attributions',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # add percentage labels and sample sizes (with extra space calculated)
        max_val = max(values)
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            n_size = [len(target_bias), len(target_normal), len(peer_bias), len(peer_normal)][i]
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}%\n(n={n_size:,})', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # set y-axis limit to provide space for labels
        ax.set_ylim(0, max_val * 1.25)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'sab_proportion_by_period.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # chart 2: histogram of sab proportion distribution (snippet-level)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # calculate snippet-level sab proportions per company-quarter
        target_sab_by_quarter = target_df.groupby(['Company', 'Year', 'Quarter']).agg({
            'is_sab': lambda x: x.sum() / len(x) * 100
        }).reset_index()
        
        peer_sab_by_quarter = peer_df.groupby(['Company', 'Year', 'Quarter']).agg({
            'is_sab': lambda x: x.sum() / len(x) * 100
        }).reset_index()
        
        # plot histograms
        ax1.hist(target_sab_by_quarter['is_sab'], bins=20, color='#d62728', alpha=0.7, edgecolor='black')
        ax1.axvline(target_sab_by_quarter['is_sab'].mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {target_sab_by_quarter["is_sab"].mean():.1f}%')
        ax1.set_xlabel('SAB Proportion (%)', fontsize=11)
        ax1.set_ylabel('Frequency (Company-Quarters)', fontsize=11)
        ax1.set_title('Target Firms: SAB Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        ax2.hist(peer_sab_by_quarter['is_sab'], bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
        ax2.axvline(peer_sab_by_quarter['is_sab'].mean(), color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {peer_sab_by_quarter["is_sab"].mean():.1f}%')
        ax2.set_xlabel('SAB Proportion (%)', fontsize=11)
        ax2.set_ylabel('Frequency (Company-Quarters)', fontsize=11)
        ax2.set_title('Peer Firms: SAB Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        fig.suptitle('Distribution of SAB Proportions Across Company-Quarters', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'sab_proportion_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # chart 3: histograms for other key metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribution of Attribution Rate Components: Target vs Peer', fontsize=14, fontweight='bold')
        
        metrics_to_plot = [
            ('attribution_outcome', 'Positive', 'attribution_locus', 'Internal', 'Positive-Internal Rate'),
            ('attribution_outcome', 'Positive', 'attribution_locus', 'External', 'Positive-External Rate'),
            ('attribution_outcome', 'Negative', 'attribution_locus', 'Internal', 'Negative-Internal Rate'),
            ('attribution_outcome', 'Negative', 'attribution_locus', 'External', 'Negative-External Rate')
        ]
        
        for idx, (outcome_col, outcome_val, locus_col, locus_val, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            # calculate rates per company-quarter
            target_rates = target_df.groupby(['Company', 'Year', 'Quarter']).apply(
                lambda g: ((g[outcome_col] == outcome_val) & (g[locus_col] == locus_val)).sum() / len(g) * 100
            ).reset_index(name='rate')
            
            peer_rates = peer_df.groupby(['Company', 'Year', 'Quarter']).apply(
                lambda g: ((g[outcome_col] == outcome_val) & (g[locus_col] == locus_val)).sum() / len(g) * 100
            ).reset_index(name='rate')
            
            # plot overlapping histograms
            ax.hist(target_rates['rate'], bins=15, color='#d62728', alpha=0.5, edgecolor='black', label='target')
            ax.hist(peer_rates['rate'], bins=15, color='#1f77b4', alpha=0.5, edgecolor='black', label='peer')
            
            # add mean lines
            ax.axvline(target_rates['rate'].mean(), color='darkred', linestyle='--', linewidth=2)
            ax.axvline(peer_rates['rate'].mean(), color='darkblue', linestyle='--', linewidth=2)
            
            ax.set_xlabel('Rate (%)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'attribution_rates_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n   Saved enhanced comparison charts to {self.viz_dir}")
        print(f"    - sab_proportion_by_period.png (4-bar comparison)")
        print(f"    - sab_proportion_histogram.png (distribution analysis)")
        print(f"    - attribution_rates_histogram.png (component distributions)")
        
        return comparison_df
    
    
    def calculate_bias_metrics_multilevel(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        print("\n3. Calculating Multi-Level Bias Metrics")
        print("=" * 80)
        
        metrics = {}
        
        levels = {
            'full': df.copy(),
            'management': df[df['Team'] == 'management_team'].copy() if 'Team' in df.columns else df.copy(),
            'prepared': df[df['Section'] == 'prepared_remarks'].copy() if 'Section' in df.columns else pd.DataFrame(),
            'qna': df[df['Section'] == 'qna'].copy() if 'Section' in df.columns else pd.DataFrame()
        }
        
        for level_name, level_df in levels.items():
            if len(level_df) == 0:
                continue
            
            print(f"\nCalculating {level_name} level metrics...")
            
            level_metrics = []
            
            for (company, quarter, year), group in level_df.groupby(['Company', 'Quarter', 'Year']):
                total = len(group)
                
                if total == 0:
                    continue
                
                pos_internal = len(group[
                    (group['attribution_outcome'] == 'Positive') &
                    (group['attribution_locus'] == 'Internal')
                ])
                
                neg_external = len(group[
                    (group['attribution_outcome'] == 'Negative') &
                    (group['attribution_locus'] == 'External')
                ])
                
                pos_external = len(group[
                    (group['attribution_outcome'] == 'Positive') &
                    (group['attribution_locus'] == 'External')
                ])
                
                neg_internal = len(group[
                    (group['attribution_outcome'] == 'Negative') &
                    (group['attribution_locus'] == 'Internal')
                ])
                
                positive_total = len(group[group['attribution_outcome'] == 'Positive'])
                negative_total = len(group[group['attribution_outcome'] == 'Negative'])
                
                # calculate rates (conditionals among positive or negative)
                pos_internal_rate = pos_internal / positive_total if positive_total > 0 else 0
                neg_external_rate = neg_external / negative_total if negative_total > 0 else 0
                pos_external_rate = pos_external / positive_total if positive_total > 0 else 0
                neg_internal_rate = neg_internal / negative_total if negative_total > 0 else 0
                
                # asymmetry score (range: -2 to +2)
                asymmetry_score = (pos_internal_rate + neg_external_rate) - (pos_external_rate + neg_internal_rate)
                
                # sab proportion: (pos_int + neg_ext) / all attributions
                sab_count = pos_internal + neg_external
                sab_proportion = sab_count / total if total > 0 else 0
                
                # neg-ext rate of all negative (more granular)
                neg_ext_of_negative = neg_external / negative_total if negative_total > 0 else 0
                
                # pos-int rate of all positive (more granular)  
                pos_int_of_positive = pos_internal / positive_total if positive_total > 0 else 0
                
                is_target = group['IS_TARGET'].iloc[0] if 'IS_TARGET' in group.columns else 'N'
                is_peer = group['IS_PEER'].iloc[0] if 'IS_PEER' in group.columns else 'N'
                related_firms = group['RELATED_FIRMS'].iloc[0] if 'RELATED_FIRMS' in group.columns else ''
                
                # get bias period flags (all windows)
                is_bias_exact = group['is_bias_period_exact'].iloc[0] if 'is_bias_period_exact' in group.columns else False
                is_bias_w1 = group['is_bias_period_window1'].iloc[0] if 'is_bias_period_window1' in group.columns else False
                is_bias_w2 = group['is_bias_period_window2'].iloc[0] if 'is_bias_period_window2' in group.columns else False
                is_bias_w4 = group['is_bias_period_window4'].iloc[0] if 'is_bias_period_window4' in group.columns else False
                is_bias_pre4 = group['is_bias_period_pre4'].iloc[0] if 'is_bias_period_pre4' in group.columns else False
                is_bias_post4 = group['is_bias_period_post4'].iloc[0] if 'is_bias_period_post4' in group.columns else False
                bias_type = group['bias_type'].iloc[0] if 'bias_type' in group.columns else None
                bias_label = group['bias_period_label'].iloc[0] if 'bias_period_label' in group.columns else 'Normal'
                
                # sample size flag
                sample_flag = 'Small Sample' if total < 10 else 'OK'
                
                level_metrics.append({
                    'company': company,
                    'quarter': quarter,
                    'year': year,
                    'level': level_name,
                    'total_attributions': total,
                    'pos_internal': pos_internal,
                    'neg_external': neg_external,
                    'pos_external': pos_external,
                    'neg_internal': neg_internal,
                    'positive_total': positive_total,
                    'negative_total': negative_total,
                    'pos_internal_rate': pos_internal_rate,
                    'neg_external_rate': neg_external_rate,
                    'pos_external_rate': pos_external_rate,
                    'neg_internal_rate': neg_internal_rate,
                    'asymmetry_score': asymmetry_score,
                    'sab_proportion': sab_proportion,
                    'sab_count': sab_count,
                    'neg_ext_of_negative': neg_ext_of_negative,
                    'pos_int_of_positive': pos_int_of_positive,
                    'is_target': is_target,
                    'is_peer': is_peer,
                    'related_firms': related_firms,
                    'is_bias_period_exact': is_bias_exact,
                    'is_bias_period_window1': is_bias_w1,
                    'is_bias_period_window2': is_bias_w2,
                    'is_bias_period_window4': is_bias_w4,
                    'is_bias_period_pre4': is_bias_pre4,
                    'is_bias_period_post4': is_bias_post4,
                    'bias_type': bias_type,
                    'bias_period_label': bias_label,
                    'sample_size_flag': sample_flag
                })
            
            metrics[level_name] = pd.DataFrame(level_metrics)
            print(f"  {level_name}: {len(metrics[level_name])} company-quarters")
        
        # print aggregate statistics
        if 'full' in metrics and len(metrics['full']) > 0:
            self._print_aggregate_statistics(metrics['full'])
        
        return metrics
    
    def _print_aggregate_statistics(self, full_metrics: pd.DataFrame):
        print(f"\nAggregate Attribution Statistics:")
        print("=" * 80)
        
        # overall breakdown
        total_obs = len(full_metrics)
        targets = full_metrics[full_metrics['is_target'] == 'Y']
        peers = full_metrics[full_metrics['is_target'] == 'N']
        
        print(f"\n OVERALL BREAKDOWN (All Companies, All Quarters)")
        print(f"  Total observations: {total_obs:,} company-quarters")
        print(f"  Targets: {len(targets):,} ({len(targets)/total_obs*100:.1f}%)")
        print(f"  Peers: {len(peers):,} ({len(peers)/total_obs*100:.1f}%)")
        
        # positive internal attribution
        print(f"\n POSITIVE INTERNAL ATTRIBUTION (taking credit)")
        if len(targets) > 0:
            target_pos_int = targets['pos_internal'].sum()
            target_total = targets['total_attributions'].sum()
            print(f"  Targets: {target_pos_int:,} occurrences ({target_pos_int/target_total*100:.1f}% of target attributions)")
            print(f"           Mean rate per quarter: {targets['pos_internal_rate'].mean():.1%}")
        if len(peers) > 0:
            peer_pos_int = peers['pos_internal'].sum()
            peer_total = peers['total_attributions'].sum()
            print(f"  Peers:   {peer_pos_int:,} occurrences ({peer_pos_int/peer_total*100:.1f}% of peer attributions)")
            print(f"           Mean rate per quarter: {peers['pos_internal_rate'].mean():.1%}")
        
        # negative external attribution
        print(f"\n NEGATIVE EXTERNAL ATTRIBUTION (blaming outside factors)")
        if len(targets) > 0:
            target_neg_ext = targets['neg_external'].sum()
            print(f"  Targets: {target_neg_ext:,} occurrences ({target_neg_ext/target_total*100:.1f}% of target attributions)")
            print(f"           Mean rate per quarter: {targets['neg_external_rate'].mean():.1%}")
        if len(peers) > 0:
            peer_neg_ext = peers['neg_external'].sum()
            print(f"  Peers:   {peer_neg_ext:,} occurrences ({peer_neg_ext/peer_total*100:.1f}% of peer attributions)")
            print(f"           Mean rate per quarter: {peers['neg_external_rate'].mean():.1%}")
        
        # primary metric: sab proportion
        print(f"\n PRIMARY METRIC: SAB PROPORTION")
        print(f"  Calculation: (Pos-Int + Neg-Ext) / All Attributions")
        print(f"  Range: 0% to 100%")
        print(f"  Interpretation: Higher % = more self-serving bias")
        print(f"               This is the KEY METRIC for bias detection (absolute proportion)")
        
        if len(targets) > 0 and len(peers) > 0:
            target_sab_mean = targets['sab_proportion'].mean()
            peer_sab_mean = peers['sab_proportion'].mean()
            sab_diff = target_sab_mean - peer_sab_mean
            
            if target_sab_mean > peer_sab_mean:
                sab_comparison = "Targets show HIGHER SAB proportion than peers"
                sab_symbol = "↑"
            elif target_sab_mean < peer_sab_mean:
                sab_comparison = "Targets show LOWER SAB proportion than peers"
                sab_symbol = "↓"
            else:
                sab_comparison = "Targets and peers show similar SAB proportions"
                sab_symbol = "="
            
            print(f"\n   Comparison: {sab_comparison} ({sab_symbol} {abs(sab_diff):.3f})")
        
        if len(targets) > 0:
            target_sab_mean = targets['sab_proportion'].mean()
            target_sab_median = targets['sab_proportion'].median()
            print(f"\n  Targets: Mean {target_sab_mean:.3f}, Median {target_sab_median:.3f}")
            print(f"           Range: {targets['sab_proportion'].min():.3f} to {targets['sab_proportion'].max():.3f}")
        
        if len(peers) > 0:
            peer_sab_mean = peers['sab_proportion'].mean()
            peer_sab_median = peers['sab_proportion'].median()
            print(f"  Peers:   Mean {peer_sab_mean:.3f}, Median {peer_sab_median:.3f}")
            print(f"           Range: {peers['sab_proportion'].min():.3f} to {peers['sab_proportion'].max():.3f}")
        
        # secondary metric: attribution rate asymmetry
        print(f"\n SECONDARY METRIC: ATTRIBUTION RATE ASYMMETRY")
        print(f"  Calculation: (Pos-Int rate + Neg-Ext rate) - (Pos-Ext rate + Neg-Int rate)")
        print(f"  Range: -2.0 to +2.0")
        print(f"  Interpretation: +ve = self-serving bias, -ve = reverse bias, 0 = balanced")
        print(f"               NOTE: This measures the imbalance/asymmetry between rate pairs")
        print(f"               SAB Proportion remains the primary metric for hypothesis testing")
        
        if len(targets) > 0 and len(peers) > 0:
            target_asym_mean = targets['asymmetry_score'].mean()
            peer_asym_mean = peers['asymmetry_score'].mean()
            asym_diff = target_asym_mean - peer_asym_mean
            
            print(f"\n   Comparison: Rate asymmetry difference = {asym_diff:+.3f}")
        
        if len(targets) > 0:
            target_asym_mean = targets['asymmetry_score'].mean()
            target_asym_median = targets['asymmetry_score'].median()
            print(f"\n  Targets: Mean {target_asym_mean:.3f}, Median {target_asym_median:.3f}")
            print(f"           Range: {targets['asymmetry_score'].min():.3f} to {targets['asymmetry_score'].max():.3f}")
        
        if len(peers) > 0:
            peer_asym_mean = peers['asymmetry_score'].mean()
            peer_asym_median = peers['asymmetry_score'].median()
            print(f"  Peers:   Mean {peer_asym_mean:.3f}, Median {peer_asym_median:.3f}")
            print(f"           Range: {peers['asymmetry_score'].min():.3f} to {peers['asymmetry_score'].max():.3f}")
        
        # high asymmetry periods
        high_asymmetry = full_metrics[full_metrics['asymmetry_score'] > full_metrics['asymmetry_score'].quantile(0.75)]
        high_bias_targets = high_asymmetry[high_asymmetry['is_target'] == 'Y']
        
        if len(high_bias_targets) > 0:
            print(f"\n HIGH BIAS PERIODS (top 25% asymmetry)")
            print(f"  {len(high_bias_targets)} target company-quarters show elevated bias")
            
            # show which targets appear most
            top_biased = high_bias_targets.groupby('company').size().sort_values(ascending=False).head(5)
            print(f"  Most frequent targets:")
            for company, count in top_biased.items():
                total_quarters = len(full_metrics[full_metrics['company'] == company])
                print(f"    {company}: {count}/{total_quarters} quarters ({count/total_quarters*100:.0f}%)")
    
    # =========================================================================
    # statistical helper functions
    # =========================================================================
    
    def _calculate_z_score(self, target_val: float, peer_vals: List[float]) -> float:
        """Calculate z-score: (target - peer_mean) / peer_std"""
        if len(peer_vals) == 0:
            return 0.0
        peer_mean = np.mean(peer_vals)
        peer_std = np.std(peer_vals, ddof=1) if len(peer_vals) > 1 else 0
        if peer_std == 0:
            return 0.0
        return (target_val - peer_mean) / peer_std
    
    def _calculate_t_test(self, group1: List[float], group2: List[float]) -> Tuple[float, float]:
        """
        Independent samples t-test."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0, 1.0
        try:
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)  # welch's t-test
            return float(t_stat), float(p_val)
        except:
            return 0.0, 1.0
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Cohen's d effect size: (mean1 - mean2) / pooled_std
        Interpretation: 0.2=small, 0.5=medium, 0.8=large
        """
        if len(group1) == 0 or len(group2) == 0:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _get_sample_size_flag(self, n: int) -> str:
        """Flag sample sizes for interpretation"""
        if n < 5:
            return '️ Very Small (n<5)'
        elif n < 10:
            return '️ Small (n<10)'
        elif n < 30:
            return '~ Moderate (n<30)'
        else:
            return ' Adequate (n≥30)'
    
    def _interpret_z_score(self, z: float) -> str:
        """Interpret z-score magnitude"""
        abs_z = abs(z)
        if abs_z >= 2.0:
            return 'Strong Signal'
        elif abs_z >= 1.5:
            return 'Moderate Signal'
        elif abs_z >= 1.0:
            return 'Weak Signal'
        else:
            return 'No Signal'
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d >= 0.8:
            return 'Large Effect'
        elif abs_d >= 0.5:
            return 'Medium Effect'
        elif abs_d >= 0.2:
            return 'Small Effect'
        else:
            return 'Negligible Effect'
    
    def calculate_rolling_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        print("\n4. Calculating Rolling Window Metrics")
        print("=" * 80)
        
        metrics_df = metrics_df.sort_values(['company', 'year', 'quarter'])
        
        rolling_results = []
        
        for company in metrics_df['company'].unique():
            company_data = metrics_df[metrics_df['company'] == company].reset_index(drop=True)
            
            for i in range(len(company_data)):
                row = company_data.iloc[i]
                
                window_2q = company_data.iloc[max(0, i-1):i+1]
                window_4q = company_data.iloc[max(0, i-3):i+1]
                
                rolling_results.append({
                    'company': company,
                    'quarter': row['quarter'],
                    'year': row['year'],
                    'level': row['level'],
                    'asymmetry_current': row['asymmetry_score'],
                    'asymmetry_2q_mean': window_2q['asymmetry_score'].mean(),
                    'asymmetry_2q_std': window_2q['asymmetry_score'].std(),
                    'asymmetry_4q_mean': window_4q['asymmetry_score'].mean(),
                    'asymmetry_4q_std': window_4q['asymmetry_score'].std(),
                    'is_target': row['is_target'],
                    'is_peer': row['is_peer']
                })
        
        rolling_df = pd.DataFrame(rolling_results)
        
        print(f"Calculated rolling metrics for {len(rolling_df)} observations")
        
        return rolling_df
    
    def create_rolling_timeseries_visualizations(self, rolling_df: pd.DataFrame, metrics_df: pd.DataFrame):
        """
        Create comprehensive rolling time series visualizations for each target company.
        
        Shows how multiple bias metrics evolve over time with target vs peer comparisons:
        - Asymmetry Score
        - SAB Proportion
        - Attribution rates (pos-int, pos-ext, neg-int, neg-ext)
        - Z-scores vs peers
        
        Highlights expert-identified bias periods as shaded regions.
        """
        print("\n Creating Rolling Time Series Visualizations")
        print("=" * 80)
        
        # get target companies
        targets = rolling_df[rolling_df['is_target'] == 'Y']['company'].unique()
        
        if len(targets) == 0:
            print("   No target companies found in rolling metrics")
            return
        
        print(f"  Creating enhanced time series plots for {len(targets)} target companies")
        
        viz_created = 0
        
        for target in targets:
            # get target metrics
            target_metrics = metrics_df[
                (metrics_df['company'] == target) & 
                (metrics_df['is_target'] == 'Y')
            ].sort_values(['year', 'quarter']).copy()
            
            if len(target_metrics) < 2:
                continue
            
            # get peer group for this target
            target_config = None
            for target_key, info in self.peer_groups.items():
                target_folder = info.get('target_folder', target_key)
                if target_folder == target:
                    target_config = info
                    break
            
            if not target_config:
                continue
            
            peer_folders = target_config.get('direct_competitors_folders', [])
            peer_folders = [p for p in peer_folders if p is not None]
            
            if not peer_folders:
                continue
            
            peer_metrics = metrics_df[metrics_df['company'].isin(peer_folders)].copy()
            
            # calculate peer means for each quarter
            peer_quarterly = peer_metrics.groupby(['year', 'quarter']).agg({
                'asymmetry_score': ['mean', 'std'],
                'sab_proportion': ['mean', 'std'],
                'pos_int_of_positive': ['mean', 'std'],
                'neg_ext_of_negative': ['mean', 'std'],
                'pos_internal_rate': ['mean', 'std'],
                'neg_external_rate': ['mean', 'std']
            }).reset_index()
            
            # flatten column names
            peer_quarterly.columns = ['_'.join(col).strip('_') for col in peer_quarterly.columns]
            
            # merge target with peer data
            target_metrics = target_metrics.merge(
                peer_quarterly,
                on=['year', 'quarter'],
                how='left',
                suffixes=('', '_peer')
            )
            
            # calculate z-scores
            for metric in ['asymmetry_score', 'sab_proportion', 'pos_int_of_positive', 'neg_ext_of_negative']:
                peer_mean_col = f'{metric}_mean'
                peer_std_col = f'{metric}_std'
                if peer_mean_col in target_metrics.columns and peer_std_col in target_metrics.columns:
                    target_metrics[f'{metric}_zscore'] = (
                        (target_metrics[metric] - target_metrics[peer_mean_col]) / 
                        target_metrics[peer_std_col].replace(0, np.nan)
                    ).fillna(0)
            
            # create comprehensive figure with 6 subplots (3x2 grid)
            fig = plt.figure(figsize=(18, 14))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
            fig.suptitle(f'{target} - Comprehensive Bias Evolution Over Time', fontsize=16, fontweight='bold')
            
            # create time labels and positions
            time_labels = [f"{int(row['year'])}Q{int(row['quarter'])}" 
                          for _, row in target_metrics.iterrows()]
            x_pos = range(len(time_labels))
            
            # helper function to shade bias periods
            def shade_bias_periods(ax):
                # find the ticker that corresponds to this target_folder
                bias_info = None
                ticker_found = None
                for ticker, info in self.expert_bias_periods.items():
                    if info['target_folder'] == target:
                        bias_info = info
                        ticker_found = ticker
                        break
                
                if bias_info is not None:
                    print(f"      → Found bias period for {target} (ticker: {ticker_found})")
                    quarters_exact = bias_info['quarters_exact']
                    
                    if len(quarters_exact) > 0:
                        # get start and end quarters of the bias period
                        bias_start_year, bias_start_q = quarters_exact[0]
                        bias_end_year, bias_end_q = quarters_exact[-1]
                        
                        # find the x-axis positions that bracket the bias period
                        # we want to shade even if the exact bias quarters don't exist in the data
                        start_pos = None
                        end_pos = None
                        
                        # find the position of the first quarter >= bias start
                        # use enumerate to get correct x-axis positions
                        for row_pos, (idx, row) in enumerate(target_metrics.iterrows()):
                            row_year, row_q = int(row['year']), int(row['quarter'])
                            
                            # check if this is the start of or within the bias period
                            if (row_year, row_q) >= (bias_start_year, bias_start_q):
                                if start_pos is None:
                                    start_pos = row_pos
                            
                            # check if this is within or at the end of the bias period
                            if (row_year, row_q) <= (bias_end_year, bias_end_q):
                                end_pos = row_pos
                        
                        # shade the region if we found overlapping data
                        if start_pos is not None and end_pos is not None:
                            # shade the full range from start to end
                            ax.axvspan(start_pos - 0.5, end_pos + 0.5, color='red', alpha=0.15, label='Expert Bias Period')
                            print(f"         Shading bias period from position {start_pos} to {end_pos} ({bias_start_year}Q{bias_start_q} to {bias_end_year}Q{bias_end_q})")
                        elif start_pos is not None:
                            # only found start (bias period extends beyond data)
                            ax.axvspan(start_pos - 0.5, len(target_metrics) - 0.5, color='red', alpha=0.15, label='Expert Bias Period (extends beyond data)')
                            print(f"         Shading bias period from position {start_pos} to end (extends beyond data)")
                        elif end_pos is not None:
                            # only found end (bias period started before data)
                            ax.axvspan(-0.5, end_pos + 0.5, color='red', alpha=0.15, label='Expert Bias Period (started before data)')
                            print(f"         Shading bias period from start to position {end_pos} (started before data)")
                        else:
                            print(f"         ️  No overlap found between bias period and data quarters")
                else:
                    print(f"      ℹ️  No bias period defined for {target}")
            
            # plot 1: sab proportion (primary metric)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(x_pos, target_metrics['sab_proportion'], 'o-', linewidth=2.5, 
                    markersize=6, label='Target', color='#d62728')
            ax1.plot(x_pos, target_metrics['sab_proportion_mean'], 's--', linewidth=2, 
                    markersize=5, label='Peer Mean', color='#1f77b4', alpha=0.8)
            shade_bias_periods(ax1)
            ax1.set_title('PRIMARY: SAB Proportion = (Pos-Int + Neg-Ext) / All Attributions', fontsize=10, fontweight='bold')
            ax1.set_ylabel('Proportion', fontsize=9)
            ax1.legend(loc='best', fontsize=8, framealpha=0.9)
            ax1.grid(True, alpha=0.2)
            ax1.set_xticks(x_pos[::max(1, len(x_pos)//10)])
            ax1.set_xticklabels(time_labels[::max(1, len(x_pos)//10)], rotation=45, ha='right', fontsize=8)
            
            # plot 2: attribution rate asymmetry (secondary metric)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(x_pos, target_metrics['asymmetry_score'], 'o-', linewidth=2.5, 
                    markersize=6, label='Target', color='#d62728')
            ax2.plot(x_pos, target_metrics['asymmetry_score_mean'], 's--', linewidth=2, 
                    markersize=5, label='Peer Mean', color='#1f77b4', alpha=0.8)
            shade_bias_periods(ax2)
            ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
            ax2.set_title('SECONDARY: Attribution Rate Asymmetry\n(Pos-Int + Neg-Ext) - (Pos-Ext + Neg-Int) rates', fontsize=10, fontweight='bold')
            ax2.set_ylabel('Rate Asymmetry', fontsize=9)
            ax2.legend(loc='best', fontsize=8, framealpha=0.9)
            ax2.grid(True, alpha=0.2)
            ax2.set_xticks(x_pos[::max(1, len(x_pos)//10)])
            ax2.set_xticklabels(time_labels[::max(1, len(x_pos)//10)], rotation=45, ha='right', fontsize=8)
            
            # plot 3: positive internal rate (of all positive attributions)
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(x_pos, target_metrics['pos_int_of_positive'], 'o-', linewidth=2.5, 
                    markersize=6, label='Target', color='#d62728')
            ax3.plot(x_pos, target_metrics['pos_int_of_positive_mean'], 's--', linewidth=2, 
                    markersize=5, label='Peer Mean', color='#1f77b4', alpha=0.8)
            shade_bias_periods(ax3)
            ax3.set_title('Positive-Internal Rate: Pos-Int / All Positive', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Rate', fontsize=9)
            ax3.legend(loc='best', fontsize=8, framealpha=0.9)
            ax3.grid(True, alpha=0.2)
            ax3.set_xticks(x_pos[::max(1, len(x_pos)//10)])
            ax3.set_xticklabels(time_labels[::max(1, len(x_pos)//10)], rotation=45, ha='right', fontsize=8)
            
            # plot 4: negative external rate (of all negative attributions)
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(x_pos, target_metrics['neg_ext_of_negative'], 'o-', linewidth=2.5, 
                    markersize=6, label='Target', color='#d62728')
            ax4.plot(x_pos, target_metrics['neg_ext_of_negative_mean'], 's--', linewidth=2, 
                    markersize=5, label='Peer Mean', color='#1f77b4', alpha=0.8)
            shade_bias_periods(ax4)
            ax4.set_title('Negative-External Rate: Neg-Ext / All Negative', fontsize=10, fontweight='bold')
            ax4.set_ylabel('Rate', fontsize=9)
            ax4.legend(loc='best', fontsize=8, framealpha=0.9)
            ax4.grid(True, alpha=0.2)
            ax4.set_xticks(x_pos[::max(1, len(x_pos)//10)])
            ax4.set_xticklabels(time_labels[::max(1, len(x_pos)//10)], rotation=45, ha='right', fontsize=8)
            
            # plot 5: z-scores (standardized metrics)
            ax5 = fig.add_subplot(gs[2, 0])
            if 'sab_proportion_zscore' in target_metrics.columns:
                ax5.plot(x_pos, target_metrics['sab_proportion_zscore'], 'o-', linewidth=2.5, 
                        markersize=6, label='SAB Proportion Z-Score (PRIMARY)', color='#d62728')
            if 'asymmetry_score_zscore' in target_metrics.columns:
                ax5.plot(x_pos, target_metrics['asymmetry_score_zscore'], 's--', linewidth=1.5, 
                        markersize=4, label='Rate Asymmetry Z-Score (SECONDARY)', color='#ff7f0e', alpha=0.6)
            shade_bias_periods(ax5)
            ax5.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
            ax5.axhline(y=1.0, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='±1σ')
            ax5.axhline(y=-1.0, color='orange', linestyle=':', linewidth=1, alpha=0.5)
            ax5.axhline(y=2.0, color='red', linestyle=':', linewidth=1, alpha=0.5, label='±2σ')
            ax5.axhline(y=-2.0, color='red', linestyle=':', linewidth=1, alpha=0.5)
            ax5.set_title('Standardized Metrics: Z-Scores vs Peers', fontsize=10, fontweight='bold')
            ax5.set_ylabel('Z-Score (Std Deviations)', fontsize=9)
            ax5.set_xlabel('Quarter', fontsize=9)
            ax5.legend(loc='best', fontsize=8, framealpha=0.9)
            ax5.grid(True, alpha=0.2)
            ax5.set_xticks(x_pos[::max(1, len(x_pos)//10)])
            ax5.set_xticklabels(time_labels[::max(1, len(x_pos)//10)], rotation=45, ha='right', fontsize=8)
            
            # plot 6: attribution rate components
            ax6 = fig.add_subplot(gs[2, 1])
            # calculate pos-ext and neg-int rates from available data
            target_metrics['pos_ext_rate'] = target_metrics['pos_external_rate']
            target_metrics['neg_int_rate'] = target_metrics['neg_internal_rate']
            
            ax6.plot(x_pos, target_metrics['pos_internal_rate'], 'o-', linewidth=2, 
                    markersize=4, label='Pos-Int Rate (Target)', color='#2ca02c')
            ax6.plot(x_pos, target_metrics['pos_ext_rate'], 's--', linewidth=2, 
                    markersize=4, label='Pos-Ext Rate (Target)', color='#98df8a', alpha=0.7)
            ax6.plot(x_pos, target_metrics['neg_external_rate'], '^-', linewidth=2, 
                    markersize=4, label='Neg-Ext Rate (Target)', color='#d62728')
            ax6.plot(x_pos, target_metrics['neg_int_rate'], 'v--', linewidth=2, 
                    markersize=4, label='Neg-Int Rate (Target)', color='#ff9896', alpha=0.7)
            shade_bias_periods(ax6)
            ax6.set_title('Attribution Rate Components (Target)', fontsize=10, fontweight='bold')
            ax6.set_ylabel('Rate', fontsize=9)
            ax6.set_xlabel('Quarter', fontsize=9)
            ax6.legend(loc='best', fontsize=7, framealpha=0.9, ncol=2)
            ax6.grid(True, alpha=0.2)
            ax6.set_xticks(x_pos[::max(1, len(x_pos)//10)])
            ax6.set_xticklabels(time_labels[::max(1, len(x_pos)//10)], rotation=45, ha='right', fontsize=8)
            
            plt.tight_layout()
            
            # save figure
            safe_name = target.replace('/', '_').replace('\\', '_').replace(':', '_')
            filename = self.viz_dir / f"timeseries_comprehensive_{safe_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_created += 1
        
        print(f"   Created {viz_created} comprehensive time series visualizations")
        print(f"   Saved to: {self.viz_dir}")
    

    
    def analyze_bias_metadata_distribution(self, df: pd.DataFrame, metrics: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        print("\n6. Analyzing Metadata Distribution in Bias Periods")
        print("=" * 80)
        
        full_metrics = metrics.get('full', pd.DataFrame())
        if len(full_metrics) == 0:
            return pd.DataFrame()
        
        high_bias_quarters = full_metrics[full_metrics['asymmetry_score'] > full_metrics['asymmetry_score'].quantile(0.75)]
        
        distribution_results = []
        
        for target in high_bias_quarters[high_bias_quarters['is_target'] == 'Y']['company'].unique():
            target_bias_quarters = high_bias_quarters[high_bias_quarters['company'] == target]
            
            target_bias_data = df[
                (df['Company'] == target) &
                (df['Quarter'].isin(target_bias_quarters['quarter'])) &
                (df['Year'].isin(target_bias_quarters['year']))
            ]
            
            if len(target_bias_data) == 0:
                continue
            
            topic_dist = target_bias_data['Primary_Topic'].value_counts().head(5).to_dict() if 'Primary_Topic' in target_bias_data.columns else {}
            temporal_dist = target_bias_data['Primary_Temporal_Context'].value_counts().to_dict() if 'Primary_Temporal_Context' in target_bias_data.columns else {}
            sentiment_dist = target_bias_data['Primary_Content_Sentiment'].value_counts().to_dict() if 'Primary_Content_Sentiment' in target_bias_data.columns else {}
            section_dist = target_bias_data['Section'].value_counts().to_dict() if 'Section' in target_bias_data.columns else {}
            
            avg_attribution_confidence = target_bias_data['attribution_locus_confidence'].mean() if 'attribution_locus_confidence' in target_bias_data.columns else 0
            
            distribution_results.append({
                'target': target,
                'bias_quarters': len(target_bias_quarters),
                'top_topics': ', '.join([f"{k}({v})" for k, v in list(topic_dist.items())[:3]]),
                'temporal_context_dist': json.dumps(temporal_dist),
                'sentiment_dist': json.dumps(sentiment_dist),
                'section_dist': json.dumps(section_dist),
                'avg_attribution_confidence': avg_attribution_confidence,
                'prepared_remarks_pct': section_dist.get('prepared_remarks', 0) / sum(section_dist.values()) * 100 if section_dist else 0,
                'qna_pct': section_dist.get('qna', 0) / sum(section_dist.values()) * 100 if section_dist else 0
            })
        
        distribution_df = pd.DataFrame(distribution_results)
        
        if len(distribution_df) > 0:
            print(f"\nBias Metadata Distribution:")
            print(f"  Mean attribution confidence in bias periods: {distribution_df['avg_attribution_confidence'].mean():.1f}%")
            print(f"  Mean Q&A percentage in bias periods: {distribution_df['qna_pct'].mean():.1f}%")
        
        return distribution_df
    

    
    def compare_targets_vs_peers_with_lag(self, metrics: Dict[str, pd.DataFrame],
                                          cause_sim_df: pd.DataFrame,
                                          effect_sim_df: pd.DataFrame,
                                          consistency_df: pd.DataFrame,
                                          coherence_df: pd.DataFrame,
                                          metadata_dist_df: pd.DataFrame) -> pd.DataFrame:
        print("\n8. Comparing Targets vs Peers (with Quarter ±1 Matching)")
        print("=" * 80)
        print("NOTE: Compares each target quarter to peers in same quarter ±1")
        print("      Accounts for fiscal calendar differences and reporting delays")
        
        full_metrics = metrics.get('full', pd.DataFrame())
        
        if len(full_metrics) == 0:
            return pd.DataFrame()
        
        comparison_results = []
        targets_analyzed = []
        targets_skipped_no_peers = []
        
        for target in full_metrics[full_metrics['is_target'] == 'Y']['company'].unique():
            target_data = full_metrics[full_metrics['company'] == target]
            
            if len(target_data) == 0:
                continue
            
            # get peer folders from config (use direct competitors only)
            target_config = None
            for target_key, info in self.peer_groups.items():
                target_folder = info.get('target_folder', target_key)
                if target_folder == target:
                    target_config = info
                    break
            
            if not target_config:
                targets_skipped_no_peers.append(target)
                continue
            
            # use direct_competitors_folders for focused peer analysis
            peer_folders = target_config.get('direct_competitors_folders', [])
            peer_folders = [p for p in peer_folders if p is not None]
            
            if not peer_folders:
                targets_skipped_no_peers.append(target)
                continue
            
            peer_data = full_metrics[full_metrics['company'].isin(peer_folders)]
            
            if len(peer_data) == 0:
                targets_skipped_no_peers.append(target)
                continue
            
            if target not in targets_analyzed:
                targets_analyzed.append(target)
            
            for _, target_row in target_data.iterrows():
                target_year = target_row['year']
                target_quarter = target_row['quarter']
                
                matching_peer_data = peer_data[
                    ((peer_data['year'] == target_year) & (peer_data['quarter'] == target_quarter)) |
                    ((peer_data['year'] == target_year) & (peer_data['quarter'] == target_quarter - 1)) |
                    ((peer_data['year'] == target_year) & (peer_data['quarter'] == target_quarter + 1)) |
                    ((peer_data['year'] == target_year - 1) & (target_quarter == 1) & (peer_data['quarter'] == 4)) |
                    ((peer_data['year'] == target_year + 1) & (target_quarter == 4) & (peer_data['quarter'] == 1))
                ]
                
                if len(matching_peer_data) == 0:
                    continue
                
                peer_mean_asymmetry = matching_peer_data['asymmetry_score'].mean()
                peer_std_asymmetry = matching_peer_data['asymmetry_score'].std()
                
                z_score = (target_row['asymmetry_score'] - peer_mean_asymmetry) / peer_std_asymmetry if peer_std_asymmetry > 0 else 0
                
                comparison_results.append({
                    'target': target,
                    'year': target_year,
                    'quarter': target_quarter,
                    'target_asymmetry': target_row['asymmetry_score'],
                    'peer_mean_asymmetry': peer_mean_asymmetry,
                    'peer_std_asymmetry': peer_std_asymmetry,
                    'z_score_with_lag': z_score,
                    'n_matching_peers': len(matching_peer_data),
                    'bias_flag': 'High' if abs(z_score) > 1.0 else ('Moderate' if abs(z_score) > 0.5 else 'Low')
                })
        
        comparison_df = pd.DataFrame(comparison_results)
        
        if len(comparison_df) > 0:
            print(f"\n Results:")
            print(f"  Targets analyzed: {len(targets_analyzed)}")
            print(f"  Total quarters analyzed: {len(comparison_df)}")
            print(f"  High bias periods (|z| > 1.0): {(comparison_df['z_score_with_lag'].abs() > 1.0).sum()}")
            
            if targets_skipped_no_peers:
                print(f"\n {len(targets_skipped_no_peers)} target(s) skipped - no peer data:")
                print(f"  {', '.join(sorted(set(targets_skipped_no_peers)))}")
            
            # detailed per-target breakdown
            self._print_target_diagnostics(comparison_df, cause_sim_df, effect_sim_df, 
                                          consistency_df, coherence_df, metadata_dist_df)
        
        return comparison_df
    
    def _print_target_diagnostics(self, comparison_df, cause_sim_df, effect_sim_df,
                                  consistency_df, coherence_df, metadata_dist_df):
        print(f"\n{'='*80}")
        print(f"PER-TARGET DIAGNOSTIC BREAKDOWN")
        print(f"{'='*80}")
        
        for target in comparison_df['target'].unique():
            target_quarters = comparison_df[comparison_df['target'] == target].sort_values(['year', 'quarter'])
            high_bias_quarters = target_quarters[target_quarters['bias_flag'] == 'High']
            
            print(f"\n TARGET: {target}")
            print(f"   Quarters analyzed: {len(target_quarters)}")
            print(f"   High bias quarters: {len(high_bias_quarters)} ({len(high_bias_quarters)/len(target_quarters)*100:.0f}%)")
            
            # get target-specific metrics (only from available analyses)
            # note: removed analyses will have empty dataframes, so we check before accessing
            cause_data = pd.DataFrame()
            effect_data = pd.DataFrame()
            consistency_data = pd.DataFrame()
            
            if len(cause_sim_df) > 0 and 'target' in cause_sim_df.columns:
                cause_data = cause_sim_df[cause_sim_df['target'] == target]
            
            if len(effect_sim_df) > 0 and 'target' in effect_sim_df.columns:
                effect_data = effect_sim_df[effect_sim_df['target'] == target]
            
            if len(consistency_df) > 0 and 'company' in consistency_df.columns:
                consistency_data = consistency_df[consistency_df['company'] == target]
            
            # calculate signal flags (only from available metrics)
            flags = []
            
            # z-score signal (always available)
            mean_z = target_quarters['z_score_with_lag'].abs().mean()
            if mean_z > 1.0:
                flags.append(f" High Z-Score ({mean_z:.2f})")
            
            # q&a concentration (if available)
            if len(metadata_dist_df) > 0 and 'target' in metadata_dist_df.columns:
                meta_data = metadata_dist_df[metadata_dist_df['target'] == target]
                if len(meta_data) > 0 and 'qna_pct' in meta_data.columns:
                    qna_pct = meta_data['qna_pct'].values[0]
                    if qna_pct > 60:
                        flags.append(f" Q&A Concentration ({qna_pct:.0f}%)")
            
            # note: removed cause/effect overlap, consistency, and coherence signals
            # as those analyses have been removed from the pipeline
            
            print(f"\n   BIAS SIGNALS DETECTED ({len(flags)}/2):")
            if flags:
                for flag in flags:
                    print(f"      {flag}")
            else:
                print(f"      No strong bias signals detected")
            
            # show high bias quarters
            if len(high_bias_quarters) > 0:
                print(f"\n   HIGH BIAS QUARTERS:")
                for _, quarter_row in high_bias_quarters.head(10).iterrows():
                    year, q = quarter_row['year'], quarter_row['quarter']
                    z = quarter_row['z_score_with_lag']
                    asym = quarter_row['target_asymmetry']
                    peer_asym = quarter_row['peer_mean_asymmetry']
                    
                    # determine bias type
                    if asym > peer_asym:
                        bias_type = "Self-serving bias (positive internal + negative external)"
                    else:
                        bias_type = "Reverse bias (positive external + negative internal)"
                    
                    print(f"      {year} Q{int(q)}: z-score={z:+.2f}, asymmetry={asym:.3f} vs peers={peer_asym:.3f}")
                    print(f"                 → {bias_type}")
                
                if len(high_bias_quarters) > 10:
                    print(f"      ... and {len(high_bias_quarters)-10} more quarters")
    
    def _create_methodology_dataframe(self) -> pd.DataFrame:
        """Create a comprehensive methodology explanation for Excel output."""
        methodology = [
            {
                'Metric': 'PEER SELECTION',
                'Formula': 'Uses direct_competitors_folders from config',
                'Interpretation': 'Analysis focuses on DIRECT competitors only (not indirect). Excludes null entries (missing data).',
                'Components': 'Provides cleaner signal by comparing targets to closest industry peers. Ticker-folder mapping via direct_competitors_folders_dict.'
            },
            {
                'Metric': 'SAB Proportion (PRIMARY METRIC)',
                'Formula': '(Positive-Internal + Negative-External) / All Attributions',
                'Interpretation': 'PRIMARY metric for bias detection. Range: 0% to 100%. Higher = more self-serving bias. Absolute proportion measure.',
                'Components': 'Measures what percentage of all attributions are self-serving (taking credit OR blaming externals). Comparable across contexts.'
            },
            {
                'Metric': 'Attribution Rate Asymmetry (SECONDARY METRIC)',
                'Formula': '(pos_internal_rate + neg_external_rate) - (pos_external_rate + neg_internal_rate)',
                'Interpretation': 'SECONDARY metric showing rate imbalance. Range: -2.0 to +2.0. >0.5 = high asymmetry, 0 = balanced, <0 = reverse pattern',
                'Components': 'Measures the asymmetry/imbalance between self-serving rates vs. modest rates. Difference-based measure.'
            },
            {
                'Metric': 'Positive Internal Rate',
                'Formula': '(Positive Internal attributions) / (Total Positive attributions)',
                'Interpretation': 'Percentage of positive outcomes attributed to internal factors (taking credit)',
                'Components': 'Higher values indicate tendency to claim success as own doing'
            },
            {
                'Metric': 'Negative External Rate',
                'Formula': '(Negative External attributions) / (Total Negative attributions)',
                'Interpretation': 'Percentage of negative outcomes attributed to external factors (blaming others)',
                'Components': 'Higher values indicate tendency to blame external factors for failures'
            },
            {
                'Metric': 'SAB Proportion Z-Score (PRIMARY)',
                'Formula': '(target_sab_proportion - peer_mean_sab) / peer_std_sab',
                'Interpretation': 'Standardized SAB Proportion. |z| >1.0 = weak signal, >1.5 = moderate, >2.0 = strong. Primary metric for statistical testing.',
                'Components': 'Compares target to peer distribution. Shows how many standard deviations target is from peer mean.'
            },
            {
                'Metric': 'Rate Asymmetry Z-Score (SECONDARY)',
                'Formula': '(target_asymmetry - peer_mean_asymmetry) / peer_std_asymmetry',
                'Interpretation': 'Standardized rate asymmetry. Same thresholds as above. Secondary validation metric.',
                'Components': 'Alternative standardized measure using rate differences instead of proportions.'
            },
            {
                'Metric': 'Cause/Effect Overlap',
                'Formula': '(Shared items between target and peers) / (Total target items) * 100',
                'Interpretation': '<30% suggests target uses unique causes/effects not shared by peers (potentially fabricated)',
                'Components': 'NOTE: Requires cause taxonomy for better discrimination. Current text-based approach has limitations.'
            },
            {
                'Metric': 'Narrative Stability',
                'Formula': 'Mean Jaccard similarity of causes/effects between adjacent quarters',
                'Interpretation': '>0.7 = consistent story, <0.3 = changes narrative each quarter (suspicious)',
                'Components': 'Measures if company repeats same explanations or constantly changes excuses'
            },
            {
                'Metric': 'Composite Bias Score',
                'Formula': 'Weighted sum: Z-score(3pts) + Low cause overlap(3pts) + Low effect overlap(2pts) + Low stability(2pts) + Suspect pairs(2pts) + Q&A bias(1pt)',
                'Interpretation': '12-15 = Extreme Bias, 8-11 = High Bias, 5-7 = Moderate Bias, 0-4 = Low Bias',
                'Components': 'Multi-signal approach combining statistical, temporal, and semantic indicators'
            },
            {
                'Metric': 'Peer Similarity Ratio',
                'Formula': '(Target-to-peer similarity) / (Target-to-non-peer similarity)',
                'Interpretation': '>1.2 = good peer group (target similar to peers), <1.0 = poor peer selection',
                'Components': 'Validates if defined peer groups show similar attribution patterns'
            },
            {
                'Metric': 'Suspect Cause-Effect Pairs',
                'Formula': 'Pairs with peer_frequency < 0.5% or unique to target',
                'Interpretation': 'High percentage suggests illogical or fabricated attributions',
                'Components': 'NOTE: With thousands of unique pairs, threshold may be too sensitive. Consider taxonomy.'
            },
            {
                'Metric': 'Potential Periods of Interest',
                'Formula': 'High Bias Period: |z_score| > 1.0; Adjacent: Quarters immediately before/after high bias',
                'Interpretation': 'Flags quarters for deeper investigation. Adjacent periods provide context (pre-bias setup, post-bias consequences).',
                'Components': 'Enables period-specific analysis: Target vs Peers in bias period; Target in bias vs normal periods; Peers in bias vs normal periods.'
            },
            {
                'Metric': 'TODO: Future Period-Specific Analysis',
                'Formula': 'Compare: (1) Target in high-bias periods vs peers in same periods; (2) Target high-bias vs target normal periods; (3) Peers in high-bias periods vs peers in normal periods',
                'Interpretation': 'Will reveal if bias manifests differently in specific quarters or is a persistent pattern',
                'Components': 'TODO: Implement comparative analysis for periods flagged in timeseries tabs. Will show if attribution patterns change significantly during bias periods.'
            }
        ]
        
        return pd.DataFrame(methodology)
    
    def _create_target_timeseries_data(self, metrics: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Create per-target time-series data for visualization.
        
        Includes 'potential_periods_of_interest' column that flags:
        - High bias quarters (|z_score| > 1.0)
        - Adjacent quarters (±1) around high bias periods for context
        """
        full_metrics = metrics.get('full', pd.DataFrame())
        
        if len(full_metrics) == 0:
            return {}
        
        target_timeseries = {}
        
        for target in full_metrics[full_metrics['is_target'] == 'Y']['company'].unique():
            target_data = full_metrics[full_metrics['company'] == target].copy()
            
            if len(target_data) == 0:
                continue
            
            # get peer folders from config (use direct competitors only)
            target_config = None
            for target_key, info in self.peer_groups.items():
                target_folder = info.get('target_folder', target_key)
                if target_folder == target:
                    target_config = info
                    break
            
            if not target_config:
                continue
            
            # use direct_competitors_folders for focused peer analysis
            peer_folders = target_config.get('direct_competitors_folders', [])
            peer_folders = [p for p in peer_folders if p is not None]
            
            if not peer_folders:
                continue
            
            peer_data = full_metrics[full_metrics['company'].isin(peer_folders)].copy()
            
            # create quarter-level comparison
            target_quarters = target_data.sort_values(['year', 'quarter'])
            
            timeseries_rows = []
            
            for _, target_row in target_quarters.iterrows():
                year = target_row['year']
                quarter = target_row['quarter']
                
                # get matching peer data (same quarter ±1)
                matching_peers = peer_data[
                    ((peer_data['year'] == year) & (peer_data['quarter'] == quarter)) |
                    ((peer_data['year'] == year) & (peer_data['quarter'] == quarter - 1)) |
                    ((peer_data['year'] == year) & (peer_data['quarter'] == quarter + 1)) |
                    ((peer_data['year'] == year - 1) & (quarter == 1) & (peer_data['quarter'] == 4)) |
                    ((peer_data['year'] == year + 1) & (quarter == 4) & (peer_data['quarter'] == 1))
                ]
                
                peer_mean_asym = matching_peers['asymmetry_score'].mean() if len(matching_peers) > 0 else 0
                peer_std_asym = matching_peers['asymmetry_score'].std() if len(matching_peers) > 0 else 0
                peer_mean_pos_int = matching_peers['pos_internal_rate'].mean() if len(matching_peers) > 0 else 0
                peer_mean_neg_ext = matching_peers['neg_external_rate'].mean() if len(matching_peers) > 0 else 0
                
                z_score = (target_row['asymmetry_score'] - peer_mean_asym) / peer_std_asym if peer_std_asym > 0 else 0
                
                timeseries_rows.append({
                    'year': year,
                    'quarter': quarter,
                    'period': f"{year}Q{int(quarter)}",
                    'target_asymmetry': target_row['asymmetry_score'],
                    'peer_mean_asymmetry': peer_mean_asym,
                    'peer_std_asymmetry': peer_std_asym,
                    'z_score': z_score,
                    # all four attribution rates
                    'target_pos_internal_rate': target_row['pos_internal_rate'],
                    'target_pos_external_rate': target_row['pos_external_rate'],
                    'target_neg_internal_rate': target_row['neg_internal_rate'],
                    'target_neg_external_rate': target_row['neg_external_rate'],
                    # peer rates for comparison
                    'peer_mean_pos_internal_rate': peer_mean_pos_int,
                    'peer_mean_neg_external_rate': peer_mean_neg_ext,
                    # all four attribution counts
                    'target_pos_internal_count': target_row['pos_internal'],
                    'target_pos_external_count': target_row['pos_external'],
                    'target_neg_internal_count': target_row['neg_internal'],
                    'target_neg_external_count': target_row['neg_external'],
                    'target_total_attributions': target_row['total_attributions'],
                    'n_matching_peers': len(matching_peers),
                    'bias_flag': 'High' if abs(z_score) > 1.0 else ('Moderate' if abs(z_score) > 0.5 else 'Low'),
                    'bias_type': 'Self-serving' if (target_row['asymmetry_score'] > peer_mean_asym and z_score > 0) else 'Reverse' if z_score < -1.0 else 'Balanced'
                })
            
            # convert to dataframe
            df = pd.DataFrame(timeseries_rows)
            
            # add 'potential_periods_of_interest' column
            # flag high bias periods (|z_score| > 1.0) and their ±1 quarters
            if len(df) > 0:
                df['potential_periods_of_interest'] = 'No'
                
                # find high bias quarters
                high_bias_mask = df['z_score'].abs() > 1.0
                high_bias_indices = df[high_bias_mask].index.tolist()
                
                # mark high bias quarters and ±1 quarters
                for idx in high_bias_indices:
                    # mark the high bias quarter itself
                    df.at[idx, 'potential_periods_of_interest'] = 'High Bias Period'
                    
                    # mark ±1 quarters if they exist
                    if idx > 0:  # previous quarter
                        if df.at[idx-1, 'potential_periods_of_interest'] == 'No':
                            df.at[idx-1, 'potential_periods_of_interest'] = 'Adjacent (Pre-Bias)'
                    if idx < len(df) - 1:  # next quarter
                        if df.at[idx+1, 'potential_periods_of_interest'] == 'No':
                            df.at[idx+1, 'potential_periods_of_interest'] = 'Adjacent (Post-Bias)'
            
            target_timeseries[target] = df
        
        return target_timeseries
    
    def _create_signal_decomposition_table(self, metrics: Dict[str, pd.DataFrame], 
                                           cause_sim_df: pd.DataFrame,
                                           effect_sim_df: pd.DataFrame,
                                           consistency_df: pd.DataFrame,
                                           coherence_df: pd.DataFrame,
                                           comparison_df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate signal decomposition showing which signals fire for each target.
        
        Helps answer: Do high-scorers share common patterns (low cause overlap, high Z-scores, etc.)?
        """
        full_metrics = metrics.get('full', pd.DataFrame())
        targets = full_metrics[full_metrics['is_target'] == 'Y']['company'].unique()
        
        decomposition_rows = []
        
        for target in targets:
            target_data = full_metrics[
                (full_metrics['company'] == target) & 
                (full_metrics['is_target'] == 'Y')
            ].copy()
            
            if len(target_data) == 0:
                continue
            
            # get peer data for comparison
            target_config = None
            for target_key, info in self.peer_groups.items():
                target_folder = info.get('target_folder', target_key)
                if target_folder == target:
                    target_config = info
                    break
            
            if not target_config:
                continue
            
            peer_folders = target_config.get('direct_competitors_folders', [])
            peer_folders = [p for p in peer_folders if p is not None]
            
            if not peer_folders:
                continue
            
            peer_data = full_metrics[full_metrics['company'].isin(peer_folders)].copy()
            
            # calculate aggregate signals
            mean_asymmetry = target_data['asymmetry_score'].mean()
            mean_z_score = 0
            n_high_z = 0
            n_mod_z = 0
            
            # calculate z-scores per quarter
            for _, target_row in target_data.iterrows():
                qtr = target_row['quarter']
                yr = target_row['year']
                
                peer_qtr = peer_data[
                    (peer_data['quarter'] == qtr) & 
                    (peer_data['year'] == yr)
                ]
                
                if len(peer_qtr) > 1:
                    peer_mean = peer_qtr['asymmetry_score'].mean()
                    peer_std = peer_qtr['asymmetry_score'].std()
                    
                    if peer_std > 0.01:
                        z_score = (target_row['asymmetry_score'] - peer_mean) / peer_std
                        mean_z_score += abs(z_score)
                        if abs(z_score) >= 2.0:
                            n_high_z += 1
                        elif abs(z_score) >= 1.5:
                            n_mod_z += 1
            
            if len(target_data) > 0:
                mean_z_score /= len(target_data)
            
            # get cause/effect overlap (these use 'target' column, no quarterly breakdown)
            # note: these analyses have been removed, so dataframes will be empty
            target_cause_sim = pd.DataFrame()
            target_effect_sim = pd.DataFrame()
            target_consistency = pd.DataFrame()
            target_coherence = pd.DataFrame()
            
            if len(cause_sim_df) > 0 and 'target' in cause_sim_df.columns:
                target_cause_sim = cause_sim_df[cause_sim_df['target'] == target]
            
            if len(effect_sim_df) > 0 and 'target' in effect_sim_df.columns:
                target_effect_sim = effect_sim_df[effect_sim_df['target'] == target]
            
            if len(consistency_df) > 0 and 'company' in consistency_df.columns:
                target_consistency = consistency_df[consistency_df['company'] == target]
            
            if len(coherence_df) > 0 and 'target' in coherence_df.columns:
                target_coherence = coherence_df[coherence_df['target'] == target]
            
            mean_cause_overlap = target_cause_sim['cause_overlap_pct'].mean() / 100 if len(target_cause_sim) > 0 and 'cause_overlap_pct' in target_cause_sim.columns else 0
            mean_effect_overlap = target_effect_sim['effect_overlap_pct'].mean() / 100 if len(target_effect_sim) > 0 and 'effect_overlap_pct' in target_effect_sim.columns else 0
            mean_cause_stability = target_consistency['cause_consistency_mean'].mean() if len(target_consistency) > 0 and 'cause_consistency_mean' in target_consistency.columns else 0
            mean_effect_stability = target_consistency['effect_consistency_mean'].mean() if len(target_consistency) > 0 and 'effect_consistency_mean' in target_consistency.columns else 0
            n_suspect_pairs = len(target_coherence[target_coherence['is_suspect'] == True]) if len(target_coherence) > 0 and 'is_suspect' in target_coherence.columns else 0
            
            # get composite score from comparison_df
            target_comparison = comparison_df[comparison_df['target'] == target]
            composite_score = target_comparison['z_score'].iloc[0] if len(target_comparison) > 0 else 0
            
            # determine which signals are true
            decomposition_rows.append({
                'target': target,
                'composite_score': composite_score,
                'mean_asymmetry': mean_asymmetry,
                'mean_abs_z_score': mean_z_score,
                'quarters_high_z': n_high_z,
                'quarters_mod_z': n_mod_z,
                'signal_high_z': n_high_z > 0,
                'signal_mod_z': n_mod_z > 0,
                'signal_high_asymmetry': mean_asymmetry >= 0.7,
                'signal_low_cause_overlap': mean_cause_overlap < 0.3,
                'signal_low_effect_overlap': mean_effect_overlap < 0.3,
                'signal_cause_instability': mean_cause_stability < 0.3,
                'signal_effect_instability': mean_effect_stability < 0.3,
                'signal_suspect_pairs': n_suspect_pairs > 0,
                'mean_cause_overlap': mean_cause_overlap,
                'mean_effect_overlap': mean_effect_overlap,
                'mean_cause_stability': mean_cause_stability,
                'mean_effect_stability': mean_effect_stability,
                'n_suspect_pairs': n_suspect_pairs,
                'n_quarters': len(target_data)
            })
        
        decomposition_df = pd.DataFrame(decomposition_rows)
        
        if len(decomposition_df) > 0:
            # sort by composite score descending
            decomposition_df = decomposition_df.sort_values('composite_score', ascending=False)
            
            # print summary
            print("\n" + "="*80)
            print("SIGNAL DECOMPOSITION SUMMARY")
            print("="*80)
            print(f"Total targets analyzed: {len(decomposition_df)}")
            
            # count how many targets have each signal
            signal_cols = [c for c in decomposition_df.columns if c.startswith('signal_')]
            print("\nSignal Prevalence Across Targets:")
            for col in signal_cols:
                count = decomposition_df[col].sum()
                pct = (count / len(decomposition_df)) * 100
                signal_name = col.replace('signal_', '').replace('_', ' ').title()
                print(f"  {signal_name}: {count}/{len(decomposition_df)} targets ({pct:.1f}%)")
            
            # pattern analysis for high scorers
            high_scorers = decomposition_df[decomposition_df['composite_score'] >= 5.0]
            if len(high_scorers) > 0:
                print(f"\nPattern Analysis for High Scorers (score ≥ 5.0, n={len(high_scorers)}):")
                for col in signal_cols:
                    count = high_scorers[col].sum()
                    pct = (count / len(high_scorers)) * 100
                    signal_name = col.replace('signal_', '').replace('_', ' ').title()
                    print(f"  {signal_name}: {count}/{len(high_scorers)} ({pct:.1f}%)")
        
        return decomposition_df
    
    def _create_quarterly_signal_tracking(self, metrics: Dict[str, pd.DataFrame],
                                         cause_sim_df: pd.DataFrame,
                                         effect_sim_df: pd.DataFrame,
                                         consistency_df: pd.DataFrame,
                                         coherence_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create per-quarter signal tracking for each target.
        
        Shows which signals were active in each quarter through time.
        """
        full_metrics = metrics.get('full', pd.DataFrame())
        targets = full_metrics[full_metrics['is_target'] == 'Y']['company'].unique()
        
        target_signals = {}
        
        for target in targets:
            target_data = full_metrics[
                (full_metrics['company'] == target) & 
                (full_metrics['is_target'] == 'Y')
            ].copy()
            
            if len(target_data) == 0:
                continue
            
            # get peer data
            target_config = None
            for target_key, info in self.peer_groups.items():
                target_folder = info.get('target_folder', target_key)
                if target_folder == target:
                    target_config = info
                    break
            
            if not target_config:
                continue
            
            peer_folders = target_config.get('direct_competitors_folders', [])
            peer_folders = [p for p in peer_folders if p is not None]
            
            if not peer_folders:
                continue
            
            peer_data = full_metrics[full_metrics['company'].isin(peer_folders)].copy()
            
            signal_rows = []
            
            for _, target_row in target_data.iterrows():
                qtr = target_row['quarter']
                yr = target_row['year']
                period = f"{int(yr)}Q{int(qtr)}"
                
                # calculate z-score for this quarter
                peer_qtr = peer_data[
                    (peer_data['quarter'] == qtr) & 
                    (peer_data['year'] == yr)
                ]
                
                z_score = 0
                if len(peer_qtr) > 1:
                    peer_mean = peer_qtr['asymmetry_score'].mean()
                    peer_std = peer_qtr['asymmetry_score'].std()
                    
                    if peer_std > 0.01:
                        z_score = (target_row['asymmetry_score'] - peer_mean) / peer_std
                
                # get cause/effect overlap (aggregated per target, not per quarter)
                # note: these analyses have been removed, so dataframes will be empty
                target_cause = pd.DataFrame()
                target_effect = pd.DataFrame()
                target_coherence = pd.DataFrame()
                
                if len(cause_sim_df) > 0 and 'target' in cause_sim_df.columns:
                    target_cause = cause_sim_df[cause_sim_df['target'] == target]
                
                if len(effect_sim_df) > 0 and 'target' in effect_sim_df.columns:
                    target_effect = effect_sim_df[effect_sim_df['target'] == target]
                
                if len(coherence_df) > 0 and 'target' in coherence_df.columns:
                    target_coherence = coherence_df[coherence_df['target'] == target]
                
                cause_overlap = target_cause['cause_overlap_pct'].iloc[0] / 100 if len(target_cause) > 0 and 'cause_overlap_pct' in target_cause.columns else 0
                effect_overlap = target_effect['effect_overlap_pct'].iloc[0] / 100 if len(target_effect) > 0 and 'effect_overlap_pct' in target_effect.columns else 0
                suspect_pairs = len(target_coherence[target_coherence['is_suspect'] == True]) if len(target_coherence) > 0 and 'is_suspect' in target_coherence.columns else 0
                
                # determine signals
                signal_rows.append({
                    'year': yr,
                    'quarter': qtr,
                    'period': period,
                    'asymmetry_score': target_row['asymmetry_score'],
                    'z_score': z_score,
                    'cause_overlap': cause_overlap,
                    'effect_overlap': effect_overlap,
                    'suspect_pairs': suspect_pairs,
                    'signal_high_z': abs(z_score) >= 2.0,
                    'signal_mod_z': 1.5 <= abs(z_score) < 2.0,
                    'signal_high_asymmetry': target_row['asymmetry_score'] >= 0.7,
                    'signal_low_cause_overlap': cause_overlap < 0.3,
                    'signal_low_effect_overlap': effect_overlap < 0.3,
                    'signal_suspect_pairs': suspect_pairs > 0,
                    'total_signals_active': 0  # will calculate next
                })
            
            # calculate total active signals per quarter
            signal_df = pd.DataFrame(signal_rows)
            if len(signal_df) > 0:
                signal_cols = [c for c in signal_df.columns if c.startswith('signal_')]
                signal_df['total_signals_active'] = signal_df[signal_cols].sum(axis=1)
                target_signals[target] = signal_df
        
        return target_signals
    
    def _create_target_peer_detail(self, metrics: Dict[str, pd.DataFrame], target: str) -> pd.DataFrame:
        """Create detailed peer-level data for a specific target.
        
        Shows both ticker and folder name for each peer using direct_competitors_folders_dict.
        Includes target firm's stats as first row for easy comparison.
        """
        full_metrics = metrics.get('full', pd.DataFrame())
        
        if len(full_metrics) == 0:
            return pd.DataFrame()
        
        target_data = full_metrics[full_metrics['company'] == target]
        
        if len(target_data) == 0:
            return pd.DataFrame()
        
        # get peer folders and ticker mapping from config
        target_config = None
        target_ticker = None
        for target_key, info in self.peer_groups.items():
            target_folder = info.get('target_folder', target_key)
            if target_folder == target:
                target_config = info
                target_ticker = target_key
                break
        
        if not target_config:
            return pd.DataFrame()
        
        # use direct_competitors_folders for peer list
        peer_folders = target_config.get('direct_competitors_folders', [])
        peer_folders = [p for p in peer_folders if p is not None]
        
        # get ticker mapping (ticker -> folder name)
        peer_dict = target_config.get('direct_competitors_folders_dict', {})
        
        if not peer_folders:
            return pd.DataFrame()
        
        peer_data = full_metrics[full_metrics['company'].isin(peer_folders)]
        
        # start with target firm's stats as first row
        peer_details = []
        
        # add target as first row
        peer_details.append({
            'peer_ticker': f" {target_ticker if target_ticker else target}",
            'peer_folder': target,
            'n_quarters': len(target_data),
            'mean_asymmetry': target_data['asymmetry_score'].mean(),
            'std_asymmetry': target_data['asymmetry_score'].std(),
            'mean_pos_internal_rate': target_data['pos_internal_rate'].mean(),
            'mean_neg_external_rate': target_data['neg_external_rate'].mean(),
            'total_attributions': target_data['total_attributions'].sum(),
            'total_pos_internal': target_data['pos_internal'].sum(),
            'total_neg_external': target_data['neg_external'].sum()
        })
        
        # add peer stats
        for peer_folder in peer_folders:
            peer_company_data = peer_data[peer_data['company'] == peer_folder]
            
            if len(peer_company_data) == 0:
                continue
            
            # find ticker for this folder (reverse lookup)
            peer_ticker = None
            for ticker, folder in peer_dict.items():
                if folder == peer_folder:
                    peer_ticker = ticker
                    break
            
            peer_details.append({
                'peer_ticker': peer_ticker if peer_ticker else peer_folder,
                'peer_folder': peer_folder,
                'n_quarters': len(peer_company_data),
                'mean_asymmetry': peer_company_data['asymmetry_score'].mean(),
                'std_asymmetry': peer_company_data['asymmetry_score'].std(),
                'mean_pos_internal_rate': peer_company_data['pos_internal_rate'].mean(),
                'mean_neg_external_rate': peer_company_data['neg_external_rate'].mean(),
                'total_attributions': peer_company_data['total_attributions'].sum(),
                'total_pos_internal': peer_company_data['pos_internal'].sum(),
                'total_neg_external': peer_company_data['neg_external'].sum()
            })
        
        return pd.DataFrame(peer_details)
    

  
    
    # =========================================================================
    # bias period validation methods
    # =========================================================================
    
    def compare_target_bias_vs_peers_same_period(self, metrics: Dict[str, pd.DataFrame], 
                                                  window: str = 'exact') -> pd.DataFrame:
        """
        Compare target firms during bias periods vs peers during same time periods.
        
        SIGNIFICANT DIFFERENCE CRITERIA:
        - Z-score: |z| > 1.0 = weak signal, |z| > 1.5 = moderate, |z| > 2.0 = strong
          (measures how many standard deviations target is from peer mean)
        - P-value: p < 0.05 = statistically significant (95% confidence)
          (probability that difference is due to chance)
        - Cohen's d: |d| > 0.2 = small effect, |d| > 0.5 = medium, |d| > 0.8 = large
          (standardized effect size, independent of sample size)"""
        print(f"\n Bias Period Validation: Target vs Peers (Same Period, {window})")
        print("=" * 80)
        print(f"  NOTE: Significant = p<0.05, |z|>1.0 (weak), |z|>1.5 (moderate), |z|>2.0 (strong)")
        
        full_metrics = metrics.get('full', pd.DataFrame())
        if len(full_metrics) == 0:
            return pd.DataFrame()
        
        # determine which bias flag to use
        bias_col = f'is_bias_period_{window}'
        if bias_col not in full_metrics.columns:
            print(f"  ️  Bias flag '{bias_col}' not found in metrics")
            return pd.DataFrame()
        
        comparison_results = []
        
        # metrics to analyze
        metric_cols = ['asymmetry_score', 'sab_proportion', 'neg_ext_of_negative', 'pos_int_of_positive',
                      'pos_internal_rate', 'neg_external_rate']
        
        # get targets with bias periods
        targets_with_bias = full_metrics[
            (full_metrics['is_target'] == 'Y') & 
            (full_metrics[bias_col] == True)
        ]['company'].unique()
        
        print(f"   Found {len(targets_with_bias)} target companies with bias periods")
        
        for target in targets_with_bias:
            # get target's peer group
            target_config = None
            for ticker, info in self.expert_bias_periods.items():
                if info['target_folder'] == target:
                    target_config = info
                    break
            
            if not target_config:
                continue
            
            # get direct competitors
            peer_folders = set()
            for peer_key, peer_info in self.peer_groups.items():
                if peer_info.get('target_folder') == target:
                    peer_folders = set(peer_info.get('direct_competitors_folders', []))
                    peer_folders = {p for p in peer_folders if p is not None}
                    break
            
            if not peer_folders:
                continue
            
            # get target bias periods
            target_bias = full_metrics[
                (full_metrics['company'] == target) & 
                (full_metrics[bias_col] == True)
            ]
            
            for _, target_row in target_bias.iterrows():
                year, quarter = target_row['year'], target_row['quarter']
                
                # get peers during same period
                peer_same_period = full_metrics[
                    (full_metrics['company'].isin(peer_folders)) &
                    (full_metrics['year'] == year) &
                    (full_metrics['quarter'] == quarter)
                ]
                
                if len(peer_same_period) == 0:
                    continue
                
                # calculate statistics for each metric
                result_row = {
                    'target': target,
                    'year': year,
                    'quarter': quarter,
                    'bias_type': target_row['bias_type'],
                    'window': window,
                    'n_peers': len(peer_same_period),
                    'sample_flag': self._get_sample_size_flag(len(peer_same_period))
                }
                
                for metric in metric_cols:
                    if metric not in target_row or metric not in peer_same_period.columns:
                        continue
                    
                    target_val = target_row[metric]
                    peer_vals = peer_same_period[metric].dropna().tolist()
                    
                    if len(peer_vals) == 0:
                        continue
                    
                    # calculate statistics
                    peer_mean = np.mean(peer_vals)
                    peer_std = np.std(peer_vals, ddof=1) if len(peer_vals) > 1 else 0
                    z_score = self._calculate_z_score(target_val, peer_vals)
                    t_stat, p_val = self._calculate_t_test([target_val], peer_vals)
                    cohens_d = self._calculate_cohens_d([target_val], peer_vals)
                    
                    result_row[f'{metric}_target'] = target_val
                    result_row[f'{metric}_peer_mean'] = peer_mean
                    result_row[f'{metric}_peer_std'] = peer_std
                    result_row[f'{metric}_z_score'] = z_score
                    result_row[f'{metric}_t_stat'] = t_stat
                    result_row[f'{metric}_p_value'] = p_val
                    result_row[f'{metric}_cohens_d'] = cohens_d
                    result_row[f'{metric}_difference'] = target_val - peer_mean
                    result_row[f'{metric}_z_interp'] = self._interpret_z_score(z_score)
                    result_row[f'{metric}_d_interp'] = self._interpret_cohens_d(cohens_d)
                
                comparison_results.append(result_row)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        if len(comparison_df) > 0:
            print(f"\n   Analyzed {len(comparison_df)} bias period quarters across {len(targets_with_bias)} targets")
            
            # summary statistics
            for metric in ['asymmetry_score', 'sab_proportion']:
                if f'{metric}_z_score' in comparison_df.columns:
                    significant = (comparison_df[f'{metric}_p_value'] < 0.05).sum()
                    mean_z = comparison_df[f'{metric}_z_score'].mean()
                    print(f"    {metric}: Mean z-score={mean_z:.2f}, Significant differences={significant}/{len(comparison_df)}")
        
        return comparison_df
    
    def compare_target_bias_vs_target_normal(self, metrics: Dict[str, pd.DataFrame],
                                             window: str = 'exact') -> pd.DataFrame:
        """
        Compare target firms during bias periods vs their own normal periods.
        Within-firm comparison to see if attribution patterns shift.
        """
        print(f"\n Bias Period Validation: Target Bias vs Target Normal ({window})")
        print("=" * 80)
        
        full_metrics = metrics.get('full', pd.DataFrame())
        if len(full_metrics) == 0:
            return pd.DataFrame()
        
        bias_col = f'is_bias_period_{window}'
        if bias_col not in full_metrics.columns:
            return pd.DataFrame()
        
        comparison_results = []
        metric_cols = ['asymmetry_score', 'sab_proportion', 'neg_ext_of_negative', 'pos_int_of_positive']
        
        targets_with_bias = full_metrics[
            (full_metrics['is_target'] == 'Y') & 
            (full_metrics[bias_col] == True)
        ]['company'].unique()
        
        for target in targets_with_bias:
            target_data = full_metrics[
                (full_metrics['company'] == target) &
                (full_metrics['is_target'] == 'Y')
            ]
            
            bias_periods = target_data[target_data[bias_col] == True]
            normal_periods = target_data[target_data[bias_col] == False]
            
            if len(bias_periods) == 0 or len(normal_periods) == 0:
                continue
            
            # get bias type
            bias_type = bias_periods['bias_type'].iloc[0] if 'bias_type' in bias_periods.columns else None
            
            result_row = {
                    'target': target,
                'bias_type': bias_type,
                'window': window,
                'n_bias_quarters': len(bias_periods),
                'n_normal_quarters': len(normal_periods),
                'bias_sample_flag': self._get_sample_size_flag(len(bias_periods)),
                'normal_sample_flag': self._get_sample_size_flag(len(normal_periods))
            }
            
            for metric in metric_cols:
                if metric not in bias_periods.columns or metric not in normal_periods.columns:
                    continue
                
                bias_vals = bias_periods[metric].dropna().tolist()
                normal_vals = normal_periods[metric].dropna().tolist()
                
                if len(bias_vals) == 0 or len(normal_vals) == 0:
                    continue
                
                bias_mean = np.mean(bias_vals)
                normal_mean = np.mean(normal_vals)
                t_stat, p_val = self._calculate_t_test(bias_vals, normal_vals)
                cohens_d = self._calculate_cohens_d(bias_vals, normal_vals)
                
                result_row[f'{metric}_bias_mean'] = bias_mean
                result_row[f'{metric}_normal_mean'] = normal_mean
                result_row[f'{metric}_difference'] = bias_mean - normal_mean
                result_row[f'{metric}_pct_change'] = ((bias_mean - normal_mean) / normal_mean * 100) if normal_mean != 0 else 0
                result_row[f'{metric}_t_stat'] = t_stat
                result_row[f'{metric}_p_value'] = p_val
                result_row[f'{metric}_cohens_d'] = cohens_d
                result_row[f'{metric}_d_interp'] = self._interpret_cohens_d(cohens_d)
                result_row[f'{metric}_significant'] = p_val < 0.05
            
            comparison_results.append(result_row)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        if len(comparison_df) > 0:
            print(f"   Analyzed {len(comparison_df)} targets")
            
            # count significant differences
            for metric in ['asymmetry_score', 'sab_proportion']:
                if f'{metric}_significant' in comparison_df.columns:
                    sig_count = comparison_df[f'{metric}_significant'].sum()
                    print(f"    {metric}: {sig_count}/{len(comparison_df)} targets show significant change")
        
        return comparison_df
    
    def aggregate_validation_results(self, 
                                     target_vs_peer: pd.DataFrame,
                                     target_bias_vs_normal: pd.DataFrame) -> Dict:
        """
        Aggregate validation results to provide overall assessment of bias period hypothesis.
        """
        print(f"\n Aggregating Validation Results")
        print("=" * 80)
        
        summary = {
            'n_targets_analyzed': 0,
            'metrics': {},
            'overall_validation': 'Unknown'
        }
        
        if len(target_vs_peer) == 0 and len(target_bias_vs_normal) == 0:
            print("  ️  No validation data available")
            return summary
        
        metrics_to_check = ['asymmetry_score', 'sab_proportion', 'neg_ext_of_negative', 'pos_int_of_positive']
        
        # analyze target vs peer results
        if len(target_vs_peer) > 0:
            unique_targets = target_vs_peer['target'].nunique()
            summary['n_targets_analyzed'] = unique_targets
            
            print(f"\n   Target vs Peer Analysis ({unique_targets} targets)")
            
            for metric in metrics_to_check:
                z_col = f'{metric}_z_score'
                p_col = f'{metric}_p_value'
                
                if z_col not in target_vs_peer.columns:
                    continue
                
                # count quarters with elevated values
                elevated = (target_vs_peer[z_col] > 1.0).sum()
                strong_signal = (target_vs_peer[z_col] > 2.0).sum()
                significant = (target_vs_peer[p_col] < 0.05).sum() if p_col in target_vs_peer.columns else 0
                
                mean_z = target_vs_peer[z_col].mean()
                
                summary['metrics'][metric] = {
                    'mean_z_score': mean_z,
                    'elevated_count': elevated,
                    'strong_signal_count': strong_signal,
                    'significant_count': significant,
                    'total_quarters': len(target_vs_peer)
                }
                
                print(f"    {metric}:")
                print(f"      Mean z-score: {mean_z:.2f}")
                print(f"      Elevated (z>1.0): {elevated}/{len(target_vs_peer)} quarters ({elevated/len(target_vs_peer)*100:.1f}%)")
                print(f"      Strong (z>2.0): {strong_signal}/{len(target_vs_peer)} quarters")
        
        # analyze within-target results
        if len(target_bias_vs_normal) > 0:
            print(f"\n   Within-Target Analysis ({len(target_bias_vs_normal)} targets)")
            
            for metric in metrics_to_check:
                sig_col = f'{metric}_significant'
                diff_col = f'{metric}_difference'
                
                if sig_col not in target_bias_vs_normal.columns:
                    continue
                
                significant = target_bias_vs_normal[sig_col].sum()
                mean_diff = target_bias_vs_normal[diff_col].mean() if diff_col in target_bias_vs_normal.columns else 0
                
                print(f"    {metric}:")
                print(f"      Significant changes: {significant}/{len(target_bias_vs_normal)} targets ({significant/len(target_bias_vs_normal)*100:.1f}%)")
                print(f"      Mean difference: {mean_diff:+.3f}")
        
        # overall assessment
        validation_score = 0
        if 'asymmetry_score' in summary['metrics']:
            if summary['metrics']['asymmetry_score']['mean_z_score'] > 1.0:
                validation_score += 2
            if summary['metrics']['asymmetry_score']['elevated_count'] > len(target_vs_peer) * 0.5:
                validation_score += 1
        
        if 'sab_proportion' in summary['metrics']:
            if summary['metrics']['sab_proportion']['mean_z_score'] > 1.0:
                validation_score += 2
        
        if validation_score >= 4:
            summary['overall_validation'] = 'Strong Support'
        elif validation_score >= 2:
            summary['overall_validation'] = 'Moderate Support'
        else:
            summary['overall_validation'] = 'Weak/No Support'
        
        print(f"\n   Overall Validation: {summary['overall_validation']}")
        
        return summary
    
    def analyze_by_bias_type(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """
        Group validation results by bias type to see if patterns differ.
        """
        print(f"\n Analyzing Results by Bias Type")
        print("=" * 80)
        
        if len(comparison_df) == 0 or 'bias_type' not in comparison_df.columns:
            print("  ️  No bias type data available")
            return pd.DataFrame()
        
        bias_types = comparison_df['bias_type'].dropna().unique()
        
        results = []
        
        for bias_type in bias_types:
            type_data = comparison_df[comparison_df['bias_type'] == bias_type]
            
            result = {
                'bias_type': bias_type,
                'n_observations': len(type_data),
                'n_targets': type_data['target'].nunique() if 'target' in type_data.columns else 0
            }
            
            # calculate mean metrics for this bias type
            for metric in ['asymmetry_score', 'sab_proportion', 'neg_ext_of_negative', 'pos_int_of_positive']:
                target_col = f'{metric}_target'
                diff_col = f'{metric}_difference'
                z_col = f'{metric}_z_score'
                
                if target_col in type_data.columns:
                    result[f'{metric}_mean'] = type_data[target_col].mean()
                
                if diff_col in type_data.columns:
                    result[f'{metric}_diff_mean'] = type_data[diff_col].mean()
                
                if z_col in type_data.columns:
                    result[f'{metric}_z_mean'] = type_data[z_col].mean()
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            print(f"   Analyzed {len(bias_types)} bias types")
            
            for _, row in results_df.iterrows():
                print(f"\n  {row['bias_type']} (n={row['n_observations']}, targets={row['n_targets']}):")
                if 'asymmetry_score_mean' in row:
                    print(f"    Mean asymmetry: {row['asymmetry_score_mean']:.3f}")
                if 'sab_proportion_mean' in row:
                    print(f"    Mean SAB proportion: {row['sab_proportion_mean']:.1%}")
        
        return results_df
    
    def compare_targets_vs_peers(self, metrics: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        print("\n10. Comparing Targets vs Peers (Aggregate)")
        print("=" * 80)
        
        full_metrics = metrics.get('full', pd.DataFrame())
        
        if len(full_metrics) == 0:
            print("No metrics available for comparison")
            return pd.DataFrame()
        
        # first, get all targets found in the data
        all_targets_in_data = full_metrics[full_metrics['is_target'] == 'Y']['company'].unique()
        print(f"\n Found {len(all_targets_in_data)} targets with attribution data:")
        print(f"  {', '.join(sorted(all_targets_in_data))}")
        
        company_agg = full_metrics.groupby(['company', 'is_target', 'is_peer', 'related_firms']).agg({
            'asymmetry_score': ['mean', 'std', 'count'],
            'pos_internal_rate': 'mean',
            'neg_external_rate': 'mean'
        }).reset_index()
        
        company_agg.columns = ['_'.join(col).strip('_') for col in company_agg.columns]
        
        comparison_results = []
        targets_without_peers = []
        targets_without_peer_data = []
        
        for target in company_agg[company_agg['is_target'] == 'Y']['company'].unique():
            target_row = company_agg[company_agg['company'] == target].iloc[0]
            
            # get peer_folders from config for this target
            target_config = None
            for target_key, info in self.peer_groups.items():
                target_folder = info.get('target_folder', target_key)
                if target_folder == target:
                    target_config = info
                    break
            
            if not target_config:
                print(f"   {target}: No config found for this target folder name")
                targets_without_peers.append(target)
                continue
            
            # use direct_competitors_folders (focused peer analysis, not all peers)
            peer_folders = target_config.get('direct_competitors_folders', [])
            peer_folders = [p for p in peer_folders if p is not None]
            
            if not peer_folders:
                print(f"   {target}: No peers defined in config (will still include in target analysis)")
                targets_without_peers.append(target)
                # don't continue - include target even without peers
            
            # check which peers are in the data
            peer_rows = company_agg[company_agg['company'].isin(peer_folders)] if peer_folders else pd.DataFrame()
            
            if len(peer_folders) > 0 and len(peer_rows) == 0:
                print(f"   {target}: peers={peer_folders[:3]}... but NONE found in data")
                targets_without_peer_data.append(target)
                # don't continue - include target even without peer data
            
            if len(peer_rows) > 0:
                print(f"   {target}: Found {len(peer_rows)}/{len(peer_folders)} peers in data")
                
                peer_mean_asymmetry = peer_rows['asymmetry_score_mean'].mean()
                peer_std_asymmetry = peer_rows['asymmetry_score_mean'].std()
                z_score = (target_row['asymmetry_score_mean'] - peer_mean_asymmetry) / peer_std_asymmetry if peer_std_asymmetry > 0 else 0
                peer_pos_internal = peer_rows['pos_internal_rate_mean'].mean()
                peer_neg_external = peer_rows['neg_external_rate_mean'].mean()
            else:
                # no peers available - still include target with nan for peer comparisons
                print(f"   {target}: No peer data available for comparison")
                peer_mean_asymmetry = float('nan')
                peer_std_asymmetry = float('nan')
                z_score = float('nan')
                peer_pos_internal = float('nan')
                peer_neg_external = float('nan')
            
            comparison_results.append({
                'target': target,
                'n_peers': len(peer_rows),
                'target_asymmetry': target_row['asymmetry_score_mean'],
                'peer_mean_asymmetry': peer_mean_asymmetry,
                'peer_std_asymmetry': peer_std_asymmetry,
                'z_score': z_score,
                'target_pos_internal_rate': target_row['pos_internal_rate_mean'],
                'target_neg_external_rate': target_row['neg_external_rate_mean'],
                'peer_pos_internal_rate': peer_pos_internal,
                'peer_neg_external_rate': peer_neg_external,
                'bias_flag': 'High' if abs(z_score) > 1.0 else ('Moderate' if abs(z_score) > 0.5 else 'Low') if not pd.isna(z_score) else 'No Peers'
            })
        
        comparison_df = pd.DataFrame(comparison_results)
        
        if len(comparison_df) > 0:
            print(f"\n Target vs Peer Comparison:")
            print(f"  Targets analyzed: {len(comparison_df)}")
            print(f"  High bias (|z| > 1.0): {(comparison_df['z_score'].abs() > 1.0).sum()}")
            print(f"  Mean z-score: {comparison_df['z_score'].mean():.2f}")
        
        # show which targets were filtered out and why
        if targets_without_peers:
            print(f"\n {len(targets_without_peers)} target(s) excluded - no peer group defined in config:")
            print(f"  {', '.join(sorted(targets_without_peers))}")
        
        if targets_without_peer_data:
            print(f"\n {len(targets_without_peer_data)} target(s) excluded - peer folders not in dataset:")
            print(f"  NOTE: Check if peer_folders in config match actual CSV filenames")
            print(f"  {', '.join(sorted(targets_without_peer_data))}")
        
        return comparison_df
    
    def calculate_composite_bias_scores(self,
                                        comparison_df: pd.DataFrame,
                                        cause_sim_df: pd.DataFrame,
                                        effect_sim_df: pd.DataFrame,
                                        consistency_df: pd.DataFrame,
                                        coherence_df: pd.DataFrame,
                                        metadata_dist_df: pd.DataFrame) -> pd.DataFrame:
        print("\n12. Calculating Composite Bias Scores")
        print("=" * 80)
        
        composite_results = []
        
        for _, row in comparison_df.iterrows():
            target = row['target']
            
            score = 0
            flags = []
            
            z_score = row.get('z_score', 0)
            if abs(z_score) > 1.0:
                score += 3
                flags.append(f"High Z-Score ({z_score:.2f})")
            elif abs(z_score) > 0.5:
                score += 1
                flags.append(f"Moderate Z-Score ({z_score:.2f})")
            
            # note: the following analyses have been removed, so dataframes will be empty
            # only access them if they contain data with the expected columns
            
            if len(cause_sim_df) > 0 and 'target' in cause_sim_df.columns:
                cause_overlap = cause_sim_df[cause_sim_df['target'] == target]['cause_overlap_pct'].values
                if len(cause_overlap) > 0 and 'cause_overlap_pct' in cause_sim_df.columns and cause_overlap[0] < 30:
                    score += 3
                    flags.append(f"Low Cause Overlap ({cause_overlap[0]:.1f}%)")
            
            if len(effect_sim_df) > 0 and 'target' in effect_sim_df.columns:
                effect_overlap = effect_sim_df[effect_sim_df['target'] == target]['effect_overlap_pct'].values
                if len(effect_overlap) > 0 and 'effect_overlap_pct' in effect_sim_df.columns and effect_overlap[0] < 30:
                    score += 2
                    flags.append(f"Low Effect Overlap ({effect_overlap[0]:.1f}%)")
            
            if len(consistency_df) > 0 and 'company' in consistency_df.columns:
                stability = consistency_df[consistency_df['company'] == target]['narrative_stability'].values
                if len(stability) > 0 and 'narrative_stability' in consistency_df.columns and stability[0] < 0.3:
                    score += 2
                    flags.append(f"Low Narrative Stability ({stability[0]:.2f})")
            
            if len(metadata_dist_df) > 0 and 'target' in metadata_dist_df.columns:
                qna_pct = metadata_dist_df[metadata_dist_df['target'] == target]['qna_pct'].values
                if len(qna_pct) > 0 and 'qna_pct' in metadata_dist_df.columns and qna_pct[0] > 60:
                    score += 1
                    flags.append(f"High Q&A Bias ({qna_pct[0]:.1f}%)")
            
            if len(coherence_df) > 0 and 'target' in coherence_df.columns:
                target_coherence = coherence_df[coherence_df['target'] == target]
                if len(target_coherence) > 0 and 'is_suspect' in target_coherence.columns:
                    suspect_pct = (target_coherence['is_suspect'].sum() / len(target_coherence) * 100)
                    if suspect_pct > 20:
                        score += 2
                        flags.append(f"Suspect Pairs ({suspect_pct:.1f}%)")
            
            if score >= 12:
                classification = "Extreme Bias"
            elif score >= 8:
                classification = "High Bias"
            elif score >= 5:
                classification = "Moderate Bias"
            else:
                classification = "Low Bias"
            
            composite_results.append({
                'target': target,
                'composite_score': score,
                'bias_classification': classification,
                'n_flags': len(flags),
                'key_flags': '; '.join(flags) if flags else 'None',
                'z_score': z_score,
                'target_asymmetry': row.get('target_asymmetry', 0),
                'peer_mean_asymmetry': row.get('peer_mean_asymmetry', 0)
            })
        
        composite_df = pd.DataFrame(composite_results)
        
        if len(composite_df) > 0:
            composite_df = composite_df.sort_values('composite_score', ascending=False)
            print(f"\nComposite Bias Scoring:")
            print(f"  Extreme Bias (12-15 pts): {(composite_df['composite_score'] >= 12).sum()}")
            print(f"  High Bias (8-11 pts): {((composite_df['composite_score'] >= 8) & (composite_df['composite_score'] < 12)).sum()}")
            print(f"  Moderate Bias (5-7 pts): {((composite_df['composite_score'] >= 5) & (composite_df['composite_score'] < 8)).sum()}")
            print(f"  Low Bias (0-4 pts): {(composite_df['composite_score'] < 5).sum()}")
        else:
            print(f"\nNo targets to score (comparison_df was empty)")
        
        return composite_df
    
    def create_excel_workbook(self,
                             composite_df: pd.DataFrame,
                             comparison_df: pd.DataFrame,
                             metrics: Dict[str, pd.DataFrame],
                             rolling_df: pd.DataFrame,
                             cause_sim_df: pd.DataFrame,
                             effect_sim_df: pd.DataFrame,
                             coherence_df: pd.DataFrame,
                             consistency_df: pd.DataFrame,
                             metadata_dist_df: pd.DataFrame,
                             lag_comparison_df: pd.DataFrame,
                             similarity_df: pd.DataFrame,
                             signal_decomposition_df: pd.DataFrame,
                             target_signals: Dict[str, pd.DataFrame],
                             timestamp: str,
                             validation_results: Optional[Dict] = None,
                             distribution_results: Optional[Dict] = None,
                             attribution_comparison_df: Optional[pd.DataFrame] = None):
        
        if not EXCEL_AVAILABLE:
            print("Openpyxl not available, skipping Excel consolidation")
            return
        
        print("\n13. Creating Consolidated Excel Workbook")
        print("=" * 80)
        
        excel_file = self.output_dir / f"peer_benchmark_consolidated_{timestamp}.xlsx"
        
        # generate time-series data for all targets
        print("\nGenerating per-target time-series data...")
        target_timeseries = self._create_target_timeseries_data(metrics)
        print(f"  Generated time-series for {len(target_timeseries)} targets")
        
        print("\nCreating Excel tabs:")
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            
            # tab 0: methodology
            print("  - 0_METHODOLOGY")
            methodology_df = self._create_methodology_dataframe()
            methodology_df.to_excel(writer, sheet_name='0_METHODOLOGY', index=False)
            
            # tab 1: summary (composite scores)
            if len(composite_df) > 0:
                print(f"  - 1_SUMMARY ({len(composite_df)} targets)")
                composite_df.to_excel(writer, sheet_name='1_SUMMARY', index=False)
            else:
                print(f"  - 1_SUMMARY (SKIPPED - no composite scores)")
            
            # tab 2: target comparison (aggregate view)
            if len(comparison_df) > 0:
                print(f"  - 2_TARGET_COMPARISON ({len(comparison_df)} targets)")
                comparison_df.to_excel(writer, sheet_name='2_TARGET_COMPARISON', index=False)
            else:
                print(f"  - 2_TARGET_COMPARISON (SKIPPED - no comparison data)")
            
            # tab 3: all quarterly detail
            if 'full' in metrics and len(metrics['full']) > 0:
                metrics['full'].to_excel(writer, sheet_name='3_QUARTERLY_DETAIL', index=False)
            
            # tab 4: rolling trends
            if len(rolling_df) > 0:
                rolling_df.to_excel(writer, sheet_name='4_ROLLING_TRENDS', index=False)
            
            # tab 5-8: analysis tabs
            if len(cause_sim_df) > 0:
                cause_sim_df.to_excel(writer, sheet_name='5_CAUSE_ANALYSIS', index=False)
            
            if len(effect_sim_df) > 0:
                effect_sim_df.to_excel(writer, sheet_name='6_EFFECT_ANALYSIS', index=False)
            
            if len(coherence_df) > 0:
                coherence_df.to_excel(writer, sheet_name='7_COHERENCE', index=False)
            
            if len(consistency_df) > 0:
                consistency_df.to_excel(writer, sheet_name='8_CONSISTENCY', index=False)
            
            # tab 9: metadata
            if len(metadata_dist_df) > 0:
                metadata_dist_df.to_excel(writer, sheet_name='9_METADATA', index=False)
            
            # tab 10: section levels
            level_data = []
            for level_name in ['full', 'management', 'prepared', 'qna']:
                if level_name in metrics and len(metrics[level_name]) > 0:
                    level_df = metrics[level_name].copy()
                    level_data.append(level_df)
            if level_data:
                combined_levels = pd.concat(level_data, ignore_index=True)
                combined_levels.to_excel(writer, sheet_name='10_SECTION_LEVELS', index=False)
            
            # tab 11: lag comparison
            if len(lag_comparison_df) > 0:
                lag_comparison_df.to_excel(writer, sheet_name='11_LAG_COMPARISON', index=False)
            
            # tab 12: peer validation
            if len(similarity_df) > 0:
                similarity_df.to_excel(writer, sheet_name='12_PEER_VALIDATION', index=False)
            
            # tab 13: signal decomposition (aggregate by target)
            print("  - 13_SIGNAL_DECOMPOSITION")
            if len(signal_decomposition_df) > 0:
                signal_decomposition_df.to_excel(writer, sheet_name='13_SIGNAL_DECOMP', index=False)
            
            # new validation tabs (14-20)
            tab_count = 13
            
            if validation_results:
                print("\n   Adding Bias Validation Tabs:")
                
                # exact window
                if 'exact' in validation_results:
                    tab_count += 1
                    if len(validation_results['exact'].get('target_vs_peer', pd.DataFrame())) > 0:
                        print(f"  - {tab_count}_VALIDATION_EXACT_TvP")
                        validation_results['exact']['target_vs_peer'].to_excel(
                            writer, sheet_name=f'{tab_count}_VAL_EXACT_TvP', index=False)
                    
                    tab_count += 1
                    if len(validation_results['exact'].get('target_bias_vs_normal', pd.DataFrame())) > 0:
                        print(f"  - {tab_count}_VALIDATION_EXACT_BiasVsNormal")
                        validation_results['exact']['target_bias_vs_normal'].to_excel(
                            writer, sheet_name=f'{tab_count}_VAL_EXACT_BvN', index=False)
                    
                    tab_count += 1
                    if len(validation_results['exact'].get('bias_type_analysis', pd.DataFrame())) > 0:
                        print(f"  - {tab_count}_VALIDATION_BY_BIAS_TYPE")
                        validation_results['exact']['bias_type_analysis'].to_excel(
                            writer, sheet_name=f'{tab_count}_VAL_BY_TYPE', index=False)
                
                # window 1 (±1q)
                if 'window1' in validation_results:
                    tab_count += 1
                    if len(validation_results['window1'].get('target_vs_peer', pd.DataFrame())) > 0:
                        print(f"  - {tab_count}_VALIDATION_W1_TvP")
                        validation_results['window1']['target_vs_peer'].to_excel(
                            writer, sheet_name=f'{tab_count}_VAL_W1_TvP', index=False)
                    
                    tab_count += 1
                    if len(validation_results['window1'].get('target_bias_vs_normal', pd.DataFrame())) > 0:
                        print(f"  - {tab_count}_VALIDATION_W1_BvN")
                        validation_results['window1']['target_bias_vs_normal'].to_excel(
                            writer, sheet_name=f'{tab_count}_VAL_W1_BvN', index=False)
                
                # window 2 (±2q)
                if 'window2' in validation_results:
                    tab_count += 1
                    if len(validation_results['window2'].get('target_vs_peer', pd.DataFrame())) > 0:
                        print(f"  - {tab_count}_VALIDATION_W2_TvP")
                        validation_results['window2']['target_vs_peer'].to_excel(
                            writer, sheet_name=f'{tab_count}_VAL_W2_TvP', index=False)
                    
                    tab_count += 1
                    if len(validation_results['window2'].get('target_bias_vs_normal', pd.DataFrame())) > 0:
                        print(f"  - {tab_count}_VALIDATION_W2_BvN")
                        validation_results['window2']['target_bias_vs_normal'].to_excel(
                            writer, sheet_name=f'{tab_count}_VAL_W2_BvN', index=False)
            
            # attribution comparison tab
            if attribution_comparison_df is not None and len(attribution_comparison_df) > 0:
                tab_count += 1
                print(f"  - {tab_count}_ATTRIBUTION_COMPARISON")
                attribution_comparison_df.to_excel(writer, sheet_name=f'{tab_count}_ATTR_COMP', index=False)
            
            # per-target tabs
            print(f"\nCreating per-target tabs:")

            targets_with_tabs = []
            targets_without_tabs = []
            
            for target, target_ts_df in target_timeseries.items():
                # clean target name for sheet name (max 31 chars, no special chars)
                clean_target = target.replace('.', '_').replace(' ', '_')[:25]
                
                if len(target_ts_df) > 0:
                    # time-series tab for this target
                    sheet_name = f"T_{clean_target}_timeseries"
                    target_ts_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    tab_count += 1
                    
                    # peer detail tab for this target
                    peer_detail_df = self._create_target_peer_detail(metrics, target)
                    if len(peer_detail_df) > 0:
                        sheet_name = f"T_{clean_target}_peers"
                        peer_detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        tab_count += 1
                    
                    # signal tracking tab for this target
                    if target in target_signals:
                        signal_df = target_signals[target]
                        if len(signal_df) > 0:
                            sheet_name = f"T_{clean_target}_signals"
                            signal_df.to_excel(writer, sheet_name=sheet_name, index=False)
                            tab_count += 1
                            print(f"   {target}: Created timeseries + peers + signals tabs ({len(peer_detail_df)} peers)")
                        else:
                            print(f"  ~ {target}: Created timeseries + peers tabs (no signal data)")
                    else:
                        print(f"  ~ {target}: Created timeseries + peers tabs (no signal tracking)")
                    targets_with_tabs.append(target)
                else:
                    print(f"   {target}: SKIPPED (no time-series data)")
                    targets_without_tabs.append(target)
            
            if targets_without_tabs:
                print(f"\n {len(targets_without_tabs)} targets excluded from per-target tabs:")
                print(f"  {', '.join(targets_without_tabs)}")
        
        print(f" Saved consolidated workbook: {excel_file.name}")
        print(f"  Total tabs: {tab_count}")
        print(f"  Includes: Methodology, Summary, Signal Decomposition, 11 analysis tabs")
        print(f"  Plus {len(target_timeseries)*3} target-specific tabs (timeseries + peers + signals per target)")
        print(f"  Main summary tab: 1_SUMMARY (composite scores)")
        print(f"  Signal analysis tab: 13_SIGNAL_DECOMP (which signals fire for each target)")
        print(f"  Per-target tabs: T_<company>_timeseries, T_<company>_peers, T_<company>_signals")
    
    def save_results(self, 
                    similarity_df: pd.DataFrame,
                    metrics: Dict[str, pd.DataFrame],
                    rolling_df: pd.DataFrame,
                    cause_sim_df: pd.DataFrame,
                    effect_sim_df: pd.DataFrame,
                    metadata_dist_df: pd.DataFrame,
                    consistency_df: pd.DataFrame,
                    lag_comparison_df: pd.DataFrame,
                    coherence_df: pd.DataFrame,
                    comparison_df: pd.DataFrame,
                    composite_df: pd.DataFrame,
                    timestamp: str,
                    validation_results: Optional[Dict] = None,
                    distribution_results: Optional[Dict] = None,
                    attribution_comparison_df: Optional[pd.DataFrame] = None):
        
        print("\n14. Saving Individual CSV Files")
        print("=" * 80)
        
        outputs = [
            (composite_df, "composite_bias_scores"),
            (similarity_df, "peer_similarity_validation"),
            (cause_sim_df, "cause_similarity"),
            (effect_sim_df, "effect_similarity"),
            (metadata_dist_df, "bias_metadata_distribution"),
            (consistency_df, "temporal_consistency"),
            (lag_comparison_df, "quarterly_comparison_with_lag"),
            (coherence_df, "cause_effect_coherence"),
            (rolling_df, "rolling_metrics"),
            (comparison_df, "target_peer_comparison")
        ]
        
        # add new validation results
        if validation_results:
            for window, data in validation_results.items():
                if 'target_vs_peer' in data and len(data['target_vs_peer']) > 0:
                    outputs.append((data['target_vs_peer'], f"validation_target_vs_peer_{window}"))
                if 'target_bias_vs_normal' in data and len(data['target_bias_vs_normal']) > 0:
                    outputs.append((data['target_bias_vs_normal'], f"validation_target_bias_vs_normal_{window}"))
                if 'bias_type_analysis' in data and len(data['bias_type_analysis']) > 0:
                    outputs.append((data['bias_type_analysis'], f"validation_by_bias_type_{window}"))
        
        # add attribution comparison
        if attribution_comparison_df is not None and len(attribution_comparison_df) > 0:
            outputs.append((attribution_comparison_df, "attribution_distribution_comparison"))
        
        for df, name in outputs:
            if len(df) > 0:
                file = self.output_dir / f"{name}_{timestamp}.csv"
                df.to_csv(file, index=False)
                print(f"Saved: {file.name}")
        
        for level_name, level_df in metrics.items():
            if len(level_df) > 0:
                file = self.output_dir / f"bias_metrics_{level_name}_{timestamp}.csv"
                level_df.to_csv(file, index=False)
                print(f"Saved: {file.name}")
        
        if len(comparison_df) > 0:
            latest_file = self.output_dir / "target_peer_comparison_latest.csv"
            comparison_df.to_csv(latest_file, index=False)
            print(f"Saved: {latest_file.name}")
        
        summary = {
            'timestamp': timestamp,
            'targets_analyzed': len(comparison_df) if len(comparison_df) > 0 else 0,
            'extreme_bias_count': int((composite_df['composite_score'] >= 12).sum()) if len(composite_df) > 0 else 0,
            'high_bias_count': int(((composite_df['composite_score'] >= 8) & (composite_df['composite_score'] < 12)).sum()) if len(composite_df) > 0 else 0,
            'moderate_bias_count': int(((composite_df['composite_score'] >= 5) & (composite_df['composite_score'] < 8)).sum()) if len(composite_df) > 0 else 0,
            'low_bias_count': int((composite_df['composite_score'] < 5).sum()) if len(composite_df) > 0 else 0,
            'mean_composite_score': float(composite_df['composite_score'].mean()) if len(composite_df) > 0 else 0,
            'mean_z_score': float(comparison_df['z_score'].mean()) if len(comparison_df) > 0 else 0,
            'low_consistency_count': int((consistency_df['narrative_stability'] < 0.3).sum()) if len(consistency_df) > 0 else 0,
            'suspect_cause_effect_pairs': int(coherence_df['is_suspect'].sum()) if len(coherence_df) > 0 else 0
        }
        
        summary_file = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {summary_file.name}")
        
        # save validation summaries
        if validation_results:
            for window, data in validation_results.items():
                if 'summary' in data:
                    val_summary_file = self.output_dir / f"validation_summary_{window}_{timestamp}.json"
                    # convert any non-serializable objects (recursive)
                    def make_serializable(obj):
                        """Recursively convert numpy types to Python types for JSON serialization"""
                        if isinstance(obj, dict):
                            return {k: make_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, (list, tuple)):
                            return [make_serializable(item) for item in obj]
                        elif isinstance(obj, (np.integer, np.int64, np.int32)):
                            return int(obj)
                        elif isinstance(obj, (np.floating, np.float64, np.float32)):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif pd.isna(obj):
                            return None
                        else:
                            return obj
                    
                    serializable_summary = make_serializable(data['summary'])
                    
                    with open(val_summary_file, 'w') as f:
                        json.dump(serializable_summary, f, indent=2)
                    print(f"Saved: {val_summary_file.name}")
        
        print(f"\n Saved visualization PNG files to: {self.viz_dir}")
        print(f"  (Generated during distribution analysis)")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Visualizations saved to: {self.viz_dir}")
    
    def run_full_analysis(self):
        """
        Main analysis pipeline - refactored to validate bias periods rather than derive them.
        
        Flow:
        1. Load data with bias period flags (TWO datasets: filtered for attribution, unfiltered for general labels)
        2. Distribution analysis (GPT labels, Target vs Peer)
        3. Calculate metrics (with SAB proportion, detailed rates)
        4. Bias period validation (exact, ±1Q, ±2Q, ±4Q, -4Q, +4Q windows)
        5. Traditional analyses (rolling, coherence, etc.)
        6. Save all results
        """
        # step 1: load data (includes bias flags) - returns two dataframes
        df_attribution, df_all_labels = self.load_attribution_data()
        
        if len(df_attribution) == 0 and len(df_all_labels) == 0:
            print("\nNo data available")
            return
        
        # step 2: distribution analysis (new - uses both datasets)
        print("\n" + "="*80)
        print("DISTRIBUTION ANALYSIS")
        print("="*80)
        
        dist_results = self.analyze_gpt_label_distributions(df_attribution, df_all_labels)
        attribution_comparison_df = self.analyze_attribution_outcome_distributions(df_attribution)
        
        # step 3: calculate metrics (enhanced with sab proportion - uses filtered data)
        # note: peer similarity validation removed (attribution rates not a valid proxy for business similarity)
        similarity_df = pd.DataFrame()
        metrics = self.calculate_bias_metrics_multilevel(df_attribution)
        
        rolling_df = pd.DataFrame()
        if 'full' in metrics and len(metrics['full']) > 0:
            rolling_df = self.calculate_rolling_metrics(metrics['full'])
            # create rolling time series visualizations
            self.create_rolling_timeseries_visualizations(rolling_df, metrics['full'])
        
        # step 4: bias period validation (new - core analysis)
        print("\n" + "="*80)
        print("BIAS PERIOD VALIDATION")
        print("="*80)
        print("Running validation across 6 window sizes (exact, ±1Q, ±2Q, ±4Q, -4Q pre, +4Q post)")
        
        # run for exact periods
        target_vs_peer_exact = self.compare_target_bias_vs_peers_same_period(metrics, window='exact')
        target_bias_vs_normal_exact = self.compare_target_bias_vs_target_normal(metrics, window='exact')
        
        # run for ±1q window
        target_vs_peer_w1 = self.compare_target_bias_vs_peers_same_period(metrics, window='window1')
        target_bias_vs_normal_w1 = self.compare_target_bias_vs_target_normal(metrics, window='window1')
        
        # run for ±2q window
        target_vs_peer_w2 = self.compare_target_bias_vs_peers_same_period(metrics, window='window2')
        target_bias_vs_normal_w2 = self.compare_target_bias_vs_target_normal(metrics, window='window2')
        
        # run for ±4q window
        target_vs_peer_w4 = self.compare_target_bias_vs_peers_same_period(metrics, window='window4')
        target_bias_vs_normal_w4 = self.compare_target_bias_vs_target_normal(metrics, window='window4')
        
        # run for -4q pre-bias period (to detect early emergence)
        target_vs_peer_pre4 = self.compare_target_bias_vs_peers_same_period(metrics, window='pre4')
        target_bias_vs_normal_pre4 = self.compare_target_bias_vs_target_normal(metrics, window='pre4')
        
        # run for +4q post-bias period (to detect persistence)
        target_vs_peer_post4 = self.compare_target_bias_vs_peers_same_period(metrics, window='post4')
        target_bias_vs_normal_post4 = self.compare_target_bias_vs_target_normal(metrics, window='post4')
        
        # aggregate validation results for each window
        validation_summary_exact = self.aggregate_validation_results(
            target_vs_peer_exact, target_bias_vs_normal_exact
        )
        validation_summary_w1 = self.aggregate_validation_results(
            target_vs_peer_w1, target_bias_vs_normal_w1
        )
        validation_summary_w2 = self.aggregate_validation_results(
            target_vs_peer_w2, target_bias_vs_normal_w2
        )
        validation_summary_w4 = self.aggregate_validation_results(
            target_vs_peer_w4, target_bias_vs_normal_w4
        )
        validation_summary_pre4 = self.aggregate_validation_results(
            target_vs_peer_pre4, target_bias_vs_normal_pre4
        )
        validation_summary_post4 = self.aggregate_validation_results(
            target_vs_peer_post4, target_bias_vs_normal_post4
        )
        
        # analyze by bias type
        bias_type_analysis_exact = self.analyze_by_bias_type(target_vs_peer_exact)
        
        # step 5: additional analyses
        # note: removed methods that weren't providing actionable insights.
        # empty dataframes maintained for compatibility with downstream methods.
        cause_sim_df = pd.DataFrame()
        effect_sim_df = pd.DataFrame()
        consistency_df = pd.DataFrame()
        coherence_df = pd.DataFrame()
        
        # use filtered data for metadata analysis (attribution-specific)
        metadata_dist_df = self.analyze_bias_metadata_distribution(df_attribution, metrics)
        
        lag_comparison_df = self.compare_targets_vs_peers_with_lag(
            metrics, cause_sim_df, effect_sim_df, consistency_df, coherence_df, metadata_dist_df
        )
        comparison_df = self.compare_targets_vs_peers(metrics)
        
        #  composite scoring 
        composite_df = self.calculate_composite_bias_scores(
            comparison_df,
            cause_sim_df,
            effect_sim_df,
            consistency_df,
            coherence_df,
            metadata_dist_df
        )
        
        # generate signal decomposition analysis
        print("\n" + "="*80)
        print("Generating Signal Decomposition Analysis")
        print("="*80)
        
        signal_decomposition_df = self._create_signal_decomposition_table(
            metrics,
            cause_sim_df,
            effect_sim_df,
            consistency_df,
            coherence_df,
            comparison_df
        )
        
        target_signals = self._create_quarterly_signal_tracking(
            metrics,
            cause_sim_df,
            effect_sim_df,
            consistency_df,
            coherence_df
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # create comprehensive excel workbook with new validation tabs
        self.create_excel_workbook(
            composite_df,
            comparison_df,
            metrics,
            rolling_df,
            cause_sim_df,
            effect_sim_df,
            coherence_df,
            consistency_df,
            metadata_dist_df,
            lag_comparison_df,
            similarity_df,
            signal_decomposition_df,
            target_signals,
            timestamp,
            # new validation results
            validation_results={
                'exact': {
                    'target_vs_peer': target_vs_peer_exact,
                    'target_bias_vs_normal': target_bias_vs_normal_exact,
                    'summary': validation_summary_exact,
                    'bias_type_analysis': bias_type_analysis_exact
                },
                'window1': {
                    'target_vs_peer': target_vs_peer_w1,
                    'target_bias_vs_normal': target_bias_vs_normal_w1,
                    'summary': validation_summary_w1
                },
                'window2': {
                    'target_vs_peer': target_vs_peer_w2,
                    'target_bias_vs_normal': target_bias_vs_normal_w2,
                    'summary': validation_summary_w2
                },
                'window4': {
                    'target_vs_peer': target_vs_peer_w4,
                    'target_bias_vs_normal': target_bias_vs_normal_w4,
                    'summary': validation_summary_w4
                },
                'pre4': {
                    'target_vs_peer': target_vs_peer_pre4,
                    'target_bias_vs_normal': target_bias_vs_normal_pre4,
                    'summary': validation_summary_pre4
                },
                'post4': {
                    'target_vs_peer': target_vs_peer_post4,
                    'target_bias_vs_normal': target_bias_vs_normal_post4,
                    'summary': validation_summary_post4
                }
            },
            distribution_results=dist_results,
            attribution_comparison_df=attribution_comparison_df
        )
        
        # save all results
        self.save_results(
            similarity_df, 
            metrics, 
            rolling_df, 
            cause_sim_df, 
            effect_sim_df,
            metadata_dist_df,
            consistency_df,
            lag_comparison_df,
            coherence_df,
            comparison_df,
            composite_df,
            timestamp,
            # new validation results
            validation_results={
                'exact': {
                    'target_vs_peer': target_vs_peer_exact,
                    'target_bias_vs_normal': target_bias_vs_normal_exact,
                    'summary': validation_summary_exact,
                    'bias_type_analysis': bias_type_analysis_exact
                },
                'window1': {
                    'target_vs_peer': target_vs_peer_w1,
                    'target_bias_vs_normal': target_bias_vs_normal_w1,
                    'summary': validation_summary_w1
                },
                'window2': {
                    'target_vs_peer': target_vs_peer_w2,
                    'target_bias_vs_normal': target_bias_vs_normal_w2,
                    'summary': validation_summary_w2
                },
                'window4': {
                    'target_vs_peer': target_vs_peer_w4,
                    'target_bias_vs_normal': target_bias_vs_normal_w4,
                    'summary': validation_summary_w4
                },
                'pre4': {
                    'target_vs_peer': target_vs_peer_pre4,
                    'target_bias_vs_normal': target_bias_vs_normal_pre4,
                    'summary': validation_summary_pre4
                },
                'post4': {
                    'target_vs_peer': target_vs_peer_post4,
                    'target_bias_vs_normal': target_bias_vs_normal_post4,
                    'summary': validation_summary_post4
                }
            },
            distribution_results=dist_results,
            attribution_comparison_df=attribution_comparison_df
        )


class TeeOutput:
    """Redirect stdout to both console and file"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

def main():
    parser = argparse.ArgumentParser(description='Peer Benchmark Analysis for Attribution Bias')
    parser.add_argument('--input', type=str, default='classification_results',
                       help='Input directory with instance folders')
    parser.add_argument('--config', type=str, default='company_config.json',
                       help='Path to company config with peer groups')
    parser.add_argument('--output', type=str, default='output/gpt_classification_validation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # setup output logging (captures all console output to file)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = output_dir / f"analysis_log_{timestamp}.txt"
    
    # open log file and redirect stdout
    log_file = open(log_file_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(sys.stdout, log_file)
    
    try:
        print("\n" + "=" * 80)
        print("PEER BENCHMARK ANALYSIS")
        print(f"Log file: {log_file_path}")
        print("=" * 80)
        
        benchmark = PeerBenchmark(
            input_dir=args.input,
            config_path=args.config,
            output_dir=args.output
        )
        
        benchmark.run_full_analysis()
        
        print(f"\n Full analysis log saved to: {log_file_path}")
    
    finally:
        # restore stdout and close log file
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    main()


#    python conclusion/peer_benchmark.py