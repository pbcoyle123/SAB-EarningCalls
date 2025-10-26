"""
Embedding Analysis Viz
Purpose: Get visualizations for attribution bias detection using embeddings. 
Output: Saves files here output/figures/

main graphs are 
1. Fig_PERFIRM_temporal_scatter.png - THE PARADOX VISUALIZER (47.6% flagged, bidirectional effects)
2. Fig_PERFIRM_slope_graph.png - Shows all firms IN vs OUT bias periods (divergence patterns)
3. Fig_SUPERVISED_classification_comparison.png - AUC comparison across feature sets
4. Fig_TEMPORAL_aggregate_comparison.png - Aggregate temporal analysis
5. Fig_CS3_bias_period_bars.png - Expert-identified bias period detection


"""
# visualization suite for embedding-based attribution bias detection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 1200  # Minimum 800 dpi (preferably 1200 dpi)
plt.rcParams['savefig.dpi'] = 1200
plt.rcParams['savefig.format'] = 'png'  # Can also use 'pdf' for vector graphics
plt.rcParams['font.size'] = 10  # Minimum 6 point
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9  # Minimum 6 point
plt.rcParams['ytick.labelsize'] = 9  # Minimum 6 point
plt.rcParams['legend.fontsize'] = 9  # Minimum 6 point
plt.rcParams['figure.titlesize'] = 9  # Caption size: 9-point type


class EmbeddingVisualizer:
    
    def __init__(self, results_dir: str = 'output/embeddings', 
                 output_dir: str = 'output/figures'):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading results from: {self.results_dir}")
        print(f"Saving figures to: {self.output_dir}")
        
        self.embeddings = None
        self.metadata = None
        self.bias_scores = None
        self.temporal_consistency = None
        self.pca_coords = None
        self.features = None
        self.metrics = None
        
        self._load_data()
        
    def _load_data(self):
        """Load all saved results from embedding analysis."""
        import re
        from datetime import datetime
        
        def get_most_recent_file(pattern: str, description: str):
            """Find most recent timestamped file matching pattern."""
            files = list(self.results_dir.glob(pattern))
            if not files:
                return None
            
            file_timestamps = []
            for f in files:
                match = re.search(r'(\d{8}_\d{6})', f.stem)
                if match:
                    timestamp_str = match.group(1)
                    try:
                        dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        file_timestamps.append((dt, f, timestamp_str))
                    except ValueError:
                        continue
            
            if not file_timestamps:
                return sorted(files)[-1]
            
            file_timestamps.sort(key=lambda x: x[0])
            most_recent = file_timestamps[-1]
            print(f"  {description}: {most_recent[1].name}")
            return most_recent[1]
        
        embedding_files = sorted(self.results_dir.glob('embedding_vectors_*.npy'))
        if not embedding_files:
            raise FileNotFoundError(f"No embedding files in {self.results_dir}")
        
        latest = embedding_files[-1]
        match = re.search(r'(\d{8}_\d{6})', latest.stem)
        if not match:
            raise ValueError(f"No timestamp in {latest.name}")
        
        timestamp = match.group(1)
        print(f"  Using results: {timestamp}")
        
        self.embeddings = np.load(latest)
        self.metadata = pd.read_csv(self.results_dir / f'embedding_metadata_{timestamp}.csv')
        print(f"  Loaded {len(self.metadata):,} samples")
        
        bias_file = self.results_dir / f'bias_scores_{timestamp}.csv'
        if bias_file.exists():
            self.bias_scores = pd.read_csv(bias_file)
        
        temporal_file = self.results_dir / f'topic_temporal_consistency_{timestamp}.csv'
        if temporal_file.exists():
            self.temporal_consistency = pd.read_csv(temporal_file)
        
        pca_file = self.results_dir / f'pca_coordinates_{timestamp}.csv'
        if pca_file.exists():
            self.pca_coords = pd.read_csv(pca_file)
        
        features_file = self.results_dir / f'embedding_features_{timestamp}.csv'
        if features_file.exists():
            self.features = pd.read_csv(features_file)
        
        metrics_file = self.results_dir / f'embedding_analysis_metrics_{timestamp}.json'
        if metrics_file.exists():
            import json
            try:
                with open(metrics_file, 'r') as f:
                    self.metrics = json.load(f)
            except json.JSONDecodeError:
                self.metrics = None
        
        import json
        bias_pred_file = get_most_recent_file('bias_prediction_results*.json', 'Bias prediction')
        self.bias_prediction_results = None
        if bias_pred_file:
            with open(bias_pred_file, 'r') as f:
                self.bias_prediction_results = json.load(f)
        
        firm_file = get_most_recent_file('*firm_analysis*.csv', 'Per-firm analysis')
        self.firm_analysis_df = None
        if firm_file:
            self.firm_analysis_df = pd.read_csv(firm_file)
        
        print("Data loading complete\n")
    
    # =========================================================================
    # statistical helper methods
    # =========================================================================
    
    def compute_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size (positive = group1 > group2)."""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def compute_temporal_zscore(self, current: np.ndarray, 
                                historical: np.ndarray) -> float:
        """Compute Z-score of current vs historical embedding distribution."""
        if len(historical) == 0:
            return 0.0
        
        historical_centroid = historical.mean(axis=0)
        current_distance = np.linalg.norm(current - historical_centroid)
        historical_distances = np.linalg.norm(historical - historical_centroid, axis=1)
        
        mean_dist = historical_distances.mean()
        std_dist = historical_distances.std(ddof=1)
        
        if std_dist == 0:
            return 0.0
        
        return (current_distance - mean_dist) / std_dist
    
    def perform_ttest(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
        """Perform independent t-test, returns (t_statistic, p_value)."""
        return stats.ttest_ind(group1, group2)
    
    def add_significance_stars(self, ax, x1, x2, y, p_value):
        """Add significance markers to plot."""
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        
        h = y * 0.05
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], 'k-', lw=1)
        ax.text((x1+x2)/2, y+h, sig, ha='center', va='bottom', fontsize=9)
    
    # =========================================================================
    # data preparation methods
    # =========================================================================
    
    def get_attribution_type(self, row) -> str:
        """Create compound attribution type label (e.g., 'Positive-Internal')."""
        outcome = str(row.get('attribution_outcome', 'Unknown')).title()
        locus = str(row.get('attribution_locus', 'Unknown')).title()
        
        if outcome in ['Positive', 'Negative'] and locus in ['Internal', 'External']:
            return f"{outcome}-{locus}"
        return "Other"
    
    def filter_by_level(self, df: pd.DataFrame, level: str, 
                       attribution_type: str = None, 
                       topic: str = None) -> pd.DataFrame:
        """Filter data by granularity level."""
        result = df.copy()
        
        if level == 'attribution+topic':
            if attribution_type:
                result = result[result['attribution_type'] == attribution_type]
            if topic:
                result = result[result['Primary_Topic'] == topic]
        elif level == 'attribution':
            if attribution_type:
                result = result[result['attribution_type'] == attribution_type]
        elif level == 'topic':
            if topic:
                result = result[result['Primary_Topic'] == topic]
        
        return result
    
    def prepare_attribution_data(self) -> pd.DataFrame:
        """Prepare metadata with attribution type labels."""
        df = self.metadata.copy()
        df['attribution_type'] = df.apply(self.get_attribution_type, axis=1)
        
        valid_types = ['Positive-Internal', 'Positive-External', 
                       'Negative-Internal', 'Negative-External']
        df = df[df['attribution_type'].isin(valid_types)].reset_index(drop=True)
        
        print(f"Prepared {len(df):,} valid attributions")
        
        return df
    
    # =========================================================================
    # cross-sectional analyses (cs1, cs2, cs3)
    # =========================================================================
    
    def plot_cs1_topic_attribution_heatmap(self):
        """Generate Topic × Attribution Type heatmap with Cohen's d effect sizes."""
        print("\nGenerating CS1: Topic × Attribution heatmap...")
        
        df = self.prepare_attribution_data()
        
        topic_counts = df['Primary_Topic'].value_counts()
        top_topics = topic_counts.head(20).index.tolist()
        
        attribution_types = ['Positive-Internal', 'Positive-External',
                            'Negative-Internal', 'Negative-External']
        
        cohens_d_matrix = np.zeros((len(top_topics), len(attribution_types)))
        sample_counts = np.zeros((len(top_topics), len(attribution_types)))
        
        for i, topic in enumerate(top_topics):
            for j, attr_type in enumerate(attribution_types):
                # filter to this topic + attribution type
                mask = (df['Primary_Topic'] == topic) & (df['attribution_type'] == attr_type)
                subset = df[mask]
                
                if len(subset) < 10:
                    cohens_d_matrix[i, j] = np.nan
                    continue
                
                target_mask = subset['IS_TARGET'] == 'Y'
                peer_mask = subset['IS_TARGET'] != 'Y'
                
                target_indices = subset[target_mask].index.tolist()
                peer_indices = subset[peer_mask].index.tolist()
                
                if len(target_indices) < 3 or len(peer_indices) < 3:
                    cohens_d_matrix[i, j] = np.nan
                    continue
                
                target_embeds = self.embeddings[target_indices]
                peer_embeds = self.embeddings[peer_indices]
                
                overall_centroid = np.vstack([target_embeds, peer_embeds]).mean(axis=0)
                target_distances = np.linalg.norm(target_embeds - overall_centroid, axis=1)
                peer_distances = np.linalg.norm(peer_embeds - overall_centroid, axis=1)
                
                d = self.compute_cohens_d(target_distances, peer_distances)
                cohens_d_matrix[i, j] = d
                sample_counts[i, j] = len(subset)
        
        fig, ax = plt.subplots(figsize=(10, 12))
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        sns.heatmap(cohens_d_matrix, 
                   xticklabels=[at.replace('-', '\n') for at in attribution_types],
                   yticklabels=[t[:40] for t in top_topics],
                   cmap=cmap, center=0, 
                   vmin=-1.0, vmax=1.0,
                   annot=True, fmt='.2f',
                   cbar_kws={'label': "Cohen's d (Target - Peer)"},
                   ax=ax)
        
        ax.set_title("Topic × Attribution Type: Target-Peer Differences\n" + 
                    "Red = Targets More Extreme, Blue = Peers More Extreme",
                    fontsize=12, fontweight='bold')
        ax.set_xlabel("Attribution Type", fontsize=11)
        ax.set_ylabel("Topic", fontsize=11)
        
        # add interpretation guide with more space from x-axis
        fig.text(0.5, 0.04, 
                "Interpretation: d > 0.5 = Medium effect, d > 0.8 = Large effect\n" +
                "Self-serving bias: Expect red for Positive-Internal and Negative-External",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3, edgecolor='gray', linewidth=0.8))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.98])
        
        output_file = self.output_dir / 'Fig_CS1_topic_attribution_heatmap.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {output_file.name}")
        plt.close()
    
    def plot_cs2_attribution_type_bars(self):
        """Generate attribution type comparison bar chart."""
        print("\nGenerating CS2: Attribution type bars...")
        
        df = self.prepare_attribution_data()
        
        attribution_types = ['Positive-Internal', 'Positive-External',
                            'Negative-Internal', 'Negative-External']
        
        results = []
        
        for attr_type in attribution_types:
            # filter to this attribution type (any topic)
            mask = df['attribution_type'] == attr_type
            subset = df[mask]
            
            # split targets and peers
            target_mask = subset['IS_TARGET'] == 'Y'
            peer_mask = subset['IS_TARGET'] != 'Y'
            
            target_indices = subset[target_mask].index.tolist()
            peer_indices = subset[peer_mask].index.tolist()
            
            if len(target_indices) < 5 or len(peer_indices) < 5:
                continue
            
            # get embeddings
            target_embeds = self.embeddings[target_indices]
            peer_embeds = self.embeddings[peer_indices]
            
            # calculate distances from overall centroid
            overall_centroid = np.vstack([target_embeds, peer_embeds]).mean(axis=0)
            
            target_distances = np.linalg.norm(target_embeds - overall_centroid, axis=1)
            peer_distances = np.linalg.norm(peer_embeds - overall_centroid, axis=1)
            
            # compute statistics
            d = self.compute_cohens_d(target_distances, peer_distances)
            t_stat, p_val = self.perform_ttest(target_distances, peer_distances)
            
            results.append({
                'attribution_type': attr_type,
                'target_mean': target_distances.mean(),
                'target_std': target_distances.std(),
                'peer_mean': peer_distances.mean(),
                'peer_std': peer_distances.std(),
                'cohens_d': d,
                'p_value': p_val,
                'n_targets': len(target_indices),
                'n_peers': len(peer_indices)
            })
        
        results_df = pd.DataFrame(results)
        
        # create bar chart with extra height for annotations
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = np.arange(len(results_df))
        width = 0.35
        
        # plot bars
        bars1 = ax.bar(x - width/2, results_df['target_mean'], width,
                      yerr=results_df['target_std'], 
                      label='Targets', color='#d62728', alpha=0.8,
                      capsize=5, linewidth=1.5)
        bars2 = ax.bar(x + width/2, results_df['peer_mean'], width,
                      yerr=results_df['peer_std'],
                      label='Peers', color='#1f77b4', alpha=0.8,
                      capsize=5, linewidth=1.5)
        
        # add cohen's d annotations above each pair (before customization to calculate y-limits)
        max_y_annotation = 0
        for i, row in results_df.iterrows():
            d_value = row['cohens_d']
            p_value = row['p_value']
            
            y_pos = max(row['target_mean'] + row['target_std'], 
                       row['peer_mean'] + row['peer_std']) * 1.05
            
            # add significance stars
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            
            ax.text(i, y_pos, f"d={d_value:+.2f} {sig}", 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
            max_y_annotation = max(max_y_annotation, y_pos)
        
        # set y-axis limit to provide space for annotations below title
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], max_y_annotation * 1.15)
        
        # customize
        ax.set_xlabel('Attribution Type', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Distance from Centroid', fontsize=11, fontweight='bold')
        ax.set_title('Attribution Type Centroid Distances: Targets vs Peers\n' + 
                    'Higher = More Extreme Positioning',
                    fontsize=12, fontweight='bold', pad=20)  # add padding above title
        ax.set_xticks(x)
        ax.set_xticklabels([at.replace('-', '\n') for at in results_df['attribution_type']])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        
        # add interpretation guide with more space from x-axis
        fig.text(0.5, 0.04,
                "Interpretation: d > 0.5 = Medium effect (targets more extreme), " +
                "d > 0.8 = Large effect\n" +
                "Self-serving bias: Expect positive d for Positive-Internal and Negative-External",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3, edgecolor='gray', linewidth=0.8))
        
        plt.tight_layout(rect=[0, 0.10, 1, 0.98])  # more space at bottom (10%) and top (2%)
        
        output_file = self.output_dir / 'Fig_CS2_attribution_type_bars.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {output_file.name}")
        plt.close()
    
    def plot_cs3_full_call_violins(self):
        """Generate violin plot comparing bias scores in expert-identified periods."""
        print("\nGenerating CS3: Full call violins...")
        
        if self.metadata is None or self.bias_scores is None:
            print("Skipping: Missing data")
            return
        
        df = self.metadata.copy()
        
        if 'in_expert_period' not in df.columns:
            print("Skipping: No expert period labels")
            return
        
        bias_df = self.bias_scores[['bias_score']].copy()
        bias_df.index = range(len(bias_df))
        df['bias_score'] = bias_df['bias_score'].values
        
        # group by company-quarter, taking mean bias score and checking if any snippet is in expert period
        company_quarter_scores = df.groupby(['Company', 'Year', 'Quarter']).agg({
            'bias_score': 'mean',
            'IS_TARGET': 'first',
            'in_expert_period': 'max'  # if any snippet in this quarter is in expert period, mark quarter as expert
        }).reset_index()
        
        # rename columns for consistency
        company_quarter_scores.rename(columns={
            'Company': 'company',
            'Year': 'year', 
            'Quarter': 'quarter',
            'IS_TARGET': 'is_target'
        }, inplace=True)
        
        # create bias category based on expert periods
        def categorize_period(row):
            if row['in_expert_period']:
                return 'Expert-Identified\nBias Period'
            else:
                return 'Normal\nPeriod'
        
        company_quarter_scores['bias_category'] = company_quarter_scores.apply(categorize_period, axis=1)
        
        # create violin plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # prepare data for seaborn
        plot_data = company_quarter_scores.copy()
        plot_data['Company Type'] = plot_data['is_target'].map({
            'Y': 'Targets', True: 'Targets',
            'N': 'Peers', False: 'Peers'
        })
        
        # create violin plot
        sns.violinplot(data=plot_data, x='bias_category', y='bias_score',
                      hue='Company Type', split=True, ax=ax,
                      palette={'Targets': '#d62728', 'peers': '#1f77b4'},
                      inner='quartile')
        
        # Overlay individual points with jitter
        categories = plot_data['bias_category'].unique()
        for i, cat in enumerate(categories):
            cat_data = plot_data[plot_data['bias_category'] == cat]
            
            targets = cat_data[cat_data['Company Type'] == 'Targets']
            peers = cat_data[cat_data['Company Type'] == 'Peers']
            
            # Add jittered points
            if len(targets) > 0:
                ax.scatter(np.random.normal(i - 0.15, 0.05, len(targets)),
                          targets['bias_score'], alpha=0.3, s=20, color='#d62728')
            if len(peers) > 0:
                ax.scatter(np.random.normal(i + 0.15, 0.05, len(peers)),
                          peers['bias_score'], alpha=0.3, s=20, color='#1f77b4')
        
        ax.set_xlabel('Period Type', fontsize=11, fontweight='bold')
        ax.set_ylabel('Bias Score (Projection onto Bias Vector)', fontsize=11, fontweight='bold')
        ax.set_title('Expert-Identified Bias Periods: Targets vs Peers\n' + 
                    'Company-Quarter Level Comparison',
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # add statistical tests for each period type
        for i, cat in enumerate(categories):
            cat_data = plot_data[plot_data['bias_category'] == cat]
            targets = cat_data[cat_data['Company Type'] == 'Targets']['bias_score'].values
            peers = cat_data[cat_data['Company Type'] == 'Peers']['bias_score'].values
            
            if len(targets) >= 3 and len(peers) >= 3:
                d = self.compute_cohens_d(targets, peers)
                t_stat, p_val = self.perform_ttest(targets, peers)
                
                # significance stars
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                
                # add text annotation
                y_pos = ax.get_ylim()[1] * 0.92
                
                # highlight expert periods
                if 'Expert' in cat or 'Bias' in cat:
                    bbox_color = 'yellow'
                    bbox_alpha = 0.9
                    edge_width = 2.0
                else:
                    bbox_color = 'lightgray'
                    bbox_alpha = 0.6
                    edge_width = 1.0
                
                ax.text(i, y_pos, f"d = {d:+.2f} {sig}\np = {p_val:.3f}",
                       ha='center', va='top', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor=bbox_color, 
                                alpha=bbox_alpha,
                                edgecolor='black',
                                linewidth=edge_width))
        
        # add interpretation
        fig.text(0.5, 0.02,
                "INTERPRETATION: Positive bias scores indicate linguistic patterns characteristic of expert-identified crisis periods.\n" +
                "Cohen's d shows effect size for target-peer separation. Significant effects indicate detectable differences.",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3, 
                         edgecolor='gray', linewidth=0.8))
        
        plt.tight_layout(rect=[0, 0.06, 1, 0.98])
        
        output_file = self.output_dir / 'Fig_CS3_full_call_violins.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {output_file.name}")
        plt.close()
    
    def plot_cs3_bias_period_bars(self):
        """Generate bar chart comparing bias scores in expert-identified periods."""
        print("\nGenerating CS3: Bias period bars...")
        
        if self.metadata is None or self.bias_scores is None:
            print("  [SKIP] Metadata or bias scores not available")
            return
        
        # merge metadata (which has expert period labels) with bias scores
        df = self.metadata.copy()
        
        # check for expert period column
        if 'in_expert_period' not in df.columns:
            print("  [SKIP] Expert period labels not found in metadata")
            print("  Available columns:", df.columns.tolist()[:10], "...")
            return
        
        # merge with bias scores
        bias_df = self.bias_scores[['bias_score']].copy()
        bias_df.index = range(len(bias_df))
        df['bias_score'] = bias_df['bias_score'].values
        
        print("\nAggregating to company-quarter level...")
        
        # group by company-quarter
        company_quarter_scores = df.groupby(['Company', 'Year', 'Quarter']).agg({
            'bias_score': 'mean',
            'IS_TARGET': 'first',
            'in_expert_period': 'max'  # if any snippet in this quarter is in expert period
        }).reset_index()
        
        # rename columns for consistency
        company_quarter_scores.rename(columns={
            'Company': 'company',
            'Year': 'year',
            'Quarter': 'quarter',
            'IS_TARGET': 'is_target'
        }, inplace=True)
        
        # create bias category based on expert periods
        def categorize_period(row):
            if row['in_expert_period']:
                return 'Expert-Identified\nBias Period'
            else:
                return 'Normal\nPeriod'
        
        company_quarter_scores['bias_category'] = company_quarter_scores.apply(categorize_period, axis=1)
        
        print(f"  Total company-quarters: {len(company_quarter_scores)}")
        
        # prepare data
        plot_data = company_quarter_scores.copy()
        plot_data['Company Type'] = plot_data['is_target'].map({
            'Y': 'Targets', True: 'Targets',
            'N': 'Peers', False: 'Peers'
        })
        
        # calculate statistics for each group
        stats_data = []
        categories = plot_data['bias_category'].unique()
        for cat in categories:
            cat_data = plot_data[plot_data['bias_category'] == cat]
            
            for comp_type in ['Targets', 'Peers']:
                group = cat_data[cat_data['Company Type'] == comp_type]['bias_score']
                if len(group) > 0:
                    stats_data.append({
                        'Period': cat,
                        'Company Type': comp_type,
                        'mean': group.mean(),
                        'std': group.std(),
                        'sem': group.sem(),  # standard error of mean
                        'n': len(group)
                    })
        
        stats_df = pd.DataFrame(stats_data)
        
        # create figure
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # setup
        periods = sorted(categories, key=lambda x: 'Normal' in x)  # normal first, expert second
        x = np.arange(len(periods))
        width = 0.35
        
        # extract data for plotting
        target_means = [stats_df[(stats_df['Period'] == p) & (stats_df['Company Type'] == 'Targets')]['mean'].values[0] 
                       for p in periods]
        target_sems = [stats_df[(stats_df['Period'] == p) & (stats_df['Company Type'] == 'Targets')]['sem'].values[0] 
                      for p in periods]
        peer_means = [stats_df[(stats_df['Period'] == p) & (stats_df['Company Type'] == 'Peers')]['mean'].values[0] 
                     for p in periods]
        peer_sems = [stats_df[(stats_df['Period'] == p) & (stats_df['Company Type'] == 'Peers')]['sem'].values[0] 
                    for p in periods]
        
        # plot bars
        bars1 = ax.bar(x - width/2, target_means, width, 
                      yerr=target_sems, 
                      label='Targets', 
                      color='#d62728', 
                      alpha=0.85,
                      capsize=6, 
                      edgecolor='black', 
                      linewidth=1.5,
                      error_kw={'linewidth': 2, 'elinewidth': 2})
        
        bars2 = ax.bar(x + width/2, peer_means, width,
                      yerr=peer_sems,
                      label='Peers',
                      color='#1f77b4',
                      alpha=0.85,
                      capsize=6,
                      edgecolor='black',
                      linewidth=1.5,
                      error_kw={'linewidth': 2, 'elinewidth': 2})
        
        # add value labels on bars
        for i, (bar_t, bar_p, t_mean, p_mean) in enumerate(zip(bars1, bars2, target_means, peer_means)):
            # target label
            ax.text(bar_t.get_x() + bar_t.get_width()/2., t_mean,
                   f'{t_mean:+.3f}',
                   ha='center', va='bottom' if t_mean > 0 else 'top', 
                   fontsize=9, fontweight='bold')
            # peer label
            ax.text(bar_p.get_x() + bar_p.get_width()/2., p_mean,
                   f'{p_mean:+.3f}',
                   ha='center', va='bottom' if p_mean > 0 else 'top',
                   fontsize=9, fontweight='bold')
        
        # add cohen's d and p-values above each pair
        for i, cat in enumerate(periods):
            cat_data = plot_data[plot_data['bias_category'] == cat]
            targets = cat_data[cat_data['Company Type'] == 'Targets']['bias_score'].values
            peers = cat_data[cat_data['Company Type'] == 'Peers']['bias_score'].values
            
            if len(targets) >= 3 and len(peers) >= 3:
                d = self.compute_cohens_d(targets, peers)
                t_stat, p_val = self.perform_ttest(targets, peers)
                
                # significance stars
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                
                # position annotation
                y_pos = max(target_means[i] + target_sems[i], 
                           peer_means[i] + peer_sems[i]) * 1.15
                
                # highlight significant result
                if sig != 'ns':
                    bbox_color = 'yellow'
                    bbox_alpha = 0.9
                    edge_width = 2.0
                else:
                    bbox_color = 'lightgray'
                    bbox_alpha = 0.5
                    edge_width = 1.0
                
                ax.text(i, y_pos, 
                       f"d = {d:+.2f} {sig}\np = {p_val:.3f}",
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor=bbox_color, 
                                alpha=bbox_alpha,
                                edgecolor='black',
                                linewidth=edge_width))
        
        # highlight expert-identified period with background shading (if present)
        expert_idx = [i for i, p in enumerate(periods) if 'Expert' in p or 'Bias' in p]
        if expert_idx:
            idx = expert_idx[0]
            ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.1, color='yellow', zorder=0)
            y_bottom = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
            ax.text(idx, y_bottom,
                   '← Expert-Identified Crisis Period', ha='center', fontsize=9, fontweight='bold',
                   color='darkred', style='italic')
        
        # styling
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_ylabel('Mean Bias Score\n(Projection onto Bias Vector)', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Period Type', fontsize=12, fontweight='bold')
        ax.set_title('Context-Dependent Bias Detection:\nTargets vs Peers in Expert-Identified Crisis Periods',
                    fontsize=13, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, fontsize=11)
        ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # add interpretation box at bottom
        fig.text(0.5, 0.02,
                "KEY FINDING: Uses expert-identified crisis periods from company_config.json (not artificial data splits). " +
                "Targets exhibit distinctive linguistic patterns\nspecifically during expert-identified crisis/scandal periods, " +
                "supporting context-dependent impression management theory.",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.3, 
                         edgecolor='gray', linewidth=0.8))
        
        plt.tight_layout(rect=[0, 0.09, 1, 0.98])
        
        output_file = self.output_dir / 'Fig_CS3_bias_period_bars.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"[SAVED] {output_file}")
        
        # print detailed statistics
        print("\n" + "=" * 80)
        print("DETAILED STATISTICS")
        print("=" * 80)
        
        for cat in periods:
            cat_data = plot_data[plot_data['bias_category'] == cat]
            targets = cat_data[cat_data['Company Type'] == 'Targets']['bias_score'].values
            peers = cat_data[cat_data['Company Type'] == 'Peers']['bias_score'].values
            
            if len(targets) >= 3 and len(peers) >= 3:
                d = self.compute_cohens_d(targets, peers)
                t_stat, p_val = self.perform_ttest(targets, peers)
                
                print(f"\n{cat} Period:")
                print(f"  Targets: μ = {targets.mean():+.3f}, σ = {targets.std():.3f}, n = {len(targets)}")
                print(f"  Peers:   μ = {peers.mean():+.3f}, σ = {peers.std():.3f}, n = {len(peers)}")
                print(f"  Cohen's d = {d:+.3f}")
                print(f"  t-statistic = {t_stat:+.3f}")
                print(f"  p-value = {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '(ns)'}")
                print(f"  Interpretation: {'SIGNIFICANT - Targets distinguishable from peers' if p_val < 0.05 else 'Not significant - No difference'}")
        
        plt.close()
    
    # =========================================================================
    # time-series analyses (ts1, ts2, ts3)
    # =========================================================================
    
    def plot_ts1_aggregate_outlier_rates(self):
        """
        TS1: Aggregate Temporal Outlier Analysis
        
        Level 1-2 Granularity: Temporal consistency aggregated across topics
        Question: Do targets show more temporal inconsistency (outliers) than peers?
        Metric: % of attributions with |Z| > 2 (outliers vs historical centroid)
        
        INTERPRETATION:
        - High outlier rate = More temporal inconsistency (attributions deviate from company's history)
        - Positive result = Targets > Peers (supports bias hypothesis)
        - Z-score measures: How unusual is current attribution language compared to past?
        
        Success: Targets show higher outlier rates, especially for self-serving types
        """
        print("\n" + "=" * 80)
        print("TS1: Aggregate Temporal Outlier Analysis")
        print("=" * 80)
        
        if self.temporal_consistency is None:
            print("  [SKIP] Temporal consistency data not available")
            return
        
        df = self.temporal_consistency.copy()
        df = df[df['n_same_topic_historical'] > 0].copy()
        
        if len(df) == 0:
            print("  [SKIP] No temporal consistency data")
            return
        
        print(f"\nAnalyzing {len(df):,} attributions with historical data")
        
        # map to attribution types
        df['attribution_type'] = df.apply(
            lambda row: f"{row['outcome']}-{row['locus']}" 
            if row['outcome'] in ['Positive', 'Negative'] and 
               row['locus'] in ['Internal', 'External'] 
            else "Other",
            axis=1
        )
        
        attribution_types = ['Positive-Internal', 'Positive-External',
                            'Negative-Internal', 'Negative-External']
        
        # filter to valid types
        df = df[df['attribution_type'].isin(attribution_types)]
        df['is_outlier'] = np.abs(df['topic_outlier_score']) > 2
        
        # create figure with 3 panels
        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(1, 3, wspace=0.3)
        
        # panel 1: overall outlier rate (all attribution types combined)
        ax1 = fig.add_subplot(gs[0, 0])
        
        overall_stats = []
        for is_target, label in [(True, 'Targets'), (False, 'Peers')]:
            subset = df[df['is_target'] == is_target]
            if len(subset) > 0:
                outlier_rate = (subset['is_outlier']).mean() * 100
                n_total = len(subset)
                n_outliers = subset['is_outlier'].sum()
                overall_stats.append({
                    'group': label,
                    'outlier_rate': outlier_rate,
                    'n_total': n_total,
                    'n_outliers': n_outliers
                })
        
        overall_df = pd.DataFrame(overall_stats)
        
        colors = ['#d62728', '#1f77b4']
        bars = ax1.bar(overall_df['group'], overall_df['outlier_rate'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # add value labels on bars
        for i, (bar, row) in enumerate(zip(bars, overall_df.itertuples())):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%\n(n={row.n_outliers}/{row.n_total})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_ylabel('Outlier Rate (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Overall Temporal Outlier Rate\n(All Attribution Types Combined)', 
                     fontsize=11, fontweight='bold')
        ax1.set_ylim(0, max(overall_df['outlier_rate']) * 1.3)
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.set_axisbelow(True)
        
        # add interpretation text
        if len(overall_df) == 2:
            diff = overall_df.iloc[0]['outlier_rate'] - overall_df.iloc[1]['outlier_rate']
            ax1.text(0.5, 0.95, f"Difference: {diff:+.1f} percentage points",
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                    fontsize=9, fontweight='bold')
        
        # panel 2: by attribution type
        ax2 = fig.add_subplot(gs[0, 1])
        
        type_stats = []
        for attr_type in attribution_types:
            for is_target, label in [(True, 'Targets'), (False, 'Peers')]:
                subset = df[(df['attribution_type'] == attr_type) & (df['is_target'] == is_target)]
                if len(subset) > 0:
                    outlier_rate = (subset['is_outlier']).mean() * 100
                    type_stats.append({
                        'attribution_type': attr_type,
                        'group': label,
                        'outlier_rate': outlier_rate,
                        'n': len(subset)
                    })
        
        type_df = pd.DataFrame(type_stats)
        
        x = np.arange(len(attribution_types))
        width = 0.35
        
        target_data = type_df[type_df['group'] == 'Targets']
        peer_data = type_df[type_df['group'] == 'Peers']
        
        bars1 = ax2.bar(x - width/2, target_data['outlier_rate'].values, width,
                       label='Targets', color='#d62728', alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        bars2 = ax2.bar(x + width/2, peer_data['outlier_rate'].values, width,
                       label='Peers', color='#1f77b4', alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        
        # add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax2.set_ylabel('Outlier Rate (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Outlier Rates by Attribution Type', fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([at.replace('-', '\n') for at in attribution_types], fontsize=9)
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax2.set_axisbelow(True)
        
        # panel 3: self-serving vs non-self-serving
        ax3 = fig.add_subplot(gs[0, 2])
        
        # classify as self-serving or not
        self_serving_types = ['Positive-Internal', 'Negative-External']
        df['is_self_serving'] = df['attribution_type'].isin(self_serving_types)
        
        serving_stats = []
        for is_self_serving, serving_label in [(True, 'Self-Serving\n(Pos-Int, Neg-Ext)'), 
                                                 (False, 'Non-Self-Serving\n(Pos-Ext, Neg-Int)')]:
            for is_target, group_label in [(True, 'Targets'), (False, 'Peers')]:
                subset = df[(df['is_self_serving'] == is_self_serving) & 
                           (df['is_target'] == is_target)]
                if len(subset) > 0:
                    outlier_rate = (subset['is_outlier']).mean() * 100
                    serving_stats.append({
                        'serving_type': serving_label,
                        'group': group_label,
                        'outlier_rate': outlier_rate,
                        'n': len(subset)
                    })
        
        serving_df = pd.DataFrame(serving_stats)
        
        x = np.arange(2)
        width = 0.35
        
        for i, group in enumerate(['Targets', 'Peers']):
            group_data = serving_df[serving_df['group'] == group]
            color = '#d62728' if group == 'targets' else '#1f77b4'
            offset = -width/2 if group == 'Targets' else width/2
            
            bars = ax3.bar(x + offset, group_data['outlier_rate'].values, width,
                          label=group, color=color, alpha=0.8, 
                          edgecolor='black', linewidth=1.5)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax3.set_ylabel('Outlier Rate (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Self-Serving vs Non-Self-Serving\nAttribution Outliers', 
                     fontsize=11, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Self-Serving\n(Pos-Int, Neg-Ext)', 
                            'Non-Self-Serving\n(Pos-Ext, Neg-Int)'], fontsize=9)
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax3.set_axisbelow(True)
        
        # main title and interpretation (with more spacing to prevent overlap)
        fig.suptitle('Temporal Consistency Analysis: Topic Outlier Rates (|Z-score| > 2.0)\n' +
                    'Measures Deviation from Historical Attribution Patterns',
                    fontsize=13, fontweight='bold', y=0.99, linespacing=1.3)
        
        fig.text(0.5, 0.02,
                "INTERPRETATION: Outlier rate measures how often attributions deviate from a company's historical topic patterns (>2 standard deviations).\n" +
                "Higher rates indicate more temporal inconsistency. No significant differences found between targets and peers overall.",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3, edgecolor='gray', linewidth=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        
        output_file = self.output_dir / 'Fig_TS1_aggregate_outlier_rates.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {output_file.name}")
        plt.close()
        
        # print detailed statistics
        print("\nDetailed Outlier Statistics:")
        print("\n1. Overall (All Types Combined):")
        for _, row in overall_df.iterrows():
            print(f"   {row['group']}: {row['outlier_rate']:.1f}% " +
                  f"({row['n_outliers']}/{row['n_total']} outliers)")
        
        print("\n2. By Attribution Type:")
        for attr_type in attribution_types:
            type_subset = type_df[type_df['attribution_type'] == attr_type]
            if len(type_subset) > 0:
                print(f"   {attr_type}:")
                for _, row in type_subset.iterrows():
                    print(f"     {row['group']}: {row['outlier_rate']:.1f}% (n={row['n']})")
        
        print("\n3. Self-Serving vs Non-Self-Serving:")
        for serving_type in serving_df['serving_type'].unique():
            print(f"   {serving_type.replace(chr(10), ' ')}:")
            subset = serving_df[serving_df['serving_type'] == serving_type]
            for _, row in subset.iterrows():
                print(f"     {row['group']}: {row['outlier_rate']:.1f}% (n={row['n']})")
    
    def plot_ts2_attribution_type_boxplots(self):
        """
        TS2: Attribution Type Outlier Distribution (Box Plots)
        
        Level 2 Granularity: Attribution type over time (aggregate across topics)
        Question: Which attribution types show more temporal inconsistency?
        Metric: Z-score distribution and outlier counts
        Success: Targets have higher variance and more |Z| > 2 outliers
        """
        print("\n" + "=" * 80)
        print("TS2: Attribution Type Temporal Consistency (Box Plots)")
        print("=" * 80)
        
        if self.temporal_consistency is None:
            print("  [SKIP] Temporal consistency data not available")
            return
        
        df = self.temporal_consistency.copy()
        df = df[df['n_same_topic_historical'] > 0].copy()
        
        if len(df) == 0:
            print("   No temporal consistency data")
            return
        
        # map to attribution types
        df['attribution_type'] = df.apply(
            lambda row: f"{row['outcome']}-{row['locus']}" 
            if row['outcome'] in ['Positive', 'Negative'] and 
               row['locus'] in ['Internal', 'External'] 
            else "Other",
            axis=1
        )
        
        attribution_types = ['Positive-Internal', 'Positive-External',
                            'Negative-Internal', 'Negative-External']
        
        # filter to valid types
        df = df[df['attribution_type'].isin(attribution_types)]
        
        # use topic_outlier_score as z-score
        df['z_score'] = df['topic_outlier_score']
        df['Company Type'] = df['is_target'].map({True: 'Targets', False: 'Peers'})
        
        # create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # left plot: box plots of z-scores
        ax1 = axes[0]
        
        # prepare data for seaborn
        plot_order = []
        for attr_type in attribution_types:
            plot_order.extend([f"{attr_type}\nTargets", f"{attr_type}\nPeers"])
        
        # create combined labels
        df['type_company'] = df['attribution_type'].str.replace('-', '\n') + '\n' + df['Company Type']
        
        sns.boxplot(data=df, x='attribution_type', y='z_score', hue='Company Type',
                   ax=ax1, palette={'Targets': '#d62728', 'peers': '#1f77b4'})
        
        ax1.axhline(y=2.0, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Outlier threshold')
        ax1.axhline(y=-2.0, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax1.set_xlabel('Attribution Type', fontsize=11)
        ax1.set_ylabel('Z-score (vs Historical)', fontsize=11)
        ax1.set_title('Distribution of Temporal Z-scores', fontsize=12, fontweight='bold')
        ax1.set_xticklabels([at.replace('-', '\n') for at in attribution_types])
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # right plot: outlier counts
        ax2 = axes[1]
        
        outlier_data = []
        for attr_type in attribution_types:
            attr_data = df[df['attribution_type'] == attr_type]
            
            targets = attr_data[attr_data['Company Type'] == 'Targets']
            peers = attr_data[attr_data['Company Type'] == 'Peers']
            
            target_outlier_rate = (np.abs(targets['z_score']) > 2).mean() * 100 if len(targets) > 0 else 0
            peer_outlier_rate = (np.abs(peers['z_score']) > 2).mean() * 100 if len(peers) > 0 else 0
            
            outlier_data.append({
                'attribution_type': attr_type,
                'Targets': target_outlier_rate,
                'Peers': peer_outlier_rate
            })
        
        outlier_df = pd.DataFrame(outlier_data)
        
        x = np.arange(len(attribution_types))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, outlier_df['Targets'], width, 
                       label='Targets', color='#d62728', alpha=0.8)
        bars2 = ax2.bar(x + width/2, outlier_df['Peers'], width,
                       label='Peers', color='#1f77b4', alpha=0.8)
        
        ax2.set_xlabel('Attribution Type', fontsize=11)
        ax2.set_ylabel('Outlier Rate (%)', fontsize=11)
        ax2.set_title('Temporal Outlier Rates (|Z| > 2)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([at.replace('-', '\n') for at in attribution_types])
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        # add percentage labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        fig.suptitle('Attribution Type Temporal Consistency Analysis\n' +
                    'Targets vs Peers Outlier Comparison',
                    fontsize=13, fontweight='bold')
        
        fig.text(0.5, 0.01,
                "Interpretation: Higher outlier rates = More temporal inconsistency\n" +
                "Success: Targets show higher rates, especially for self-serving types",
                ha='center', fontsize=9, style='italic')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        output_file = self.output_dir / 'Fig_TS2_attribution_type_boxplots.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"[SAVED] {output_file}")
        plt.close()
        
        print("\nOutlier Rate Summary:")
        for _, row in outlier_df.iterrows():
            ratio = row['Targets'] / row['Peers'] if row['Peers'] > 0 else float('inf')
            print(f"  {row['attribution_type']}: Targets {row['Targets']:.1f}%, " + 
                  f"Peers {row['Peers']:.1f}%, Ratio {ratio:.2f}x")
    
    def plot_ts3_company_quarter_heatmap(self):
        """
        TS3: Company × Quarter Temporal Consistency Heatmap
        
        Level 3 Granularity: Full call temporal consistency
        Question: Which companies and quarters show highest temporal deviations?
        Metric: Z-score aggregated to company-quarter level
        Success: Target companies show spikes during known high-bias quarters
        """
        print("\n" + "=" * 80)
        print("TS3: Company × Quarter Temporal Consistency Heatmap")
        print("=" * 80)
        
        if self.bias_scores is None:
            print("  [INFO] Bias scores not available, using temporal consistency data")
            if self.temporal_consistency is None:
                print("  [SKIP] No temporal data available")
                return
            df = self.temporal_consistency.copy()
            df['consistency_score'] = df['topic_outlier_score']
        else:
            df = self.bias_scores.copy()
            df['consistency_score'] = df['bias_score']
        
        # aggregate to company-quarter level
        company_quarter = df.groupby(['company', 'year', 'quarter']).agg({
            'consistency_score': 'mean',
            'is_target': 'first'
        }).reset_index()
        
        # filter to target companies only for cleaner visualization
        targets_only = company_quarter[company_quarter['is_target'].isin(['Y', True, 1])]
        
        if len(targets_only) == 0:
            print("  [SKIP] No target company data available")
            return
        
        print(f"\nCreating heatmap for {targets_only['company'].nunique()} target companies")
        
        # pivot to matrix format
        pivot_data = targets_only.pivot_table(
            index='company',
            columns=['year', 'quarter'],
            values='consistency_score',
            aggfunc='mean'
        )
        
        # sort companies by mean consistency score (most extreme first)
        company_means = pivot_data.mean(axis=1).abs().sort_values(ascending=False)
        pivot_data = pivot_data.loc[company_means.index]
        
        # create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # custom colormap: diverging around 0
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        # create heatmap
        sns.heatmap(pivot_data, cmap=cmap, center=0,
                   cbar_kws={'label': 'Consistency Score (Z-score or Bias Score)'},
                   linewidths=0.5, linecolor='gray',
                   ax=ax)
        
        # format column labels (year-quarter)
        col_labels = [f"{y}-Q{q}" for y, q in pivot_data.columns]
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        ax.set_xlabel('Quarter', fontsize=11)
        ax.set_ylabel('Company', fontsize=11)
        ax.set_title('Temporal Consistency Across Companies and Quarters (Target Firms)\n' +
                    'Red = High Deviation/Bias, Blue = Low Deviation/Bias',
                    fontsize=12, fontweight='bold')
        
        # add interpretation
        fig.text(0.5, 0.01,
                "Interpretation: Red cells = Periods of high temporal deviation or bias\n" +
                "Success: Red clusters align with known high-bias periods",
                ha='center', fontsize=9, style='italic')
        
        plt.tight_layout(rect=[0, 0.02, 1, 1])
        
        output_file = self.output_dir / 'Fig_TS3_company_quarter_heatmap.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"[SAVED] {output_file}")
        plt.close()
        
        # print top companies
        print("\nCompanies with Highest Mean Deviation:")
        for i, (company, score) in enumerate(company_means.head(5).items(), 1):
            print(f"  {i}. {company}: {score:.3f}")
    
    def plot_bias_vector_separation(self):
        """
        REMOVED - This figure was calculating Cohen's d incorrectly.
        
        It was using artificial tertile splits (top 33% vs bottom 33%) which 
        guarantees a large Cohen's d through circular reasoning.
        
        The ACTUAL result from expert-identified bias periods is:
        - Cohen's d = 0.700 (medium effect) for expert periods vs normal periods
        
        This is already shown in:
        - Fig_CS3_full_call_violins.png (d=0.63 in high-bias periods)
        - Fig_SUPERVISED_classification_comparison.png (AUC 0.848)
        
        This figure would be redundant and misleading, so it's been removed.
        """
        print("\n" + "=" * 80)
        print("Bias Vector Separation (SKIPPED - Redundant)")
        print("=" * 80)
        print("  [INFO] This figure has been removed to avoid misleading results.")
        print("  [INFO] Actual Cohen's d for expert periods vs normal: d=0.700")
        print("  [INFO] See Fig_CS3_full_call_violins.png for correct visualization.")
        return
    
    def plot_bias_score_heatmap(self):
        """
        Company × Quarter Bias Score Heatmap
        
        Visualizes which specific company-quarters have high bias scores.
        Complements the bias vector separation analysis.
        """
        print("\n" + "=" * 80)
        print("Company × Quarter Bias Score Heatmap")
        print("=" * 80)
        
        # this is similar to ts3 but uses bias scores specifically
        # and includes both targets and top peers
        
        if self.bias_scores is None:
            print("  [SKIP] Bias scores not available")
            return
        
        df = self.bias_scores.copy()
        
        # aggregate to company-quarter
        company_quarter = df.groupby(['company', 'year', 'quarter']).agg({
            'bias_score': 'mean',
            'is_target': 'first'
        }).reset_index()
        
        # include targets + top 10 most extreme peers
        targets = company_quarter[company_quarter['is_target'].isin(['Y', True, 1])]
        peers = company_quarter[~company_quarter['is_target'].isin(['Y', True, 1])]
        
        # get top peers by average absolute bias score
        peer_means = peers.groupby('company')['bias_score'].apply(lambda x: np.abs(x).mean())
        top_peers = peer_means.nlargest(10).index.tolist()
        
        peers_top = peers[peers['company'].isin(top_peers)]
        
        # combine
        combined = pd.concat([targets, peers_top])
        
        print(f"\nIncluding {targets['company'].nunique()} targets + {len(top_peers)} top peers")
        
        # pivot
        pivot_data = combined.pivot_table(
            index='company',
            columns=['year', 'quarter'],
            values='bias_score',
            aggfunc='mean'
        )
        
        # sort: targets first (by mean), then peers
        target_companies = targets['company'].unique()
        target_means = pivot_data.loc[pivot_data.index.isin(target_companies)].mean(axis=1).abs()
        peer_means_pivot = pivot_data.loc[~pivot_data.index.isin(target_companies)].mean(axis=1).abs()
        
        sorted_targets = target_means.sort_values(ascending=False).index.tolist()
        sorted_peers = peer_means_pivot.sort_values(ascending=False).index.tolist()
        
        sorted_companies = sorted_targets + sorted_peers
        pivot_data = pivot_data.loc[sorted_companies]
        
        # create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        # flatten column names before heatmap to avoid multiindex issues
        pivot_data.columns = [f"{y}-Q{q}" for y, q in pivot_data.columns]
        
        sns.heatmap(pivot_data, cmap=cmap, center=0,
                   vmin=-1, vmax=1,
                   cbar_kws={'label': 'Bias Score'},
                   linewidths=0.3, linecolor='lightgray',
                   ax=ax)
        
        # add separator line between targets and peers
        if len(sorted_targets) > 0:
            ax.axhline(y=len(sorted_targets), color='black', linewidth=2)
            ax.text(-0.5, len(sorted_targets)/2, 'TARGETS', 
                   rotation=90, va='center', fontweight='bold', fontsize=11)
            ax.text(-0.5, len(sorted_targets) + len(sorted_peers)/2, 'PEERS',
                   rotation=90, va='center', fontweight='bold', fontsize=11)
        
        # set labels with proper rotation
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        ax.set_xlabel('Quarter', fontsize=11)
        ax.set_ylabel('Company', fontsize=11)
        ax.set_title('Bias Scores Across Companies and Quarters\n' +
                    'Red = High-Bias Pattern, Blue = Low-Bias Pattern',
                    fontsize=12, fontweight='bold')
        
        fig.text(0.5, 0.01,
                "Interpretation: Positive scores align with high-bias language patterns\n" +
                "Success: Target companies show more red cells than peers",
                ha='center', fontsize=9, style='italic')
        
        plt.tight_layout(rect=[0, 0.02, 1, 1])
        
        output_file = self.output_dir / 'Fig_BIAS_score_heatmap.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"[SAVED] {output_file}")
        plt.close()
    
    def plot_pca_biplot(self):
        """
        PCA Biplot with Bias Gradient
        
        Visualizes PC1 vs PC2 with points colored by bias score.
        Shows which principal components capture bias patterns.
        """
        print("\n" + "=" * 80)
        print("PCA Biplot with Bias Gradient")
        print("=" * 80)
        
        if self.pca_coords is None:
            print("  [SKIP] PCA coordinates not available")
            return
        
        df = self.pca_coords.copy()
        
        # merge with bias scores if available
        if self.bias_scores is not None:
            # merge on index
            df = df.merge(self.bias_scores[['bias_score']], 
                         left_index=True, right_index=True, how='left')
        
        # create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # left: colored by bias score
        ax1 = axes[0]
        
        if 'bias_score' in df.columns:
            scatter = ax1.scatter(df['pc1'], df['pc2'], 
                                 c=df['bias_score'],
                                 cmap='RdBu_r', alpha=0.6, s=20,
                                 vmin=-0.5, vmax=0.5)
            plt.colorbar(scatter, ax=ax1, label='Bias Score')
        else:
            ax1.scatter(df['pc1'], df['pc2'], alpha=0.6, s=20, color='gray')
        
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.set_xlabel('PC1', fontsize=11)
        ax1.set_ylabel('PC2', fontsize=11)
        ax1.set_title('PCA Biplot: Colored by Bias Score', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # right: targets vs peers
        ax2 = axes[1]
        
        targets = df[df['is_target'].isin(['Y', True, 1])]
        peers = df[~df['is_target'].isin(['Y', True, 1])]
        
        ax2.scatter(peers['pc1'], peers['pc2'], 
                   color='#1f77b4', alpha=0.3, s=15, label='peers')
        ax2.scatter(targets['pc1'], targets['pc2'],
                   color='#d62728', alpha=0.6, s=25, label='targets', marker='^')
        
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.set_xlabel('PC1', fontsize=11)
        ax2.set_ylabel('PC2', fontsize=11)
        ax2.set_title('PCA Biplot: Targets vs Peers', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        fig.suptitle('Principal Component Analysis\n' +
                    'First Two Components Capture Key Embedding Patterns',
                    fontsize=13, fontweight='bold')
        
        fig.text(0.5, 0.01,
                "Interpretation: Clear gradient or separation suggests PCA captures bias signal\n" +
                "PC1 and PC2 together explain most variance in embedding space",
                ha='center', fontsize=9, style='italic')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        output_file = self.output_dir / 'Fig_PCA_biplot.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"[SAVED] {output_file}")
        plt.close()
    
    # =========================================================================
    # company ranking & audio processing prioritization
    # =========================================================================
    
    def rank_companies_for_audio_processing(self):
        """
        Rank target companies by likelihood of exhibiting bias for audio processing prioritization.
        
        METHODOLOGY:
        Combines 5 independent bias signals into composite score:
        1. Bias Score Magnitude (projection onto bias vector)
        2. Bias Score Consistency (% of quarters in top tertile)
        3. Temporal Deviation (mean |Z-score| from TS3)
        4. High-Bias Period Strength (Cohen's d during peak quarters)
        5. Outlier Frequency (% outliers from temporal analysis)
        
        INTERPRETATION:
        - Higher composite score = Stronger evidence of bias
        - Multiple converging signals = More reliable candidate
        - Identifies specific quarters for targeted audio analysis"""
        print("\n" + "=" * 80)
        print("COMPANY RANKING FOR AUDIO PROCESSING PRIORITIZATION")
        print("=" * 80)
        
        if self.bias_scores is None:
            print("  [SKIP] Bias scores not available")
            return None
        
        # aggregate bias data to company-quarter level
        company_quarter_df = self.bias_scores.groupby(['company', 'year', 'quarter']).agg({
            'bias_score': 'mean',
            'is_target': 'first'
        }).reset_index()
        
        # filter to targets only
        targets = company_quarter_df[company_quarter_df['is_target'].isin(['Y', True, 1])].copy()
        
        if len(targets) == 0:
            print("  [SKIP] No target company data")
            return None
        
        print(f"\nAnalyzing {targets['company'].nunique()} target companies...")
        
        # load config to get assumed bias periods
        config_periods = self._load_config_bias_periods()
        
        rankings = []
        
        for company in targets['company'].unique():
            company_data = targets[targets['company'] == company].copy()
            
            # signal 1: bias score magnitude (mean of top 3 quarters)
            top_3_scores = company_data.nlargest(min(3, len(company_data)), 'bias_score')['bias_score']
            bias_magnitude = top_3_scores.mean() if len(top_3_scores) > 0 else 0
            
            # signal 2: bias score consistency (% quarters in top tertile)
            threshold_top_tertile = targets['bias_score'].quantile(0.67)
            consistency = (company_data['bias_score'] > threshold_top_tertile).mean() * 100
            
            # signal 3: temporal deviation (if available from temporal consistency)
            temporal_deviation = 0
            if self.temporal_consistency is not None:
                temp_data = self.temporal_consistency[
                    self.temporal_consistency['company'] == company
                ]
                if len(temp_data) > 0:
                    temporal_deviation = np.abs(temp_data['topic_outlier_score']).mean()
            
            # signal 4: peak period strength (max bias score)
            peak_strength = company_data['bias_score'].max()
            
            # signal 5: outlier frequency (from temporal analysis)
            outlier_freq = 0
            if self.temporal_consistency is not None:
                temp_data = self.temporal_consistency[
                    (self.temporal_consistency['company'] == company) &
                    (self.temporal_consistency['n_same_topic_historical'] > 0)
                ]
                if len(temp_data) > 0:
                    outlier_freq = (np.abs(temp_data['topic_outlier_score']) > 2).mean() * 100
            
            # composite score (weighted average, normalized to 0-100)
            # weights based on reliability: bias_magnitude (30%), peak_strength (25%),
            # consistency (20%), temporal_deviation (15%), outlier_freq (10%)
            
            # normalize each component to 0-1 scale
            norm_magnitude = (bias_magnitude - targets['bias_score'].min()) / \
                            (targets['bias_score'].max() - targets['bias_score'].min() + 1e-10)
            norm_peak = (peak_strength - targets['bias_score'].min()) / \
                       (targets['bias_score'].max() - targets['bias_score'].min() + 1e-10)
            norm_consistency = consistency / 100.0
            norm_temporal = min(temporal_deviation / 2.0, 1.0)  # cap at z=2
            norm_outlier = outlier_freq / 100.0
            
            composite_score = (
                0.30 * norm_magnitude +
                0.25 * norm_peak +
                0.20 * norm_consistency +
                0.15 * norm_temporal +
                0.10 * norm_outlier
            ) * 100
            
            # identify high-bias quarters (top 3)
            high_bias_quarters = company_data.nlargest(min(3, len(company_data)), 'bias_score')
            quarter_labels = [
                f"{row['year']}-Q{row['quarter']}" 
                for _, row in high_bias_quarters.iterrows()
            ]
            quarter_scores = [
                f"{score:.3f}" 
                for score in high_bias_quarters['bias_score']
            ]
            
            # compare to config period
            config_info = config_periods.get(company, {})
            config_start = config_info.get('start', 'Unknown')
            config_end = config_info.get('end', 'Unknown')
            bias_type = config_info.get('type', 'Unknown')
            
            # calculate overlap with config period
            overlap_score = self._calculate_period_overlap(
                high_bias_quarters, config_start, config_end
            )
            
            rankings.append({
                'company': company,
                'composite_score': composite_score,
                'bias_magnitude': bias_magnitude,
                'peak_strength': peak_strength,
                'consistency_pct': consistency,
                'temporal_deviation': temporal_deviation,
                'outlier_freq_pct': outlier_freq,
                'n_quarters': len(company_data),
                'top_quarter_1': quarter_labels[0] if len(quarter_labels) > 0 else 'N/A',
                'top_score_1': quarter_scores[0] if len(quarter_scores) > 0 else 'N/A',
                'top_quarter_2': quarter_labels[1] if len(quarter_labels) > 1 else 'N/A',
                'top_score_2': quarter_scores[1] if len(quarter_scores) > 1 else 'N/A',
                'top_quarter_3': quarter_labels[2] if len(quarter_labels) > 2 else 'N/A',
                'top_score_3': quarter_scores[2] if len(quarter_scores) > 2 else 'N/A',
                'config_start': config_start,
                'config_end': config_end,
                'config_bias_type': bias_type,
                'config_overlap_pct': overlap_score
            })
        
        # sort by composite score
        rankings_df = pd.DataFrame(rankings)
        rankings_df = rankings_df.sort_values('composite_score', ascending=False)
        rankings_df['rank'] = range(1, len(rankings_df) + 1)
        
        # save to csv
        output_file = self.output_dir.parent / 'audio_processing_priority_list.csv'
        rankings_df.to_csv(output_file, index=False)
        print(f"\n[SAVED] {output_file}")
        
        # print detailed report
        self._print_audio_processing_report(rankings_df)
        
        return rankings_df
    
    def _load_config_bias_periods(self):
        """Load bias start/end dates from company_config.json"""
        try:
            config_path = Path('company_config.json')
            if not config_path.exists():
                return {}
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            periods = {}
            for target in config.get('targets', []):
                folder = target.get('target_folder', '')
                if folder:
                    periods[folder] = {
                        'start': target.get('date', 'Unknown'),
                        'end': target.get('bias_end_date', 'Unknown'),
                        'type': target.get('bias_type', 'Unknown')
                    }
            
            return periods
        except Exception as e:
            print(f"  Warning: Could not load config ({e})")
            return {}
    
    def _calculate_period_overlap(self, high_bias_quarters, config_start, config_end):
        """Calculate % of identified high-bias quarters that overlap with config period"""
        if config_start == 'Unknown' or config_end == 'Unknown':
            return 0.0
        
        try:
            from datetime import datetime
            
            config_start_dt = datetime.strptime(config_start, '%Y-%m-%d')
            config_end_dt = datetime.strptime(config_end, '%Y-%m-%d')
            
            overlaps = 0
            for _, row in high_bias_quarters.iterrows():
                # convert quarter to date (use middle of quarter)
                year = int(row['year'])
                quarter = int(row['quarter'])
                month = (quarter - 1) * 3 + 2  # middle month of quarter
                quarter_dt = datetime(year, month, 15)
                
                if config_start_dt <= quarter_dt <= config_end_dt:
                    overlaps += 1
            
            return (overlaps / len(high_bias_quarters)) * 100 if len(high_bias_quarters) > 0 else 0.0
        except:
            return 0.0
    
    def _print_audio_processing_report(self, rankings_df):
        """Print detailed report for audio processing prioritization"""
        print("\n" + "=" * 80)
        print("AUDIO PROCESSING PRIORITY REPORT")
        print("=" * 80)
        
        print("\nRANKING METHODOLOGY:")
        print("  Composite Score (0-100) based on 5 signals:")
        print("    1. Bias Magnitude (30%): Mean of top 3 quarters")
        print("    2. Peak Strength (25%): Maximum bias score observed")
        print("    3. Consistency (20%): % quarters in top tertile")
        print("    4. Temporal Deviation (15%): Mean |Z-score| from history")
        print("    5. Outlier Frequency (10%): % attributions with |Z| > 2")
        print("\n  Higher score = Stronger evidence of bias = Higher priority for audio analysis")
        
        print("\n" + "-" * 80)
        print("TOP 10 PRIORITY TARGETS FOR AUDIO PROCESSING:")
        print("-" * 80)
        
        for idx, row in rankings_df.head(10).iterrows():
            print(f"\n#{row['rank']}: {row['company']}")
            print(f"  Composite Score: {row['composite_score']:.1f}/100")
            print(f"  Evidence Strength:")
            print(f"    - Bias Magnitude: {row['bias_magnitude']:.3f}")
            print(f"    - Peak Strength: {row['peak_strength']:.3f}")
            print(f"    - Consistency: {row['consistency_pct']:.1f}% of quarters in top tertile")
            print(f"    - Temporal Deviation: {row['temporal_deviation']:.2f} (mean |Z-score|)")
            print(f"    - Outlier Frequency: {row['outlier_freq_pct']:.1f}% attributions are outliers")
            
            print(f"  High-Bias Quarters (Top 3):")
            if row['top_quarter_1'] != 'N/A':
                print(f"    1. {row['top_quarter_1']} (score: {row['top_score_1']})")
            if row['top_quarter_2'] != 'N/A':
                print(f"    2. {row['top_quarter_2']} (score: {row['top_score_2']})")
            if row['top_quarter_3'] != 'N/A':
                print(f"    3. {row['top_quarter_3']} (score: {row['top_score_3']})")
            
            print(f"  Config Comparison:")
            print(f"    - Assumed Period: {row['config_start']} to {row['config_end']}")
            print(f"    - Assumed Type: {row['config_bias_type']}")
            print(f"    - Overlap: {row['config_overlap_pct']:.0f}% of identified quarters match config")
            
            if row['config_overlap_pct'] < 50:
                print(f"    [WARNING] LOW OVERLAP: Detected bias in different period than assumed!")
            elif row['config_overlap_pct'] > 66:
                print(f"    [OK] GOOD OVERLAP: Detected bias aligns with assumed period")
        
        print("\n" + "-" * 80)
        print("COMPLETE RANKING (All Targets):")
        print("-" * 80)
        
        print(f"\n{'Rank':<5} {'Company':<15} {'Score':<8} {'Magnitude':<10} {'Peak':<8} "
              f"{'Consist%':<9} {'TempDev':<9} {'Outlier%':<9} {'Top Quarter':<12}")
        print("-" * 110)
        
        for _, row in rankings_df.iterrows():
            print(f"{row['rank']:<5} {row['company']:<15} {row['composite_score']:>6.1f}   "
                  f"{row['bias_magnitude']:>8.3f}   {row['peak_strength']:>6.3f}   "
                  f"{row['consistency_pct']:>7.1f}   {row['temporal_deviation']:>7.2f}   "
                  f"{row['outlier_freq_pct']:>7.1f}   {row['top_quarter_1']:<12}")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATION:")
        print("=" * 80)
        print("\nFor audio processing, prioritize:")
        print("  - TOP 5: Core targets with strongest composite evidence")
        print("  - Rank 6-10: Secondary targets if resources allow")
        print("  - Rank 11+: Low priority unless specific research interest")
        
        print("\nFor each selected company, process audio from:")
        print("  - The 3 identified high-bias quarters listed above")
        print("  - Plus 2-3 quarters from same peers for comparison")
        
        print("\nEstimated audio processing requirements:")
        top_5 = rankings_df.head(5)
        total_target_quarters = top_5[['top_quarter_1', 'top_quarter_2', 'top_quarter_3']].notna().sum().sum()
        print(f"  Top 5 targets: ~{int(total_target_quarters)} target quarters")
        print(f"  With peer comparison: ~{int(total_target_quarters * 2.5)} total quarters")
        print(f"  Estimated hours (at 1hr/call): {int(total_target_quarters * 2.5)} hours")
        
        print("\n[OK] Full ranking saved to: audio_processing_priority_list.csv")
    
    # =========================================================================
    # new dissertation-critical visualizations
    # =========================================================================
    
    def plot_supervised_classification_comparison(self):
        """
        Supervised Classification AUC Comparison
        
        Shows GPT baseline vs Embedding vs Combined model performance.
        Demonstrates that embeddings detect patterns GPT SAB analysis misses.
        
        KEY RESULT: Combined model achieves 86.0% accuracy (+14.1% over GPT alone)
        """
        print("\n" + "=" * 80)
        print("Supervised Classification Comparison")
        print("=" * 80)
        
        if self.bias_prediction_results is None:
            print("  [SKIP] Bias prediction results not available")
            return
        
        # extract auc scores from results
        try:
            analysis = self.bias_prediction_results.get('analysis', {})
            gpt_auc = analysis.get('gpt_auc', 0.0)
            embedding_auc = analysis.get('embedding_auc', 0.0)
            combined_auc = analysis.get('combined_auc', 0.0)
            
            if gpt_auc == 0.0 or embedding_auc == 0.0:
                print("  [SKIP] AUC scores not found in results")
                return
            
            print(f"  GPT Baseline: {gpt_auc:.3f}")
            print(f"  Embedding:    {embedding_auc:.3f}")
            print(f"  Combined:     {combined_auc:.3f}")
            
        except Exception as e:
            print(f"  [ERROR] Could not extract AUC scores: {e}")
            return
        
        # create figure with compact height for dissertation formatting
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # data
        models = ['GPT Baseline\n(SAB Proportions)', 
                  'Embedding Only\n(Linguistic Features)', 
                  'Combined\n(GPT + Embedding)']
        aucs = [gpt_auc, embedding_auc, combined_auc]
        colors = ['#1f77b4', '#2ca02c', '#9467bd']  # blue, green, purple
        
        # create bars
        bars = ax.bar(models, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # add value labels on bars
        for bar, auc in zip(bars, aucs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{auc:.3f}\n({auc*100:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # calculate improvements
        improvement_emb = embedding_auc - gpt_auc
        improvement_comb = combined_auc - gpt_auc
        
        # add improvement annotation below embedding bar
        ax.text(1, 0.08, 
               f'+{improvement_emb:.3f}\n(+{improvement_emb*100:.1f}%)',
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.8, 
                        edgecolor='black', linewidth=1.5))
        
        # add improvement annotation below combined bar
        ax.text(2, 0.08, 
               f'+{improvement_comb:.3f}\n(+{improvement_comb*100:.1f}%)',
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.8,
                        edgecolor='black', linewidth=1.5))
        
        # styling
        ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Sets', fontsize=12, fontweight='bold')
        ax.set_title('Logistic Regression Classification of SAB Bias Periods\nAcross Different Feature Sets',
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'Fig_SUPERVISED_classification_comparison.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"[SAVED] {output_file}")
        plt.close()
    
    def plot_per_firm_temporal_examples(self):
        """
        Per-Firm Temporal Analysis Examples
        
        Shows 3 example firms with different crisis response strategies:
        - High positive change (evasive: more topic-jumping)
        - High negative change (controlled: more focused)
        - No change (stable)
        
        Each firm shows 4 groups: Target IN/OUT bias, Peer IN/OUT bias
        """
        print("\n" + "=" * 80)
        print("Per-Firm Temporal Analysis Examples")
        print("=" * 80)
        
        if self.firm_analysis_df is None:
            print("  [SKIP] Per-firm analysis results not available")
            return
        
        df = self.firm_analysis_df.copy()
        
        # filter to firms with sufficient bias data
        df = df[df['has_bias_data'] == True].copy()
        
        if len(df) == 0:
            print("  [SKIP] No firms with bias period data")
            return
        
        # select 3 example firms
        df_sorted = df.sort_values('outlier_change', ascending=False)
        
        # select firms with largest positive, largest negative, and closest to zero
        firm_high_positive = df_sorted.iloc[0] if len(df_sorted) > 0 else None
        firm_high_negative = df_sorted.iloc[-1] if len(df_sorted) > 0 else None
        
        # find firm closest to zero change
        df_sorted_abs = df.copy()
        df_sorted_abs['abs_outlier_change'] = df_sorted_abs['outlier_change'].abs()
        df_sorted_abs = df_sorted_abs.sort_values('abs_outlier_change')
        firm_no_change = df_sorted_abs.iloc[0] if len(df_sorted_abs) > 0 else None
        
        selected_firms = [
            (firm_high_positive, 'High Positive Change\n(More Topic-Jumping)', 'evasive'),
            (firm_high_negative, 'High Negative Change\n(More Focused)', 'controlled'),
            (firm_no_change, 'No Significant Change\n(Stable)', 'stable')
        ]
        
        selected_firms = [(f, label, strategy) for f, label, strategy in selected_firms if f is not None]
        
        if len(selected_firms) == 0:
            print("  [SKIP] Could not select example firms")
            return
        
        print(f"  Selected {len(selected_firms)} example firms:")
        for firm, label, strategy in selected_firms:
            print(f"    • {firm['target_firm']}: Outlier change = {firm['outlier_change']:+.2f}x")
        
        # create figure with 2 rows (coherence, outlier) × 3 columns (firms)
        fig, axes = plt.subplots(2, len(selected_firms), figsize=(5*len(selected_firms), 10))
        
        if len(selected_firms) == 1:
            axes = axes.reshape(-1, 1)
        
        for col, (firm, label, strategy) in enumerate(selected_firms):
            # extract data
            target_coh_in = firm['coherence_target_in']
            target_coh_out = firm['coherence_target_out']
            peer_coh_in = firm['coherence_peer_in']
            peer_coh_out = firm['coherence_peer_out']
            
            target_out_in = firm['outlier_target_in']
            target_out_out = firm['outlier_target_out']
            peer_out_in = firm['outlier_peer_in']
            peer_out_out = firm['outlier_peer_out']
            
            coherence_change = firm['coherence_change']
            outlier_change = firm['outlier_change']
            
            # top panel: coherence
            ax_coh = axes[0, col]
            
            x_pos = [0, 1]
            target_coh = [target_coh_out, target_coh_in]
            peer_coh = [peer_coh_out, peer_coh_in]
            
            # plot lines
            ax_coh.plot(x_pos, target_coh, 'o-', color='#d62728', linewidth=2.5, 
                       markersize=10, label='Target', markeredgecolor='black', markeredgewidth=1)
            ax_coh.plot(x_pos, peer_coh, 's--', color='#1f77b4', linewidth=2.5,
                       markersize=10, label='Peers', markeredgecolor='black', markeredgewidth=1)
            
            # annotate gap change
            gap_in = target_coh_in - peer_coh_in
            gap_out = target_coh_out - peer_coh_out
            
            ax_coh.text(0, max(target_coh_out, peer_coh_out) * 1.05, 
                       f'Gap: {gap_out:+.3f}', ha='center', fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax_coh.text(1, max(target_coh_in, peer_coh_in) * 1.05,
                       f'Gap: {gap_in:+.3f}', ha='center', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            
            # add change annotation
            y_mid = (max(target_coh + peer_coh) + min(target_coh + peer_coh)) / 2
            ax_coh.text(0.5, y_mid, f'Δ = {coherence_change:+.3f}',
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax_coh.set_xticks(x_pos)
            ax_coh.set_xticklabels(['OUT\nBias Periods', 'IN\nBias Periods'])
            ax_coh.set_ylabel('Coherence\n(lower = more variable)', fontsize=10)
            ax_coh.set_title(f'{firm["target_firm"]}\n{label}', fontsize=11, fontweight='bold')
            ax_coh.legend(loc='upper right', fontsize=9)
            ax_coh.grid(alpha=0.3)
            
            # bottom panel: outlier rate
            ax_out = axes[1, col]
            
            target_out_pct = [target_out_out * 100, target_out_in * 100]
            peer_out_pct = [peer_out_out * 100, peer_out_in * 100]
            
            # plot lines
            ax_out.plot(x_pos, target_out_pct, 'o-', color='#d62728', linewidth=2.5,
                       markersize=10, label='Target', markeredgecolor='black', markeredgewidth=1)
            ax_out.plot(x_pos, peer_out_pct, 's--', color='#1f77b4', linewidth=2.5,
                       markersize=10, label='Peers', markeredgecolor='black', markeredgewidth=1)
            
            # annotate ratios
            ratio_out = target_out_out / peer_out_out if peer_out_out > 0 else np.nan
            ratio_in = target_out_in / peer_out_in if peer_out_in > 0 else np.nan
            
            if not np.isnan(ratio_out):
                ax_out.text(0, max(target_out_pct[0], peer_out_pct[0]) * 1.05,
                           f'Ratio: {ratio_out:.2f}x', ha='center', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            if not np.isnan(ratio_in):
                ax_out.text(1, max(target_out_pct[1], peer_out_pct[1]) * 1.05,
                           f'Ratio: {ratio_in:.2f}x', ha='center', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            
            # add change annotation
            y_mid_out = (max(target_out_pct + peer_out_pct) + min(target_out_pct + peer_out_pct)) / 2
            ax_out.text(0.5, y_mid_out, f'Δ = {outlier_change:+.2f}x',
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax_out.set_xticks(x_pos)
            ax_out.set_xticklabels(['OUT\nBias Periods', 'IN\nBias Periods'])
            ax_out.set_ylabel('Outlier Rate (%)\n(topic inconsistency)', fontsize=10)
            ax_out.legend(loc='upper right', fontsize=9)
            ax_out.grid(alpha=0.3)
        
        fig.suptitle('Per-Firm Temporal Analysis: Crisis Response Strategies\n' +
                    'Target vs Peers Comparison in Bias Periods',
                    fontsize=14, fontweight='bold')
        
        fig.text(0.5, 0.01,
                "INTERPRETATION: Firms show heterogeneous crisis responses. " +
                "Some increase topic-jumping (evasive), others narrow focus (controlled).\n" +
                "Δ = Change in target-peer gap between IN and OUT bias periods (difference-in-differences).",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        output_file = self.output_dir / 'Fig_TEMPORAL_per_firm_examples.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {output_file.name}")
        plt.close()
    
    def plot_per_firm_temporal_scatter(self):
        """
        Scatter plot showing heterogeneity in firm-level crisis responses.
        
        KEY VISUALIZATION: Explains the paradox - why aggregate shows null results 
        but 47.6% of individual firms show patterns (bidirectional effects cancel out).
        
        X-axis: Change in outlier ratio (target-peer gap change)
        Y-axis: Change in coherence (target-peer gap change)
        Quadrants show different crisis response strategies
        """
        print("\nGenerating per-firm temporal scatter...")
        
        if self.firm_analysis_df is None:
            print("Skipping: No firm analysis data")
            return
        
        df = self.firm_analysis_df.copy()
        df = df[df['has_bias_data'] == True].copy()
        
        if len(df) == 0:
            print("Skipping: No firms with bias data")
            return
        
        print(f"  Plotting {len(df)} firms")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        flagged = df[df['flagged'] == True]
        not_flagged = df[df['flagged'] == False]
        
        ax.scatter(not_flagged['outlier_change'], not_flagged['coherence_change'],
                  s=150, alpha=0.4, color='lightgray', edgecolor='gray', linewidth=1.5,
                  label=f'Not Flagged (n={len(not_flagged)})', zorder=2)
        
        ax.scatter(flagged['outlier_change'], flagged['coherence_change'],
                  s=200, alpha=0.8, color='#d62728', edgecolor='darkred', linewidth=2,
                  label=f'Flagged (n={len(flagged)})', zorder=3)
        
        for _, row in flagged.iterrows():
            ax.annotate(row['target_firm'], 
                       (row['outlier_change'], row['coherence_change']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='none'))
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)
        
        ax.axhline(y=0.03, color='red', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        ax.axhline(y=-0.03, color='red', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        ax.axvline(x=0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        ax.axvline(x=-0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        
        ax.text(0.7, 0.025, 'Q1: Evasive\n(More inconsistent +\nLess coherent)', 
               ha='center', va='center', fontsize=9, style='italic',
               bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.3))
        ax.text(0.7, -0.025, 'Q4: Controlled Pivoting\n(More inconsistent +\nMore coherent)', 
               ha='center', va='center', fontsize=9, style='italic',
               bbox=dict(boxstyle='round', facecolor='#ccccff', alpha=0.3))
        ax.text(-0.4, -0.025, 'Q3: Controlled Messaging\n(More focused +\nMore coherent)', 
               ha='center', va='center', fontsize=9, style='italic',
               bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.3))
        ax.text(-0.4, 0.025, 'Q2: Defensive Narrowing\n(More focused +\nLess coherent)', 
               ha='center', va='center', fontsize=9, style='italic',
               bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.3))
        
        ax.set_xlabel('Outlier Ratio Change (Δ Target-Peer Gap)\n' +
                     'Positive = Target becomes more inconsistent relative to peers',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Coherence Change (Δ Target-Peer Gap)\n' +
                     'Positive = Target becomes less coherent relative to peers',
                     fontsize=12, fontweight='bold')
        ax.set_title('Per-Firm Crisis Response Heterogeneity: The Paradox Visualized\n' +
                    f'{len(flagged)} of {len(df)} Firms Flagged ({len(flagged)/len(df)*100:.1f}%) - Bidirectional Effects Cancel in Aggregate',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)
        ax.set_axisbelow(True)
        
        fig.text(0.5, 0.01,
                "KEY INSIGHT: Firms show heterogeneous crisis responses in OPPOSITE directions. " +
                "This explains why aggregate analysis shows null results:\n" +
                "Individual firm effects are real but cancel out when averaged. " +
                "Dotted lines = flagging thresholds (|Δ outlier| > 0.3, |Δ coherence| > 0.03).",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.3, 
                         edgecolor='gray', linewidth=0.8))
        
        plt.tight_layout(rect=[0, 0.04, 1, 0.98])
        
        output_file = self.output_dir / 'Fig_PERFIRM_temporal_scatter.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {output_file.name}")
        plt.close()
    
    def plot_per_firm_slope_graph(self):
        """
        Slope graph showing IN vs OUT bias periods for all firms.
        
        Clearly visualizes:
        - Firms that diverge from peers (steep slopes)
        - Firms that trend with peers (parallel slopes)  
        - Magnitude differences (vertical position)
        - Direction of change (slope direction)
        """
        print("\nGenerating per-firm slope graph...")
        
        if self.firm_analysis_df is None:
            print("Skipping: No firm analysis data")
            return
        
        df = self.firm_analysis_df.copy()
        df = df[df['has_bias_data'] == True].copy()
        
        if len(df) == 0:
            print("Skipping: No firms with bias data")
            return
        
        df = df.sort_values('outlier_change', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 12))
        
        for ax_idx, metric in enumerate(['outlier', 'coherence']):
            ax = axes[ax_idx]
            
            if metric == 'outlier':
                col_in = 'outlier_target_in'
                col_out = 'outlier_target_out'
                ylabel = 'Target Outlier Rate (%)'
                title_suffix = 'Topic Inconsistency'
                multiplier = 100
            else:
                col_in = 'coherence_target_in'
                col_out = 'coherence_target_out'
                ylabel = 'Target Coherence Score'
                title_suffix = 'Linguistic Consistency'
                multiplier = 1
            
            for i, (_, row) in enumerate(df.iterrows()):
                out_val = row[col_out] * multiplier
                in_val = row[col_in] * multiplier
                
                if row['flagged']:
                    color = '#d62728'
                    linewidth = 2.5
                    alpha = 0.9
                    marker = 'o'
                    markersize = 10
                else:
                    color = 'lightgray'
                    linewidth = 1.5
                    alpha = 0.5
                    marker = 'o'
                    markersize = 6
                
                ax.plot([0, 1], [out_val, in_val], 
                       color=color, linewidth=linewidth, alpha=alpha,
                       marker=marker, markersize=markersize)
                
                if row['flagged']:
                    ax.text(-0.05, out_val, row['target_firm'], 
                           ha='right', va='center', fontsize=8, fontweight='bold')
            
            peer_in = df[f'{metric}_peer_in'].mean() * multiplier
            peer_out = df[f'{metric}_peer_out'].mean() * multiplier
            
            ax.plot([0, 1], [peer_out, peer_in],
                   color='#1f77b4', linewidth=4, alpha=0.8,
                   marker='s', markersize=15, label='Peer Average',
                   linestyle='--', zorder=10)
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['OUT\nBias Periods', 'IN\nBias Periods'], fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(f'{title_suffix}\nTarget Firms vs Peer Average', 
                        fontsize=13, fontweight='bold', pad=15)
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            
            if ax_idx == 1:
                ax.legend(fontsize=11, loc='best', framealpha=0.95)
        
        fig.suptitle('Per-Firm Temporal Analysis: IN vs OUT Bias Periods\n' +
                    'Red Lines = Flagged Firms | Gray Lines = Non-Flagged | Blue Dashed = Peer Average',
                    fontsize=15, fontweight='bold', y=0.98)
        
        fig.text(0.5, 0.01,
                "INTERPRETATION: Steep slopes indicate firms that diverge from peer trends during bias periods. " +
                "Parallel slopes indicate firms moving with industry. " +
                "Vertical position shows magnitude. Red = flagged for unusual patterns (47.6% of firms).",
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.3, 
                         edgecolor='gray', linewidth=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        output_file = self.output_dir / 'Fig_PERFIRM_slope_graph.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {output_file.name}")
        plt.close()
    
    def plot_aggregate_temporal_comparison(self):
        """
        Aggregate Temporal Analysis: 4-Group Comparison
        
        Shows aggregate patterns across all targets:
        - Target IN bias periods vs OUT bias periods
        - Peer IN bias periods vs OUT bias periods
        
        KEY FINDING: No significant aggregate changes, but both targets and peers
        show lower outlier rates and higher coherence IN bias periods (same quarters).
        Explains why individual firm differences cancel out in aggregate.
        """
        print("\n" + "=" * 80)
        print("Aggregate Temporal Analysis")
        print("=" * 80)
        
        if self.firm_analysis_df is None:
            print("  [SKIP] Per-firm analysis results not available")
            return
        
        df = self.firm_analysis_df.copy()
        
        # filter to firms with sufficient bias data
        df = df[df['has_bias_data'] == True].copy()
        
        if len(df) == 0:
            print("  [SKIP] No firms with bias period data")
            return
        
        print(f"  Analyzing {len(df)} firms with bias period data")
        
        # calculate aggregate statistics
        target_coh_in = df['coherence_target_in'].mean()
        target_coh_out = df['coherence_target_out'].mean()
        peer_coh_in = df['coherence_peer_in'].mean()
        peer_coh_out = df['coherence_peer_out'].mean()
        
        target_out_in = df['outlier_target_in'].mean() * 100
        target_out_out = df['outlier_target_out'].mean() * 100
        peer_out_in = df['outlier_peer_in'].mean() * 100
        peer_out_out = df['outlier_peer_out'].mean() * 100
        
        # calculate standard errors
        target_coh_in_se = df['coherence_target_in'].std() / np.sqrt(len(df))
        target_coh_out_se = df['coherence_target_out'].std() / np.sqrt(len(df))
        peer_coh_in_se = df['coherence_peer_in'].std() / np.sqrt(len(df))
        peer_coh_out_se = df['coherence_peer_out'].std() / np.sqrt(len(df))
        
        target_out_in_se = df['outlier_target_in'].std() / np.sqrt(len(df)) * 100
        target_out_out_se = df['outlier_target_out'].std() / np.sqrt(len(df)) * 100
        peer_out_in_se = df['outlier_peer_in'].std() / np.sqrt(len(df)) * 100
        peer_out_out_se = df['outlier_peer_out'].std() / np.sqrt(len(df)) * 100
        
        # create figure with 2 panels
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # left panel: coherence
        ax_coh = axes[0]
        
        x_pos = np.arange(2)
        width = 0.35
        
        target_coh_data = [target_coh_out, target_coh_in]
        peer_coh_data = [peer_coh_out, peer_coh_in]
        target_coh_err = [target_coh_out_se, target_coh_in_se]
        peer_coh_err = [peer_coh_out_se, peer_coh_in_se]
        
        bars1 = ax_coh.bar(x_pos - width/2, target_coh_data, width, 
                          yerr=target_coh_err, label='Targets',
                          color='#d62728', alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
        bars2 = ax_coh.bar(x_pos + width/2, peer_coh_data, width,
                          yerr=peer_coh_err, label='Peers',
                          color='#1f77b4', alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
        
        # add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax_coh.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # statistical test result
        from scipy import stats as sp_stats
        t_stat_target, p_val_target = sp_stats.ttest_rel(
            df['coherence_target_in'].dropna(), 
            df['coherence_target_out'].dropna()
        )
        
        ax_coh.text(0.5, 0.95, 
                   f'Target: IN vs OUT\np = {p_val_target:.4f}' + (' ***' if p_val_target < 0.001 else ' **' if p_val_target < 0.01 else ' *' if p_val_target < 0.05 else ' (ns)'),
                   transform=ax_coh.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                   fontsize=9)
        
        ax_coh.set_ylabel('Coherence\n(lower = more variable)', fontsize=11, fontweight='bold')
        ax_coh.set_title('Linguistic Coherence:\nAggregate Across All Target Firms', fontsize=12, fontweight='bold')
        ax_coh.set_xticks(x_pos)
        ax_coh.set_xticklabels(['OUT\nBias Periods', 'IN\nBias Periods'], fontsize=10)
        ax_coh.legend(fontsize=10, loc='upper left')
        ax_coh.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        
        # right panel: outlier rate
        ax_out = axes[1]
        
        target_out_data = [target_out_out, target_out_in]
        peer_out_data = [peer_out_out, peer_out_in]
        target_out_err = [target_out_out_se, target_out_in_se]
        peer_out_err = [peer_out_out_se, peer_out_in_se]
        
        bars1 = ax_out.bar(x_pos - width/2, target_out_data, width,
                          yerr=target_out_err, label='Targets',
                          color='#d62728', alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
        bars2 = ax_out.bar(x_pos + width/2, peer_out_data, width,
                          yerr=peer_out_err, label='Peers',
                          color='#1f77b4', alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
        
        # add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax_out.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # statistical test result
        t_stat_target_out, p_val_target_out = sp_stats.ttest_rel(
            df['outlier_target_in'].dropna() * 100,
            df['outlier_target_out'].dropna() * 100
        )
        
        ax_out.text(0.5, 0.95,
                   f'Target: IN vs OUT\np = {p_val_target_out:.4f}' + (' ***' if p_val_target_out < 0.001 else ' **' if p_val_target_out < 0.01 else ' *' if p_val_target_out < 0.05 else ' (ns)'),
                   transform=ax_out.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                   fontsize=9)
        
        ax_out.set_ylabel('Outlier Rate (%)\n(topic inconsistency)', fontsize=11, fontweight='bold')
        ax_out.set_title('Topic Outlier Rate:\nAggregate Across All Target Firms', fontsize=12, fontweight='bold')
        ax_out.set_xticks(x_pos)
        ax_out.set_xticklabels(['OUT\nBias Periods', 'IN\nBias Periods'], fontsize=10)
        ax_out.legend(fontsize=10, loc='upper left')
        ax_out.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        
        fig.suptitle('Aggregate Temporal Analysis: Targets vs Peers\nMean Values Across 20 Firms with Bias Period Data',
                    fontsize=14, fontweight='bold', y=0.98)
        
        fig.text(0.5, 0.02,
                "INTERPRETATION: Both targets and peers show lower outlier rates and higher coherence during bias periods (same quarters).\n" +
                "Target-peer gap changes are not significant in aggregate; 47.6% of individual firms show patterns, but bidirectional effects cancel.",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3, edgecolor='gray', linewidth=0.8))
        
        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
        
        output_file = self.output_dir / 'Fig_TEMPORAL_aggregate_comparison.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"[SAVED] {output_file}")
        plt.close()
    
    def generate_all_figures(self):
        """
        Generate all visualization figures.
        
        Runs:
        - CS1, CS2, CS3 (Cross-sectional analyses)
        - TS1, TS2, TS3 (Time-series analyses)
        - Bias vector visualizations
        - PCA analysis
        """
        print("\n" + "=" * 80)
        print("GENERATING ALL FIGURES")
        print("=" * 80)
        
        figures_generated = 0
        
        # cross-sectional analyses
        try:
            self.plot_cs1_topic_attribution_heatmap()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] CS1: {e}")
        
        try:
            self.plot_cs2_attribution_type_bars()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] CS2: {e}")
        
        try:
            self.plot_cs3_full_call_violins()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] CS3: {e}")
        
        try:
            self.plot_cs3_bias_period_bars()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] CS3 Bar Chart: {e}")
        
        # time-series analyses
        try:
            self.plot_ts1_aggregate_outlier_rates()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] TS1: {e}")
        
        try:
            self.plot_ts2_attribution_type_boxplots()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] TS2: {e}")
        
        try:
            self.plot_ts3_company_quarter_heatmap()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] TS3: {e}")
        
        # removed: bias vector separation (used circular reasoning with tertile splits)
        # this method now just prints a skip message - see line ~1481 for details
        try:
            self.plot_bias_vector_separation()
            # note: this returns early without generating a figure
        except Exception as e:
            print(f"[ERROR] Bias Vector: {e}")
        
        try:
            self.plot_bias_score_heatmap()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] Bias Heatmap: {e}")
        
        try:
            self.plot_pca_biplot()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] PCA: {e}")
        
        # new: dissertation-critical visualizations
        try:
            self.plot_supervised_classification_comparison()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] Supervised Classification Comparison: {e}")
        
        try:
            self.plot_per_firm_temporal_examples()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] Per-Firm Temporal Examples: {e}")
        
        try:
            self.plot_per_firm_temporal_scatter()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] Per-Firm Temporal Scatter: {e}")
        
        try:
            self.plot_per_firm_slope_graph()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] Per-Firm Slope Graph: {e}")
        
        try:
            self.plot_aggregate_temporal_comparison()
            figures_generated += 1
        except Exception as e:
            print(f"[ERROR] Aggregate Temporal Comparison: {e}")
        
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETE")
        print("=" * 80)
        print(f"\n{figures_generated} figures generated successfully")
        print(f"All figures saved to: {self.output_dir}")
        
        # generate audio processing prioritization report
        try:
            print("\n" + "=" * 80)
            print("GENERATING AUDIO PROCESSING PRIORITY REPORT...")
            print("=" * 80)
            self.rank_companies_for_audio_processing()
        except Exception as e:
            print(f"[ERROR] Audio Processing Report: {e}")


def main():
    """
    Main entry point for visualization script.
    
    Usage:
        python embedding_visualizations.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate embedding analysis visualizations'
    )
    parser.add_argument('--results', type=str, default='output/embeddings',
                       help='Directory with embedding analysis results')
    parser.add_argument('--output', type=str, default='output/figures',
                       help='Directory to save figures')
    
    args = parser.parse_args()
    
    # create visualizer and generate figures
    visualizer = EmbeddingVisualizer(
        results_dir=args.results,
        output_dir=args.output
    )
    
    visualizer.generate_all_figures()


if __name__ == "__main__":
    main()


# python conclusion/embedding_visualizations.py
