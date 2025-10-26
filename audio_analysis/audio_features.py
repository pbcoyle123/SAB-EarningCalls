"""
Audio Feature Extraction Pipeline for Earnings Call Analysis

This module extracts audio features from earnings call recordings to analyze vocal characteristics
that may correlate with bias patterns identified in text analysis.

Data Flow:
1. Input: Bias-classified segments from SNAP_simple_merged_bias_class.csv
2. Audio: Raw MP3 files from test_output/raw_audio_files/ .. update if folder names change
3. Output: Audio features for each segment in 20-second chunks

Implementation Strategy:
- Primary: openSMILE with eGeMAPS feature set for proven acoustic features
- Secondary: Wav2Vec2 embeddings for advanced vocal characteristics
- Tertiary: Custom VDQ (Vocal Delivery Quality) metrics
"""

import pandas as pd
import numpy as np
import os
import json
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    import soundfile as sf
    from pydub import AudioSegment
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("Warning: Audio processing libraries not available. Install with: pip install librosa soundfile pydub")

try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False
    print("Warning: OpenSMILE not available. Install with: pip install opensmile")

try:
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers torch")

class AudioFeatureExtractor:
    
    def __init__(self, 
                 bias_data_path: str = "test_output/bias_classification/SNAP_simple_merged_bias_class.csv",
                 audio_base_path: str = "test_output/raw_audio_files",
                 output_path: str = "test_output/audio_features",
                 chunk_duration: int = 20):
        self.bias_data_path = bias_data_path
        self.audio_base_path = Path(audio_base_path)
        self.output_path = Path(output_path)
        self.chunk_duration = chunk_duration
        
        self.output_path.mkdir(exist_ok=True)
        self._setup_logging()
        self._initialize_extractors()
        
    def _setup_logging(self):
        log_file = self.output_path / f"audio_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_extractors(self):
        self.extractors = {}
        
        if OPENSMILE_AVAILABLE:
            try:
                self.extractors['opensmile'] = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.LowLevelDescriptors
                )
                self.logger.info("OpenSMILE extractor initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenSMILE: {e}")
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.extractors['wav2vec2'] = {
                    'model': Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base'),
                    'feature_extractor': Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
                }
                self.logger.info("Wav2Vec2 extractor initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Wav2Vec2: {e}")
    
    def load_bias_data(self) -> pd.DataFrame:
        self.logger.info(f"Loading bias data from {self.bias_data_path}")
        
        try:
            df = pd.read_csv(self.bias_data_path)
            self.logger.info(f"Loaded {len(df)} bias classification records")
            df = self._map_audio_files(df)
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load bias data: {e}")
            raise
    
    def _map_audio_files(self, df: pd.DataFrame) -> pd.DataFrame:
        df['audio_file'] = df.apply(self._get_audio_file_path, axis=1)
        
        initial_count = len(df)
        df = df[df['audio_file'].notna()]
        filtered_count = len(df)
        
        self.logger.info(f"Audio file mapping complete: {filtered_count}/{initial_count} records have audio files")
        
        return df
    
    def _get_audio_file_path(self, row: pd.Series) -> Optional[str]:
        company = row.get('Company', '').strip()
        quarter = row.get('quarter', '')
        year = row.get('year', '')
        
        if not all([company, quarter, year]):
            return None
        
        audio_filename = f"{company} Q{quarter} {year}.mp3"
        
        possible_paths = [
            self.audio_base_path / audio_filename,
            self.audio_base_path / company / audio_filename,
            self.audio_base_path / f"{company}/" / audio_filename
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def extract_segment_features(self, audio_file: str, start_time: float, end_time: float, feature_types: List[str] = None) -> Dict:
        if feature_types is None:
            feature_types = ['opensmile', 'wav2vec2', 'basic', 'vdq']
        
        features = {}
        
        try:
            audio_segment = self._load_audio_segment(audio_file, start_time, end_time)
            
            if audio_segment is None:
                return features
            
            if 'opensmile' in feature_types and 'opensmile' in self.extractors:
                features.update(self._extract_opensmile_features(audio_segment))
            
            if 'wav2vec2' in feature_types and 'wav2vec2' in self.extractors:
                features.update(self._extract_wav2vec2_features(audio_segment))
            
            if 'basic' in feature_types:
                features.update(self._extract_basic_features(audio_segment))
            
            if 'vdq' in feature_types:
                features.update(self._extract_vdq_features(audio_segment))
            
        except Exception as e:
            self.logger.error(f"Error extracting features from {audio_file}: {e}")
        
        return features
    
    def _load_audio_segment(self, audio_file: str, start_time: float, end_time: float) -> Optional[np.ndarray]:
        try:
            audio = AudioSegment.from_mp3(audio_file)
            
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            segment = audio[start_ms:end_ms]
            
            samples = np.array(segment.get_array_of_samples())
            
            if segment.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)
            
            return samples.astype(np.float32) / 32768.0
            
        except Exception as e:
            self.logger.error(f"Error loading audio segment: {e}")
            return None
    
    def _extract_opensmile_features(self, audio: np.ndarray) -> Dict:
        features = {}
        
        try:
            temp_file = self.output_path / "temp_audio.wav"
            sf.write(temp_file, audio, 16000)
            
            smile_features = self.extractors['opensmile'].process_file(str(temp_file))
            
            if not smile_features.empty:
                for col in smile_features.columns:
                    features[f'opensmile_{col}_mean'] = float(smile_features[col].mean())
                    features[f'opensmile_{col}_std'] = float(smile_features[col].std())
            
            temp_file.unlink(missing_ok=True)
            
        except Exception as e:
            self.logger.error(f"Error extracting OpenSMILE features: {e}")
        
        return features
    
    def _extract_wav2vec2_features(self, audio: np.ndarray) -> Dict:
        features = {}
        
        try:
            # process long audio in 30-second chunks to avoid memory issues
            max_samples = 30 * 16000
            if len(audio) > max_samples:
                self.logger.info(f"Audio too long ({len(audio)/16000:.1f}s), using chunked processing")
                
                chunk_size = 30 * 16000
                all_embeddings = []
                
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                    
                    inputs = self.extractors['wav2vec2']['feature_extractor'](
                        chunk, 
                        sampling_rate=16000, 
                        return_tensors="pt"
                    )
                    
                    with torch.no_grad():
                        outputs = self.extractors['wav2vec2']['model'](**inputs)
                        hidden_states = outputs.last_hidden_state.squeeze()
                        chunk_embedding = hidden_states.mean(dim=0)
                        all_embeddings.append(chunk_embedding.cpu().numpy())
                
                all_embeddings = np.array(all_embeddings)
                features['wav2vec2_embedding_mean'] = float(all_embeddings.mean())
                features['wav2vec2_embedding_std'] = float(all_embeddings.std())
                
                for i in range(min(10, all_embeddings.shape[1])):
                    features[f'wav2vec2_pc_{i}'] = float(all_embeddings[:, i].mean())
                
            else:
                inputs = self.extractors['wav2vec2']['feature_extractor'](
                    audio, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    outputs = self.extractors['wav2vec2']['model'](**inputs)
                    hidden_states = outputs.last_hidden_state.squeeze()
                    
                    features['wav2vec2_embedding_mean'] = float(hidden_states.mean().item())
                    features['wav2vec2_embedding_std'] = float(hidden_states.std().item())
                    
                    if hidden_states.shape[0] > 10:
                        for i in range(10):
                            features[f'wav2vec2_pc_{i}'] = float(hidden_states[:, i].mean().item())
            
        except Exception as e:
            self.logger.error(f"Error extracting Wav2Vec2 features: {e}")
        
        return features
    
    def _extract_basic_features(self, audio: np.ndarray) -> Dict:
        features = {}
        
        if not AUDIO_LIBS_AVAILABLE:
            return features
        
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=16000)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 95)]
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
                features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
            
            rms = librosa.feature.rms(y=audio)[0]
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            features['energy_range'] = float(np.max(rms) - np.min(rms))
            
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=16000)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zero_crossing_rate_mean'] = float(np.mean(zcr))
            features['zero_crossing_rate_std'] = float(np.std(zcr))
            
        except Exception as e:
            self.logger.error(f"Error extracting basic features: {e}")
        
        return features
    
    def _extract_vdq_features(self, audio: np.ndarray) -> Dict:
        features = {}
        
        try:
            energy = librosa.feature.rms(y=audio)[0]
            peaks = librosa.util.peak_pick(energy, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
            speech_rate = len(peaks) / (len(audio) / 16000)
            features['speech_rate'] = float(speech_rate)
            
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=16000)[0]
            features['spectral_rolloff_mean'] = float(np.mean(rolloff))
            
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=16000)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
            
            # todo: gop (goodness of pronunciation) metric - requires wav2vec2 phoneme head
            
        except Exception as e:
            self.logger.error(f"Error extracting VDQ features: {e}")
        
        return features
    
    def process_all_segments(self, max_segments: Optional[int] = None) -> pd.DataFrame:
        df = self.load_bias_data()
        
        if max_segments:
            df = df.head(max_segments)
        
        self.logger.info(f"Processing {len(df)} segments for audio feature extraction")
        
        audio_features_list = []
        
        for idx, row in df.iterrows():
            self.logger.info(f"Processing segment {idx + 1}/{len(df)}: {row.get('Company', 'Unknown')} Q{row.get('quarter', '?')} {row.get('year', '?')}")
            
            features = self.extract_segment_features(
                row['audio_file'],
                start_time=0,
                end_time=self.chunk_duration
            )
            
            segment_data = row.to_dict()
            segment_data.update(features)
            audio_features_list.append(segment_data)
        
        result_df = pd.DataFrame(audio_features_list)
        
        output_file = self.output_path / f"audio_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_df.to_csv(output_file, index=False)
        
        self.logger.info(f"Audio feature extraction complete. Results saved to {output_file}")
        
        return result_df

def test_single_audio_file(feature_types=None):
    if feature_types is None:
        feature_types = ['opensmile', 'wav2vec2', 'basic', 'vdq']
    
    print("=== audio feature extraction test ===")
    print(f"testing with features: {', '.join(feature_types)}")
    
    extractor = AudioFeatureExtractor()
    
    audio_files = list(extractor.audio_base_path.glob("*.mp3"))
    if not audio_files:
        print("no mp3 files found in test_output/raw_audio_files/")
        return
    
    test_audio_file = str(audio_files[0])
    print(f"using audio file: {test_audio_file}")
    
    try:
        from pydub import AudioSegment
        full_audio = AudioSegment.from_mp3(test_audio_file)
        duration_seconds = len(full_audio) / 1000.0
        print(f"audio duration: {duration_seconds:.2f}s ({duration_seconds/60:.1f}min)")
        
        if duration_seconds <= 300:
            process_duration = duration_seconds
            print(f"processing full file ({process_duration:.1f}s)")
        else:
            process_duration = 300
            print(f"processing first 5 minutes ({process_duration}s)")
        
    except Exception as e:
        print(f"error loading audio file: {e}")
        return
    
    print(f"\nextracting features from {process_duration:.1f}s of audio...")
    start_time = time.time()
    
    try:
        features = extractor.extract_segment_features(
            test_audio_file,
            start_time=0,
            end_time=process_duration,
            feature_types=feature_types
        )
        
        processing_time = time.time() - start_time
        print(f"extracted {len(features)} features")
        print(f"processing time: {processing_time:.2f}s")
        print(f"speed: {process_duration/processing_time:.2f}x realtime")
        
        feature_categories = {
            'opensmile': [k for k in features.keys() if k.startswith('opensmile_')],
            'wav2vec2': [k for k in features.keys() if k.startswith('wav2vec2_')],
            'basic': [k for k in features.keys() if k.startswith(('pitch_', 'energy_', 'mfcc_', 'spectral_', 'zero_crossing'))],
            'vdq': [k for k in features.keys() if k.startswith(('speech_rate', 'spectral_rolloff', 'spectral_bandwidth'))]
        }
        
        for category, feature_list in feature_categories.items():
            if feature_list:
                print(f"   {category}: {len(feature_list)} features")
        
        print("\nsample features:")
        sample_features = list(features.items())[:10]
        for name, value in sample_features:
            if isinstance(value, float):
                print(f"   {name}: {value:.6f}")
            else:
                print(f"   {name}: {value}")
        
    except Exception as e:
        print(f"error extracting features: {e}")
        return
    
    print(f"\ncreating test output...")
    try:
        test_data = {
            'audio_file': [test_audio_file],
            'duration_seconds': [process_duration],
            'processing_time_seconds': [processing_time],
            'processing_speed_audio_per_second': [process_duration/processing_time],
            'test_timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            **features
        }
        
        test_df = pd.DataFrame([test_data])
        
        test_output_file = extractor.output_path / f"test_audio_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        test_df.to_csv(test_output_file, index=False)
        
        print(f"results saved to: {test_output_file}")
        
    except Exception as e:
        print(f"error saving test results: {e}")

def main():
    parser = argparse.ArgumentParser(description='audio feature extraction for earnings call analysis')
    parser.add_argument('-test', '--test', action='store_true', 
                       help='run test mode: process only the first audio file found')
    parser.add_argument('-test_only_smile', '--test_only_smile', action='store_true',
                       help='run test mode with only opensmile features')
    parser.add_argument('-test_smile_wav2vec', '--test_smile_wav2vec', action='store_true',
                       help='run test mode with opensmile + wav2vec2 features')
    parser.add_argument('-test_basic_only', '--test_basic_only', action='store_true',
                       help='run test mode with only basic audio features (librosa)')
    parser.add_argument('-segments', '--segments', type=int, default=10,
                       help='number of segments to process (default: 10)')
    
    args = parser.parse_args()
    
    if args.test:
        test_single_audio_file()
    elif args.test_only_smile:
        test_single_audio_file(feature_types=['opensmile'])
    elif args.test_smile_wav2vec:
        test_single_audio_file(feature_types=['opensmile', 'wav2vec2'])
    elif args.test_basic_only:
        test_single_audio_file(feature_types=['basic'])
    else:
        print("starting audio feature extraction pipeline...")
        
        extractor = AudioFeatureExtractor()
        
        results = extractor.process_all_segments(max_segments=args.segments)
        
        print(f"extraction complete! processed {len(results)} segments.")
        print(f"features extracted: {len([col for col in results.columns if col.startswith(('opensmile_', 'wav2vec2_', 'pitch_', 'energy_', 'mfcc_', 'spectral_', 'speech_rate'))])}")

if __name__ == "__main__":
    main()


# to test the code on creating features for a single audio file
# python audio_features.py -test
# python audio_features.py --test_only_smile
# python audio_features.py --test_basic_only

# python audio_features.py -h

# options:
#   -h, --help            show this help message and exit
#   -test, --test         Run test mode: process only the first audio file found
#   -test_only_smile, --test_only_smile
#                         Run test mode with only OpenSMILE features
#   -test_smile_wav2vec, --test_smile_wav2vec
#                         Run test mode with OpenSMILE + Wav2Vec2 features
#   -test_basic_only, --test_basic_only
#                         Run test mode with only basic audio features (librosa)
#   -segments, --segments SEGMENTS
#                         Number of segments to process (default: 10)
