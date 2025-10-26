# audio transcription with timestamps for earnings calls
import os
import glob
import json
from datetime import datetime
import whisper
from pydub import AudioSegment
import pandas as pd
import subprocess

def configure_ffmpeg():
    ffmpeg_dir = r'C:\Users\pbcoy\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin'
    ffmpeg_path = os.path.join(ffmpeg_dir, 'ffmpeg.exe')
    ffprobe_path = os.path.join(ffmpeg_dir, 'ffprobe.exe')
    
    if os.path.exists(ffmpeg_path):
        current_path = os.environ.get('PATH', '')
        if ffmpeg_dir not in current_path:
            os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path
        
        AudioSegment.converter = ffmpeg_path
        AudioSegment.ffprobe = ffprobe_path
        
        os.environ['FFMPEG_BINARY'] = ffmpeg_path
        os.environ['FFPROBE_BINARY'] = ffprobe_path
        
        print(f"configured ffmpeg at: {ffmpeg_path}")
        print(f"configured ffprobe at: {ffprobe_path}")
        print(f"added ffmpeg directory to path: {ffmpeg_dir}")
        return True
    return False

AUDIO_FOLDER = "test_output/raw_audio_files"
OUTPUT_FOLDER = "test_output/transcriptions"
COMPANY_KEYWORDS = ["SNAP"]

def check_ffmpeg():
    ffmpeg_paths = [
        'ffmpeg',
        r'C:\Users\pbcoy\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\ffmpeg\bin\ffmpeg.exe'
    ]
    
    for ffmpeg_path in ffmpeg_paths:
        try:
            result = subprocess.run([ffmpeg_path, '-version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"ffmpeg found at: {ffmpeg_path}")
                os.environ['FFMPEG_BINARY'] = ffmpeg_path
                return True
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
    
    print("ffmpeg not found in any expected location.")
    print("please ensure ffmpeg is installed and accessible.")
    return False

def create_output_directory():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"created output directory: {OUTPUT_FOLDER}")

def find_audio_files(company_keywords):
    audio_files = []
    
    all_mp3_files = glob.glob(os.path.join(AUDIO_FOLDER, "*.mp3"))
    
    for file_path in all_mp3_files:
        filename = os.path.basename(file_path)
        
        for keyword in company_keywords:
            if keyword.upper() in filename.upper():
                audio_files.append({
                    'file_path': file_path,
                    'filename': filename,
                    'company': keyword,
                    'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
                })
                break
    
    return audio_files

def load_whisper_model():
    print("loading whisper model...")
    model = whisper.load_model("base")
    print("whisper model loaded successfully!")
    return model

def transcribe_audio_with_timestamps(model, audio_file_path):
    print(f"transcribing: {os.path.basename(audio_file_path)}")
    
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"audio file not found: {audio_file_path}")
    
    print(f"file size: {os.path.getsize(audio_file_path) / (1024*1024):.1f} mb")
    
    result = model.transcribe(
        audio_file_path,
        verbose=True,
        language="en"
    )
    
    return result

def process_transcription_result(result, filename):
    transcription_data = {
        'filename': filename,
        'full_text': result['text'],
        'segments': []
    }
    
    for segment in result['segments']:
        segment_data = {
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip(),
            'duration': segment['end'] - segment['start']
        }
        
        transcription_data['segments'].append(segment_data)
    
    print(f"timestamp granularity:")
    print(f"  - segments: {len(transcription_data['segments'])} segments")
    if transcription_data['segments']:
        avg_segment_duration = sum([s['duration'] for s in transcription_data['segments']]) / len(transcription_data['segments'])
        print(f"  - average segment duration: {avg_segment_duration:.2f} seconds")
    
    return transcription_data

def save_transcription_results(transcription_data, output_folder):
    filename_base = os.path.splitext(transcription_data['filename'])[0]
    
    json_path = os.path.join(output_folder, f"{filename_base}_transcription.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcription_data, f, indent=2, ensure_ascii=False)
    
    segments_data = []
    for segment in transcription_data['segments']:
        segments_data.append({
            'start_time': segment['start'],
            'end_time': segment['end'],
            'duration': segment['end'] - segment['start'],
            'text': segment['text']
        })
    
    segments_df = pd.DataFrame(segments_data)
    csv_path = os.path.join(output_folder, f"{filename_base}_segments.csv")
    segments_df.to_csv(csv_path, index=False, encoding='utf-8')
    
    txt_path = os.path.join(output_folder, f"{filename_base}_transcription.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"transcription: {transcription_data['filename']}\n")
        f.write("=" * 50 + "\n\n")
        
        for segment in transcription_data['segments']:
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            f.write(f"[{start_time} - {end_time}] {segment['text']}\n\n")
    
    return {
        'json': json_path,
        'csv': csv_path,
        'txt': txt_path
    }

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds_remainder = int(seconds % 60)
    return f"{minutes:02d}:{seconds_remainder:02d}"

def search_text_in_transcription(transcription_data, search_terms):
    search_results = []
    
    for segment in transcription_data['segments']:
        segment_text = segment['text'].lower()
        
        for term in search_terms:
            if term.lower() in segment_text:
                search_results.append({
                    'search_term': term,
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'text': segment['text'],
                    'formatted_start': format_time(segment['start']),
                    'formatted_end': format_time(segment['end'])
                })
    
    return search_results

def main():
    print("starting audio transcription process")
    print("=" * 50)
    
    configure_ffmpeg()
    
    if not check_ffmpeg():
        print("\nplease install ffmpeg and restart your shell, then try again.")
        return
    
    create_output_directory()
    
    print(f"searching for audio files with keywords: {COMPANY_KEYWORDS}")
    audio_files = find_audio_files(COMPANY_KEYWORDS)
    
    if not audio_files:
        print("no audio files found matching the specified keywords.")
        print("available files:")
        all_files = glob.glob(os.path.join(AUDIO_FOLDER, "*.mp3"))
        for file in all_files[:10]:
            print(f"  - {os.path.basename(file)}")
        if len(all_files) > 10:
            print(f"  ... and {len(all_files) - 10} more files")
        return
    
    print(f"found {len(audio_files)} audio files to process:")
    for file_info in audio_files:
        print(f"  - {file_info['filename']} ({file_info['file_size_mb']:.1f} mb)")
    
    model = load_whisper_model()
    
    all_results = []
    
    for i, file_info in enumerate(audio_files, 1):
        print(f"\nprocessing file {i}/{len(audio_files)}: {file_info['filename']}")
        
        try:
            result = transcribe_audio_with_timestamps(model, file_info['file_path'])
            
            transcription_data = process_transcription_result(result, file_info['filename'])
            
            saved_files = save_transcription_results(transcription_data, OUTPUT_FOLDER)
            
            search_terms = ["revenue", "profit", "growth", "quarter", "earnings"]
            search_results = search_text_in_transcription(transcription_data, search_terms)
            
            file_result = {
                'file_info': file_info,
                'transcription_data': transcription_data,
                'saved_files': saved_files,
                'search_results': search_results
            }
            
            all_results.append(file_result)
            
            print(f"completed: {file_info['filename']}")
            print(f"  - saved to: {saved_files['txt']}")
            print(f"  - found {len(search_results)} search term matches")
            
        except Exception as e:
            print(f"error processing {file_info['filename']}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("transcription completed")
    print("=" * 50)
    print(f"total files processed: {len(all_results)}")
    print(f"output folder: {OUTPUT_FOLDER}")
    
    if all_results:
        print("\nexample search results (revenue/profit/growth mentions):")
        for result in all_results:
            if result['search_results']:
                print(f"\n{result['file_info']['filename']}:")
                for search_result in result['search_results'][:3]:
                    print(f"  [{search_result['formatted_start']}] {search_result['search_term']}: {search_result['text'][:100]}...")

if __name__ == "__main__":
    main()



# python audio_transcibe.py
