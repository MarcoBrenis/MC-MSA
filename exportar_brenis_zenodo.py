import os
import re
import csv
import json
import math
from pathlib import Path

def clean_filename(name):
    # Remove unicode characters or normalize
    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)

def find_json_cache(audio_path, cache_dir):
    # Try exact match first
    stem = audio_path.stem
    exact_json = cache_dir / f"{stem}.json"
    if exact_json.exists():
        return exact_json
        
    # If not, look for matching digits at the start
    match = re.match(r'^(\d+)', audio_path.name)
    if match:
        prefix = match.group(1)
        # Find all json files starting with this prefix
        candidates = list(cache_dir.glob(f"{prefix} *.json"))
        # Also try prefix without space
        candidates += list(cache_dir.glob(f"{prefix}*.json"))
        # De-duplicate and filter out .tiny.json
        candidates = list({c for c in candidates if not c.name.endswith(".tiny.json")})
        
        # If there are candidates, find the one that shares the most words
        if candidates:
            best_cand = None
            best_overlap = -1
            audio_words = set(re.findall(r'\w+', stem.lower()))
            for cand in candidates:
                cand_words = set(re.findall(r'\w+', cand.stem.lower()))
                overlap = len(audio_words.intersection(cand_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cand = cand
            if best_cand:
                return best_cand
    return None

def normalize_label(label: str) -> str:
    l = label.strip().lower()
    if l in ['pregunta', 'antecedent', 'a', 'q', 'question']:
        return 'Antecedent'
    elif l in ['respuesta', 'consequent', 'c', 'r', 'answer']:
        return 'Consequent'
    elif l in ['silencio', 'silence', 'x', 's']:
        return 'Silence'
    return label

def assign_label_to_time(t, segments):
    # Since segments are sorted, find the segment containing t
    for seg in segments:
        start = seg['start_time']
        end = seg['end_time']
        # Use small epsilon for boundaries
        if start - 1e-5 <= t <= end + 1e-5:
            return normalize_label(seg['label'])
    return "Silence" # Default to Silence if not in any segment

def export_dataset():
    base_dir = Path(".").resolve()
    dataset_dir = base_dir / "dataset_OA"
    cache_dir = base_dir / "cache" / "bs_roformer_rmvpe"
    
    output_dir = base_dir / "BRENIS_dataset"
    seq_dir = output_dir / "sequences"
    
    seq_dir.mkdir(parents=True, exist_ok=True)
    
    orig_dir = dataset_dir / "originales"
    cover_dir = dataset_dir / "covers"
    
    if not orig_dir.exists() or not cover_dir.exists():
        print(f"Error: dataset_OA directories not found in {base_dir}")
        return
        
    # Read the popular metadata csv
    metadata_in_path = base_dir / "dataset_popular.csv"
    if not metadata_in_path.exists():
        print(f"Error: {metadata_in_path} not found.")
        return
        
    records = []
    with open(metadata_in_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)
            
    print(f"Loaded {len(records)} metadata records. Processing frame-level data...")
    
    # Let's scan original and cover directories to index paths by ID and type
    orig_files = {}
    cover_files = {}
    
    # We group by track ID and match as in extraer_nya.py
    for f in orig_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp3', '.wav']:
            m = re.match(r'^(\d+)', f.name)
            if m:
                track_id = int(m.group(1))
                orig_files.setdefault(track_id, []).append(f)
                
    for f in cover_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp3', '.wav']:
            m = re.match(r'^(\d+)', f.name)
            if m:
                track_id = int(m.group(1))
                cover_files.setdefault(track_id, []).append(f)
                
    # New metadata records with sequence file paths
    new_records = []
    skipped_count = 0
    success_count = 0
    
    # Track mapping of original/cover pairings for the duplicate track 55
    # Since Calle 13 is first original with 55, and Marco Antonio Solis is second,
    # let's match Calle 13 to Muerte en Hawaii cover, and Marco Antonio Solis to Muerte en Hawaii cover too.
    for r in records:
        track_id = int(r['track_id'])
        t_type = r['type'] # 'original' or 'cover'
        title = r['performance title']
        artist = r['performance artist']
        
        # Find corresponding audio file
        audio_file = None
        candidates = orig_files.get(track_id, []) if t_type == "original" else cover_files.get(track_id, [])
        
        if len(candidates) == 1:
            audio_file = candidates[0]
        elif len(candidates) > 1:
            # Match by words in artist/title
            best_overlap = -1
            query_words = set(re.findall(r'\w+', (artist + " " + title).lower()))
            for cand in candidates:
                cand_words = set(re.findall(r'\w+', cand.name.lower()))
                overlap = len(query_words.intersection(cand_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    audio_file = cand
        
        if not audio_file:
            print(f"Warning: Could not find audio file for track {track_id} ({t_type}) - {title} by {artist}")
            skipped_count += 1
            continue
            
        # Find JSON cache file
        json_file = find_json_cache(audio_file, cache_dir)
        if not json_file or not json_file.exists():
            print(f"Warning: Could not find JSON cache for {audio_file.name}")
            skipped_count += 1
            continue
            
        # Process the JSON cache and write CSV
        with open(json_file, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            
        times = data.get('times', [])
        pitch_midi = data.get('pitch_midi', [])
        confidence = data.get('confidence', [])
        energy = data.get('energy', [])
        segments = data.get('segments', [])
        
        if not times:
            print(f"Warning: Empty times in cache for {audio_file.name}")
            skipped_count += 1
            continue
            
        # Define output filename
        safe_name = f"{track_id:02d}_{t_type}_{clean_filename(audio_file.stem)}.csv"
        out_csv_path = seq_dir / safe_name
        
        # Write sequence CSV
        with open(out_csv_path, "w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["time", "f0_midi", "energy", "voicing", "label"])
            
            for i, t in enumerate(times):
                f0 = pitch_midi[i] if i < len(pitch_midi) else 0.0
                eng = energy[i] if i < len(energy) else 0.0
                vc = confidence[i] if i < len(confidence) else 0.0
                
                # Assign label (A, C, X)
                label = assign_label_to_time(t, segments)
                
                # Represent unvoiced f0 as NaN or 0.0
                # Using standard float format
                writer.writerow([
                    round(t, 4),
                    round(f0, 4) if not math.isnan(f0) and f0 > 0 else 0.0,
                    round(eng, 4),
                    round(vc, 4),
                    label
                ])
                
        # Store in metadata
        r_new = dict(r)
        r_new['sequence_file'] = f"sequences/{safe_name}"
        new_records.append(r_new)
        success_count += 1
        
    # Write the new metadata.csv
    new_metadata_path = output_dir / "metadata.csv"
    with open(new_metadata_path, "w", encoding="utf-8", newline="") as out_f:
        fieldnames = list(records[0].keys()) + ["sequence_file"]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for r_new in new_records:
            writer.writerow(r_new)
            
    # Generate BRENIS_sequences.zip containing metadata.csv, README.txt, and the sequences folder
    import zipfile
    zip_path = output_dir / "BRENIS_sequences.zip"
    print(f"\nCreating zip archive {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add README.txt
        readme_path = output_dir / "README.txt"
        if readme_path.exists():
            zipf.write(readme_path, arcname="README.txt")
        # Add metadata.csv
        if new_metadata_path.exists():
            zipf.write(new_metadata_path, arcname="metadata.csv")
        # Add all CSVs in sequences
        for csv_file in seq_dir.glob("*.csv"):
            zipf.write(csv_file, arcname=f"sequences/{csv_file.name}")
    print(f"Archive created successfully: {zip_path}")
            
    print(f"\nProcessing Complete!")
    print(f"Successfully exported {success_count} sequences to {seq_dir}")
    print(f"Skipped: {skipped_count} tracks due to missing audio or cache files.")
    print(f"New dataset metadata written to {new_metadata_path}")

if __name__ == "__main__":
    export_dataset()
