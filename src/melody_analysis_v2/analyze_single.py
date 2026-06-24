import sys
import argparse
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.melody_analysis_v2 import MelodyAnalyzer, MelodyClassifierPaper
from src.melody_analysis_v2.pipeline import MelodyAnalysisResult
from src.melody_analysis_v2.features import MelodyFeatures
from src.melody_analysis_v2.segmenter import MelodySegment
from src.melody_analysis_v2.classifier import MelodySegmentAnnotation

def load_or_analyze_in_sub(analyzer, file_path, method, cache_dir, label_prefix=""):
    cache_path = cache_dir / method / f"{file_path.stem}.json"
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            features = MelodyFeatures(
                times=np.array(data["times"]),
                pitch_midi=np.array(data["pitch_midi"]),
                confidence=np.array(data["confidence"]),
                energy=np.array(data["energy"])
            )
            result = analyzer.analyze_features(features)
            with open(cache_path, 'w') as f:
                json.dump(result.to_dict(), f)
            return
        except Exception:
            pass
    
    result = analyzer.analyze_file(str(file_path), label_prefix=label_prefix)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(result.to_dict(), f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", required=True, type=str)
    parser.add_argument("--method", required=True, type=str)
    parser.add_argument("--cache_dir", required=True, type=str)
    parser.add_argument("--label_prefix", default="", type=str)
    args = parser.parse_args()
    
    classifier = MelodyClassifierPaper()
    analyzer = MelodyAnalyzer(extraction_method=args.method, classifier=classifier)
    
    file_path = Path(args.file_path)
    cache_dir = Path(args.cache_dir)
    
    load_or_analyze_in_sub(analyzer, file_path, args.method, cache_dir, label_prefix=args.label_prefix)

if __name__ == "__main__":
    main()
