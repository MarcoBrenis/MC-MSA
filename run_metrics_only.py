#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
import numpy as np

METHOD_CLASSIFICATION = {
    'pyin': 'Fundamental Frequency (F0 Extractor)',
    'yin': 'Fundamental Frequency (F0 Extractor)',
    'crepe': 'Fundamental Frequency (F0 Extractor)',
    'rmvpe': 'Fundamental Frequency (F0 Extractor)',
    'spice': 'Fundamental Frequency (F0 Extractor)',
    'jdc': 'Fundamental Frequency (F0 Extractor)',
    'fcn_f0': 'Fundamental Frequency (F0 Extractor)',
    'melodia': 'Melody Extraction (Salient F0)',
    'tachibana': 'Melody Extraction (Salient F0)',
    'poliner': 'Melody Extraction (Salient F0)',
    'durrieu': 'Melody Extraction (Salient F0)',
    'basic_pitch': 'Melody Extraction (Salient F0)',
    'demucs_crepe': 'Hybrid (Separation + F0)',
    'bs_roformer_rmvpe': 'Hybrid (Separation + F0)',
    'bs_roformer_crepe': 'Hybrid (Separation + F0)',
    'demucs_rmvpe': 'Hybrid (Separation + F0)',
    'bs_roformer': 'Vocal Melody (Vocal Separation + F0)',
    'demucs': 'Vocal Melody (Vocal Separation + F0)',
    'ensemble': 'Ensemble (Vocal + Salient F0)'
}

def get_audio_files(directory_path: Path, match_mode: str = "id"):
    result = {}
    if not directory_path.exists():
        return result
    for f in directory_path.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp3', '.wav']:
            if match_mode == "stem":
                name = f.stem.lower()
                name = re.sub(r'[-_](cover|originales|original|orig|ref|covers|version|var)', '', name)
                name = re.sub(r'^\d+\s*[-_]?\s*', '', name)
                name = re.sub(r'[^a-z0-9]', '', name)
                key = name.strip()
                if key:
                    result[key] = f
            elif match_mode == "fuzzy":
                result[f.name] = f
            else: # "id"
                match = re.search(r'^(\d+)', f.name)
                if match:
                    file_id = int(match.group(1))
                    result[file_id] = f
    return result

def pair_files_fuzzy(orig_files: dict, cover_files: dict) -> tuple:
    paired_orig = {}
    paired_cover = {}
    
    def get_prefix_and_text(filename: str):
        match = re.match(r'^(\d+)\s*[-_]?\s*(.*)', filename)
        if match:
            return int(match.group(1)), match.group(2)
        return 0, filename
        
    orig_by_prefix = {}
    cover_by_prefix = {}
    
    for name, path in orig_files.items():
        prefix, text = get_prefix_and_text(name)
        orig_by_prefix.setdefault(prefix, []).append((text, path))
            
    for name, path in cover_files.items():
        prefix, text = get_prefix_and_text(name)
        cover_by_prefix.setdefault(prefix, []).append((text, path))
            
    pair_id = 1
    
    def word_overlap(str1: str, str2: str) -> float:
        w1 = set(re.findall(r'[a-zA-Z0-9]+', str1.lower()))
        w2 = set(re.findall(r'[a-zA-Z0-9]+', str2.lower()))
        stopwords = {'cover', 'covers', 'original', 'originales', 'orig', 'ref', 'version', 'mp3', 'wav'}
        w1 = w1 - stopwords
        w2 = w2 - stopwords
        if not w1 or not w2:
            return 0.0
        return len(w1.intersection(w2)) / len(w1.union(w2))
        
    for prefix in sorted(orig_by_prefix.keys()):
        if prefix in cover_by_prefix:
            orig_list = orig_by_prefix[prefix]
            cover_list = cover_by_prefix[prefix]
            
            used_origs = set()
            for cov_text, cov_path in cover_list:
                best_overlap = -1.0
                best_orig_path = None
                
                for orig_text, orig_path in orig_list:
                    if orig_path in used_origs:
                        continue
                    overlap = word_overlap(cov_text, orig_text)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_orig_path = orig_path
                        
                if best_orig_path is not None and best_overlap > 0.0:
                    paired_orig[pair_id] = best_orig_path
                    paired_cover[pair_id] = cov_path
                    used_origs.add(best_orig_path)
                    pair_id += 1
                    
    return paired_orig, paired_cover

def save_dataset_comparative_table(dataset_dir: Path, output_dir: Path):
    summary_path = output_dir / "mc_msa_summary.csv"
    if not summary_path.exists():
        print(f"[Comparative Table] Summary file not found at {summary_path}")
        return
        
    dataset_name = dataset_dir.name
    
    method_rows = {}
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            header_indices = {col.strip().lower(): i for i, col in enumerate(header)}
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > 0:
                    method_rows[parts[0]] = parts
    except Exception as e:
        print(f"Error reading {summary_path}: {e}")
        return
        
    if not method_rows:
        return
        
    sorted_methods = sorted(list(method_rows.keys()))
    
    has_mr = "mr" in header_indices
    has_mdr = "mdr" in header_indices
    has_map = "map" in header_indices
    has_top10 = "top10_prec" in header_indices or "top10" in header_indices
    has_opt_params = "min_voicing_thresh" in header_indices
    
    lines = []
    divider_len = 160 if has_opt_params else 120
    lines.append("-" * divider_len)
    lines.append(f"Dataset: {dataset_name}" + (" (Optimized with Optuna)" if has_opt_params else ""))
    lines.append("-" * divider_len)
    
    hdr_cols = [f"{'Method':<25}", f"{'Avg. LCS (%)':<14}"]
    if has_mr: hdr_cols.append(f"{'MR':<8}")
    hdr_cols.append(f"{'MRR (%)':<10}")
    if has_mdr: hdr_cols.append(f"{'MDR':<8}")
    if has_map: hdr_cols.append(f"{'MAP (%)':<10}")
    hdr_cols.append(f"{'Top-5 (%)':<12}")
    if has_top10: hdr_cols.append(f"{'Top-10 (%)':<12}")
    hdr_cols.append(f"{'DTW':<10}")
    if has_opt_params:
        hdr_cols.append(f"{'Optimized Parameters (voicing, slope, energy)':<45}")
    lines.append(" | ".join(hdr_cols))
    lines.append("-" * divider_len)
    
    for method in sorted_methods:
        row = method_rows[method]
        method_disp = method.upper()
        
        def get_val(name, is_pct=False, fmt=".2f", default="-"):
            idx = header_indices.get(name.lower())
            if idx is not None and idx < len(row) and row[idx] != "":
                try:
                    val = float(row[idx])
                    if is_pct:
                        val *= 100
                    return f"{val:{fmt}}"
                except ValueError:
                    return row[idx]
            return default

        try:
            lcs = get_val("avg_lcs" if "avg_lcs" in header_indices else "lcs_promedio", is_pct=True) + "%"
            mrr = get_val("mrr", is_pct=True) + "%"
            top5 = get_val("top5_prec", is_pct=True) + "%"
            dtw = get_val("avg_dtw" if "avg_dtw" in header_indices else "dtw_promedio")
            
            row_cols = [f"{method_disp:<25}", f"{lcs:>12}"]
            if has_mr:
                mr = get_val("mr")
                row_cols.append(f"{mr:>8}")
            row_cols.append(f"{mrr:>8}")
            if has_mdr:
                mdr = get_val("mdr", fmt=".1f")
                row_cols.append(f"{mdr:>8}")
            if has_map:
                map_val = get_val("map", is_pct=True) + "%"
                row_cols.append(f"{map_val:>8}")
            row_cols.append(f"{top5:>10}")
            if has_top10:
                top10 = get_val("top10_prec" if "top10_prec" in header_indices else "top10", is_pct=True) + "%"
                row_cols.append(f"{top10:>10}")
            row_cols.append(f"{dtw:>10}")
            
            if has_opt_params:
                v_t = get_val("min_voicing_thresh", fmt=".4f")
                s_e = get_val("slope_epsilon", fmt=".4f")
                e_t = get_val("energy_tau", fmt=".4f")
                if v_t != "-" and s_e != "-" and e_t != "-":
                    params_str = f"voicing: {v_t}, slope: {s_e}, energy: {e_t}"
                else:
                    params_str = "N/A"
                row_cols.append(f"{params_str:<45}")
                
            lines.append(" | ".join(row_cols))
        except Exception as e:
            lines.append(f"{method.upper():<25} | Error formatting row: {e}")
            
    lines.append("-" * divider_len)
    
    table_content = "\n".join(lines) + "\n"
    table_filename = "comparative_table_recalculated_optuna.txt" if "optuna" in output_dir.name else "comparative_table_recalculated.txt"
    table_path = dataset_dir / table_filename
    try:
        table_path.write_text(table_content, encoding='utf-8')
        print(f"[Comparative Table] Successfully saved at {table_path}")
    except Exception as e:
        print(f"Error writing comparative table: {e}")

def find_available_datasets(directory: Path):
    datasets = []
    if directory.exists():
        for item in directory.iterdir():
            if item.is_dir() and item.name.startswith("dataset_"):
                datasets.append(item.name)
    return sorted(list(set(datasets)))

def main():
    parser = argparse.ArgumentParser(description="Recalculate Information Retrieval metrics from cached comparison JSONs.")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Path to the dataset directory (e.g., dataset_OA)")
    parser.add_argument("--method", type=str, default=None,
                        help="Recalculate only a single method, or 'all' to recalculate all methods found in cache")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Path to cache directory")
    parser.add_argument("--match_mode", type=str, default=None,
                        choices=["id", "stem", "fuzzy"],
                        help="Match mode used to pair originals and covers")
    parser.add_argument("--optuna", type=str, default=None,
                        choices=["y", "n", "s", "no", "yes"],
                        help="Process Optuna hyperparameter MC-MSA evaluations (y/n)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.absolute()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = base_dir / cache_dir

    # 1. Dataset Selection
    if args.dataset_dir is None:
        datasets = find_available_datasets(base_dir)
        if not datasets:
            print("\nNo dataset folders starting with 'dataset_' detected automatically.")
            manual = input("Please enter the path or name of the dataset to use: ").strip()
            args.dataset_dir = manual
        else:
            print("\n=== Dataset Selection ===")
            for i, d in enumerate(datasets, 1):
                print(f"{i}. {d}")
            print(f"{len(datasets) + 1}. [Enter another manual path...]")
            
            while True:
                choice = input(f"\nSelect a dataset (1-{len(datasets) + 1}): ").strip()
                if not choice:
                    continue
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(datasets):
                        args.dataset_dir = datasets[idx]
                        break
                    elif idx == len(datasets):
                        manual = input("Enter the path or name of the dataset: ").strip()
                        if manual:
                            args.dataset_dir = manual
                            break
                    else:
                        print(f"Error: Please select a number between 1 and {len(datasets) + 1}.")
                except ValueError:
                    if choice in datasets:
                        args.dataset_dir = choice
                        break
                    print("Error: Invalid input. Enter option number or exact name.")

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = base_dir / dataset_dir

    if not dataset_dir.exists():
        print(f"Dataset directory '{dataset_dir}' does not exist.")
        return

    # 2. Match Mode Selection
    if args.match_mode is None:
        print("\n=== Match Method Selection ===")
        print("1. By Numeric ID (e.g. '01 - Pedro Infante.wav' with '01 - Cover.mp3') [Default]")
        print("2. By Exact Name / Stem (e.g. 'Te_Vi_Venir_Original.wav' with 'Te Vi Venir (Covers).mp3')")
        print("3. Smart / Fuzzy Match (For complex names or classical music, e.g. '02 - Symphony No. 40...')")
        
        while True:
            choice = input("\nSelect match method (1-3) [Default: 1]: ").strip()
            if not choice or choice == "1":
                args.match_mode = "id"
                break
            elif choice == "2":
                args.match_mode = "stem"
                break
            elif choice == "3":
                args.match_mode = "fuzzy"
                break
            else:
                print("Error: Please select 1, 2 or 3.")

    # 3. Optuna Selection
    optuna_flag = False
    if args.optuna is None:
        choice = input("\nEvaluate version optimized with Optuna? (y/n) [Default: n]: ").strip().lower()
        if choice in ['y', 'yes', 's', 'si']:
            optuna_flag = True
    else:
        if args.optuna.lower() in ['y', 'yes', 's']:
            optuna_flag = True
    args.optuna = optuna_flag

    orig_dir = dataset_dir / "originales"
    cover_dir = dataset_dir / "covers"
    
    if not orig_dir.exists() or not cover_dir.exists():
        print(f"Dataset directory must contain 'originales' and 'covers' subdirectories.")
        return

    orig_files = get_audio_files(orig_dir, match_mode=args.match_mode)
    cover_files = get_audio_files(cover_dir, match_mode=args.match_mode)
    
    if args.match_mode == "fuzzy":
        orig_files, cover_files = pair_files_fuzzy(orig_files, cover_files)
        common_ids = sorted(list(orig_files.keys()))
    else:
        common_ids = sorted(list(set(orig_files.keys()).intersection(set(cover_files.keys()))))

    if not common_ids:
        print("No valid song pairs found using match mode:", args.match_mode)
        return

    # Find methods that have a comparison cache for this dataset
    methods_found = []
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            if item.is_dir():
                cache_suffix = "_optuna.json" if args.optuna else ".json"
                comp_cache_file = item / f"comparison_cache_{dataset_dir.name}{cache_suffix}"
                if comp_cache_file.exists():
                    methods_found.append((item.name, comp_cache_file))

    if not methods_found:
        print(f"No comparison cache JSON files found in cache subdirectories for dataset '{dataset_dir.name}'.")
        return

    # 4. Method Selection
    selected_methods = sorted(methods_found)
    if args.method is None:
        print(f"\n=== Method Selection for '{dataset_dir.name}' ===")
        print("1. [All methods found in cache] (Default)")
        for idx, (m_name, _) in enumerate(sorted(methods_found), 2):
            print(f"{idx:2d}. {m_name}")
            
        while True:
            choice = input(f"\nSelect an option (1-{len(methods_found) + 1}) [Default: 1]: ").strip()
            if not choice or choice == "1":
                # Keep selected_methods as all methods
                break
            try:
                idx = int(choice)
                if 2 <= idx <= len(methods_found) + 1:
                    selected_methods = [sorted(methods_found)[idx - 2]]
                    break
                else:
                    print(f"Error: Please select a number between 1 and {len(methods_found) + 1}.")
            except ValueError:
                # check if they typed the name of the method directly
                exact_match = [m for m in methods_found if m[0] == choice.lower()]
                if exact_match:
                    selected_methods = exact_match
                    break
                if choice.lower() in ['all', 'todos', 't']:
                    break
                print("Error: Invalid input. Enter option number or exact name.")
    elif args.method.lower() != 'all':
        # Filter for the specific method requested
        selected_methods = [m for m in methods_found if m[0] == args.method.lower()]
        if not selected_methods:
            print(f"Requested method '{args.method}' not found in cache for dataset '{dataset_dir.name}'.")
            return

    output_subdir = "resultados_mc_msa_optuna" if args.optuna else "resultados_mc_msa"
    output_dir = dataset_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "mc_msa_summary.csv"
    
    # Read existing parameters if we are in optuna mode to avoid losing them
    existing_params = {}
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                header_cols = [c.strip().lower() for c in f.readline().split(',')]
                param_indices = {col: idx for idx, col in enumerate(header_cols) if col in ["min_voicing_thresh", "slope_epsilon", "energy_tau"]}
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) > 0:
                        m_name = parts[0]
                        m_params = {}
                        for p_name, idx in param_indices.items():
                            if idx < len(parts):
                                m_params[p_name] = parts[idx]
                        existing_params[m_name] = m_params
        except Exception as e:
            print(f"Warning reading existing CSV: {e}")

    # Write headers / reset CSV (but keep records of methods we are NOT recalculating)
    recalculated_names = {f"{m[0]}_optuna" if args.optuna else m[0] for m in selected_methods}
    other_rows = []
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                header_line = f.readline()
                for line in f:
                    parts = line.strip().split(',')
                    if parts and parts[0] not in recalculated_names:
                        other_rows.append(line.strip())
        except Exception:
            pass

    with open(summary_path, 'w', encoding='utf-8') as f:
        if args.optuna:
            f.write("method,pairs,avg_lcs,mr,mrr,mdr,map,top5_prec,top10_prec,avg_dtw,min_voicing_thresh,slope_epsilon,energy_tau\n")
        else:
            f.write("method,pairs,avg_lcs,mr,mrr,mdr,map,top5_prec,top10_prec,avg_dtw\n")
        for row in other_rows:
            f.write(row + "\n")

    print(f"\nRecalculating metrics for dataset: {dataset_dir.name} ({len(common_ids)} pairs)")
    print(f"Output directory: {output_dir}\n")

    for method, cache_file in selected_methods:
        print(f"Processing method: {method} ...")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                comp_cache = json.load(f)
        except Exception as e:
            print(f"  Error reading {cache_file.name}: {e}")
            continue

        lcs_list = []
        dtw_list = []
        ranks_list = []
        mrr_sum = 0.0
        top5_hits = 0
        top10_hits = 0
        detailed_results = []
        valid_count = 0

        for uid_cover in common_ids:
            cover_name = cover_files[uid_cover].name
            similarities = []
            
            for uid_orig in common_ids:
                orig_name = orig_files[uid_orig].name
                
                # Check keys in comparison cache
                key1 = f"{orig_name}:::{cover_name}"
                key2 = f"{cover_name}:::{orig_name}"
                
                entry = comp_cache.get(key1) or comp_cache.get(key2)
                if entry and "lcs_similarity" in entry:
                    lcs_sim = entry["lcs_similarity"]
                    similarities.append((lcs_sim, uid_orig, entry))
            
            if not similarities:
                continue

            similarities.sort(key=lambda x: x[0], reverse=True)
            
            rank = -1
            true_sim = 0.0
            matched_entry = None
            for idx, (sim, r_uid, entry) in enumerate(similarities):
                if r_uid == uid_cover:
                    rank = idx + 1
                    true_sim = sim
                    matched_entry = entry
                    break

            if rank != -1:
                valid_count += 1
                ranks_list.append(rank)
                mrr_sum += 1.0 / rank
                if rank <= 5:
                    top5_hits += 1
                if rank <= 10:
                    top10_hits += 1
                lcs_list.append(true_sim)

                correct_dtw = -1.0
                if matched_entry:
                    dtw_val = matched_entry.get("dtw_distance")
                    if dtw_val is not None and dtw_val != "":
                        correct_dtw = float(dtw_val)
                        dtw_list.append(correct_dtw)

                id_label = f"ID {uid_cover:02d}" if isinstance(uid_cover, int) else f"ID {uid_cover}"
                detailed_results.append(f"{id_label} | LCS: {true_sim:.4f} | Rank: {rank:2d} | DTW: {correct_dtw:.4f}")

        if valid_count == 0:
            print(f"  No valid evaluations could be reconstructed for method: {method}")
            continue

        avg_lcs = np.mean(lcs_list) if lcs_list else 0
        mr = np.mean(ranks_list) if ranks_list else 0
        mrr = mrr_sum / valid_count if valid_count else 0
        mdr = np.median(ranks_list) if ranks_list else 0
        map_val = np.mean([1.0 / r for r in ranks_list]) if ranks_list else 0
        top5_prec = top5_hits / valid_count if valid_count else 0
        top10_prec = top10_hits / valid_count if valid_count else 0
        avg_dtw = np.mean(dtw_list) if dtw_list else 0

        m_key = f"{method}_optuna" if args.optuna else method
        print(f"  Results | LCS: {avg_lcs:.4f} | MR: {mr:.2f} | MRR: {mrr:.4f} | MDR: {mdr:.1f} | MAP: {map_val:.4f} | Top-5: {top5_prec:.2%} | Top-10: {top10_prec:.2%} | DTW: {avg_dtw:.4f}")

        # Fetch parameter values if in optuna mode
        opt_vals = []
        if args.optuna:
            params = existing_params.get(m_key, {})
            v_t = params.get("min_voicing_thresh", "0.0")
            s_e = params.get("slope_epsilon", "0.0")
            e_t = params.get("energy_tau", "0.0")
            opt_vals = [v_t, s_e, e_t]

        # Write to summary CSV
        with open(summary_path, 'a', encoding='utf-8') as f:
            row_str = f"{m_key},{valid_count},{avg_lcs:.6f},{mr:.6f},{mrr:.6f},{mdr:.1f},{map_val:.6f},{top5_prec:.6f},{top10_prec:.6f},{avg_dtw:.6f}"
            if args.optuna:
                row_str += f",{opt_vals[0]},{opt_vals[1]},{opt_vals[2]}"
            f.write(row_str + "\n")

        # Save detailed report
        out_method_dir = output_dir / method
        out_method_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_method_dir / "detailed_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"DETAILED REPORT - METHOD: {method}\n")
            f.write(f"==================================================\n")
            for line in detailed_results:
                f.write(line + "\n")
            f.write(f"==================================================\n")
            f.write(f"GENERAL SUMMARY:\n")
            f.write(f"Pairs evaluated: {valid_count}\n")
            f.write(f"Average LCS:    {avg_lcs:.4f}\n")
            f.write(f"Mean Rank:      {mr:.4f}\n")
            f.write(f"MRR:             {mrr:.4f}\n")
            f.write(f"Median Rank:     {mdr:.1f}\n")
            f.write(f"MAP:             {map_val:.4f}\n")
            f.write(f"Top-5 Precision: {top5_prec:.2%}\n")
            f.write(f"Top-10 Precision: {top10_prec:.2%}\n")
            f.write(f"Average DTW:    {avg_dtw:.4f}\n")

    # Generate consolidated comparative table
    save_dataset_comparative_table(dataset_dir, output_dir)
    print("\nRecalculation finished successfully.")

if __name__ == "__main__":
    main()
