import os
import re
import sys
import gc
import json
import argparse
from pathlib import Path
import numpy as np
import librosa

try:
    import essentia
    import essentia.standard as es
except ImportError:
    print("=" * 70)
    print("Error: Essentia no está instalado en tu entorno de Python.")
    print("Para instalarlo localmente en tu Mac, ejecuta:")
    print("  pip install essentia")
    print("O si usas Homebrew, puedes instalar dependencias primero:")
    print("  brew install python-essentia")
    print("=" * 70)
    sys.exit(1)

def clean_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)

def get_audio_files(directory_path: Path):
    result = {}
    if not directory_path.exists():
        return result
    for f in directory_path.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp3', '.wav']:
            # Extraer prefijo numérico como ID
            match = re.search(r'^(\d+)', f.name)
            if match:
                file_id = int(match.group(1))
                result[file_id] = f
    return result

def pair_files(orig_files: dict, cover_files: dict):
    # Encontrar IDs comunes entre originales y covers
    common_ids = sorted(list(set(orig_files.keys()).intersection(set(cover_files.keys()))))
    return common_ids

def extract_hpcp_features(audio_path: Path, sample_rate=16000, frame_size=2048, hop_size=1024):
    """
    Extrae características HPCP (Harmonic Pitch Class Profile) usando el pipeline de Essentia.
    """
    # Usar MonoLoader de Essentia para cargar el audio a mono y resamplear
    loader = es.MonoLoader(filename=str(audio_path), sampleRate=sample_rate)
    audio = loader()
    
    # Pipeline estándar de Essentia para obtener HPCP
    run_windowing = es.Windowing(type='hann')
    run_spectrum = es.Spectrum()
    run_spectral_peaks = es.SpectralPeaks(
        orderBy='magnitude', 
        magnitudeThreshold=0.00001, 
        minFrequency=40, 
        maxFrequency=5000, 
        maxPeaks=100
    )
    run_hpcp = es.HPCP(
        size=12, 
        referenceFrequency=440.0, 
        harmonics=8, 
        bandPreset=False, 
        minFrequency=40.0, 
        maxFrequency=5000.0
    )
    
    hpcp_list = []
    # Generar ventanas y procesar
    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        windowed = run_windowing(frame)
        spectrum = run_spectrum(windowed)
        frequencies, magnitudes = run_spectral_peaks(spectrum)
        hpcp = run_hpcp(frequencies, magnitudes)
        hpcp_list.append(hpcp)
        
    return np.array(hpcp_list, dtype=np.float32)

def compute_essentia_csi_distance(hpcp_cover, hpcp_orig, base_ds_factor=8):
    """
    Calcula la distancia de cover song similarity usando ChromaCrossSimilarity y CoverSongSimilarity.
    """
    # Determinar factor de downsampling adaptativo para evitar matrices demasiado grandes
    # Queremos un máximo de ~500 frames por canción
    ds_cov = base_ds_factor
    ds_orig = base_ds_factor
    
    if len(hpcp_cover) / ds_cov > 500:
        ds_cov = int(len(hpcp_cover) // 500)
    if len(hpcp_orig) / ds_orig > 500:
        ds_orig = int(len(hpcp_orig) // 500)
        
    ds_factor = max(ds_cov, ds_orig)
    
    # Downsample time axis
    hpcp_cov_ds = hpcp_cover[::ds_factor]
    hpcp_orig_ds = hpcp_orig[::ds_factor]
    
    # 1. Calcular la matriz de similitud cruzada binaria (CRP) con invarianza de clave (OTI)
    crp_generator = es.ChromaCrossSimilarity(
        frameStackSize=9, 
        frameStackStride=1, 
        binarizePercentile=0.095, 
        oti=True
    )
    
    try:
        # Generar la matriz binaria CRP
        crp = crp_generator(hpcp_cov_ds, hpcp_orig_ds)
        
        # 2. Calcular alineación local (Serra 2009)
        aligner = es.CoverSongSimilarity(
            disOnset=0.5, 
            disExtension=0.5, 
            alignmentType='serra09', 
            distanceType='asymmetric'
        )
        
        _, distance = aligner(crp)
        return float(distance)
    except Exception as e:
        # En caso de error (por ejemplo, audios excesivamente cortos) devolver distancia máxima
        return 1.0

def evaluate_binary_classification(pairwise_results, lower_is_better=True):
    """Evaluates binary classification for a given metric over a range of thresholds."""
    # Filtrar valores finitos para determinar el rango de barrido de umbrales
    finite_values = [r[0] for r in pairwise_results if r[0] is not None and np.isfinite(r[0])]
    if not finite_values:
        return 0.0, None, []
        
    min_val, max_val = min(finite_values), max(finite_values)
    thresholds = np.linspace(min_val, max_val, 21)
        
    best_f1 = -1.0
    best_thresh = 0.0
    best_metrics = {}
    curves = []
    
    for t in thresholds:
        tp, fp, fn, tn = 0, 0, 0, 0
        for val, is_correct in pairwise_results:
            if val is None:
                continue
            if lower_is_better:
                pred_positive = (val <= t)
            else:
                pred_positive = (val >= t)
                
            if pred_positive:
                if is_correct:
                    tp += 1
                else:
                    fp += 1
            else:
                if is_correct:
                    fn += 1
                else:
                    tn += 1
                    
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        curves.append({
            "threshold": t,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "fpr": fpr,
            "fnr": fnr
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "fpr": fpr,
                "fnr": fnr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            }
            
    return best_thresh, best_metrics, curves

def main():
    parser = argparse.ArgumentParser(description="Evaluación de Cover Song Identification (CSI) usando Essentia (HPCP + Serra09)")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Nombre o ruta del dataset (ej: dataset_OA o dataset_Acad)")
    parser.add_argument("--orig_subdir", type=str, default="originales", help="Subdirectorio de canciones originales")
    parser.add_argument("--cover_subdir", type=str, default="covers", help="Subdirectorio de canciones cover")
    parser.add_argument("--output_dir", type=str, default="output_essentia", help="Subdirectorio para guardar reportes y tablas")
    parser.add_argument("--cache_dir", type=str, default="cache_essentia", help="Carpeta de caché de características")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.absolute()
    
    # 1. Selección de dataset
    if args.dataset_dir is None:
        datasets = []
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.startswith("dataset_"):
                if (item / args.orig_subdir).exists() and (item / args.cover_subdir).exists():
                    datasets.append(item.name)
        if not datasets:
            print("No se encontraron carpetas que empiecen con 'dataset_' conteniendo originales y covers.")
            return
            
        print("\nDatasets locales detectados:")
        for i, ds in enumerate(datasets, 1):
            print(f"{i}. {ds}")
        while True:
            try:
                choice = input(f"Selecciona el dataset para correr Essentia CSI (1-{len(datasets)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(datasets):
                    dataset_name = datasets[idx]
                    break
            except ValueError:
                print("Selección inválida.")
        dataset_dir = base_dir / dataset_name
    else:
        dataset_name = args.dataset_dir
        dataset_dir = Path(dataset_name)
        if not dataset_dir.is_absolute():
            dataset_dir = base_dir / dataset_dir
            
    if not dataset_dir.exists():
        print(f"Error: La carpeta de dataset '{dataset_dir}' no existe.")
        return

    orig_dir = dataset_dir / args.orig_subdir
    cover_dir = dataset_dir / args.cover_subdir
    output_dir = dataset_dir / args.output_dir
    cache_dir = base_dir / args.cache_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n Cargando archivos de audio desde {dataset_dir.name}...")
    orig_files = get_audio_files(orig_dir)
    cover_files = get_audio_files(cover_dir)
    common_ids = pair_files(orig_files, cover_files)
    
    if not common_ids:
        print(f"No se pudieron emparejar archivos entre '{orig_dir.name}' y '{cover_dir.name}'.")
        print("Asegúrate de que los archivos empiecen con el mismo número ID (ej: '01 - Cancion.mp3').")
        return
        
    print(f"Parejas detectadas y emparejadas para evaluar: {len(common_ids)}")

    # 2. Extracción de características HPCP (con caché)
    hpcp_originals = {}
    hpcp_covers = {}
    
    print("\n Extrayendo características HPCP (usando caché para mayor velocidad)...")
    
    # Procesar Originales
    for i, uid in enumerate(common_ids, 1):
        file_path = orig_files[uid]
        cache_file = cache_dir / f"hpcp_orig_{dataset_dir.name}_{uid:02d}_{clean_filename(file_path.stem)}.npy"
        
        print(f"  [{i}/{len(common_ids)}] Original ID {uid:02d}: {file_path.name}", end='\r')
        
        if cache_file.exists():
            hpcp_originals[uid] = np.load(cache_file)
        else:
            try:
                features = extract_hpcp_features(file_path)
                np.save(cache_file, features)
                hpcp_originals[uid] = features
            except Exception as e:
                print(f"\n  Error extrayendo original ID {uid:02d}: {e}")
                hpcp_originals[uid] = None
        gc.collect()
    print("\n Originales cargados con éxito.")

    # Procesar Covers
    for i, uid in enumerate(common_ids, 1):
        file_path = cover_files[uid]
        cache_file = cache_dir / f"hpcp_cover_{dataset_dir.name}_{uid:02d}_{clean_filename(file_path.stem)}.npy"
        
        print(f"  [{i}/{len(common_ids)}] Cover ID {uid:02d}: {file_path.name}", end='\r')
        
        if cache_file.exists():
            hpcp_covers[uid] = np.load(cache_file)
        else:
            try:
                features = extract_hpcp_features(file_path)
                np.save(cache_file, features)
                hpcp_covers[uid] = features
            except Exception as e:
                print(f"\n  Error extrayendo cover ID {uid:02d}: {e}")
                hpcp_covers[uid] = None
        gc.collect()
    print("\n Covers cargados con éxito.")

    # 3. Calcular similitudes de todas las parejas (Inferencia)
    print("\n Calculando matriz de alineación cruzada Serra09...")
    detailed_lines = []
    ranks_list = []
    mrr_sum = 0.0
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    valid_count = 0
    pairwise_results = []

    total_pairs = len(common_ids)
    
    for i, uid_cover in enumerate(common_ids, 1):
        hpcp_cov = hpcp_covers[uid_cover]
        if hpcp_cov is None or len(hpcp_cov) == 0:
            continue
            
        print(f"  [{i}/{total_pairs}] Evaluando cover ID {uid_cover:02d} contra todos los originales...", end='\r')
        
        # Comparar con TODOS los originales del dataset
        distances = []
        for uid_orig in common_ids:
            hpcp_orig = hpcp_originals[uid_orig]
            if hpcp_orig is None or len(hpcp_orig) == 0:
                continue
                
            dist = compute_essentia_csi_distance(hpcp_cov, hpcp_orig)
            distances.append((dist, uid_orig))
            pairwise_results.append((dist, uid_cover == uid_orig))
            
        if not distances:
            continue
            
        # Ordenar distancias en orden ascendente (menor distancia = más similar)
        distances.sort(key=lambda x: x[0])
        
        # Encontrar el rango de la versión original correcta
        rank = -1
        correct_dist = 1.0
        for idx, (dist, r_uid) in enumerate(distances):
            if r_uid == uid_cover:
                rank = idx + 1
                correct_dist = dist
                break
                
        if rank != -1:
            valid_count += 1
            ranks_list.append(rank)
            mrr_sum += 1.0 / rank
            if rank == 1: top1_hits += 1
            if rank <= 5: top5_hits += 1
            if rank <= 10: top10_hits += 1
            
            detailed_lines.append(
                f"ID {uid_cover:02d} | Cover: {cover_files[uid_cover].name:<40} | Distancia Correcta: {correct_dist:.4f} | Rango: {rank:2d}"
            )
            
    # 4. Calcular métricas finales
    if valid_count == 0:
        print("\nNo se pudieron obtener métricas válidas.")
        return
        
    mr = np.mean(ranks_list)
    mrr = mrr_sum / valid_count
    mdr = np.median(ranks_list)
    top1_prec = top1_hits / valid_count
    top5_prec = top5_hits / valid_count
    top10_prec = top10_hits / valid_count
    
    # Evaluar clasificación binaria y umbrales óptimos
    best_thresh, best_metrics, curves = evaluate_binary_classification(pairwise_results, lower_is_better=True)
    
    # Imprimir en consola en formato tabla limpia
    print("\n" + "=" * 80)
    print(f" RESUMEN DE MÉTRICAS - ESSENTIA CSI (HPCP + Serra09)")
    print(f" Dataset: {dataset_dir.name}")
    print(f" Parejas evaluadas: {valid_count}")
    print("=" * 80)
    print(f" Mean Rank (MR):           {mr:.2f}")
    print(f" Median Rank (MDR):        {mdr:.1f}")
    print(f" MRR / MAP:                {mrr:.4f} ({mrr*100:.2f}%)")
    print(f" Top-1 Accuracy (Hit@1):   {top1_prec:.2%}")
    print(f" Top-5 Accuracy (Hit@5):   {top5_prec:.2%}")
    print(f" Top-10 Accuracy (Hit@10): {top10_prec:.2%}")
    if best_metrics:
        print("=" * 80)
        print(f" CLASIFICACIÓN BINARIA (UMBRAL ÓPTIMO DE DISTANCIA: {best_thresh:.4f})")
        print(f" F1-Score:                 {best_metrics['f1_score']:.4f}")
        print(f" Precision:                {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)")
        print(f" Recall / Sensibilidad:    {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)")
        print(f" FPR (False Pos. Rate):    {best_metrics['fpr']:.4f} ({best_metrics['fpr']*100:.2f}%)")
        print(f" FNR (False Neg. Rate):    {best_metrics['fnr']:.4f} ({best_metrics['fnr']*100:.2f}%)")
        print(f" Accuracy:                 {best_metrics['accuracy']:.4f}")
        print(f" Matriz de Confusión:")
        print(f"   - TP (True Pos.):       {best_metrics['tp']}")
        print(f"   - FP (False Pos.):      {best_metrics['fp']}")
        print(f"   - FN (False Neg.):      {best_metrics['fn']}")
        print(f"   - TN (True Neg.):       {best_metrics['tn']}")
    print("=" * 80)
    
    # 5. Escribir reportes
    report_path = output_dir / "detailed_report_essentia.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"REPORTE DETALLADO - ESSENTIA CSI (HPCP + Serra09)\n")
        f.write(f"Dataset: {dataset_dir.name}\n")
        f.write("=" * 80 + "\n\n")
        f.write("\n".join(detailed_lines) + "\n\n")
        f.write("=" * 80 + "\n")
        f.write("RESUMEN GLOBAL:\n")
        f.write(f"  Parejas válidas:         {valid_count}\n")
        f.write(f"  Mean Rank (MR):          {mr:.4f}\n")
        f.write(f"  Median Rank (MDR):       {mdr:.1f}\n")
        f.write(f"  MRR (MAP):               {mrr:.4f} ({mrr*100:.2f}%)\n")
        f.write(f"  Top-1 Hit Rate:          {top1_prec:.2%}\n")
        f.write(f"  Top-5 Hit Rate:          {top5_prec:.2%}\n")
        f.write(f"  Top-10 Hit Rate:         {top10_prec:.2%}\n")
        if best_metrics:
            f.write("=" * 80 + "\n")
            f.write(f"CLASIFICACIÓN BINARIA (UMBRAL ÓPTIMO DE DISTANCIA: {best_thresh:.4f}):\n")
            f.write(f"  F1-Score:                {best_metrics['f1_score']:.4f}\n")
            f.write(f"  Precision:               {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)\n")
            f.write(f"  Recall (Sens.):          {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)\n")
            f.write(f"  FPR:                     {best_metrics['fpr']:.4f} ({best_metrics['fpr']*100:.2f}%)\n")
            f.write(f"  FNR:                     {best_metrics['fnr']:.4f} ({best_metrics['fnr']*100:.2f}%)\n")
            f.write(f"  Accuracy:                {best_metrics['accuracy']:.4f}\n")
            f.write(f"  Confusion Matrix:\n")
            f.write(f"    - TP (True Pos.):      {best_metrics['tp']}\n")
            f.write(f"    - FP (False Pos.):     {best_metrics['fp']}\n")
            f.write(f"    - FN (False Neg.):     {best_metrics['fn']}\n")
            f.write(f"    - TN (True Neg.):      {best_metrics['tn']}\n")
        f.write("=" * 80 + "\n")
        
    print(f"\nReporte detallado guardado en: {report_path}")
    
    # Exportar análisis de umbrales en CSV
    curves_csv = output_dir / "threshold_analysis_essentia.csv"
    with open(curves_csv, "w", encoding="utf-8") as f:
        f.write("threshold,tp,fp,fn,tn,precision,recall,f1_score,accuracy,fpr,fnr\n")
        for c in curves:
            f.write(f"{c['threshold']:.4f},{c['tp']},{c['fp']},{c['fn']},{c['tn']},{c['precision']:.6f},{c['recall']:.6f},{c['f1_score']:.6f},{c['accuracy']:.6f},{c['fpr']:.6f},{c['fnr']:.6f}\n")
    print(f"Análisis de umbrales guardado en: {curves_csv}")
    
    # Generar archivo de resumen CSV compatible
    summary_csv = output_dir / "essentia_summary.csv"
    with open(summary_csv, "w", encoding="utf-8") as f:
        if best_metrics:
            f.write("dataset,method,mrr_pct,map_pct,mr,mdr,top1_pct,top5_pct,top10_pct,threshold,f1_score,precision_pct,recall_pct,fpr_pct,fnr_pct,accuracy,tp,fp,fn,tn\n")
            f.write(f"{dataset_dir.name},Essentia-HPCP (Serra09),{mrr*100:.2f},{mrr*100:.2f},{mr:.2f},{mdr:.1f},{top1_prec*100:.2f},{top5_prec*100:.2f},{top10_prec*100:.2f},{best_thresh:.4f},{best_metrics['f1_score']:.4f},{best_metrics['precision']*100:.2f},{best_metrics['recall']*100:.2f},{best_metrics['fpr']*100:.2f},{best_metrics['fnr']*100:.2f},{best_metrics['accuracy']:.4f},{best_metrics['tp']},{best_metrics['fp']},{best_metrics['fn']},{best_metrics['tn']}\n")
        else:
            f.write("dataset,method,mrr_pct,map_pct,mr,mdr,top1_pct,top5_pct,top10_pct\n")
            f.write(f"{dataset_dir.name},Essentia-HPCP (Serra09),{mrr*100:.2f},{mrr*100:.2f},{mr:.2f},{mdr:.1f},{top1_prec*100:.2f},{top5_prec*100:.2f},{top10_prec*100:.2f}\n")
        
    print(f"Resumen CSV guardado en: {summary_csv}")

    # Guardar en un formato de tabla de texto limpio y estandarizado
    table_txt_path = dataset_dir / "tabla_comparativa_essentia.txt"
    with open(table_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 160 + "\n")
        f.write(f"TABLA DE RESULTADOS STANDARIZADA - ESSENTIA CSI Baseline ({dataset_dir.name})\n")
        f.write("=" * 160 + "\n")
        header = f"{'Dataset':<15} | {'Method':<24} | {'MRR (%)':<8} | {'MAP (%)':<8} | {'MR':<6} | {'MDR':<5} | {'Top-1 (%)':<10} | {'Top-5 (%)':<10} | {'Top-10 (%)':<10} | {'Threshold':<10} | {'F1-Score':<9} | {'Precision (%)':<14} | {'Recall (%)':<11} | {'FPR (%)':<8} | {'FNR (%)':<8}\n"
        f.write(header)
        f.write("-" * 160 + "\n")
        if best_metrics:
            row = (f"{dataset_dir.name:<15} | {'Essentia-HPCP (Serra09)':<24} | {mrr*100:<8.2f} | {mrr*100:<8.2f} | {mr:<6.2f} | {mdr:<5.1f} | "
                   f"{top1_prec*100:<10.2f} | {top5_prec*100:<10.2f} | {top10_prec*100:<10.2f} | {best_thresh:<10.4f} | {best_metrics['f1_score']:<9.4f} | "
                   f"{best_metrics['precision']*100:<14.2f} | {best_metrics['recall']*100:<11.2f} | {best_metrics['fpr']*100:<8.2f} | {best_metrics['fnr']*100:<8.2f}\n")
        else:
            row = (f"{dataset_dir.name:<15} | {'Essentia-HPCP (Serra09)':<24} | {mrr*100:<8.2f} | {mrr*100:<8.2f} | {mr:<6.2f} | {mdr:<5.1f} | "
                   f"{top1_prec*100:<10.2f} | {top5_prec*100:<10.2f} | {top10_prec*100:<10.2f} | {'N/A':<10} | {'N/A':<9} | "
                   f"{'N/A':<14} | {'N/A':<11} | {'N/A':<8} | {'N/A':<8}\n")
        f.write(row)
        f.write("=" * 160 + "\n")
    print(f"Tabla de texto estandarizada guardada en: {table_txt_path}")

if __name__ == "__main__":
    main()
