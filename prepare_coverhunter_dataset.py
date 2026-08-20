import os
import re
import csv
import zipfile
from pathlib import Path
import librosa
import soundfile as sf

def clean_filename(name):
    # Reemplazar caracteres no alfanuméricos por guiones bajos
    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)

def get_audio_files(directory_path: Path):
    result = {}
    if not directory_path.exists():
        return result
    for f in directory_path.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp3', '.wav']:
            result[f.name] = f
    return result

def get_prefix(filename: str):
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    print("=== Herramienta de Preparación de Dataset para CoverHunterMPS ===")
    base_dir = Path(__file__).parent.absolute()
    
    # 1. Encontrar directorios de dataset disponibles
    datasets = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("dataset_"):
            if (item / "originales").exists() and (item / "covers").exists():
                datasets.append(item.name)
                
    if not datasets:
        print("No se encontraron carpetas que empiecen con 'dataset_' y contengan 'originales' y 'covers'.")
        return
        
    print("\nDatasets locales detectados:")
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds}")
        
    while True:
        try:
            choice = input(f"\nSelecciona el dataset a exportar (1-{len(datasets)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                selected_dataset = datasets[idx]
                break
        except ValueError:
            print("Selección no válida.")
            
    dataset_dir = base_dir / selected_dataset
    orig_dir = dataset_dir / "originales"
    cover_dir = dataset_dir / "covers"
    
    # Determinar qué archivo CSV/metadata corresponde
    metadata_csv = None
    if "popular" in selected_dataset.lower() or "oa" in selected_dataset.lower():
        metadata_csv = base_dir / "dataset_popular.csv"
    elif "academic" in selected_dataset.lower() or "acad" in selected_dataset.lower():
        metadata_csv = base_dir / "dataset_academic.csv"
    else:
        # Buscar cualquier csv en el directorio base
        csvs = list(base_dir.glob("*.csv"))
        if csvs:
            metadata_csv = csvs[0]
            print(f"Usando archivo de metadatos por defecto: {metadata_csv.name}")
            
    if not metadata_csv or not metadata_csv.exists():
        print(f"Error: No se encontró el archivo de metadatos (CSV) correspondiente.")
        return

    # Preguntar el nombre del dataset de destino para CoverHunter
    dest_dataset_name = input("\nIntroduce el nombre que tendrá el dataset en CoverHunter (ej. 'covers80' o 'my_dataset'): ").strip()
    if not dest_dataset_name:
        dest_dataset_name = "covers80"
        
    # Crear directorio temporal de salida
    out_dir = base_dir / f"coverhunter_prep_{selected_dataset}"
    wav_out_dir = out_dir / "wav_16k"
    wav_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLeyendo metadatos desde {metadata_csv.name}...")
    records = []
    with open(metadata_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)
            
    orig_files = get_audio_files(orig_dir)
    cover_files = get_audio_files(cover_dir)
    
    print(f"Archivos originales encontrados: {len(orig_files)}")
    print(f"Archivos covers encontrados: {len(cover_files)}")
    
    # Emparejar usando los track_ids y nombres del CSV
    paired_tracks = []
    # Indexar archivos por prefijo numérico
    orig_by_prefix = {}
    for name, path in orig_files.items():
        pref = get_prefix(name)
        if pref is not None:
            orig_by_prefix.setdefault(pref, []).append(path)
            
    cover_by_prefix = {}
    for name, path in cover_files.items():
        pref = get_prefix(name)
        if pref is not None:
            cover_by_prefix.setdefault(pref, []).append(path)
            
    # Para cada registro en el CSV, encontrar el archivo
    processed_files_count = 0
    dataset_txt_lines = []
    
    # Estructura del dataset.txt de CoverHunter:
    # perf:<perf_id>\twav:<path>\tdur_s:<duration>\twork:<work_id>\tversion:<version>
    
    print("\nProcesando y convirtiendo audios a 16kHz mono (esto puede tomar unos minutos)...")
    
    for r in records:
        track_id_str = r.get('track_id')
        if not track_id_str:
            continue
        try:
            track_id = int(track_id_str)
        except ValueError:
            continue
            
        t_type = r.get('type') # 'original' o 'cover'
        title = r.get('performance title', '')
        artist = r.get('performance artist', '')
        work_title = r.get('work title', f"work_{track_id:02d}")
        
        candidates = orig_by_prefix.get(track_id, []) if t_type == 'original' else cover_by_prefix.get(track_id, [])
        
        # Si hay más de un candidato, filtrar por coincidencia de palabras
        audio_file = None
        if len(candidates) == 1:
            audio_file = candidates[0]
        elif len(candidates) > 1:
            best_overlap = -1
            query_words = set(re.findall(r'\w+', (artist + " " + title).lower()))
            for cand in candidates:
                cand_words = set(re.findall(r'\w+', cand.name.lower()))
                overlap = len(query_words.intersection(cand_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    audio_file = cand
        else:
            # Intentar búsqueda directa
            continue
            
        if not audio_file or not audio_file.exists():
            continue
            
        # Nombre de salida limpio
        clean_stem = clean_filename(audio_file.stem)
        dest_filename = f"{track_id:02d}_{t_type}_{clean_stem}.wav"
        dest_path = wav_out_dir / dest_filename
        
        # Cargar, convertir a mono y resamplear a 16kHz usando librosa
        try:
            # librosa.load convierte automáticamente a mono si no se especifica mono=False
            y, sr = librosa.load(str(audio_file), sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Guardar como wav de 16kHz
            sf.write(str(dest_path), y, sr)
            
            # Generar datos para el dataset.txt
            # Formato de ids: {dataset}_{track_id}_{original/cover}
            perf_id = f"{dest_dataset_name}_{track_id:03d}_{'0' if t_type == 'original' else '1'}"
            wav_relative_path = f"data/{dest_dataset_name}/wav_16k/{dest_filename}"
            work_id = clean_filename(work_title.replace(" ", "_"))
            
            # Asegurarse de que no haya tabuladores en los campos
            work_id = work_id.replace("\t", " ")
            version_info = clean_filename(f"{artist}_{title}".replace(" ", "_"))
            
            line = f"perf:{perf_id}\twav:{wav_relative_path}\tdur_s:{duration:.3f}\twork:{work_id}\tversion:{version_info}"
            dataset_txt_lines.append(line)
            
            processed_files_count += 1
            print(f" [{processed_files_count}] Procesado: {audio_file.name} -> {dest_filename} ({duration:.2f}s)")
            
        except Exception as e:
            print(f"Error procesando {audio_file.name}: {e}")
            
    # Escribir dataset.txt
    dataset_txt_path = out_dir / "dataset.txt"
    with open(dataset_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dataset_txt_lines) + "\n")
        
    print(f"\nSe generó dataset.txt con {len(dataset_txt_lines)} registros.")
    
    # Crear archivo ZIP para subir a Colab
    zip_path = base_dir / f"coverhunter_dataset_{selected_dataset}.zip"
    print(f"\nComprimiendo todo en {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Añadir dataset.txt
        zipf.write(dataset_txt_path, arcname="dataset.txt")
        # Añadir todos los archivos wav de wav_16k
        for wav_file in wav_out_dir.glob("*.wav"):
            zipf.write(wav_file, arcname=f"wav_16k/{wav_file.name}")
            
    print(f"¡Dataset empaquetado con éxito en: {zip_path}!")
    print("\nInstrucciones para Google Colab:")
    print(f"1. Sube el archivo '{zip_path.name}' a tu entorno de Colab.")
    print("2. Descomprímelo en la carpeta del proyecto en la ruta correcta usando:")
    print(f"   !mkdir -p data/{dest_dataset_name}")
    print(f"   !unzip {zip_path.name} -d data/{dest_dataset_name}/")
    print(f"3. Listo! El archivo dataset.txt y los audios en data/{dest_dataset_name}/wav_16k/ estarán listos para CoverHunter.")

if __name__ == "__main__":
    main()
