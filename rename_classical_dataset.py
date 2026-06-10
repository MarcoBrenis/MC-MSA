import os
import re
from pathlib import Path

def get_audio_files(directory_path: Path):
    result = {}
    if not directory_path.exists():
        return result
    for f in directory_path.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp3', '.wav']:
            result[f.name] = f
    return result

def word_overlap(str1: str, str2: str) -> float:
    w1 = set(re.findall(r'[a-zA-Z0-9]+', str1.lower()))
    w2 = set(re.findall(r'[a-zA-Z0-9]+', str2.lower()))
    # Remove common stopwords and file extensions
    stopwords = {'cover', 'covers', 'original', 'originales', 'orig', 'ref', 'version', 'mp3', 'wav'}
    w1 = w1 - stopwords
    w2 = w2 - stopwords
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / len(w1.union(w2))

def pair_files_fuzzy(orig_files: dict, cover_files: dict):
    paired = []
    
    # Group by numeric prefix (movement number) if exists, else group under 0
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
            
    # Pair files under each movement prefix
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
                    paired.append((best_orig_path, cov_path))
                    used_origs.add(best_orig_path)
                    
    return paired

def main():
    print("=== Herramienta de Renombrado Inteligente de Datasets (Classical/Conciertos) ===")
    base_dir = Path(__file__).parent.absolute()
    
    # 1. Encontrar datasets disponibles
    datasets = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("dataset_"):
            if (item / "originales").exists() and (item / "covers").exists():
                datasets.append(item.name)
                
    if not datasets:
        print("No se encontraron carpetas con formato 'dataset_*' que contengan 'originales' y 'covers'.")
        return
        
    print("\nDatasets encontrados:")
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds}")
        
    while True:
        try:
            choice = input(f"\nSeleccione el dataset a renombrar (1-{len(datasets)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                selected_dataset = datasets[idx]
                break
        except ValueError:
            print("Selección no válida.")
            
    dataset_dir = base_dir / selected_dataset
    orig_dir = dataset_dir / "originales"
    cover_dir = dataset_dir / "covers"
    
    orig_files = get_audio_files(orig_dir)
    cover_files = get_audio_files(cover_dir)
    
    print(f"\nCargando archivos de {selected_dataset}...")
    print(f"Originales encontrados: {len(orig_files)}")
    print(f"Covers encontrados: {len(cover_files)}")
    
    pairs = pair_files_fuzzy(orig_files, cover_files)
    print(f"\nSe han detectado y emparejado automáticamente {len(pairs)} pares de archivos.")
    
    if len(pairs) == 0:
        print("No se pudieron emparejar canciones. Saliendo.")
        return
        
    # Mostrar una muestra de emparejamientos
    print("\nMuestra de los primeros 5 emparejamientos:")
    for i, (op, cp) in enumerate(pairs[:5], 1):
        print(f"  Pareja {i}:")
        print(f"    [Orig]: {op.name}")
        print(f"    [Cover]: {cp.name}")
        print("-" * 40)
        
    confirm = input("\n¿Está seguro de que desea proceder con el renombrado? (s/N): ").strip().lower()
    if confirm not in ['s', 'si', 'yes', 'y']:
        print("Operación cancelada por el usuario.")
        return
        
    print("\nRenombrando archivos...")
    for i, (orig_path, cover_path) in enumerate(pairs, 1):
        # Generar un prefijo de ID único de 2 o 3 dígitos según el tamaño
        prefix = f"{i:02d}" if len(pairs) < 100 else f"{i:03d}"
        
        # Eliminar el prefijo numérico antiguo si existía en el nombre original
        def clean_old_prefix(name: str):
            cleaned = re.sub(r'^\d+\s*[-_]?\s*', '', name)
            return cleaned
            
        new_orig_name = f"{prefix} - {clean_old_prefix(orig_path.name)}"
        new_cover_name = f"{prefix} - {clean_old_prefix(cover_path.name)}"
        
        new_orig_path = orig_path.parent / new_orig_name
        new_cover_path = cover_path.parent / new_cover_name
        
        # Renombrar en disco
        os.rename(orig_path, new_orig_path)
        os.rename(cover_path, new_cover_path)
        
    print("\n¡Renombrado inteligente completado exitosamente!")
    print(f"Ahora puedes usar el dataset '{selected_dataset}' en el benchmark con total compatibilidad utilizando el ID Numérico (Opción 1).")

if __name__ == "__main__":
    main()
