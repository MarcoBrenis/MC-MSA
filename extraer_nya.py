import csv
import os
import re
import argparse
from pathlib import Path

def escapar_latex(texto):
    """
    Cleans text to prevent breaking LaTeX compilation.
    Escapes ampersands, percentages, dollar signs, etc.
    """
    texto = str(texto).strip()
    reemplazos = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}"
    }
    for original, nuevo in reemplazos.items():
        texto = texto.replace(original, nuevo)
    return texto

def parse_id3v2(filepath):
    """
    Parses ID3v2 tags from MP3 files using pure Python.
    Returns a dictionary with 'title', 'artist', 'track', and 'year'.
    """
    tags = {}
    try:
        with open(filepath, 'rb') as f:
            header = f.read(10)
            if len(header) < 10 or header[:3] != b'ID3':
                return tags
            
            major = header[3]
            size_bytes = header[6:10]
            tag_size = (size_bytes[0] << 21) | (size_bytes[1] << 14) | (size_bytes[2] << 7) | size_bytes[3]
            
            tag_data = f.read(tag_size)
            
            offset = 0
            while offset + 10 < len(tag_data):
                if major == 2:
                    frame_id = tag_data[offset:offset+3].decode('latin1', errors='ignore')
                    if not frame_id or frame_id == '\x00\x00\x00':
                        break
                    frame_size = (tag_data[offset+3] << 16) | (tag_data[offset+4] << 8) | tag_data[offset+5]
                    frame_body = tag_data[offset+6:offset+6+frame_size]
                    offset += 6 + frame_size
                else:
                    frame_id = tag_data[offset:offset+4].decode('latin1', errors='ignore')
                    if not frame_id or frame_id == '\x00\x00\x00\x00':
                        break
                    fs_bytes = tag_data[offset+4:offset+8]
                    if major == 4:
                        frame_size = (fs_bytes[0] << 21) | (fs_bytes[1] << 14) | (fs_bytes[2] << 7) | fs_bytes[3]
                    else:
                        frame_size = (fs_bytes[0] << 24) | (fs_bytes[1] << 16) | (fs_bytes[2] << 8) | fs_bytes[3]
                    
                    frame_body = tag_data[offset+10:offset+10+frame_size]
                    offset += 10 + frame_size
                    
                if frame_id.startswith('T') and frame_id != 'TXXX':
                    if len(frame_body) > 1:
                        encoding = frame_body[0]
                        text_bytes = frame_body[1:]
                        try:
                            if encoding == 0:
                                text = text_bytes.decode('latin1', errors='ignore')
                            elif encoding == 1:
                                text = text_bytes.decode('utf-16', errors='ignore')
                            elif encoding == 2:
                                text = text_bytes.decode('utf-16-be', errors='ignore')
                            elif encoding == 3:
                                text = text_bytes.decode('utf-8', errors='ignore')
                            else:
                                text = text_bytes.decode('latin1', errors='ignore')
                        except Exception:
                            text = text_bytes.decode('latin1', errors='ignore')
                        
                        text = text.strip('\x00').strip()
                        
                        if frame_id in ['TPE1', 'TPE2', 'TP1']:
                            tags['artist'] = text
                        elif frame_id in ['TIT2', 'TT2']:
                            tags['title'] = text
                        elif frame_id in ['TRCK', 'TRK']:
                            tags['track'] = text
                        elif frame_id in ['TYER', 'TDRC', 'TYE']:
                            m = re.search(r'\b(19\d\d|20\d\d)\b', text)
                            if m:
                                tags['year'] = m.group(1)
    except Exception:
        pass
    return tags

def detect_composer(filename, tags):
    artist = tags.get('artist', '')
    if 'beethoven' in artist.lower():
        return 'Beethoven'
    if 'mozart' in artist.lower():
        return 'Mozart'
    if 'haydn' in artist.lower():
        return 'Haydn'
    if 'tchaikovsky' in artist.lower():
        return 'Tchaikovsky'
        
    fname_lower = filename.lower()
    if 'beethoven' in fname_lower:
        return 'Beethoven'
    if 'mozart' in fname_lower:
        return 'Mozart'
    if 'haydn' in fname_lower:
        return 'Haydn'
    if 'tchaikovsky' in fname_lower:
        return 'Tchaikovsky'
    if 'requiem' in fname_lower or 'k.' in fname_lower:
        return 'Mozart'
    if 'op.23' in fname_lower or 'op. 23' in fname_lower:
        return 'Tchaikovsky'
        
    return None

def clean_classical_title(title, composer):
    title = re.sub(r'\s*\((Remastered|Recorded)\s+\d{4}\)', '', title)
    if composer:
        match = re.search(rf'\s*:\s*{re.escape(composer)}\s*:\s*', title, re.IGNORECASE)
        if match:
            start_idx = match.start()
            end_idx = match.end()
            prefix = title[:start_idx].strip()
            suffix = title[end_idx:].strip()
            
            if suffix.lower().startswith(prefix.lower()):
                suffix = suffix[len(prefix):].strip()
                suffix = re.sub(r'^[:\s\-–—]+', '', suffix)
            
            if suffix:
                title = f"{prefix}: {suffix}"
            else:
                title = prefix
    return title

def clean_artist_list(artist_str):
    if not artist_str:
        return "Unknown"
    clean = artist_str.replace('\x00', ', ').replace('; ', ', ')
    return clean.strip()

def extraer_dataset_a_csv(dir_dataset, es_academico, ruta_csv):
    dir_dataset = Path(dir_dataset)
    orig_dir = dir_dataset / "originales"
    cover_dir = dir_dataset / "covers"
    
    if not orig_dir.exists():
        print(f"Error: Original tracks folder not found at {orig_dir}")
        return False
        
    files = sorted(list(orig_dir.glob("*.mp3")) + list(orig_dir.glob("*.wav")))
    if not files:
        print(f"No audio files found at {orig_dir}")
        return False
        
    cover_files = sorted(list(cover_dir.glob("*.mp3")) + list(cover_dir.glob("*.wav"))) if cover_dir.exists() else []
        
    pattern_pop = re.compile(r'^(\d+)\s*-\s*(.*?)\s*-\s*(.*?)\.(mp3|wav)$', re.IGNORECASE)
    
    rows = []
    clean_txt_lines = []
    
    for f in files:
        m_track = re.match(r'^(\d+)', f.name)
        track = m_track.group(1) if m_track else "00"
        
        f_cover = None
        for cf in cover_files:
            m_cf = re.match(r'^(\d+)', cf.name)
            if m_cf and m_cf.group(1) == track:
                f_cover = cf
                break
        
        tags_orig = parse_id3v2(f)
        tags_cover = parse_id3v2(f_cover) if f_cover else {}
        
        year = tags_orig.get('year')
        if not year:
            m_year = re.search(r'\b(19\d\d|20\d\d)\b', f.name)
            year = m_year.group(1) if m_year else "Unknown"
            
        if es_academico:
            author = detect_composer(f.name, tags_orig) or "Unknown"
            
            title = tags_orig.get('title')
            if title:
                title = clean_classical_title(title, author)
            else:
                cleaned_name = re.sub(r'^\d+\s*-\s*', '', f.name)
                cleaned_name = re.sub(r'\s*\((Remastered|Recorded)\s+\d{4}\)', '', cleaned_name)
                cleaned_name = re.sub(r'\.(mp3|wav)$', '', cleaned_name, flags=re.IGNORECASE)
                if author != "Unknown":
                    cleaned_name = re.sub(rf'\s*_\s*{author}\s*_\s*', ': ', cleaned_name, flags=re.IGNORECASE)
                title = cleaned_name.replace('_', ' ').replace('  ', ' ').strip()
        else:
            author = tags_orig.get('artist')
            title = tags_orig.get('title')
            
            if not author or not title:
                m_pop = pattern_pop.match(f.name)
                if m_pop:
                    _, parsed_artist, parsed_title, _ = m_pop.groups()
                    author = author or parsed_artist
                    title = title or parsed_title
                else:
                    cleaned_name = re.sub(r'^\d+\s*-\s*', '', f.name)
                    cleaned_name = re.sub(r'\.(mp3|wav)$', '', cleaned_name, flags=re.IGNORECASE)
                    parts = cleaned_name.split(' - ')
                    if len(parts) >= 2:
                        author = author or parts[0]
                        title = title or ' - '.join(parts[1:])
                    else:
                        author = author or "Unknown"
                        title = title or cleaned_name
            
        orig_interpreter = tags_orig.get('artist')
        if not orig_interpreter:
            orig_interpreter = author
        orig_interpreter = clean_artist_list(orig_interpreter)
        
        cover_interpreter = "Unknown"
        if f_cover:
            cover_interpreter = tags_cover.get('artist')
            if not cover_interpreter:
                m_cover = pattern_pop.match(f_cover.name)
                if m_cover:
                    _, parsed_interpreter, _, _ = m_cover.groups()
                    cover_interpreter = parsed_interpreter
                else:
                    cleaned_name = re.sub(r'^\d+\s*-\s*', '', f_cover.name)
                    cleaned_name = re.sub(r'\.(mp3|wav)$', '', cleaned_name, flags=re.IGNORECASE)
                    parts = cleaned_name.split(' - ')
                    if len(parts) >= 2:
                        cover_interpreter = parts[0]
                    else:
                        cover_interpreter = cleaned_name
                        
        cover_interpreter = clean_artist_list(cover_interpreter)
        
        if es_academico:
            rows.append({
                'Pista': track,
                'Cancion': title,
                'Autor': author,
                'Original': orig_interpreter,
                'Cover': cover_interpreter,
                'Anio': year
            })
            clean_txt_lines.append(f"{title} - {author} - {orig_interpreter} - {cover_interpreter}")
        else:
            rows.append({
                'Pista': track,
                'Cancion': title,
                'Original': orig_interpreter,
                'Cover': cover_interpreter,
                'Anio': year
            })
            clean_txt_lines.append(f"{title} - {orig_interpreter} - {cover_interpreter}")
            
    with open(ruta_csv, mode='w', encoding='utf-8', newline='') as csvfile:
        if es_academico:
            fieldnames = ['Pista', 'Cancion', 'Autor', 'Original', 'Cover', 'Anio']
        else:
            fieldnames = ['Pista', 'Cancion', 'Original', 'Cover', 'Anio']
            
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row_to_write = {k: r[k] for k in fieldnames if k in r}
            writer.writerow(row_to_write)
            
    ruta_txt = Path(ruta_csv).with_suffix('.txt')
    with open(ruta_txt, mode='w', encoding='utf-8') as txtfile:
        for line in clean_txt_lines:
            txtfile.write(line + "\n")
            
    print(f"Extraction completed!")
    print(f"  - CSV saved at: {ruta_csv}")
    print(f"  - Clean TXT saved at: {ruta_txt}")
    
    print("\n--- CONTENT OF CLEAN TXT FILE ---")
    for line in clean_txt_lines:
        print(line)
    print("----------------------------------\n")
    
    return True

def generar_tabla_latex(ruta_csv, ruta_salida, titulo_seccion, etiqueta_tabla, descripcion):
    """
    Reads a CSV file and generates the complete LaTeX code.
    If the CSV contains the 'Autor' column (Academic), generates a 4-column table.
    Otherwise (Popular), generates a 3-column table.
    """
    if not os.path.exists(ruta_csv):
        print(f"Error: File not found {ruta_csv}")
        return

    # Read headers to detect if 'Autor' is present
    with open(ruta_csv, mode='r', encoding='utf-8') as archivo_csv:
        lector = csv.DictReader(archivo_csv)
        tiene_autor = 'Autor' in lector.fieldnames

    if tiene_autor:
        latex_code = f"""\\section{{{titulo_seccion}}}

{descripcion}

\\vspace{{0.5cm}}

\\begin{{longtable}}{{p{{0.32\\textwidth}} p{{0.18\\textwidth}} p{{0.23\\textwidth}} p{{0.23\\textwidth}}}}
\\caption{{List of tracks included in the {titulo_seccion}.}} \\label{{{etiqueta_tabla}}} \\\\
\\toprule
\\textbf{{Track Title / Evaluated Work}} & \\textbf{{Composer / Author}} & \\textbf{{Original Performer}} & \\textbf{{Cover Performer}} \\\\
\\midrule
\\endfirsthead

\\multicolumn{{4}}{{c}}%
{{{{\\bfseries \\tablename\\ \\thetable{{}} -- continued from previous page}}}} \\\\
\\toprule
\\textbf{{Track Title / Evaluated Work}} & \\textbf{{Composer / Author}} & \\textbf{{Original Performer}} & \\textbf{{Cover Performer}} \\\\
\\midrule
\\endhead

\\midrule
\\multicolumn{{4}}{{r}}{{Continued on next page}} \\\\
\\endfoot

\\bottomrule
\\endlastfoot

"""
    else:
        latex_code = f"""\\section{{{titulo_seccion}}}

{descripcion}

\\vspace{{0.5cm}}

\\begin{{longtable}}{{p{{0.4\\textwidth}} p{{0.28\\textwidth}} p{{0.28\\textwidth}}}}
\\caption{{List of tracks included in the {titulo_seccion}.}} \\label{{{etiqueta_tabla}}} \\\\
\\toprule
\\textbf{{Track Title / Evaluated Work}} & \\textbf{{Original Performer}} & \\textbf{{Cover Performer}} \\\\
\\midrule
\\endfirsthead

\\multicolumn{{3}}{{c}}%
{{{{\\bfseries \\tablename\\ \\thetable{{}} -- continued from previous page}}}} \\\\
\\toprule
\\textbf{{Track Title / Evaluated Work}} & \\textbf{{Original Performer}} & \\textbf{{Cover Performer}} \\\\
\\midrule
\\endhead

\\midrule
\\multicolumn{{3}}{{r}}{{Continued on next page}} \\\\
\\endfoot

\\bottomrule
\\endlastfoot

"""

    with open(ruta_csv, mode='r', encoding='utf-8') as archivo_csv:
        lector = csv.DictReader(archivo_csv)
        
        for fila in lector:
            cancion = escapar_latex(fila.get('Cancion', 'Unknown'))
            original = escapar_latex(fila.get('Original', 'Unknown'))
            cover = escapar_latex(fila.get('Cover', 'Unknown'))
            
            if tiene_autor:
                autor = escapar_latex(fila.get('Autor', 'Unknown'))
                latex_code += f"{cancion} & {autor} & {original} & {cover} \\\\\n"
            else:
                latex_code += f"{cancion} & {original} & {cover} \\\\\n"

    latex_code += "\n\\end{longtable}\n\n"
    
    with open(ruta_salida, mode='w', encoding='utf-8') as archivo_salida:
        archivo_salida.write(latex_code)
        
    print(f"Success! Table generated at: {ruta_salida}")

def procesar_un_dataset(ds_path):
    ds_name = ds_path.name
    print(f"\nProcessing dataset: {ds_name}...")
    
    es_academico = "acad" in ds_name.lower()
    
    confirm = input(f"Is this a classical/academic music dataset? (y/N) [Default: {'Yes' if es_academico else 'No'}]: ").strip().lower()
    if confirm in ['s', 'si', 'y', 'yes']:
        es_academico = True
    elif confirm in ['n', 'no']:
        es_academico = False
        
    csv_name = f"{ds_name}.csv"
    tex_name = f"tabla_{ds_name.replace('dataset_', '')}.tex"
    
    ok = extraer_dataset_a_csv(ds_path, es_academico=es_academico, ruta_csv=csv_name)
    if ok:
        seccion_titulo = ds_name.replace("dataset_", "").replace("_", " ").title() + " Corpus"
        if es_academico:
            descripcion = f"This section details the works from the {seccion_titulo}."
        else:
            descripcion = f"The following table lists the tracks evaluated in the {seccion_titulo}."
            
        generar_tabla_latex(
            ruta_csv=csv_name,
            ruta_salida=tex_name,
            titulo_seccion=seccion_titulo,
            etiqueta_tabla=f"tab:{ds_name.replace('dataset_', '')}",
            descripcion=descripcion
        )

def procesar_predeterminados(base_path):
    # --- POPULAR CORPUS ---
    dir_popular = base_path / "dataset_clei"
    if not dir_popular.exists():
        dir_popular = base_path / "dataset_OA"
        
    print(f"\nProcessing Popular Corpus from: {dir_popular.name}...")
    popular_ok = extraer_dataset_a_csv(dir_popular, es_academico=False, ruta_csv="dataset_popular.csv")
    
    if popular_ok:
        texto_popular = (
            "The following table lists the contemporary popular tracks evaluated in the second corpus. "
            "To ensure transparency and allow for auditory verification of the original-to-acoustic pairing "
            "constraint, a curated Spotify playlist containing all the reference tracks and their respective "
            "acoustic covers is available. The playlist can be accessed by scanning the QR code at the end of this appendix."
        )
        generar_tabla_latex(
            ruta_csv="dataset_popular.csv", 
            ruta_salida="tabla_popular.tex",
            titulo_seccion="Popular Music Stress-Test Corpus (80 Pairs)",
            etiqueta_tabla="tab:popular_corpus",
            descripcion=texto_popular
        )
        
    # --- ACADEMIC CORPUS ---
    dir_academico = base_path / "dataset_Acad"
    print(f"\nProcessing Academic Corpus from: {dir_academico.name}...")
    academico_ok = extraer_dataset_a_csv(dir_academico, es_academico=True, ruta_csv="dataset_academico.csv")
    
    if academico_ok:
        texto_academico = (
            "This section details the canonical works from the Classical and early Romantic periods "
            "used to establish the theoretical ground truth for William E. Caplin's formal functions."
        )
        generar_tabla_latex(
            ruta_csv="dataset_academico.csv", 
            ruta_salida="tabla_academica.tex",
            titulo_seccion="Academic Baseline Corpus (73 Pairs)",
            etiqueta_tabla="tab:academic_corpus",
            descripcion=texto_academico
        )

def menu_seleccion_dataset():
    base_path = Path(__file__).parent.absolute()
    
    datasets = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("dataset_"):
            if (item / "originales").exists():
                datasets.append(item.name)
                
    datasets.sort()
    
    print("\n" + "=" * 60)
    print("      METADATA EXTRACTOR (Track, Song, Artist, Year)")
    print("=" * 60)
    print("Select the dataset to process:")
    for i, ds in enumerate(datasets, 1):
        orig_dir = base_path / ds / "originales"
        count = len(list(orig_dir.glob("*.mp3")) + list(orig_dir.glob("*.wav")))
        print(f"  [{i}] {ds} ({count} tracks)")
    
    print(f"  [{len(datasets) + 1}] Process default datasets (Popular + Academic)")
    print(f"  [{len(datasets) + 2}] Exit")
    print("=" * 60)
    
    while True:
        try:
            opcion = input(f"Select an option (1-{len(datasets) + 2}): ").strip()
            if not opcion:
                continue
            idx = int(opcion)
            if 1 <= idx <= len(datasets):
                ds_seleccionado = datasets[idx - 1]
                procesar_un_dataset(base_path / ds_seleccionado)
                break
            elif idx == len(datasets) + 1:
                procesar_predeterminados(base_path)
                break
            elif idx == len(datasets) + 2:
                print("Operation cancelled. Goodbye!")
                break
            else:
                print("Invalid option. Please try again.")
        except ValueError:
            print("Please enter a valid option number.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts track number, artist, song title, and year from an audio dataset, and optionally generates LaTeX tables."
    )
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory (which contains the 'originales' folder)")
    parser.add_argument("--output_csv", type=str, help="Output path for the generated CSV file")
    parser.add_argument("--es_academico", action="store_true", help="Indicates if the dataset is classical/academic music")
    
    args = parser.parse_args()
    
    if args.dataset_dir and args.output_csv:
        extraer_dataset_a_csv(args.dataset_dir, args.es_academico, args.output_csv)
    else:
        menu_seleccion_dataset()