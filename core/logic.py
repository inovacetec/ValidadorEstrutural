import ezdxf
import pandas as pd
import numpy as np
import re
import os

def sanitize_text(text):
    """Limpeza agressiva de formatação AutoCAD."""
    if not text: return ""
    text = str(text)
    text = re.sub(r'\\[ACFHQTW]\d+;', '', text) 
    text = re.sub(r'\\[ACFHQTW]\[.*?\];', '', text)
    text = re.sub(r'\\S.*?;', '', text)
    text = re.sub(r'\\P', ' ', text)
    text = re.sub(r'\\[A-Z0-9]+;', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    text = text.replace('%%c', 'ø').replace('%%C', 'ø')
    text = text.replace('\n', ' ').replace('\r', '').strip()
    return text

def extract_texts_from_dxf(dxf_path):
    """Lê DXF e retorna DataFrame direto."""
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        raw_texts = []
        for entity in msp.query('TEXT MTEXT ATTRIB'):
            if entity.dxftype() == 'MTEXT':
                text_content = entity.plain_text() 
            else:
                text_content = entity.dxf.text
            
            clean = sanitize_text(text_content)
            if clean:
                insertion = entity.dxf.insert
                rotation = getattr(entity.dxf, 'rotation', 0.0)
                
                raw_texts.append({
                    'text': clean,
                    'x': insertion.x,
                    'y': insertion.y,
                    'rotation': rotation,
                    'layer': entity.dxf.layer
                })
        
        if not raw_texts: return pd.DataFrame(), "DXF Vazio"
        return pd.DataFrame(raw_texts), ""
        
    except Exception as e:
        return None, str(e)

def extract_element_name(title_text):
    """Extrai nome completo do elemento."""
    if any(ignore in title_text.upper() for ignore in ["ESCALA", "OBS", "NOTA", "DETALHE", "CORTE", "PESO TOTAL", "RESUMO"]):
        return None, 1

    match_mult = re.search(r'\([xX]?\s*(\d+)\s*\)', title_text)
    
    if match_mult:
        mult = int(match_mult.group(1))
        clean_name = title_text.replace(match_mult.group(0), "").strip()
    else:
        mult = 1
        clean_name = title_text.strip()
    
    upper_name = clean_name.upper()
    if upper_name == "L": return "LAJE", mult
    if upper_name == "B": return "BLOCO", mult
    if upper_name == "P": return "PILAR", mult
    if upper_name == "V": return "VIGA", mult
    
    if len(clean_name) < 2: return None, 1

    return clean_name, mult

def parse_table_spatial(df_texts):
    """Abordagem Espacial com Trava Vertical Rígida."""
    if df_texts.empty: return pd.DataFrame()

    pivots = df_texts[df_texts['text'].str.contains("50A", case=False, na=False)].copy()
    if pivots.empty: return pd.DataFrame()

    pivots = pivots.sort_values(by='y', ascending=False)
    pivot_list = pivots.to_dict('records')
    extracted_data = []

    for i, current in enumerate(pivot_list):
        current_name = "INDEFINIDO"
        current_mult = 1
        has_parent = False
        
        start_search = max(0, i-20)
        for prev in pivot_list[start_search:i]:
            dist = ((current['x'] - prev['x'])**2 + (current['y'] - prev['y'])**2)**0.5
            if dist <= 0.5 and prev['y'] >= current['y']:
                has_parent = True
                current_name = prev.get('assigned_name', 'INDEFINIDO')
                current_mult = prev.get('assigned_mult', 1)
                break
        
        if not has_parent:
            candidates = df_texts[
                (df_texts['x'] < current['x']) & (df_texts['x'] > current['x'] - 5.0) &
                (df_texts['y'] > current['y']) & (df_texts['y'] < current['y'] + 5.0)
            ]
            if not candidates.empty:
                candidates = candidates.copy()
                candidates['dist'] = np.sqrt((candidates['x'] - current['x'])**2 + (candidates['y'] - current['y'])**2)
                valid_titles = candidates[candidates['dist'] <= 3.0]
                
                if not valid_titles.empty:
                    raw_title = valid_titles.sort_values('dist').iloc[0]['text']
                    name_extracted, mult_extracted = extract_element_name(raw_title)
                    if name_extracted:
                        current_name = name_extracted
                        current_mult = mult_extracted
                    else:
                        current_name = "TITULO_INVALIDO"
                else:
                    current_name = "TITULO_NAO_ENCONTRADO"
            else:
                current_name = "TITULO_NAO_ENCONTRADO"

        current['assigned_name'] = current_name
        current['assigned_mult'] = current_mult
        
        if current_name in ["INDEFINIDO", "TITULO_NAO_ENCONTRADO", "TITULO_INVALIDO"]:
            continue

        data_candidates = df_texts[
            (df_texts['x'] > current['x']) &
            (df_texts['y'] > current['y'] - 0.15) & (df_texts['y'] < current['y'] + 0.15)
        ]
        
        if not data_candidates.empty:
            valid_data = data_candidates.sort_values('x')
            cols = valid_data['text'].tolist()
            
            parsed_values = []
            for c in cols:
                if re.search(r'(?i)-?CORR?-?', c):
                    parsed_values.append("-CORR-")
                    continue
                
                clean_c = re.sub(r'[^\d\.]', '', c.replace(',', '.'))
                if clean_c:
                    try: parsed_values.append(float(clean_c))
                    except: pass
            
            if len(parsed_values) >= 5:
                try:
                    pos = int(parsed_values[0])
                    bit = float(parsed_values[1])
                    num_barras = int(parsed_values[2])
                    val_comprimento = parsed_values[3]
                    
                    if val_comprimento == "-CORR-":
                        comprimento = "-CORR-"
                        total_calculado = "-CORR-"
                        total_tab = parsed_values[-1]
                    else:
                        comprimento = int(val_comprimento)
                        val_idx_4 = parsed_values[4]
                        is_duplicate = False
                        try:
                            if float(val_idx_4) == float(comprimento):
                                is_duplicate = True
                        except: pass

                        if len(parsed_values) > 5 and is_duplicate:
                            total_tab = int(parsed_values[5])
                        else:
                            total_tab = int(val_idx_4)
                        
                        total_calculado = num_barras * comprimento
                    
                    extracted_data.append({
                        'Elemento': current_name,
                        'pos': pos,
                        'bit': bit,
                        'Quantidade': current_mult,
                        'num_barras': num_barras,    
                        'Comprimento': comprimento,  
                        'total_tab': total_tab,      
                        'total_calculado': total_calculado 
                    })
                except: pass

    return pd.DataFrame(extracted_data)

def define_quadrants(df_elems):
    if df_elems.empty: return df_elems
    
    DEFAULT_WIDTH = 5000.0  
    DEFAULT_HEIGHT = 100.0
    
    df_elems['x_min'] = df_elems['x'] - 2.5
    df_elems['y_max'] = df_elems['y'] + 1.0
    
    if len(df_elems) == 1:
        df_elems['x_max'] = df_elems['x'] + 999999.0
        df_elems['y_min'] = df_elems['y'] - 999999.0
        return df_elems
    
    x_max_list = []
    
    for idx, row in df_elems.iterrows():
        curr_x = row['x']
        curr_y = row['y']
        
        right_candidates = df_elems[df_elems['x'] > curr_x]
        aligned_candidates = right_candidates[
            (right_candidates['y'] <= curr_y + 3) & 
            (right_candidates['y'] >= curr_y - 3)
        ]
        
        if not aligned_candidates.empty:
            x_limit = aligned_candidates['x'].min()
        else:
            x_limit = curr_x + DEFAULT_WIDTH
            
        x_max_list.append(x_limit - 2.5)
    
    df_elems['x_max'] = x_max_list
    
    y_min_list = []
    for idx, row in df_elems.iterrows():
        curr_x = row['x']
        curr_y = row['y']
        
        below_candidates = df_elems[df_elems['y'] <= (curr_y - 8.0)].copy()
        
        if not below_candidates.empty:
            below_candidates['dist'] = np.sqrt(
                (below_candidates['x'] - curr_x)**2 + 
                (below_candidates['y'] - curr_y)**2
            )
            closest_neighbor_y = below_candidates.loc[below_candidates['dist'].idxmin(), 'y']
            y_limit = closest_neighbor_y
        else:
            y_limit = curr_y - DEFAULT_HEIGHT
            
        y_min_list.append(y_limit)
        
    df_elems['y_min'] = y_min_list
    
    return df_elems

def generate_debug_dxf(original_dxf_path, df_elems):
    try:
        base_name = os.path.splitext(original_dxf_path)[0]
        debug_filename = f"{base_name}_DEBUG_QUADRANTES.dxf"
        
        doc = ezdxf.readfile(original_dxf_path)
        msp = doc.modelspace()
        
        if "DEBUG_QUADRANTES" not in doc.layers:
            doc.layers.add("DEBUG_QUADRANTES", color=5) 
            
        for _, row in df_elems.iterrows():
            draw_xmin = row['x_min']
            draw_ymax = row['y_max']
            draw_xmax = row['x_max'] if row['x_max'] < 500000 else row['x_min'] + 2000
            draw_ymin = row['y_min'] if row['y_min'] > -500000 else row['y_max'] - 2000

            points = [
                (draw_xmin, draw_ymax), 
                (draw_xmax, draw_ymax),
                (draw_xmax, draw_ymin), 
                (draw_xmin, draw_ymin), 
                (draw_xmin, draw_ymax) 
            ]
            msp.add_lwpolyline(points, dxfattribs={'layer': 'DEBUG_QUADRANTES', 'lineweight': 30})
            
            msp.add_text(
                f"BOX: {row['nome']}", 
                dxfattribs={
                    'layer': 'DEBUG_QUADRANTES', 
                    'height': 10,
                    'insert': (draw_xmin, draw_ymax + 5)
                }
            )

        doc.saveas(debug_filename)
        return debug_filename
    except Exception as e:
        print(f"Erro fatal ao gerar DXF de debug: {e}")
        return None

def parse_drawing_data(df_texts, known_elements=None):
    acos = []
    elementos = []
    
    # --- HELPER: Normalização Robusta ---
    def clean_key(text):
        if not text: return ""
        t = str(text).upper().strip()
        t = t.replace("Ã", "A").replace("Á", "A").replace("Â", "A").replace("À", "A")
        t = t.replace("É", "E").replace("Ê", "E")
        t = t.replace("Í", "I")
        t = t.replace("Ó", "O").replace("Õ", "O").replace("Ô", "O")
        t = t.replace("Ú", "U")
        t = t.replace("Ç", "C")
        return re.sub(r'[^A-Z0-9]', '', t)

    known_norm_set = set()
    if known_elements:
        for k in known_elements:
            if k and isinstance(k, str):
                known_norm_set.add(clean_key(k)) 

    # --- 1. DETECÇÃO DE TÍTULOS ---
    for idx, row in df_texts.iterrows():
        txt = str(row['text'])
        txt_clean_strict = clean_key(txt)
        is_title = False; name_found = ""
        
        if known_norm_set:
            sorted_elements = sorted(list(known_elements), key=lambda x: len(x), reverse=True)
            for k in sorted_elements:
                k_clean_strict = clean_key(k)
                if not k_clean_strict: continue
                match = False
                
                if k_clean_strict in txt_clean_strict:
                    if len(k_clean_strict) < 4:
                        start_idx = txt_clean_strict.find(k_clean_strict)
                        end_idx = start_idx + len(k_clean_strict)
                        suffix_ok = True; prefix_ok = True
                        if end_idx < len(txt_clean_strict):
                            if txt_clean_strict[end_idx].isdigit(): suffix_ok = False 
                        if start_idx > 0:
                            if txt_clean_strict[start_idx - 1].isalpha(): prefix_ok = False 
                        if suffix_ok and prefix_ok: match = True
                    else:
                        ratio = len(txt_clean_strict) / len(k_clean_strict)
                        if ratio <= 1.4: match = True
                        else: match = False
                        
                        if match and len(k_clean_strict) > 10:
                            curr_x = row['x']; curr_y = row['y']
                            vizinhos_verticais = df_texts[
                                (abs(df_texts['x'] - curr_x) < 2.0) & 
                                (abs(df_texts['y'] - curr_y) > 0.5) & 
                                (abs(df_texts['y'] - curr_y) < 12.0)
                            ]
                            if not vizinhos_verticais.empty:
                                for _, v_row in vizinhos_verticais.iterrows():
                                    v_txt = str(v_row['text']).upper(); v_clean = clean_key(v_txt)
                                    if any(x in v_clean for x in ["ESC", "150", "120", "125", "DETALHE"]): continue
                                    tem_numero = any(char.isdigit() for char in v_txt)
                                    if tem_numero and len(v_clean) < 15: continue 
                                    if len(v_clean) > 5:
                                        if v_clean[:6] == k_clean_strict[:6]:
                                            match = False; break
                if match:
                    is_title = True; name_found = k; break
        
        if is_title and name_found:
            elementos.append({
                'nome': name_found, 'mult': 1, 'x': row['x'], 'y': row['y'],
                'raw_text': txt, 'qtde_des': 1
            })
    
    elementos_validos = []
    titulos_vistos = set()
    if known_norm_set:
        for el in elementos:
            el_norm = clean_key(el['nome']) 
            chave_unica = f"{el_norm}_{round(el['x'], 1)}_{round(el['y'], 1)}"
            if el_norm in known_norm_set and chave_unica not in titulos_vistos:
                elementos_validos.append(el)
                titulos_vistos.add(chave_unica)
    else:
        elementos_validos = elementos

    df_elems = pd.DataFrame(elementos_validos)
    if df_elems.empty:
        return pd.DataFrame(columns=['Elemento', 'pos', 'bit', 'comp_des', 'num_barras_des']), pd.DataFrame()

    df_elems = define_quadrants(df_elems)

    # --- 2. MULTIPLICADORES ---
    if not df_elems.empty:
        for idx, row in df_elems.iterrows():
            xmin, xmax = row['x_min'], row['x_max']; tx, ty = row['x'], row['y']
            candidates = df_texts[(df_texts['x'] > tx) & (df_texts['x'] <= xmax) & (df_texts['y'] >= ty - 3) & (df_texts['y'] <= ty + 3)]
            if not candidates.empty:
                candidates = candidates.sort_values('x')
                for _, n_row in candidates.iterrows():
                    match_qty = re.match(r'^[\(]?\s*(\d+)\s*[xX]\s*[\)]?$', n_row['text'].strip(), re.IGNORECASE)
                    if match_qty:
                        qty = int(match_qty.group(1))
                        df_elems.at[idx, 'mult'] = qty; df_elems.at[idx, 'qtde_des'] = qty; break
    
    # --- 3. SELEÇÃO DE ANCHORS (N1, N2...) ---
    df_anchors_raw = df_texts[df_texts['text'].str.contains(r'[Nn]\d+', na=False)].copy()

    if not df_anchors_raw.empty and not df_elems.empty:
        filtered_idx = set(); in_quadrant_idx = set()
        for _, elem in df_elems.iterrows():
            anchors_in_quad = df_anchors_raw[
                (df_anchors_raw['x'] >= elem['x_min']) & (df_anchors_raw['x'] <= elem['x_max']) &
                (df_anchors_raw['y'] >= elem['y_min']) & (df_anchors_raw['y'] <= elem['y_max'])
            ].copy()
            if anchors_in_quad.empty: continue
            in_quadrant_idx.update(anchors_in_quad.index.tolist())
            anchors_in_quad['pos'] = anchors_in_quad['text'].str.extract(r'(\d+)').astype(int)
            for pos_val, group in anchors_in_quad.groupby('pos'):
                candidate_infos = []
                for a_idx, a_row in group.iterrows():
                    anchor_x, anchor_y = a_row['x'], a_row['y']
                    anchor_rot = a_row.get('rotation', 0.0) or 0.0
                    neigh = df_texts[df_texts.index != a_idx].copy()
                    if neigh.empty: continue
                    dx_glob = neigh['x'] - anchor_x; dy_glob = neigh['y'] - anchor_y
                    rad = np.radians(-anchor_rot)
                    neigh['dx_loc'] = dx_glob * np.cos(rad) - dy_glob * np.sin(rad)
                    neigh['dy_loc'] = dx_glob * np.sin(rad) + dy_glob * np.cos(rad)
                    
                    neigh = neigh[(neigh['dx_loc'] >= -3.0) & (neigh['dx_loc'] <= 4.0) & (neigh['dy_loc'] >= -4.0) & (neigh['dy_loc'] <= 1.0)]
                    
                    has_c = False; has_bit = False; has_c_strict = False
                    if not neigh.empty:
                        for _, nrow in neigh.iterrows():
                            t = str(nrow['text']).lower().strip()
                            if t.startswith("c="): has_c = True; has_c_strict = True
                            elif 'c=' in t or 'l=' in t: has_c = True
                            if 'ø' in t or 'diam' in t or '%%c' in t: has_bit = True
                    candidate_infos.append({'idx': a_idx, 'anchor_x': anchor_x, 'score_items': (1 if has_c else 0) + (1 if has_bit else 0), 'has_c_strict': has_c_strict})
                anchors_with_c = [ci for ci in candidate_infos if ci['has_c_strict']]
                if len(anchors_with_c) >= 2:
                    for ci in anchors_with_c: filtered_idx.add(ci['idx'])
                else:
                    if candidate_infos:
                        best = max(candidate_infos, key=lambda ci: (ci['score_items'], ci['anchor_x']))
                        filtered_idx.add(best['idx'])
        df_anchors = df_anchors_raw.loc[list(filtered_idx)] if filtered_idx else df_anchors_raw # Fallback
    else:
        df_anchors = df_anchors_raw
    
    # --- 4. EXTRAÇÃO DE DADOS (LOOP FINAL - CORRIGIDO) ---
    for idx, row in df_anchors.iterrows():
        try: pos = int(re.search(r'[Nn](\d+)', row['text']).group(1))
        except: continue
        
        anchor_x, anchor_y = row['x'], row['y']
        anchor_rot = row.get('rotation', 0.0) or 0.0
        rad = np.radians(-anchor_rot)
        
        neighbors = df_texts[df_texts.index != row.name].copy()
        if neighbors.empty: continue
        
        dx_glob = neighbors['x'] - anchor_x
        dy_glob = neighbors['y'] - anchor_y
        neighbors['dx_loc'] = dx_glob * np.cos(rad) - dy_glob * np.sin(rad)
        neighbors['dy_loc'] = dx_glob * np.sin(rad) + dy_glob * np.cos(rad)
        
        neighbors = neighbors[
            (neighbors['dx_loc'] >= -4.0) & (neighbors['dx_loc'] <= 4.0) &
            (neighbors['dy_loc'] >= -1.0) & (neighbors['dy_loc'] <= 0.5)
        ].copy()
        
        qtd = 1; bit = 0.0; comp = 0; bit_dist = 10000.0; has_close_c_flag = False; qty_candidates = []
        
        # Tenta pegar QTD do próprio texto Anchor (ex: "5 N1")
        try:
            match_qtd_own = re.match(r'^(\d+)\s*[Nn]', row['text'].strip())
            if match_qtd_own: qtd = int(match_qtd_own.group(1))
        except: pass

        for _, n_row in neighbors.iterrows():
            txt_n = n_row['text'].strip()
            dist = np.sqrt((n_row['x'] - anchor_x)**2 + (n_row['y'] - anchor_y)**2)
            dx_loc = n_row['dx_loc']; dy_loc = n_row['dy_loc']
            txt_lower = txt_n.lower()
            
            if dist <= 3.0 and re.match(r'^c=', txt_n, re.IGNORECASE): has_close_c_flag = True
            
            # Bitola (mesma linha)
            if (-1.0 <= dy_loc <= 1.0) and (txt_lower.startswith('ø') or txt_lower.startswith('diam') or txt_lower.startswith('%%c')):
                clean_bit = re.sub(r'[^\d\.,]', '', txt_n).replace(',', '.')
                if clean_bit: bit = float(clean_bit); bit_dist = dist 
            
            elif (-3.0 <= dx_loc <= -0.1) and (-0.5 <= dy_loc <= 0.5):
                if re.match(r'^[\d]+([xX\*][\d]+)?$', txt_n):
                    has_mult = ('x' in txt_lower) or ('*' in txt_lower)
                    qty_candidates.append({'text': txt_n, 'has_mult': has_mult, 'dist': dist})
            
            else:
                is_right_aligned = (0 < dx_loc <= 4.0) and (-0.5 <= dy_loc <= 0.5)
                is_below_aligned = (-2.0 <= dx_loc <= 2.5) and (-1.0 <= dy_loc <= -0.5)
                
                if (is_right_aligned or is_below_aligned):
                    if txt_lower.startswith("c="):
                        clean_comp = re.sub(r'[^\d]', '', txt_lower)
                        if clean_comp: comp = int(clean_comp)
        
        if qty_candidates and qtd == 1:
            mults = [c for c in qty_candidates if c['has_mult']]
            best = min(mults, key=lambda c: c['dist']) if mults else min(qty_candidates, key=lambda c: c['dist'])
            txt_best = best['text'].strip()
            if 'x' in txt_best.lower() or '*' in txt_best:
                try: 
                    parts = re.split(r'[xX\*]', txt_best)
                    qtd = int(parts[0]) * int(parts[1])
                except: pass
            else:
                try: qtd = int(re.sub(r'\D', '', txt_best))
                except: pass

        if pos > 0:
            acos.append({
                'pos': pos, 'qtd': qtd, 'bit': bit, 'comp': comp,
                'x': anchor_x, 'y': anchor_y,
                'bit_dist': bit_dist, 'has_close_c': has_close_c_flag
            })

    # --- 5. ASSOCIAÇÃO FINAL ---
    candidates = []
    if len(df_elems) == 1:
        single_elem = df_elems.iloc[0]
        for aco in acos:
            qtd_final = aco['qtd'] * single_elem['mult']
            candidates.append({
                'Elemento': single_elem['nome'], 'pos': aco['pos'], 'bit': aco['bit'], 'comp_des': aco['comp'],
                'qtd_barras_des': qtd_final, 'num_barras_des': aco['qtd'], 'anchor_x': aco['x'], 'bit_dist': aco.get('bit_dist', 10000.0),
                'has_close_c': aco.get('has_close_c', False), 'qtde_des': single_elem['qtde_des'] 
            })
    else:
        for aco in acos:
            ax, ay = aco['x'], aco['y']; found = False
            for _, elem in df_elems.iterrows():
                if (elem['x_min'] <= ax <= elem['x_max']) and (elem['y_min'] <= ay <= elem['y_max']):
                    qtd_final = aco['qtd'] * elem['mult']
                    candidates.append({
                        'Elemento': elem['nome'], 'pos': aco['pos'], 'bit': aco['bit'], 'comp_des': aco['comp'],
                        'qtd_barras_des': qtd_final, 'num_barras_des': aco['qtd'], 'anchor_x': aco['x'], 'bit_dist': aco.get('bit_dist', 10000.0),
                        'has_close_c': aco.get('has_close_c', False), 'qtde_des': elem['qtde_des']
                    })
                    found = True; break
            if not found:
                dists = np.sqrt((df_elems['x'] - ax)**2 + (df_elems['y'] - ay)**2)
                if not dists.empty:
                    idx_min = dists.idxmin()
                    if dists.min() < 2000:
                        elem = df_elems.iloc[idx_min]
                        qtd_final = aco['qtd'] * elem['mult']
                        candidates.append({
                            'Elemento': elem['nome'], 'pos': aco['pos'], 'bit': aco['bit'], 'comp_des': aco['comp'],
                            'qtd_barras_des': qtd_final, 'num_barras_des': aco['qtd'], 'anchor_x': aco['x'], 'bit_dist': aco.get('bit_dist', 10000.0),
                            'has_close_c': aco.get('has_close_c', False), 'qtde_des': elem['qtde_des']
                        })

    results = []
    if candidates:
        df_cand = pd.DataFrame(candidates)
        df_cand['is_complete_def'] = ((df_cand['bit_dist'] <= 1.0) & (df_cand['has_close_c']))
        df_cand = df_cand.sort_values(by=['Elemento', 'pos', 'is_complete_def', 'anchor_x'], ascending=[True, True, False, False])
        results = df_cand.to_dict('records')

    return pd.DataFrame(results), df_elems

def process_project(dxf_path):
    df_raw, warning = extract_texts_from_dxf(dxf_path)
    if df_raw is None: return None, None, warning, None
    if df_raw.empty: return None, pd.DataFrame(), "DXF Vazio", None

    count_50A = df_raw[df_raw['text'].str.contains("50A", case=False, na=False)].shape[0]
    msg_validacao = f"Balizadores '50A': {count_50A}. "

    df_tabela = parse_table_spatial(df_raw)
    
    known_elements = set()
    if not df_tabela.empty and 'Elemento' in df_tabela.columns:
        known_elements = set(df_tabela['Elemento'].unique())
        known_elements.discard("INDEFINIDO")
        known_elements.discard("TITULO_NAO_ENCONTRADO")
        known_elements.discard("TITULO_INVALIDO")

    df_desenho, df_elems_detected = parse_drawing_data(df_raw, known_elements)
    
    debug_path = None
    if not df_elems_detected.empty:
        debug_path = generate_debug_dxf(dxf_path, df_elems_detected)
        if debug_path:
            msg_validacao += f"ARQUIVO DEBUG GERADO. "
        else:
            msg_validacao += "FALHA DEBUG. "

    if df_tabela.empty:
        return pd.DataFrame(), df_raw, f"{msg_validacao}Tabela errada. {warning}", df_elems_detected
        
    if df_desenho.empty:
        df_draw_agg = pd.DataFrame(columns=['Elemento', 'pos', 'bit_des', 'qtde_des'])
        warning += " (Nenhuma armadura encontrada)"
    else:
        df_draw_agg = df_desenho.groupby(['Elemento', 'pos']).agg({
            'bit': 'first',
            'comp_des': 'first',
            'num_barras_des': 'sum',
            'qtde_des': 'max'
        }).reset_index().rename(columns={'bit': 'bit_des'})

    def norm(n): 
        if not n: return ""
        t = str(n).upper().strip()
        t = t.replace("Ã", "A").replace("Á", "A").replace("Â", "A").replace("À", "A")
        t = t.replace("É", "E").replace("Ê", "E")
        t = t.replace("Í", "I")
        t = t.replace("Ó", "O").replace("Õ", "O").replace("Ô", "O")
        t = t.replace("Ú", "U")
        t = t.replace("Ç", "C")
        return re.sub(r'[^A-Z0-9]', '', t)
    
    if df_elems_detected is not None and not df_elems_detected.empty:
        titulos_desenho_set = set(df_elems_detected['nome'].apply(norm).unique())
    else:
        titulos_desenho_set = set()

    if not df_draw_agg.empty:
        df_draw_agg['join_key'] = df_draw_agg['Elemento'].apply(norm)
    else:
        df_draw_agg['join_key'] = []
        
    df_tabela['join_key'] = df_tabela['Elemento'].apply(norm)
    
    if not df_tabela.empty:
        df_tabela = df_tabela.drop_duplicates(subset=['join_key', 'pos', 'bit'])

    cols_tabela = ['join_key', 'pos', 'bit', 'total_calculado', 'Quantidade', 'num_barras', 'Comprimento', 'total_tab', 'Elemento']

    df_final = pd.merge(
        df_draw_agg,
        df_tabela[cols_tabela],
        on=['join_key', 'pos'],
        how='right', 
        suffixes=('', '_tab')
    ).fillna(0)
    
    df_final['Elemento'] = df_final.apply(
        lambda x: x['Elemento_tab'] if (x['Elemento'] == 0 or pd.isna(x['Elemento'])) else x['Elemento'], 
        axis=1
    )
    
    df_final = df_final.drop(columns=['join_key', 'Elemento_tab'])

    if not df_elems_detected.empty:
        df_coords = df_elems_detected[['nome', 'x_min', 'y_min', 'x_max', 'y_max']].copy()
        df_coords['join_key_coords'] = df_coords['nome'].apply(norm)
        df_coords = df_coords.drop_duplicates(subset=['join_key_coords'])
        
        df_final['join_key_temp'] = df_final['Elemento'].apply(norm)
        df_final = pd.merge(df_final, df_coords, left_on='join_key_temp', right_on='join_key_coords', how='left')
        df_final = df_final.drop(columns=['join_key_temp', 'join_key_coords', 'nome'])
    else:
        df_final['x_min'] = 0; df_final['y_min'] = 0; df_final['x_max'] = 0; df_final['y_max'] = 0

    def check_title_found(row):
        if norm(row['Elemento']) in titulos_desenho_set: return "✅"
        return "❌"
    df_final['Tit. Des.'] = df_final.apply(check_title_found, axis=1)

    def check_comp_safe(r):
        try:
            c_des = float(r.get('comp_des', 0))
            c_tab = r.get('Comprimento', 0)
            
            if isinstance(c_tab, str): return f"⚠️ Verif. ({c_tab})"
            c_tab = float(c_tab)
            
            if c_des == 0: 
                return "❌ (Não Lido)" 
            
            if abs(c_des - c_tab) < 0.05: 
                return f"✅ ({int(c_des)})"
            
            return f"❌ ({int(c_des)} ≠ {int(c_tab)})"
        except: return "⚠️ Erro"

    df_final['Check Comp'] = df_final.apply(check_comp_safe, axis=1)
    
    def check_qtd_safe(r):
        try:
            q_unit_des = float(r.get('num_barras_des', 0))
            q_elem_mult = float(r.get('qtde_des', 0))
            q_tab_total = float(r.get('num_barras', 0))    
            
            mult = q_elem_mult if q_elem_mult > 0 else 1.0
            q_total_des = q_unit_des * mult
            
            if q_total_des == 0: 
                return "❌ (Não Lido)"
            
            if abs(q_total_des - q_tab_total) < 0.1: 
                return f"✅ ({int(q_total_des)})"
            
            return f"❌ ({int(q_unit_des)}*{int(mult)}={int(q_total_des)} ≠ {int(q_tab_total)})"
        except: return "✅"

    df_final['Check num_barras'] = df_final.apply(check_qtd_safe, axis=1)

    def check_qtde_elem_safe(r):
        try:
            q_elem_des = float(r.get('qtde_des', 0))
            q_elem_tab = float(r.get('Quantidade', 0))
            
            if q_elem_des == 0: 
                return "❌ (Não Lido)"
            
            if q_elem_des == q_elem_tab: 
                return f"✅ ({int(q_elem_des)})"
            
            return f"❌ ({int(q_elem_des)} ≠ {int(q_elem_tab)})"
        except: return "✅"
        
    df_final['Check Qtde'] = df_final.apply(check_qtde_elem_safe, axis=1)

    def check_bit_safe(r):
        try:
            b_des = float(r.get('bit_des', 0))
            b_tab = float(r.get('bit', 0))
            
            if b_des == 0: 
                return "❌ (-)"
            
            if abs(b_des - b_tab) < 0.1: 
                return f"✅ ({b_des})"
            return f"❌ ({b_des} ≠ {b_tab})"
        except: return "⚠️"

    df_final['Check Bit'] = df_final.apply(check_bit_safe, axis=1)

    def get_status(row):
        if "❌" in str(row['Check Comp']) or "❌" in str(row['Check num_barras']) or "❌" in str(row['Check Bit']) or "❌" in str(row['Check Qtde']):
            return "❌ ERRO"
        return "✅ OK"

    df_final['status'] = df_final.apply(get_status, axis=1)

    cols_to_str = ['Comprimento', 'total_calculado', 'total_tab', 'Check Comp', 'Check num_barras', 'Check Bit', 'Check Qtde']
    for col in cols_to_str:
        if col in df_final.columns:
            df_final[col] = df_final[col].astype(str)

    cols = ['Elemento', 'Tit. Des.', 'pos', 'bit', 'Check Bit', 'Quantidade', 'Check Qtde', 'num_barras', 'Check num_barras', 'Comprimento', 'Check Comp', 'total_tab', 'total_calculado', 'status', 'x_min', 'y_min', 'x_max', 'y_max']
    cols = [c for c in cols if c in df_final.columns]
    df_final = df_final[cols]
    
    return df_final, df_raw, f"{msg_validacao}Processado. {warning}", df_elems_detected