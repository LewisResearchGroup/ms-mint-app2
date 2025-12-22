
from ms_mint_app.tools import sparsify_chrom
import itertools

def generate_chromatogram_traces(chrom_df, use_megatrace=False):
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')

    legend_groups = set()
    traces = []
    
    if chrom_df is None or len(chrom_df) == 0:
        return traces, x_min, x_max, y_min, y_max

    if not use_megatrace:
        # ------------------------------------
        # MODO NORMAL: una trace por cromatograma
        # ------------------------------------
        for i, row in enumerate(chrom_df.iter_rows(named=True)):

            scan_time_sparse, intensity_sparse = sparsify_chrom(
                row['scan_time_sliced'], row['intensity_sliced'], w=1, baseline=1.0, eps=0.0
            )
            
            trace = {
                'type': 'scattergl',
                'mode': 'lines',
                'x': scan_time_sparse,
                'y': intensity_sparse,
                'line': {'color': row['color']},
                'name': row['label'] or row['ms_file_label'],
                'legendgroup': row['sample_type'],
                'hoverlabel': dict(namelength=-1)
            }

            if row['sample_type'] not in legend_groups:
                trace['legendgrouptitle'] = {'text': row['sample_type']}
                legend_groups.add(row['sample_type'])

            traces.append(trace)
            
            # Check for None values before min/max calculations
            if row['scan_time_min_in_range'] is not None:
                x_min = min(x_min, row['scan_time_min_in_range'])
            if row['scan_time_max_in_range'] is not None:
                x_max = max(x_max, row['scan_time_max_in_range'])
            if row['intensity_min_in_range'] is not None:
                y_min = min(y_min, row['intensity_min_in_range'])
            if row['intensity_max_in_range'] is not None:
                y_max = max(y_max, row['intensity_max_in_range'])

    else:
        # ------------------------------------
        # MODO REDUCIDO: una trace por sample_type
        # ------------------------------------
        grouped = {}
        for row in chrom_df.iter_rows(named=True):
            stype = row['sample_type']
            if stype not in grouped:
                grouped[stype] = {
                    'x': [],
                    'y': [],
                    'color': row['color'],
                    'count': 0
                }

            # Esparcificar individualmente antes de unir (opcional, para ahorrar más)
            st, ints = sparsify_chrom(
                row['scan_time_sliced'], row['intensity_sliced'], w=1, baseline=1.0, eps=0.0
            )

            grouped[stype]['x'].append(st)
            grouped[stype]['x'].append([None])
            grouped[stype]['y'].append(ints)
            grouped[stype]['y'].append([None])
            grouped[stype]['count'] += 1

            if row['scan_time_min_in_range'] is not None:
                x_min = min(x_min, row['scan_time_min_in_range'])
            if row['scan_time_max_in_range'] is not None:
                x_max = max(x_max, row['scan_time_max_in_range'])
            if row['intensity_min_in_range'] is not None:
                y_min = min(y_min, row['intensity_min_in_range'])
            if row['intensity_max_in_range'] is not None:
                y_max = max(y_max, row['intensity_max_in_range'])

        # Construir traces
        for stype, data in grouped.items():
            # Flatten arrays
            x_flat = list(itertools.chain(*data['x']))
            y_flat = list(itertools.chain(*data['y']))

            trace = {
                'type': 'scattergl',
                'mode': 'lines',
                'x': x_flat,
                'y': y_flat,
                'line': {'color': data['color']},
                'name': f"{stype} ({data['count']})",
                'legendgroup': stype,
                'hoverlabel': dict(namelength=-1),
                'connectgaps': False,
                'hoverinfo': 'skip',
                'hovertemplate': None
            }
            traces.append(trace)
            
    # Fix infinite values if data was empty or all Nones
    if x_min == float('inf'): x_min = 0
    if x_max == float('-inf'): x_max = 1
    if y_min == float('inf'): y_min = 0
    if y_max == float('-inf'): y_max = 1
            
    return traces, x_min, x_max, y_min, y_max
