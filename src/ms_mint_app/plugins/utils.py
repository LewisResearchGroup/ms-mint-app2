import os
from pathlib import Path

import pandas as pd


def process_file(file_path: str, wdir, time_unit='min'):
    # move converted file to processed folder
    file_path = Path(file_path)
    ms_dir = Path(wdir).joinpath("ms_files")

    # TODO: is this needed?
    # T.fix_first_emtpy_line_after_upload_workaround(file_path)

    output_file = Path(ms_dir).joinpath(file_path.name).with_suffix(".parquet")
    from pyteomics import mzxml

    ms_level = 0
    polarity = '-'
    time_factor = 60 if time_unit in ['minutes', 'min'] else 1

    with mzxml.read(file_path.as_posix()) as ms_file_data:
        ms_data = []
        for i, data in enumerate(ms_file_data, start=1):
            ms_level = int(data.get("msLevel", 0))

            filterLine_ELMAVEN = None
            mz_precursor = None

            if ms_level == 2:
                polarity_str = 'Positive' if data.get("polarity") == '+' else 'Negative'
                mz_precursor = float(data['precursorMz'][0]['precursorMz'])
                mz = data["m/z array"][0]
                filterLine_ELMAVEN = ' '.join([
                    polarity_str,
                    # 'ESI', 'SRM', 'ms2',
                    f"{mz_precursor:.3f}",
                    f"[{mz:.3f}]"
                ])

            ms_data.append(
                dict(
                    scan_id=int(data.get("num") or 0),  # scan id
                    mz=[float(v) for v in data.get("m/z array", [])],  # mz
                    intensity=[float(v) for v in data.get("intensity array", [])],  # intensity
                    scan_time=float(data.get("retentionTime", 0.0)) * time_factor,  # scan time
                    mz_precursor=mz_precursor,  # mz precursor
                    filterLine=data.get("filterLine"),  # filter line
                    filterLine_ELMAVEN=filterLine_ELMAVEN  # filter line ELMAVEN
                )
            )
        df = pd.json_normalize(ms_data)
        df = df.explode(["mz", "intensity"])
        try:
            df.to_parquet(output_file)
        except Exception as e:
            print(f"{e = }")
        os.remove(file_path)
    return file_path.stem, ms_level, polarity, output_file