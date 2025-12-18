GROUP_COLUMNS = [f"group_{i}" for i in range(1, 6)]

GROUP_LABELS = {
    "sample_type": "Sample Type",
    **{col: f"Group {i}" for i, col in enumerate(GROUP_COLUMNS, start=1)},
}

GROUP_DESCRIPTIONS = {
    col: f"User-defined grouping field {i} for analysis/grouping (free text)." for i, col in enumerate(GROUP_COLUMNS, start=1)
}
