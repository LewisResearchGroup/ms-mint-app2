from PyInstaller.utils.hooks import collect_data_files

from PyInstaller.utils.hooks import collect_submodules


datas = collect_data_files("ms_mint_app", excludes=["**/*.py", "**/*.pyc"])

hiddenimports = (
    ["ms_mint_app.static"]
    + collect_submodules("ms_mint_app.plugins")
    + collect_submodules("ms_mint_app.plugins.analysis_tools")
)
