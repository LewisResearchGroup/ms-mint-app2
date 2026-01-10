from PyInstaller.utils.hooks import collect_dynamic_libs

datas = collect_dynamic_libs("numpy")
