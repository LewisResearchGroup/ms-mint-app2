# Runtime hook to set OPENMS_DATA_PATH for pyopenms in frozen PyInstaller app
import os
import sys

if getattr(sys, 'frozen', False):
    # In frozen app, set OPENMS_DATA_PATH to bundled share directory
    _MEIPASS = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    openms_share = os.path.join(_MEIPASS, 'pyopenms_share', 'OpenMS')
    
    if os.path.exists(openms_share):
        os.environ['OPENMS_DATA_PATH'] = openms_share
