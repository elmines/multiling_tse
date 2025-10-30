import os
import stat

def chmodx(out_path):
    # All this just to do the equivalent of `chmod +x` ...
    os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)