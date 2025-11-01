import os
import stat

def chmodx(out_path):
    # All this just to do the equivalent of `chmod +x` ...
    os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

def write_code(out_path, bash_code):
    with open(out_path, 'w') as w:
        w.write(bash_code)
    chmodx(out_path)