import os
import platform
import argparse
import glob
from pathlib import Path
import re
import subprocess
import shutil
import sys

oskeys = {"Darwin": "macos", "Windows": "win", "Linux": "linux"}


def dir_path(string):
    if os.path.isdir(string):
        return Path(string)
    else:
        raise NotADirectoryError(string)


parser = argparse.ArgumentParser()
parser.add_argument(
    "path",
    type=dir_path,
)
args = parser.parse_args()

oskey = oskeys[platform.system()]

for f in glob.glob(f'{args.path/"*.whl"}'):
    if oskey in f:
        match = re.search("-cp(.)(.)-", f)
        try:
            env = os.environ.copy()
            env["PYVERSION"] = f"={match[1]}.{match[2]}"
            env["WHEEL"] = str(Path(f).absolute())
            print("building", f)
            subprocess.Popen(["conda-build", os.path.dirname(os.path.realpath(sys.argv[0])), "--output-folder", str((args.path/'tmp').absolute())], env=env).wait()

        except Exception as e:
            print("failed", f, e)
            continue

outdir=args.path/'output'
outdir.mkdir(exist_ok=True)
res=f"{(args.path/'tmp').absolute()}/**/*.tar.bz2"
for f in glob.glob(res):
  print(f)   
  shutil.copy(f,outdir)    
