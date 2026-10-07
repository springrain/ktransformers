#!/usr/bin/env python3
# Compile and run the GEC C++ test suite without requiring CMake.
# Usage: python test/gec/run_tests.py
# Exit code 0 = all checks passed; 2 = no C++ compiler found.
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'test' / 'gec' / 'test_gec.cpp'
INCLUDE = ROOT / 'include'


def find_compiler():
    for name in ('g++', 'clang++', 'c++'):
        path = shutil.which(name)
        if path:
            return [path, '-std=c++17', '-Wall', '-Wextra']
    if shutil.which('cl'):
        return ['cl', '/std:c++17', '/W4', '/EHsc']
    return None


def main():
    compiler = find_compiler()
    if compiler is None:
        print('No C++ compiler found (g++/clang++/c++/cl)', file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix='gec_test_') as tmp:
        binary = Path(tmp) / ('test_gec.exe' if sys.platform == 'win32' else 'test_gec')
        if compiler[0] == 'cl':
            cmd = compiler + ['/I', str(INCLUDE), str(SOURCE), '/Fe:' + str(binary)]
        else:
            cmd = compiler + ['-I', str(INCLUDE), str(SOURCE), '-o', str(binary)]
        print('Compiling:', ' '.join(cmd))
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.stdout:
            print(build.stdout, end='')
        if build.stderr:
            print(build.stderr, end='', file=sys.stderr)
        if build.returncode != 0:
            return build.returncode
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        print(run.stdout, end='')
        if run.stderr:
            print(run.stderr, end='', file=sys.stderr)
        return run.returncode


if __name__ == '__main__':
    sys.exit(main())
