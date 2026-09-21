#!/usr/bin/env python3
"""Start a command in its own session so it survives the terminal or agent process going away.
usage: launch_detached.py <logfile> <cwd> <cmd> [args...]"""
import subprocess, sys, os
log, cwd, cmd = sys.argv[1], sys.argv[2], sys.argv[3:]
p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True)
print(p.pid)
