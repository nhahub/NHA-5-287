#!/bin/sh
# Capture exact versions after a successful install (run on a machine with network access):
pip install -r requirements.txt && pip freeze > requirements-lock.txt && echo "wrote requirements-lock.txt"
