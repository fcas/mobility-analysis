#!/bin/bash
until superset runserver; do
    echo "Superset crashed with exit code $?.  Respawning.." >&2
    sleep 1
done