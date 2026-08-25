#!/bin/bash
echo "Lancement du serveur API..."

# Tentative avec python3, puis python
if command -v python3 &> /dev/null; then
    python3 server.py
elif command -v python &> /dev/null; then
    python server.py
else
    echo "Erreur: Python n'est pas installé ou introuvable."
    exit 1
fi
