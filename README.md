# MAIA_API v3.0 — Distributed AI Load Balancer

**MAIA_API** est un équilibreur de charge (Load Balancer) et routeur ultra-léger pour scaler vos modèles d'IA locaux (LLM et Vision) sous **llama.cpp server** et **Ollama** sur plusieurs ordinateurs de votre réseau local.

---

## 🌟 Points Forts

* 🚀 **Scaling Multi-Ordinateurs** : Répartissez la charge de vos requêtes IA entre plusieurs PC de votre réseau.
* 🤖 **Support Ollama & llama.cpp** : Découverte automatique des modèles via `/api/tags` (Ollama) et `/v1/models` (llama.cpp server).
* ⚖️ **Load Balancing & Failover** : Algorithme **Round-Robin** & suivi de latence avec basculement automatique en cas de panne d'une machine.
* 🖼️ **Support Vision Native** : Transmission transparente des requêtes d'images vers les modèles vision (`llava`, `qwen2-vl`, etc.).
* 🔌 **Compatibilité OpenAI & Ollama** : Endpoints `/v1/chat/completions`, `/api/chat` et `/api/generate` avec streaming temps réel.
* 📊 **Dashboard Web Moderne** : Interface glassmorphism pour suivre l'état des machines, les latences, les modèles disponibles et ajouter des nœuds dynamiquement.

---

## 🚀 Démarrage Rapide

### 1. Lancement de l'API
```bash
python server.py
```
L'API démarre sur `http://localhost:5000`.

### 2. Accès au Dashboard Web
Ouvrez votre navigateur sur `http://localhost:5000/` pour accéder au tableau de bord de surveillance et de gestion des ordinateurs.

---

## 📡 Utilisation des Endpoints API

### API Compatible OpenAI
```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1",
    "messages": [{"role": "user", "content": "Bonjour!"}]
  }'
```

### API Compatible Ollama
```bash
curl http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [{"role": "user", "content": "Explique la gravité."}]
  }'
```

---

## 🧪 Exécution des Tests
```bash
python test_api.py
```
