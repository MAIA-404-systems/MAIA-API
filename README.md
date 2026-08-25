# MAIA_API v3.0 — Distributed AI Load Balancer

**MAIA_API** est un équilibreur de charge (Load Balancer) et routeur ultra-léger conçu pour faciliter la mise à l'échelle de vos modèles d'IA locaux (LLM et Vision) exécutés via **llama.cpp server** et **Ollama** sur plusieurs ordinateurs au sein de votre réseau local.

---

## Fonctionnalités Principales

* **Mise à l'échelle distribuée** : Répartissez efficacement la charge de vos requêtes d'intelligence artificielle entre plusieurs postes de votre réseau.
* **Support Ollama & llama.cpp** : Découverte automatique des modèles via les endpoints `/api/tags` (Ollama) et `/v1/models` (llama.cpp server).
* **Équilibrage de charge et tolérance aux pannes** : Implémentation de l'algorithme Round-Robin, suivi des latences en temps réel et basculement automatique en cas d'indisponibilité d'un nœud.
* **Support natif pour la vision par ordinateur** : Transmission transparente des requêtes d'analyse d'images vers les modèles spécialisés (tels que `llava`, `qwen2-vl`, etc.).
* **Compatibilité avec les standards OpenAI & Ollama** : Prise en charge des endpoints `/v1/chat/completions`, `/api/chat` et `/api/generate` avec support du streaming en temps réel.
* **Interface d'administration Web** : Tableau de bord moderne permettant la supervision de l'état des nœuds, des latences, la consultation des modèles disponibles et l'ajout dynamique de nouvelles ressources.

---

## Démarrage Rapide

### 1. Lancement de l'API
```bash
python server.py
```
Le serveur API sera accessible à l'adresse : `http://localhost:11345`.

### 2. Accès au Tableau de Bord
Ouvrez votre navigateur web et accédez à l'URL `http://localhost:11345/` pour visualiser l'interface de surveillance et gérer les nœuds de calcul.

---

## Utilisation des Endpoints API

### API Compatible avec le standard OpenAI
```bash
curl http://localhost:11345/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1",
    "messages": [{"role": "user", "content": "Bonjour!"}]
  }'
```

### API Compatible avec Ollama
```bash
curl http://localhost:11345/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [{"role": "user", "content": "Explique la gravité."}]
  }'
```

---

## Exécution des Tests
```bash
python test_api.py
```
