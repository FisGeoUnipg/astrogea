# Quick Start: Astrogea su Kubernetes - Guida Rapida

Guida veloce per installare e usare astrogea sul cluster Kubernetes uninuvola-dev.

## ⚡ Quick Start (3 Passi)

### 1️⃣ Installazione Sulla Macchina Kubernetes Host

```bash
# Scarica o copia astrogea sulla macchina
cd ~
# [opzionale: git clone o scp del progetto]

# Esegui installer automatico
chmod +x install-on-kubernetes-host.sh
./install-on-kubernetes-host.sh

# Attiva environment
source ~/astrogea-activate.sh
```

### 2️⃣ Build Immagine Docker

```bash
# Build immagine (vedi BUILD_DOCKER_IMAGE.md per dettagli)
cd ~/astrogea
docker build -t astrogea:latest .

# Se usi registry remoto:
docker tag astrogea:latest your-registry.uninuvola.cloud/astrogea:latest
docker push your-registry.uninuvola.cloud/astrogea:latest
export ASTROGEA_IMAGE=your-registry.uninuvola.cloud/astrogea:latest
```

### 3️⃣ Esegui Test

```bash
# Test con un file S3
python examples/simple_kubernetes.py \
  --input "s3://vamorini-test/path/to/file.hdr" \
  --output "s3://vamorini-test/results/"
```

## 📋 Configurazione Completa

### Prerequisiti Verificati

La configurazione usa questi valori dal `worker.yaml`:

- **Cluster**: uninuvola-dev
- **Workers**: 3 nodi (uninuvola-dev-01, 02, 03)
- **Resources per worker**: 4 CPU, 16GB RAM
- **Resources aggregate**: 12 CPU, 48GB RAM
- **Storage**: rook-ceph-block
- **Priority Classes**: 
  - worker-high-priority (1000)
  - worker-normal-priority (500) [default]

### Configurazione S3/MinIO

Già configurata nel codice:
- **Bucket**: vamorini-test
- **Endpoint**: https://minio-api.eagleprojects.cloud
- **Credentials**: incluse nel codice (con hash sicuro)

### Parametri Predefiniti

```bash
# Valori già configurati nel codice
--bucket vamorini-test
--endpoint https://minio-api.eagleprojects.cloud
--namespace astrogea
--storage-class rook-ceph-block
--priority-class worker-normal-priority
--concurrency 3

# Credenziali S3 già incluse
AWS_ACCESS_KEY_ID: 1ka7a5gUF5ZaNtHIBd7X
AWS_SECRET_ACCESS_KEY: PVEA7bXbUaFDq6X69xHkFX82k6CB5wgIDUWCT3i1
```

## 🚀 Esecuzione

### Comando Base

```bash
python examples/simple_kubernetes.py \
  --input "s3://vamorini-test/file.hdr" \
  --output "s3://vamorini-test/results/"
```

### Comando Completo

```bash
python examples/simple_kubernetes.py \
  --bucket vamorini-test \
  --endpoint https://minio-api.eagleprojects.cloud \
  --input "s3://vamorini-test/file1.hdr,s3://vamorini-test/file2.hdr" \
  --output "s3://vamorini-test/results/" \
  --namespace astrogea \
  --image astrogea:latest \
  --storage-class rook-ceph-block \
  --priority-class worker-normal-priority \
  --concurrency 3 \
  --timeout 1800 \
  --wait 600
```

### Con Variabili d'Ambiente

```bash
export ASTROGEA_INPUT_FILES="s3://vamorini-test/file.hdr"
export ASTROGEA_OUTPUT_PATH="s3://vamorini-test/results/"
export ASTROGEA_IMAGE=astrogea:latest

python examples/simple_kubernetes.py
```

## 📊 Monitoraggio

### Durante Esecuzione

```bash
# In un altro terminale
watch kubectl get pods -n astrogea

# O segui eventi
kubectl get events -n astrogea --sort-by='.lastTimestamp' -w

# Log di tutti i pod
kubectl logs -f -n astrogea -l app=astrogea-worker
```

### Verifica Risultati

```bash
# Lista job completati
kubectl get jobs -n astrogea

# Controlla risultati in S3
aws s3 ls s3://vamorini-test/results/ \
  --endpoint-url https://minio-api.eagleprojects.cloud
```

## 🐛 Troubleshooting

### "Module astrogea not found"

```bash
# Attiva environment
source ~/astrogea-activate.sh

# Oppure
cd ~/astrogea
source venv/bin/activate
```

### "ImagePullBackOff"

```bash
# Verifica immagine esiste
docker images | grep astrogea

# Se no, build:
docker build -t astrogea:latest .
```

### "PriorityClass not found"

```bash
# Crea PriorityClasses
kubectl apply -f k8s/priorityclasses.yaml
```

### "PVC not bound"

```bash
# Verifica storage class
kubectl get storageclass

# Crea PVC
kubectl apply -f k8s/storage.yaml
kubectl get pvc -n astrogea
```

## 📚 Guide Dettagliate

Per approfondimenti, leggi:

1. **[SETUP_KUBERNETES_HOST.md](SETUP_KUBERNETES_HOST.md)** - Installazione completa Python
2. **[BUILD_DOCKER_IMAGE.md](BUILD_DOCKER_IMAGE.md)** - Build e deploy immagine
3. **[KUBERNETES_TEST.md](KUBERNETES_TEST.md)** - Testing e debug
4. **k8s/** - File YAML Kubernetes
5. **examples/simple_kubernetes.py** - Codice sorgente

## 🔧 Modifiche Apportate al Codice

### File Modificati

1. **examples/simple_kubernetes.py**
   - Aggiunti valori default MinIO e credenziali
   - Aggiunti parametri `--storage-class` e `--priority-class`
   - Configurato `node_selector` per worker nodes
   - Risorse impostate secondo LimitRange

2. **astrogea/config.py**
   - Aggiunti campi `priority_class` e `node_selector` a `KubernetesConfig`

3. **astrogea/distributed.py**
   - Esteso `KubernetesJobManager` per supportare `priority_class`, `node_selector`, `storage_class`
   - Job spec ora include configurazioni complete

4. **k8s/priorityclasses.yaml** (nuovo)
   - PriorityClasses per cluster uninuvola-dev

5. **setup-kubernetes-test.sh** (nuovo)
   - Script setup risorse Kubernetes

6. **install-on-kubernetes-host.sh** (nuovo)
   - Installer completo Python environment

## ✅ Checklist Pre-Esecuzione

Prima di eseguire, verifica:

- [ ] Python 3.8+ installato
- [ ] kubectl configurato e connesso
- [ ] Docker installato
- [ ] Astrogea installato (`pip install -e .`)
- [ ] Namespace `astrogea` esiste
- [ ] PriorityClasses esistono
- [ ] StorageClass `rook-ceph-block` esiste
- [ ] PVC `astrogea-data-pvc` è Bound
- [ ] Immagine Docker `astrogea:latest` disponibile
- [ ] Accesso a MinIO/S3 funzionante

Verifica rapida:
```bash
kubectl get namespace astrogea
kubectl get priorityclass
kubectl get storageclass rook-ceph-block
kubectl get pvc -n astrogea
docker images | grep astrogea
python -c "import astrogea; print('OK')"
```

## 🎯 Prossimi Passi

1. Esegui test con file piccolo
2. Monitora performance e risorse
3. Ottimizza `concurrency` basandosi su risultati
4. Configura alerting se necessario
5. Automatizza con CI/CD se opportuno

## 📞 Supporto

Per problemi:
1. Controlla log pod: `kubectl logs -n astrogea <pod-name>`
2. Verifica eventi: `kubectl get events -n astrogea`
3. Consulta guide dettagliate sopra
4. Controlla configurazione cluster uninuvola-dev

---

**Cluster**: uninuvola-dev  
**Workers**: uninuvola-dev-01/02/03  
**Storage**: MinIO (vamorini-test)

