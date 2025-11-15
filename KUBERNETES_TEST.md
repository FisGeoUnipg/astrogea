# Guida Test Kubernetes - simple_kubernetes.py

## Prerequisiti sulla macchina SSH

1. **Verifica accesso al cluster Kubernetes:**
```bash
kubectl cluster-info
kubectl get nodes
```

2. **Verifica che esistano i PriorityClass:**
```bash
kubectl get priorityclass
# Dovresti vedere: worker-high-priority e worker-normal-priority
```

3. **Verifica che esista lo StorageClass:**
```bash
kubectl get storageclass
# Dovresti vedere: rook-ceph-block
```

## Setup Namespace e Risorse

1. **Crea il namespace (se non esiste):**
```bash
kubectl apply -f k8s/namespace.yaml
```

2. **Crea i PriorityClass se non esistono:**
```bash
kubectl apply -f - <<EOF
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: worker-high-priority
value: 1000
globalDefault: false
description: "Alta priorità per workload critici sui worker"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: worker-normal-priority
value: 500
globalDefault: true
description: "Priorità normale per workload standard sui worker"
EOF
```

3. **Crea il ConfigMap (se non esiste):**
```bash
kubectl apply -f k8s/configmap.yaml
```

4. **Crea i PVC richiesti:**
```bash
kubectl apply -f k8s/storage.yaml
```

5. **Verifica che i PVC siano bound:**
```bash
kubectl get pvc -n astrogea
```

## Preparazione Immagine Docker

**IMPORTANTE:** Prima di eseguire il programma, devi avere un'immagine Docker con astrogea installato.

### Opzione 1: Build e Push al Registry del cluster
```bash
# Build dell'immagine
docker build -t astrogea:latest .

# Se il cluster usa un registry interno
docker tag astrogea:latest your-registry/astrogea:latest
docker push your-registry/astrogea:latest
```

### Opzione 2: Build direttamente sui worker nodes
Se i worker nodes sono accessibili, puoi fare il build lì.

## Test del Programma

### 1. Verifica Prerequisiti Python

```bash
cd /path/to/astrogea
python3 -c "import sys; sys.path.insert(0, '.'); from astrogea.config import create_config; print('OK')"
```

### 2. Test con File di Input

Esempio minimo:
```bash
python3 examples/simple_kubernetes.py \
  --input "s3://vamorini-test/path/to/file.hdr" \
  --output "s3://vamorini-test/results/"
```

Esempio completo con tutti i parametri:
```bash
python3 examples/simple_kubernetes.py \
  --bucket vamorini-test \
  --endpoint https://minio-api.eagleprojects.cloud \
  --input "s3://vamorini-test/input/file1.hdr" \
  --output "s3://vamorini-test/results/" \
  --namespace astrogea \
  --image astrogea:latest \
  --storage-class rook-ceph-block \
  --priority-class worker-normal-priority \
  --concurrency 3 \
  --timeout 1800 \
  --wait 600
```

### 3. Monitoraggio Job

Durante l'esecuzione, in un altro terminale:
```bash
# Vedi i job creati
kubectl get jobs -n astrogea

# Vedi i pod
kubectl get pods -n astrogea

# Log di un pod specifico
kubectl logs -n astrogea <pod-name>

# Segui i log in tempo reale
kubectl logs -f -n astrogea -l job-name=<job-name>
```

### 4. Debug

Se i job non partono:
```bash
# Verifica eventi del namespace
kubectl get events -n astrogea --sort-by='.lastTimestamp'

# Descrizione di un pod per vedere errori
kubectl describe pod -n astrogea <pod-name>

# Verifica ConfigMap
kubectl get configmap -n astrogea astrogea-config -o yaml
```

## Possibili Problemi e Soluzioni

### Problema: "PriorityClass non trovato"
```bash
kubectl apply -f k8s/priorityclasses.yaml  # Se esiste
# Oppure crea manualmente come sopra
```

### Problema: "StorageClass non trovato"
Verifica che rook-ceph-block esista:
```bash
kubectl get storageclass rook-ceph-block
```

Se non esiste, crealo o modifica il default nel codice.

### Problema: "Immagine non trovata"
L'immagine Docker deve essere disponibile nel cluster. Verifica:
```bash
# Se usi un registry locale
docker build -t astrogea:latest .
# E poi importa nei worker nodes

# Oppure usa un registry remoto
docker tag astrogea:latest registry.uninuvola.cloud/astrogea:latest
docker push registry.uninuvola.cloud/astrogea:latest
```

### Problema: "PVC non bound"
```bash
kubectl describe pvc -n astrogea astrogea-data-pvc
```

Verifica che lo storage class sia configurato correttamente.

### Problema: "NodeSelector non match"
I pod si schedulano solo su worker nodes. Verifica:
```bash
kubectl get nodes --show-labels | grep worker
```

## Esempio Completo con File Multipli

```bash
python3 examples/simple_kubernetes.py \
  --input "s3://vamorini-test/file1.hdr,s3://vamorini-test/file2.hdr,s3://vamorini-test/file3.hdr" \
  --output "s3://vamorini-test/results/" \
  --concurrency 3 \
  --priority-class worker-high-priority \
  --timeout 3600
```

Questo esegue 3 job in parallelo sui worker nodes.

## Variabili d'Ambiente Alternative

Invece di passare i parametri da CLI, puoi usare env vars:

```bash
export AWS_BUCKET=vamorini-test
export AWS_ENDPOINT_URL=https://minio-api.eagleprojects.cloud
export ASTROGEA_INPUT_FILES="s3://vamorini-test/file.hdr"
export ASTROGEA_OUTPUT_PATH="s3://vamorini-test/results/"
export ASTROGEA_NAMESPACE=astrogea
export ASTROGEA_IMAGE=astrogea:latest
export ASTROGEA_STORAGE_CLASS=rook-ceph-block
export ASTROGEA_PRIORITY_CLASS=worker-normal-priority

python3 examples/simple_kubernetes.py
```

## Verifica Finale

Dopo l'esecuzione, verifica i risultati:
```bash
# Controlla i job completati
kubectl get jobs -n astrogea

# Verifica i risultati in S3/MinIO
aws s3 ls s3://vamorini-test/results/ --endpoint-url https://minio-api.eagleprojects.cloud
```

