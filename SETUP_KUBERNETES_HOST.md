# Setup Astrogea sulla Macchina Kubernetes

Guida completa per installare e configurare astrogea sulla macchina SSH che controlla il cluster Kubernetes.

## Prerequisiti Sistema

### 1. Verifica Python

```bash
python3 --version  # Deve essere >= 3.8
which python3
```

### 2. Verifica kubectl

```bash
kubectl version --client
kubectl cluster-info  # Deve funzionare
```

### 3. Verifica Docker (opzionale ma consigliato)

```bash
docker --version
```

## Setup Ambiente Python

### Opzione 1: Setup Completo con Virtual Environment (CONSIGLIATO)

```bash
# 1. Crea directory per il progetto
mkdir -p ~/astrogea-workspace
cd ~/astrogea-workspace

# 2. Clona o copia il progetto astrogea
# Se hai già il codice locale:
scp -r /path/to/local/astrogea ./

# Se è su git:
git clone https://github.com/FisGeoUnipg/astrogea.git
cd astrogea

# 3. Crea virtual environment
python3 -m venv venv

# 4. Attiva virtual environment
source venv/bin/activate  # Linux/Mac
# oppure su Windows PowerShell:
# venv\Scripts\Activate.ps1

# 5. Aggiorna pip
pip install --upgrade pip setuptools wheel

# 6. Installa dipendenze base
pip install numpy scipy astropy spectral xarray matplotlib netCDF4

# 7. Installa dipendenze Kubernetes
pip install kubernetes boto3 dask[complete] pyyaml

# 8. Installa astrogea in modalità sviluppo
pip install -e .

# 9. Verifica installazione
python -c "import astrogea; print('Astrogea version:', astrogea.__version__)"
python -c "from astrogea.config import create_config; print('Config OK')"
python -c "from astrogea.orchestrator import create_job_orchestrator; print('Orchestrator OK')"
```

### Opzione 2: Setup Manuale senza Virtual Environment

```bash
# 1. Installa dipendenze sistema (se necessario)
sudo apt-get update  # Debian/Ubuntu
sudo apt-get install -y python3-pip python3-venv python3-dev
sudo apt-get install -y gcc g++ libgdal-dev  # Per GDAL se serve

# 2. Installa Python packages
pip3 install --user numpy scipy astropy spectral xarray matplotlib netCDF4
pip3 install --user kubernetes boto3 dask[complete] pyyaml

# 3. Configura PYTHONPATH
export PYTHONPATH=~/astrogea:$PYTHONPATH

# 4. Test import
python3 -c "import astrogea; print('OK')"
```

### Opzione 3: Setup con Conda/Miniconda

```bash
# 1. Crea environment conda
conda create -n astrogea python=3.9
conda activate astrogea

# 2. Installa pacchetti base
conda install -c conda-forge numpy scipy astropy matplotlib netcdf4 xarray dask

# 3. Installa pacchetti Python aggiuntivi
pip install spectral kubernetes boto3 pyyaml

# 4. Installa astrogea
cd ~/astrogea
pip install -e .
```

## Setup Kubernetes Resources

### 1. Crea le risorse nel cluster

```bash
cd ~/astrogea

# Opzione A: Usa lo script automatico
chmod +x setup-kubernetes-test.sh
./setup-kubernetes-test.sh

# Opzione B: Setup manuale
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/priorityclasses.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/storage.yaml

# Verifica
kubectl get namespace astrogea
kubectl get priorityclass worker-high-priority worker-normal-priority
kubectl get pvc -n astrogea
```

### 2. Verifica StorageClass

```bash
kubectl get storageclass
# Deve esistere "rook-ceph-block"
```

## Preparazione Immagine Docker

**IMPORTANTE:** I pod Kubernetes eseguono codice dentro un container Docker. Devi preparare l'immagine.

### Opzione 1: Build e Push Registry

```bash
cd ~/astrogea

# 1. Build immagine
docker build -t astrogea:latest .

# 2. Tag per registry (se usi registry remoto)
docker tag astrogea:latest your-registry.uninuvola.cloud/astrogea:latest

# 3. Push al registry
docker push your-registry.uninuvola.cloud/astrogea:latest

# 4. Aggiorna reference nell'applicazione
# Modifica simple_kubernetes.py --image oppure variabile d'ambiente
export ASTROGEA_IMAGE=your-registry.uninuvola.cloud/astrogea:latest
```

### Opzione 2: Build su Worker Nodes (se non hai registry)

```bash
# Copia il progetto su un worker node
scp -r ~/astrogea user@worker-node:~/astrogea

# SSH sul worker node
ssh user@worker-node

# Build immagine
cd ~/astrogea
docker build -t astrogea:latest .

# Verifica che l'immagine sia disponibile
docker images | grep astrogea
```

### Opzione 3: Usa Immagine Pre-esistente

Se hai già un'immagine astrogea in un registry:
```bash
export ASTROGEA_IMAGE=existing-registry.io/astrogea:v0.1.14
```

## Test Configurazione

### 1. Test Import Python

```bash
# Attiva venv se usi opzione 1
source venv/bin/activate

# Test import
python -c "
from astrogea.config import create_config, Environment
from astrogea.orchestrator import create_job_orchestrator
from astrogea.storage import create_storage_manager
print('✓ All imports OK')
"
```

### 2. Test Accesso Kubernetes

```bash
# Verifica namespace
kubectl get namespace astrogea

# Verifica PriorityClasses
kubectl get priorityclass

# Verifica StorageClass
kubectl get storageclass rook-ceph-block

# Verifica PVC
kubectl get pvc -n astrogea
```

### 3. Test con Python Kubernetes Client

```bash
python -c "
from kubernetes import client, config
try:
    config.load_kube_config()
    v1 = client.CoreV1Api()
    ns = v1.read_namespace(name='astrogea')
    print('✓ Kubernetes connection OK')
except Exception as e:
    print('✗ Kubernetes error:', e)
"
```

## Esecuzione Test

### 1. Verifica File di Input

```bash
# Verifica accesso a MinIO/S3
aws s3 ls s3://vamorini-test/ --endpoint-url https://minio-api.eagleprojects.cloud

# Lista file disponibili
aws s3 ls s3://vamorini-test/input/ --endpoint-url https://minio-api.eagleprojects.cloud
```

### 2. Esegui Test Semplice

```bash
# Attiva venv se necessario
source venv/bin/activate

# Test con un file
python examples/simple_kubernetes.py \
  --input "s3://vamorini-test/path/to/file.hdr" \
  --output "s3://vamorini-test/results/" \
  --concurrency 1 \
  --wait 120

# Monitora in tempo reale
kubectl get pods -n astrogea -w
```

### 3. Test Completo

```bash
python examples/simple_kubernetes.py \
  --bucket vamorini-test \
  --endpoint https://minio-api.eagleprojects.cloud \
  --input "s3://vamorini-test/file1.hdr" \
  --output "s3://vamorini-test/results/" \
  --namespace astrogea \
  --image astrogea:latest \
  --storage-class rook-ceph-block \
  --priority-class worker-normal-priority \
  --concurrency 3 \
  --timeout 1800 \
  --wait 600
```

## Troubleshooting

### Problema: "Module not found: astrogea"

```bash
# Verifica PYTHONPATH
echo $PYTHONPATH

# Aggiungi manualmente
export PYTHONPATH=~/astrogea:$PYTHONPATH

# Oppure usa python -m
cd ~/astrogea
python -m examples.simple_kubernetes --input ...
```

### Problema: "kubernetes module not found"

```bash
pip install kubernetes
# oppure
pip3 install --user kubernetes
```

### Problema: "Permission denied" su kubectl

```bash
# Verifica kubeconfig
ls -la ~/.kube/config

# Fix permissions
chmod 600 ~/.kube/config

# Verifica context
kubectl config current-context
```

### Problema: "ImagePullBackOff" nei pod

```bash
# Verifica immagine Docker
docker images | grep astrogea

# Verifica che l'immagine sia disponibile nei worker
kubectl describe pod <pod-name> -n astrogea

# Se usi registry, verifica credenziali
kubectl get secret -n astrogea
```

### Problema: "PVC not bound"

```bash
# Verifica storage class
kubectl get storageclass rook-ceph-block

# Verifica PVC
kubectl describe pvc -n astrogea astrogea-data-pvc

# Verifica eventi
kubectl get events -n astrogea --sort-by='.lastTimestamp'
```

## Script Quick Start Completo

Crea uno script per automatizzare tutto:

```bash
cat > ~/astrogea-setup.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Astrogea Kubernetes Setup ==="

# Variabili
PROJECT_DIR=~/astrogea-workspace
ASTROGEA_DIR=$PROJECT_DIR/astrogea

# Crea directory
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Clona/copia astrogea se non esiste
if [ ! -d "astrogea" ]; then
    echo "Clonare astrogea qui o copiare da altra posizione"
    exit 1
fi

cd astrogea

# Setup Python venv
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Installa dipendenze
pip install numpy scipy astropy spectral xarray matplotlib netCDF4
pip install kubernetes boto3 dask[complete] pyyaml

# Installa astrogea
pip install -e .

# Setup Kubernetes
echo "Setup Kubernetes resources..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/priorityclasses.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/storage.yaml

# Attendi PVC
kubectl wait --for=condition=Bound pvc -n astrogea --all --timeout=300s

# Verifica
python -c "import astrogea; print('✓ Astrogea version:', astrogea.__version__)"

echo "=== Setup Complete ==="
echo "Attiva venv con: source $ASTROGEA_DIR/venv/bin/activate"
echo "Poi esegui: python examples/simple_kubernetes.py --input <file> --output <path>"
EOF

chmod +x ~/astrogea-setup.sh
```

Esegui:
```bash
~/astrogea-setup.sh
```

## Prossimi Passi

1. **Build immagine Docker** (se non già fatto)
2. **Test con un file piccolo** per verificare il flusso completo
3. **Monitora risorse** durante l'esecuzione
4. **Ottimizza risorse** se necessario

## Riferimenti

- [KUBERNETES_TEST.md](KUBERNETES_TEST.md) - Guida test dettagliata
- [k8s/](k8s/) - File YAML Kubernetes
- [Dockerfile](Dockerfile) - Build immagine Docker

