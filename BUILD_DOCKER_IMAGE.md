# Build e Deploy Immagine Docker per Astrogea

Guida per creare e deployare l'immagine Docker necessaria per i pod Kubernetes.

## Requisiti

- Docker installato sulla macchina o sui worker nodes
- Accesso a un registry Docker (opzionale ma consigliato)

## Opzione 1: Build Locale e Push Registry

### Su Macchina di Sviluppo

```bash
# 1. Naviga nella directory astrogea
cd ~/astrogea

# 2. Build immagine
docker build -t astrogea:latest .

# 3. Test immagine localmente
docker run --rm astrogea:latest python -c "import astrogea; print(astrogea.__version__)"

# 4. Tag per registry
docker tag astrogea:latest your-registry.uninuvola.cloud/astrogea:latest
# oppure con versione
docker tag astrogea:latest your-registry.uninuvola.cloud/astrogea:0.1.14

# 5. Login al registry
docker login your-registry.uninuvola.cloud

# 6. Push
docker push your-registry.uninuvola.cloud/astrogea:latest
```

### Su Macchina Kubernetes Host

```bash
# 1. Copia progetto se non già presente
scp -r /local/path/astrogea user@kubernetes-host:~/astrogea

# 2. SSH sul host
ssh user@kubernetes-host

# 3. Build (stesso processo di sopra)
cd ~/astrogea
docker build -t astrogea:latest .
docker tag astrogea:latest your-registry.uninuvola.cloud/astrogea:latest
docker push your-registry.uninuvola.cloud/astrogea:latest
```

### Aggiorna simple_kubernetes.py

```bash
export ASTROGEA_IMAGE=your-registry.uninuvola.cloud/astrogea:latest
```

Oppure modifica il default in `examples/simple_kubernetes.py`:

```python
parser.add_argument("--image", default="your-registry.uninuvola.cloud/astrogea:latest", ...)
```

## Opzione 2: Build Diretto su Worker Nodes (No Registry)

### Per ogni Worker Node

```bash
# 1. Copia progetto su worker
scp -r ~/astrogea user@uninuvola-dev-01:~/astrogea
scp -r ~/astrogea user@uninuvola-dev-02:~/astrogea
scp -r ~/astrogea user@uninuvola-dev-03:~/astrogea

# 2. SSH su ogni worker
ssh user@uninuvola-dev-01
cd ~/astrogea
docker build -t astrogea:latest .

# Ripeti per altri worker nodes
```

### Verifica Immagine Disponibile

Su ogni worker:
```bash
docker images | grep astrogea
```

### Usa Immagine Locale

```bash
export ASTROGEA_IMAGE=astrogea:latest
```

**Nota:** Con questa opzione, l'immagine deve esistere su TUTTI i worker nodes.

## Opzione 3: Build con Docker BuildX (Multi-arch)

Per supportare architetture multiple (AMD64, ARM64, etc.):

```bash
# 1. Setup buildx
docker buildx create --use --name multiarch

# 2. Build per multiple architetture
docker buildx build --platform linux/amd64,linux/arm64 \
    -t your-registry.uninuvola.cloud/astrogea:latest \
    -t your-registry.uninuvola.cloud/astrogea:0.1.14 \
    --push .
```

## Verifica Immagine

### Test Locale

```bash
# Avvia container
docker run -it --rm astrogea:latest bash

# Dentro il container
python -c "import astrogea; print(astrogea.__version__)"
python -c "from astrogea.config import create_config; print('OK')"
python -c "from kubernetes import client; print('K8s client OK')"
```

### Test in Kubernetes

```bash
# Crea un test pod
kubectl run astrogea-test --image=astrogea:latest -n astrogea --command -- /bin/sh -c "python -c 'import astrogea; print(astrogea.__version__); exit 0'"

# Verifica log
kubectl logs astrogea-test -n astrogea

# Elimina pod
kubectl delete pod astrogea-test -n astrogea
```

## Modifiche Dockerfile Personalizzate

### Esempio: Registry Privato con Autenticazione

Se usi un registry privato, devi creare un `imagePullSecret`:

```bash
# Crea secret
kubectl create secret docker-registry registry-credential \
  --docker-server=your-registry.uninuvola.cloud \
  --docker-username=username \
  --docker-password=password \
  --docker-email=user@example.com \
  -n astrogea

# Aggiorna imagePullSecrets nei pod (modifica distributed.py)
```

### Esempio: Cache Build Ottimizzato

```bash
# Usa BuildKit per cache layers
DOCKER_BUILDKIT=1 docker build --cache-from astrogea:latest -t astrogea:latest .
```

## Script Completo Build e Deploy

Crea `build-and-deploy.sh`:

```bash
#!/bin/bash
set -e

REGISTRY="${REGISTRY:-your-registry.uninuvola.cloud}"
IMAGE_NAME="astrogea"
VERSION="${VERSION:-latest}"

echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${VERSION} .

echo "Testing image..."
docker run --rm ${IMAGE_NAME}:${VERSION} python -c "import astrogea; print('OK')"

echo "Tagging for registry..."
docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:${VERSION}

echo "Pushing to registry..."
docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}

echo "Setting default image..."
export ASTROGEA_IMAGE=${REGISTRY}/${IMAGE_NAME}:${VERSION}

echo "To use this image, set:"
echo "  export ASTROGEA_IMAGE=${REGISTRY}/${IMAGE_NAME}:${VERSION}"
```

Esegui:
```bash
chmod +x build-and-deploy.sh
./build-and-deploy.sh
```

## Troubleshooting

### Problema: "Image pull errors"

```bash
# Verifica che l'immagine esista
docker images | grep astrogea

# Verifica registry
docker pull your-registry.uninuvola.cloud/astrogea:latest

# Verifica credenziali
docker login your-registry.uninuvola.cloud
```

### Problema: "Layer cache errors"

```bash
# Build senza cache
docker build --no-cache -t astrogea:latest .

# Pulizia cache
docker builder prune -a
```

### Problema: "Out of disk space"

```bash
# Pulizia immagini non usate
docker system prune -a --volumes

# Verifica spazio
df -h
```

### Problema: "Build troppo lento"

Usa BuildKit:
```bash
export DOCKER_BUILDKIT=1
docker build -t astrogea:latest .
```

Oppure build su machine più potente e push al registry.

## Multi-stage Build (Avanzato)

Per ridurre dimensioni immagine finale:

```dockerfile
# Stage 1: Build dependencies
FROM python:3.9-slim as builder
WORKDIR /build
COPY requirements.txt pyproject.toml ./
RUN pip install --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim
WORKDIR /opt/astrogea
COPY --from=builder /root/.local /root/.local
COPY astrogea/ ./astrogea/
ENV PATH=/root/.local/bin:$PATH
```

## CI/CD Integration (Avanzato)

### GitHub Actions

```yaml
name: Build and Push Docker Image

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build and push
        uses: docker/build-push-action@v2
        with:
          context: .
          push: true
          tags: |
            your-registry.uninuvola.cloud/astrogea:${{ github.ref_name }}
            your-registry.uninuvola.cloud/astrogea:latest
```

## Riferimenti

- [Dockerfile](Dockerfile) - Dockerfile principale
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Images](https://kubernetes.io/docs/concepts/containers/images/)

