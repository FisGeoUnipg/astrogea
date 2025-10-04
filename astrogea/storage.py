"""
Storage abstraction layer for astrogea supporting local and cloud storage.
Supports S3, Google Cloud Storage, Azure Blob Storage, and local filesystem.
"""

import os
import io
import logging
import hashlib
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, BinaryIO
from pathlib import Path
import tempfile
import shutil

logger = logging.getLogger(__name__)

class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file exists at the given path."""
        pass
    
    @abstractmethod
    def read(self, path: str) -> bytes:
        """Read file content as bytes."""
        pass
    
    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Write bytes to file."""
        pass
    
    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete file at path."""
        pass
    
    @abstractmethod
    def list_files(self, prefix: str = "") -> list:
        """List files with optional prefix."""
        pass
    
    @abstractmethod
    def get_size(self, path: str) -> int:
        """Get file size in bytes."""
        pass
    
    @abstractmethod
    def get_modified_time(self, path: str) -> float:
        """Get file modification time as timestamp."""
        pass

class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str = "/data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized local storage at {self.base_path}")
    
    def _get_full_path(self, path: str) -> Path:
        """Get full path by joining with base path."""
        return self.base_path / path.lstrip('/')
    
    def exists(self, path: str) -> bool:
        return self._get_full_path(path).exists()
    
    def read(self, path: str) -> bytes:
        full_path = self._get_full_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return full_path.read_bytes()
    
    def write(self, path: str, data: bytes) -> None:
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)
        logger.debug(f"Written {len(data)} bytes to {path}")
    
    def delete(self, path: str) -> None:
        full_path = self._get_full_path(path)
        if full_path.exists():
            full_path.unlink()
            logger.debug(f"Deleted {path}")
    
    def list_files(self, prefix: str = "") -> list:
        search_path = self._get_full_path(prefix)
        if not search_path.exists():
            return []
        
        files = []
        for item in search_path.rglob('*'):
            if item.is_file():
                rel_path = item.relative_to(self.base_path)
                files.append(str(rel_path))
        return files
    
    def get_size(self, path: str) -> int:
        full_path = self._get_full_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return full_path.stat().st_size
    
    def get_modified_time(self, path: str) -> float:
        full_path = self._get_full_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return full_path.stat().st_mtime

class S3StorageBackend(StorageBackend):
    """Amazon S3 storage backend."""
    
    def __init__(self, bucket_name: str, aws_access_key_id: str = None, 
                 aws_secret_access_key: str = None, region: str = "us-east-1",
                 endpoint_url: str = None):
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
        
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize S3 client
        session_kwargs = {'region_name': region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key
            })
        
        self.s3_client = boto3.client('s3', endpoint_url=endpoint_url, **session_kwargs)
        logger.info(f"Initialized S3 storage for bucket {bucket_name}")
    
    def exists(self, path: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=path)
            return True
        except ClientError:
            return False
    
    def read(self, path: str) -> bytes:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=path)
            return response['Body'].read()
        except ClientError as e:
            raise FileNotFoundError(f"File not found in S3: {path}") from e
    
    def write(self, path: str, data: bytes) -> None:
        self.s3_client.put_object(Bucket=self.bucket_name, Key=path, Body=data)
        logger.debug(f"Written {len(data)} bytes to S3://{self.bucket_name}/{path}")
    
    def delete(self, path: str) -> None:
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=path)
            logger.debug(f"Deleted S3://{self.bucket_name}/{path}")
        except ClientError as e:
            logger.warning(f"Failed to delete S3 object {path}: {e}")
    
    def list_files(self, prefix: str = "") -> list:
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return []
    
    def get_size(self, path: str) -> int:
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=path)
            return response['ContentLength']
        except ClientError as e:
            raise FileNotFoundError(f"File not found in S3: {path}") from e
    
    def get_modified_time(self, path: str) -> float:
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=path)
            return response['LastModified'].timestamp()
        except ClientError as e:
            raise FileNotFoundError(f"File not found in S3: {path}") from e

class GCSStorageBackend(StorageBackend):
    """Google Cloud Storage backend."""
    
    def __init__(self, bucket_name: str, project_id: str = None, 
                 credentials_path: str = None):
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError("google-cloud-storage is required for GCS. Install with: pip install google-cloud-storage")
        
        self.bucket_name = bucket_name
        
        # Initialize GCS client
        client_kwargs = {}
        if project_id:
            client_kwargs['project'] = project_id
        if credentials_path:
            client_kwargs['credentials'] = credentials_path
        
        self.storage_client = storage.Client(**client_kwargs)
        self.bucket = self.storage_client.bucket(bucket_name)
        logger.info(f"Initialized GCS storage for bucket {bucket_name}")
    
    def exists(self, path: str) -> bool:
        blob = self.bucket.blob(path)
        return blob.exists()
    
    def read(self, path: str) -> bytes:
        blob = self.bucket.blob(path)
        if not blob.exists():
            raise FileNotFoundError(f"File not found in GCS: {path}")
        return blob.download_as_bytes()
    
    def write(self, path: str, data: bytes) -> None:
        blob = self.bucket.blob(path)
        blob.upload_from_string(data)
        logger.debug(f"Written {len(data)} bytes to GCS://{self.bucket_name}/{path}")
    
    def delete(self, path: str) -> None:
        blob = self.bucket.blob(path)
        try:
            blob.delete()
            logger.debug(f"Deleted GCS://{self.bucket_name}/{path}")
        except Exception as e:
            logger.warning(f"Failed to delete GCS object {path}: {e}")
    
    def list_files(self, prefix: str = "") -> list:
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Failed to list GCS objects: {e}")
            return []
    
    def get_size(self, path: str) -> int:
        blob = self.bucket.blob(path)
        if not blob.exists():
            raise FileNotFoundError(f"File not found in GCS: {path}")
        blob.reload()
        return blob.size
    
    def get_modified_time(self, path: str) -> float:
        blob = self.bucket.blob(path)
        if not blob.exists():
            raise FileNotFoundError(f"File not found in GCS: {path}")
        blob.reload()
        return blob.time_created.timestamp()

class AzureBlobStorageBackend(StorageBackend):
    """Azure Blob Storage backend."""
    
    def __init__(self, account_name: str, account_key: str, container_name: str):
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            raise ImportError("azure-storage-blob is required for Azure. Install with: pip install azure-storage-blob")
        
        self.account_name = account_name
        self.container_name = container_name
        
        # Initialize Azure client
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        logger.info(f"Initialized Azure Blob storage for container {container_name}")
    
    def exists(self, path: str) -> bool:
        try:
            blob_client = self.container_client.get_blob_client(path)
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False
    
    def read(self, path: str) -> bytes:
        try:
            blob_client = self.container_client.get_blob_client(path)
            return blob_client.download_blob().readall()
        except Exception as e:
            raise FileNotFoundError(f"File not found in Azure Blob: {path}") from e
    
    def write(self, path: str, data: bytes) -> None:
        blob_client = self.container_client.get_blob_client(path)
        blob_client.upload_blob(data, overwrite=True)
        logger.debug(f"Written {len(data)} bytes to Azure://{self.container_name}/{path}")
    
    def delete(self, path: str) -> None:
        try:
            blob_client = self.container_client.get_blob_client(path)
            blob_client.delete_blob()
            logger.debug(f"Deleted Azure://{self.container_name}/{path}")
        except Exception as e:
            logger.warning(f"Failed to delete Azure Blob object {path}: {e}")
    
    def list_files(self, prefix: str = "") -> list:
        try:
            blobs = self.container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Failed to list Azure Blob objects: {e}")
            return []
    
    def get_size(self, path: str) -> int:
        try:
            blob_client = self.container_client.get_blob_client(path)
            properties = blob_client.get_blob_properties()
            return properties.size
        except Exception as e:
            raise FileNotFoundError(f"File not found in Azure Blob: {path}") from e
    
    def get_modified_time(self, path: str) -> float:
        try:
            blob_client = self.container_client.get_blob_client(path)
            properties = blob_client.get_blob_properties()
            return properties.last_modified.timestamp()
        except Exception as e:
            raise FileNotFoundError(f"File not found in Azure Blob: {path}") from e

class StorageManager:
    """Unified storage manager supporting multiple backends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backends = {}
        self.cache_dir = Path(tempfile.gettempdir()) / "astrogea_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize backends based on config
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize storage backends based on configuration."""
        
        # Local storage
        if 'local' in self.config:
            local_config = self.config['local']
            self.backends['local'] = LocalStorageBackend(
                base_path=local_config.get('base_path', '/data/local')
            )
        
        # S3 storage
        if 's3' in self.config and self.config['s3'].get('bucket_name'):
            s3_config = self.config['s3']
            self.backends['s3'] = S3StorageBackend(
                bucket_name=s3_config['bucket_name'],
                aws_access_key_id=s3_config.get('aws_access_key_id'),
                aws_secret_access_key=s3_config.get('aws_secret_access_key'),
                region=s3_config.get('region', 'us-east-1'),
                endpoint_url=s3_config.get('endpoint_url')
            )
        
        # GCS storage
        if 'gcs' in self.config and self.config['gcs'].get('bucket_name'):
            gcs_config = self.config['gcs']
            self.backends['gcs'] = GCSStorageBackend(
                bucket_name=gcs_config['bucket_name'],
                project_id=gcs_config.get('project_id'),
                credentials_path=gcs_config.get('credentials_path')
            )
        
        # Azure Blob storage
        if 'azure' in self.config and self.config['azure'].get('container_name'):
            azure_config = self.config['azure']
            self.backends['azure'] = AzureBlobStorageBackend(
                account_name=azure_config['account_name'],
                account_key=azure_config['account_key'],
                container_name=azure_config['container_name']
            )
        
        logger.info(f"Initialized storage backends: {list(self.backends.keys())}")
    
    def _parse_uri(self, uri: str) -> tuple:
        """Parse storage URI to extract backend and path."""
        if '://' in uri:
            backend_name, path = uri.split('://', 1)
            return backend_name, path
        else:
            # Default to local storage
            return 'local', uri
    
    def _get_backend(self, backend_name: str) -> StorageBackend:
        """Get storage backend by name."""
        if backend_name not in self.backends:
            raise ValueError(f"Storage backend '{backend_name}' not configured")
        return self.backends[backend_name]
    
    def _get_cache_path(self, uri: str) -> Path:
        """Get cache path for a URI."""
        uri_hash = hashlib.md5(uri.encode()).hexdigest()
        return self.cache_dir / f"{uri_hash}"
    
    def exists(self, uri: str) -> bool:
        """Check if file exists in any backend."""
        backend_name, path = self._parse_uri(uri)
        backend = self._get_backend(backend_name)
        return backend.exists(path)
    
    def read(self, uri: str, use_cache: bool = True) -> bytes:
        """Read file from storage with optional caching."""
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(uri)
            if cache_path.exists():
                logger.debug(f"Reading from cache: {uri}")
                return cache_path.read_bytes()
        
        # Read from backend
        backend_name, path = self._parse_uri(uri)
        backend = self._get_backend(backend_name)
        data = backend.read(path)
        
        # Cache the data
        if use_cache:
            cache_path.write_bytes(data)
            logger.debug(f"Cached {len(data)} bytes for {uri}")
        
        return data
    
    def write(self, uri: str, data: bytes) -> None:
        """Write data to storage."""
        backend_name, path = self._parse_uri(uri)
        backend = self._get_backend(backend_name)
        backend.write(path, data)
        
        # Update cache
        cache_path = self._get_cache_path(uri)
        cache_path.write_bytes(data)
    
    def delete(self, uri: str) -> None:
        """Delete file from storage."""
        backend_name, path = self._parse_uri(uri)
        backend = self._get_backend(backend_name)
        backend.delete(path)
        
        # Remove from cache
        cache_path = self._get_cache_path(uri)
        if cache_path.exists():
            cache_path.unlink()
    
    def list_files(self, uri_prefix: str = "") -> list:
        """List files with prefix."""
        backend_name, path = self._parse_uri(uri_prefix)
        backend = self._get_backend(backend_name)
        return backend.list_files(path)
    
    def get_size(self, uri: str) -> int:
        """Get file size."""
        backend_name, path = self._parse_uri(uri)
        backend = self._get_backend(backend_name)
        return backend.get_size(path)
    
    def get_modified_time(self, uri: str) -> float:
        """Get file modification time."""
        backend_name, path = self._parse_uri(uri)
        backend = self._get_backend(backend_name)
        return backend.get_modified_time(path)
    
    def download_to_local(self, uri: str, local_path: str) -> str:
        """Download file from storage to local path."""
        data = self.read(uri)
        local_file = Path(local_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_bytes(data)
        logger.info(f"Downloaded {uri} to {local_path}")
        return str(local_file)
    
    def upload_from_local(self, local_path: str, uri: str) -> None:
        """Upload local file to storage."""
        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        data = local_file.read_bytes()
        self.write(uri, data)
        logger.info(f"Uploaded {local_path} to {uri}")

def create_storage_manager(config: Union[str, Dict[str, Any]] = None) -> StorageManager:
    """Create storage manager from configuration file or dictionary."""
    import yaml
    
    if isinstance(config, str):
        # Config is a file path
        if os.path.exists(config):
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            # Default configuration
            config_dict = {
                'local': {
                    'base_path': '/data/local'
                }
            }
    elif isinstance(config, dict):
        # Config is already a dictionary
        config_dict = config
    else:
        # Default configuration
        config_dict = {
            'local': {
                'base_path': '/data/local'
            }
        }
    
    return StorageManager(config_dict)
