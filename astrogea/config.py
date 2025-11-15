"""
Configuration management for astrogea supporting local and cloud environments.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Supported execution environments."""
    LOCAL = "local"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"

@dataclass
class StorageConfig:
    """Storage configuration."""
    local: Dict[str, Any] = field(default_factory=lambda: {
        'base_path': '/data/local',
        'cache_size': '1GB'
    })
    s3: Dict[str, Any] = field(default_factory=lambda: {
        'bucket_name': None,
        'region': 'us-east-1',
        'endpoint_url': None
    })
    gcs: Dict[str, Any] = field(default_factory=lambda: {
        'bucket_name': None,
        'project_id': None,
        'credentials_path': None
    })
    azure: Dict[str, Any] = field(default_factory=lambda: {
        'account_name': None,
        'account_key': None,
        'container_name': None
    })

@dataclass
class DaskConfig:
    """Dask configuration."""
    scheduler_url: Optional[str] = None
    dashboard_url: Optional[str] = None
    chunks: Dict[str, Union[str, int]] = field(default_factory=lambda: {
        'line': 'auto',
        'sample': 'auto',
        'wavelength': -1
    })
    memory_limit: str = "4GB"
    cpu_limit: int = 2
    timeout: int = 3600
    retry_attempts: int = 3

@dataclass
class ProcessingConfig:
    """Processing configuration."""
    use_dask: bool = True
    parallel_workers: int = 4
    chunk_size: str = "auto"
    memory_limit: str = "4GB"
    cpu_limit: int = 2
    timeout: int = 3600

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file_path: Optional[str] = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5

@dataclass
class KubernetesConfig:
    """Kubernetes-specific configuration."""
    namespace: str = "astrogea"
    image: str = "astrogea:latest"
    replicas: int = 3
    resources: Dict[str, Any] = field(default_factory=lambda: {
        'requests': {
            'memory': '2Gi',
            'cpu': '1'
        },
        'limits': {
            'memory': '4Gi',
            'cpu': '2'
        }
    })
    storage_class: str = ""
    priority_class: str = ""
    node_selector: Dict[str, str] = field(default_factory=lambda: {})
    pvc_size: str = "100Gi"

class AstrogeaConfig:
    """Main configuration class for astrogea."""
    
    def __init__(self, config_path: Optional[str] = None, environment: Environment = Environment.LOCAL):
        self.environment = environment
        self.config_path = config_path or self._get_default_config_path()
        
        # Initialize configuration sections
        self.storage = StorageConfig()
        self.dask = DaskConfig()
        self.processing = ProcessingConfig()
        self.logging = LoggingConfig()
        self.kubernetes = KubernetesConfig()
        
        # Load configuration
        self._load_config()
        self._apply_environment_overrides()
        self._setup_logging()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        # Check for config in current directory
        local_config = Path("astrogea_config.yaml")
        if local_config.exists():
            return str(local_config)
        
        # Check for config in user home directory
        home_config = Path.home() / ".astrogea" / "config.yaml"
        if home_config.exists():
            return str(home_config)
        
        # Check for config in environment variable
        env_config = os.environ.get('ASTROGEA_CONFIG')
        if env_config and Path(env_config).exists():
            return env_config
        
        # Return default path (will create if needed)
        return str(home_config)
    
    def _load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                
                # Load storage configuration
                if 'storage' in config_data:
                    self.storage = StorageConfig(**config_data['storage'])
                
                # Load dask configuration
                if 'dask' in config_data:
                    self.dask = DaskConfig(**config_data['dask'])
                
                # Load processing configuration
                if 'processing' in config_data:
                    self.processing = ProcessingConfig(**config_data['processing'])
                
                # Load logging configuration
                if 'logging' in config_data:
                    self.logging = LoggingConfig(**config_data['logging'])
                
                # Load kubernetes configuration
                if 'kubernetes' in config_data:
                    self.kubernetes = KubernetesConfig(**config_data['kubernetes'])
                
                logger.info(f"Loaded configuration from {self.config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load configuration from {self.config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info(f"Configuration file not found at {self.config_path}, using defaults")
    
    def _apply_environment_overrides(self):
        """Apply environment-specific overrides."""
        
        # Environment variables override
        if os.environ.get('ASTROGEA_S3_BUCKET'):
            self.storage.s3['bucket_name'] = os.environ['ASTROGEA_S3_BUCKET']
        
        if os.environ.get('ASTROGEA_S3_REGION'):
            self.storage.s3['region'] = os.environ['ASTROGEA_S3_REGION']
        
        if os.environ.get('ASTROGEA_GCS_BUCKET'):
            self.storage.gcs['bucket_name'] = os.environ['ASTROGEA_GCS_BUCKET']
        
        if os.environ.get('ASTROGEA_GCS_PROJECT'):
            self.storage.gcs['project_id'] = os.environ['ASTROGEA_GCS_PROJECT']
        
        if os.environ.get('ASTROGEA_AZURE_CONTAINER'):
            self.storage.azure['container_name'] = os.environ['ASTROGEA_AZURE_CONTAINER']
        
        if os.environ.get('ASTROGEA_AZURE_ACCOUNT'):
            self.storage.azure['account_name'] = os.environ['ASTROGEA_AZURE_ACCOUNT']
        
        # Dask scheduler URL
        if os.environ.get('DASK_SCHEDULER_URL'):
            self.dask.scheduler_url = os.environ['DASK_SCHEDULER_URL']
        
        # Kubernetes environment specific
        if self.environment == Environment.KUBERNETES:
            # Use Kubernetes service discovery
            if not self.dask.scheduler_url:
                self.dask.scheduler_url = "tcp://dask-scheduler.astrogea.svc.cluster.local:8786"
            
            if not self.dask.dashboard_url:
                self.dask.dashboard_url = "http://dask-scheduler.astrogea.svc.cluster.local:8787"
            
            # Override storage paths for Kubernetes
            self.storage.local['base_path'] = '/data'
            
            # Set processing defaults for Kubernetes
            self.processing.use_dask = True
            self.processing.parallel_workers = int(os.environ.get('ASTROGEA_WORKERS', '3'))
        
        # Docker environment specific
        elif self.environment == Environment.DOCKER:
            # Use Docker network for Dask
            if not self.dask.scheduler_url:
                self.dask.scheduler_url = "tcp://dask-scheduler:8786"
            
            if not self.dask.dashboard_url:
                self.dask.dashboard_url = "http://dask-scheduler:8787"
    
    def _setup_logging(self):
        """Setup logging configuration."""
        import logging.config
        
        # Create logging configuration
        log_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': self.logging.format
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': self.logging.level,
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout'
                }
            },
            'loggers': {
                'astrogea': {
                    'level': self.logging.level,
                    'handlers': ['console'],
                    'propagate': False
                }
            },
            'root': {
                'level': self.logging.level,
                'handlers': ['console']
            }
        }
        
        # Add file handler if specified
        if self.logging.file_path:
            log_config['handlers']['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': self.logging.level,
                'formatter': 'standard',
                'filename': self.logging.file_path,
                'maxBytes': self.logging.max_bytes,
                'backupCount': self.logging.backup_count
            }
            log_config['loggers']['astrogea']['handlers'].append('file')
        
        logging.config.dictConfig(log_config)
        logger.info(f"Logging configured for environment: {self.environment.value}")
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'environment': self.environment.value,
            'storage': {
                'local': self.storage.local,
                's3': self.storage.s3,
                'gcs': self.storage.gcs,
                'azure': self.storage.azure
            },
            'dask': {
                'scheduler_url': self.dask.scheduler_url,
                'dashboard_url': self.dask.dashboard_url,
                'chunks': self.dask.chunks,
                'memory_limit': self.dask.memory_limit,
                'cpu_limit': self.dask.cpu_limit,
                'timeout': self.dask.timeout,
                'retry_attempts': self.dask.retry_attempts
            },
            'processing': {
                'use_dask': self.processing.use_dask,
                'parallel_workers': self.processing.parallel_workers,
                'chunk_size': self.processing.chunk_size,
                'memory_limit': self.processing.memory_limit,
                'cpu_limit': self.processing.cpu_limit,
                'timeout': self.processing.timeout
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file_path': self.logging.file_path,
                'max_bytes': self.logging.max_bytes,
                'backup_count': self.logging.backup_count
            },
            'kubernetes': {
                'namespace': self.kubernetes.namespace,
                'image': self.kubernetes.image,
                'replicas': self.kubernetes.replicas,
                'resources': self.kubernetes.resources,
                'storage_class': self.kubernetes.storage_class,
                'priority_class': self.kubernetes.priority_class,
                'node_selector': self.kubernetes.node_selector,
                'pvc_size': self.kubernetes.pvc_size
            }
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration for StorageManager."""
        return {
            'local': self.storage.local,
            's3': self.storage.s3,
            'gcs': self.storage.gcs,
            'azure': self.storage.azure
        }
    
    def get_dask_client_config(self) -> Dict[str, Any]:
        """Get Dask client configuration."""
        config = {
            'chunks': self.dask.chunks,
            'memory_limit': self.dask.memory_limit,
            'timeout': self.dask.timeout
        }
        
        if self.dask.scheduler_url:
            config['scheduler_url'] = self.dask.scheduler_url
        
        return config
    
    def is_cloud_storage_configured(self) -> bool:
        """Check if any cloud storage is configured."""
        return (
            self.storage.s3.get('bucket_name') or
            self.storage.gcs.get('bucket_name') or
            self.storage.azure.get('container_name')
        )
    
    def get_available_storage_backends(self) -> list:
        """Get list of available storage backends."""
        backends = ['local']
        
        if self.storage.s3.get('bucket_name'):
            backends.append('s3')
        
        if self.storage.gcs.get('bucket_name'):
            backends.append('gcs')
        
        if self.storage.azure.get('container_name'):
            backends.append('azure')
        
        return backends

def create_config(environment: Union[str, Environment] = Environment.LOCAL, 
                 config_path: Optional[str] = None) -> AstrogeaConfig:
    """Create configuration instance."""
    if isinstance(environment, str):
        environment = Environment(environment)
    
    return AstrogeaConfig(config_path=config_path, environment=environment)

def get_default_config() -> AstrogeaConfig:
    """Get default configuration."""
    return create_config(Environment.LOCAL)
