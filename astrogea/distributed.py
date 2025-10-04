"""
Distributed computing support for astrogea using Dask on Kubernetes.
"""

import os
import logging
import time
from typing import Optional, Dict, Any, Union, List, Callable
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

class DaskClusterManager:
    """Manages Dask cluster connections and operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.cluster = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Dask client connection."""
        try:
            import dask
            from dask.distributed import Client, LocalCluster
            from dask_kubernetes import KubeCluster
        except ImportError as e:
            logger.error(f"Required Dask packages not installed: {e}")
            logger.error("Install with: pip install dask[complete] dask-kubernetes")
            raise
        
        scheduler_url = self.config.get('scheduler_url')
        
        if scheduler_url:
            # Connect to existing cluster
            logger.info(f"Connecting to Dask cluster at {scheduler_url}")
            self.client = Client(scheduler_url)
        else:
            # Create local cluster
            logger.info("Creating local Dask cluster")
            self.cluster = LocalCluster(
                n_workers=self.config.get('parallel_workers', 4),
                threads_per_worker=self.config.get('cpu_limit', 2),
                memory_limit=self.config.get('memory_limit', '4GB'),
                dashboard_address=':8787'
            )
            self.client = Client(self.cluster)
        
        logger.info(f"Dask client initialized: {self.client}")
        logger.info(f"Dashboard available at: {self.client.dashboard_link}")
    
    def get_client(self):
        """Get Dask client instance."""
        return self.client
    
    def get_dashboard_url(self) -> str:
        """Get dashboard URL."""
        return self.client.dashboard_link if self.client else None
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information."""
        if not self.client:
            return {}
        
        try:
            info = {
                'scheduler': str(self.client.scheduler_info()),
                'workers': len(self.client.scheduler_info()['workers']),
                'dashboard': self.client.dashboard_link,
                'status': 'connected'
            }
            
            # Get worker details
            workers = self.client.scheduler_info()['workers']
            worker_info = []
            for worker_id, worker_data in workers.items():
                worker_info.append({
                    'id': worker_id,
                    'memory': worker_data.get('memory_limit', 'unknown'),
                    'nthreads': worker_data.get('nthreads', 'unknown'),
                    'status': worker_data.get('status', 'unknown')
                })
            
            info['worker_details'] = worker_info
            
        except Exception as e:
            logger.warning(f"Failed to get cluster info: {e}")
            info = {'status': 'error', 'error': str(e)}
        
        return info
    
    def scale_workers(self, n_workers: int):
        """Scale number of workers."""
        if self.cluster:
            self.cluster.scale(n_workers)
            logger.info(f"Scaled cluster to {n_workers} workers")
        else:
            logger.warning("Cannot scale external cluster")
    
    def close(self):
        """Close client and cluster connections."""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()
        logger.info("Dask cluster connections closed")

class DistributedProcessor:
    """Distributed processing manager for astrogea operations."""
    
    def __init__(self, cluster_manager: DaskClusterManager, storage_manager=None):
        self.cluster_manager = cluster_manager
        self.storage_manager = storage_manager
        self.client = cluster_manager.get_client()
    
    def process_file_distributed(self, file_uri: str, output_uri: str, 
                               processing_func: Callable, **kwargs) -> Dict[str, Any]:
        """Process file using distributed computing."""
        logger.info(f"Starting distributed processing of {file_uri}")
        
        start_time = time.time()
        
        try:
            # Load file data
            if self.storage_manager:
                file_data = self.storage_manager.read(file_uri)
            else:
                # Fallback to local file
                with open(file_uri, 'rb') as f:
                    file_data = f.read()
            
            # Convert to Dask array for distributed processing
            import dask.array as da
            import numpy as np
            
            # For now, assume we're processing image data
            # In practice, this would be more sophisticated
            data_array = np.frombuffer(file_data, dtype=np.uint8)
            
            # Create Dask array with appropriate chunks
            chunks = self.cluster_manager.config.get('chunks', {'auto': 'auto'})
            dask_array = da.from_array(data_array, chunks='auto')
            
            # Apply processing function
            result = processing_func(dask_array, **kwargs)
            
            # Compute result
            computed_result = result.compute()
            
            # Save result
            if self.storage_manager:
                self.storage_manager.write(output_uri, computed_result.tobytes())
            else:
                with open(output_uri, 'wb') as f:
                    f.write(computed_result.tobytes())
            
            processing_time = time.time() - start_time
            
            result_info = {
                'input_file': file_uri,
                'output_file': output_uri,
                'processing_time': processing_time,
                'status': 'completed',
                'cluster_info': self.cluster_manager.get_cluster_info()
            }
            
            logger.info(f"Distributed processing completed in {processing_time:.2f}s")
            return result_info
            
        except Exception as e:
            logger.error(f"Distributed processing failed: {e}")
            return {
                'input_file': file_uri,
                'output_file': output_uri,
                'status': 'failed',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def process_batch_distributed(self, file_uris: List[str], output_dir: str,
                                 processing_func: Callable, **kwargs) -> List[Dict[str, Any]]:
        """Process multiple files in parallel."""
        logger.info(f"Starting batch processing of {len(file_uris)} files")
        
        # Create output directory
        if self.storage_manager:
            # For cloud storage, we'll handle paths differently
            pass
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Process files in parallel using Dask
        futures = []
        for i, file_uri in enumerate(file_uris):
            output_uri = f"{output_dir}/result_{i}.nc"
            future = self.client.submit(
                self._process_single_file,
                file_uri, output_uri, processing_func, kwargs
            )
            futures.append(future)
        
        # Collect results
        for future in futures:
            try:
                result = future.result(timeout=3600)  # 1 hour timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                results.append({
                    'status': 'failed',
                    'error': str(e)
                })
        
        logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def _process_single_file(self, file_uri: str, output_uri: str, 
                            processing_func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single file (to be executed on worker)."""
        try:
            # This function runs on Dask workers
            # Load file
            if self.storage_manager:
                file_data = self.storage_manager.read(file_uri)
            else:
                with open(file_uri, 'rb') as f:
                    file_data = f.read()
            
            # Process data
            result = processing_func(file_data, **kwargs)
            
            # Save result
            if self.storage_manager:
                self.storage_manager.write(output_uri, result)
            else:
                with open(output_uri, 'wb') as f:
                    f.write(result)
            
            return {
                'input_file': file_uri,
                'output_file': output_uri,
                'status': 'completed'
            }
            
        except Exception as e:
            return {
                'input_file': file_uri,
                'output_file': output_uri,
                'status': 'failed',
                'error': str(e)
            }

class KubernetesJobManager:
    """Manages Kubernetes jobs for astrogea processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.namespace = config.get('namespace', 'astrogea')
        self.image = config.get('image', 'astrogea:latest')
        
        try:
            from kubernetes import client, config as k8s_config
            from kubernetes.client.rest import ApiException
        except ImportError:
            logger.error("Kubernetes Python client not installed")
            logger.error("Install with: pip install kubernetes")
            raise
        
        # Load Kubernetes configuration
        try:
            k8s_config.load_incluster_config()  # In-cluster
        except:
            try:
                k8s_config.load_kube_config()  # Local
            except Exception as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                raise
        
        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.V1Api()
    
    def create_processing_job(self, job_name: str, file_uris: List[str], 
                             output_uri: str, processing_config: Dict[str, Any]) -> str:
        """Create Kubernetes job for processing."""
        
        # Create job specification
        job_spec = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.namespace
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "astrogea-processor",
                            "image": self.image,
                            "command": ["python", "-m", "astrogea.worker"],
                            "args": [
                                "--input-files", ",".join(file_uris),
                                "--output", output_uri,
                                "--config", "/opt/astrogea/config.yaml"
                            ],
                            "env": [
                                {"name": "ASTROGEA_MODE", "value": "kubernetes"},
                                {"name": "ASTROGEA_JOB_ID", "value": job_name}
                            ],
                            "resources": self.config.get('resources', {
                                "requests": {"memory": "2Gi", "cpu": "1"},
                                "limits": {"memory": "4Gi", "cpu": "2"}
                            }),
                            "volumeMounts": [
                                {
                                    "name": "config-volume",
                                    "mountPath": "/opt/astrogea"
                                },
                                {
                                    "name": "data-volume",
                                    "mountPath": "/data"
                                }
                            ]
                        }],
                        "volumes": [
                            {
                                "name": "config-volume",
                                "configMap": {
                                    "name": "astrogea-config"
                                }
                            },
                            {
                                "name": "data-volume",
                                "persistentVolumeClaim": {
                                    "claimName": "astrogea-data-pvc"
                                }
                            }
                        ],
                        "restartPolicy": "Never"
                    }
                },
                "backoffLimit": 3
            }
        }
        
        try:
            # Create job
            response = self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=job_spec
            )
            
            job_id = response.metadata.name
            logger.info(f"Created Kubernetes job: {job_id}")
            return job_id
            
        except ApiException as e:
            logger.error(f"Failed to create Kubernetes job: {e}")
            raise
    
    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get job status."""
        try:
            job = self.batch_v1.read_namespaced_job(
                name=job_name,
                namespace=self.namespace
            )
            
            status = {
                'name': job_name,
                'status': 'unknown',
                'start_time': None,
                'completion_time': None,
                'pods': []
            }
            
            if job.status.start_time:
                status['start_time'] = job.status.start_time.isoformat()
            
            if job.status.completion_time:
                status['completion_time'] = job.status.completion_time.isoformat()
            
            # Get pod information
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={job_name}"
            )
            
            for pod in pods.items:
                pod_status = {
                    'name': pod.metadata.name,
                    'phase': pod.status.phase,
                    'start_time': pod.status.start_time.isoformat() if pod.status.start_time else None
                }
                status['pods'].append(pod_status)
            
            # Determine overall status
            if job.status.succeeded:
                status['status'] = 'succeeded'
            elif job.status.failed:
                status['status'] = 'failed'
            elif job.status.active:
                status['status'] = 'running'
            else:
                status['status'] = 'pending'
            
            return status
            
        except ApiException as e:
            logger.error(f"Failed to get job status: {e}")
            return {'name': job_name, 'status': 'error', 'error': str(e)}
    
    def delete_job(self, job_name: str):
        """Delete job and associated pods."""
        try:
            self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                propagation_policy="Foreground"
            )
            logger.info(f"Deleted Kubernetes job: {job_name}")
        except ApiException as e:
            logger.error(f"Failed to delete job: {e}")
            raise
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs in namespace."""
        try:
            jobs = self.batch_v1.list_namespaced_job(namespace=self.namespace)
            
            job_list = []
            for job in jobs.items:
                job_info = {
                    'name': job.metadata.name,
                    'status': 'unknown',
                    'creation_time': job.metadata.creation_timestamp.isoformat()
                }
                
                if job.status.succeeded:
                    job_info['status'] = 'succeeded'
                elif job.status.failed:
                    job_info['status'] = 'failed'
                elif job.status.active:
                    job_info['status'] = 'running'
                else:
                    job_info['status'] = 'pending'
                
                job_list.append(job_info)
            
            return job_list
            
        except ApiException as e:
            logger.error(f"Failed to list jobs: {e}")
            return []

def create_distributed_processor(config: Dict[str, Any], 
                                storage_manager=None) -> DistributedProcessor:
    """Create distributed processor instance."""
    cluster_manager = DaskClusterManager(config)
    return DistributedProcessor(cluster_manager, storage_manager)

def create_kubernetes_job_manager(config: Dict[str, Any]) -> KubernetesJobManager:
    """Create Kubernetes job manager instance."""
    return KubernetesJobManager(config)
