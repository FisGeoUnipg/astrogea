"""
Job orchestration system for astrogea distributed processing.
"""

import os
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class JobSpec:
    """Job specification."""
    job_id: str
    job_type: str
    input_files: List[str]
    output_path: str
    processing_params: Dict[str, Any]
    priority: int = 0
    timeout: int = 3600
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    error_message: Optional[str] = None
    result_files: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.result_files is None:
            self.result_files = []

class JobQueue:
    """Thread-safe job queue."""
    
    def __init__(self):
        self._queue = queue.PriorityQueue()
        self._jobs = {}  # job_id -> JobSpec
        self._lock = threading.Lock()
    
    def add_job(self, job: JobSpec):
        """Add job to queue."""
        with self._lock:
            self._jobs[job.job_id] = job
            # Priority queue uses negative priority for higher priority
            self._queue.put((-job.priority, job.created_at.timestamp(), job.job_id))
        logger.info(f"Added job {job.job_id} to queue")
    
    def get_job(self, timeout: Optional[float] = None) -> Optional[JobSpec]:
        """Get next job from queue."""
        try:
            _, _, job_id = self._queue.get(timeout=timeout)
            with self._lock:
                return self._jobs.get(job_id)
        except queue.Empty:
            return None
    
    def get_job_by_id(self, job_id: str) -> Optional[JobSpec]:
        """Get job by ID."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = status
                
                if status == "running" and job.started_at is None:
                    job.started_at = datetime.utcnow()
                elif status in ["completed", "failed"]:
                    job.completed_at = datetime.utcnow()
                
                # Update other fields
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                
                logger.info(f"Updated job {job_id} status to {status}")
    
    def list_jobs(self, status_filter: Optional[str] = None) -> List[JobSpec]:
        """List jobs with optional status filter."""
        with self._lock:
            jobs = list(self._jobs.values())
            if status_filter:
                jobs = [job for job in jobs if job.status == status_filter]
            return jobs
    
    def remove_job(self, job_id: str):
        """Remove job from queue."""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                logger.info(f"Removed job {job_id} from queue")

class JobOrchestrator:
    """Main job orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.job_queue = JobQueue()
        self.active_jobs = {}  # job_id -> thread
        self.max_concurrent_jobs = config.get('max_concurrent_jobs', 5)
        self.job_timeout = config.get('job_timeout', 3600)
        
        # Initialize storage and distributed processing
        from astrogea.storage import create_storage_manager
        from astrogea.distributed import create_distributed_processor, create_kubernetes_job_manager
        
        self.storage_manager = create_storage_manager(config.get('storage', {}))
        
        # Choose processing backend based on environment
        environment = config.get('environment', 'local')
        if environment == 'kubernetes':
            self.job_manager = create_kubernetes_job_manager(config.get('kubernetes', {}))
            self.distributed_processor = None
        else:
            self.distributed_processor = create_distributed_processor(config.get('dask', {}), self.storage_manager)
            self.job_manager = None
        
        # Start job processing thread
        self._processing_thread = threading.Thread(target=self._process_jobs, daemon=True)
        self._processing_thread.start()
        
        logger.info(f"Job orchestrator initialized for environment: {environment}")
        logger.info(f"Max concurrent jobs: {self.max_concurrent_jobs}")
    
    def submit_job(self, job_type: str, input_files: List[str], output_path: str,
                  processing_params: Dict[str, Any] = None, priority: int = 0) -> str:
        """Submit a new job."""
        job_id = str(uuid.uuid4())
        
        job_spec = JobSpec(
            job_id=job_id,
            job_type=job_type,
            input_files=input_files,
            output_path=output_path,
            processing_params=processing_params or {},
            priority=priority,
            timeout=self.job_timeout
        )
        
        self.job_queue.add_job(job_spec)
        logger.info(f"Submitted job {job_id} of type {job_type}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        job = self.job_queue.get_job_by_id(job_id)
        if job:
            return asdict(job)
        return None
    
    def list_jobs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List jobs."""
        jobs = self.job_queue.list_jobs(status_filter)
        return [asdict(job) for job in jobs]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.job_queue.get_job_by_id(job_id)
        if not job:
            return False
        
        if job.status == "running":
            # Try to cancel running job
            if self.job_manager:
                try:
                    self.job_manager.delete_job(job_id)
                except Exception as e:
                    logger.error(f"Failed to cancel Kubernetes job {job_id}: {e}")
            
            # Remove from active jobs
            if job_id in self.active_jobs:
                thread = self.active_jobs[job_id]
                # Note: Python threads can't be forcefully stopped
                # The job will need to check for cancellation status
                del self.active_jobs[job_id]
        
        self.job_queue.update_job_status(job_id, "cancelled")
        logger.info(f"Cancelled job {job_id}")
        return True
    
    def _process_jobs(self):
        """Main job processing loop."""
        while True:
            try:
                # Check for completed jobs
                self._check_completed_jobs()
                
                # Start new jobs if we have capacity
                if len(self.active_jobs) < self.max_concurrent_jobs:
                    job = self.job_queue.get_job(timeout=1.0)
                    if job:
                        self._start_job(job)
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in job processing loop: {e}")
                time.sleep(5)
    
    def _start_job(self, job: JobSpec):
        """Start processing a job."""
        if job.status != "pending":
            return
        
        # Check if we have capacity
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            return
        
        # Update job status
        self.job_queue.update_job_status(job.job_id, "running")
        
        # Start job processing thread
        thread = threading.Thread(
            target=self._execute_job,
            args=(job,),
            name=f"job-{job.job_id}"
        )
        thread.start()
        
        self.active_jobs[job.job_id] = thread
        logger.info(f"Started job {job.job_id}")
    
    def _execute_job(self, job: JobSpec):
        """Execute a job."""
        try:
            logger.info(f"Executing job {job.job_id}")
            
            # Choose execution method based on environment
            environment = self.config.get('environment', 'local')
            
            if environment == 'kubernetes':
                result = self._execute_kubernetes_job(job)
            else:
                result = self._execute_local_job(job)
            
            # Update job status
            if result['status'] == 'success':
                self.job_queue.update_job_status(
                    job.job_id, "completed",
                    result_files=result.get('result_files', [])
                )
            else:
                self.job_queue.update_job_status(
                    job.job_id, "failed",
                    error_message=result.get('error', 'Unknown error')
                )
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed with exception: {e}")
            self.job_queue.update_job_status(
                job.job_id, "failed",
                error_message=str(e)
            )
        finally:
            # Remove from active jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    def _execute_kubernetes_job(self, job: JobSpec) -> Dict[str, Any]:
        """Execute job on Kubernetes."""
        try:
            # Create Kubernetes job
            k8s_job_id = self.job_manager.create_processing_job(
                job_name=job.job_id,
                file_uris=job.input_files,
                output_uri=job.output_path,
                processing_config={
                    'processing_type': job.job_type,
                    'processing_params': job.processing_params
                }
            )
            
            # Wait for job completion
            start_time = time.time()
            while time.time() - start_time < job.timeout:
                status = self.job_manager.get_job_status(k8s_job_id)
                
                if status['status'] == 'succeeded':
                    return {
                        'status': 'success',
                        'result_files': [f"{job.output_path}/result_{i}.nc" 
                                       for i in range(len(job.input_files))]
                    }
                elif status['status'] == 'failed':
                    return {
                        'status': 'error',
                        'error': f"Kubernetes job failed: {status.get('error', 'Unknown error')}"
                    }
                
                time.sleep(10)  # Check every 10 seconds
            
            # Timeout
            self.job_manager.delete_job(k8s_job_id)
            return {
                'status': 'error',
                'error': f"Job timeout after {job.timeout} seconds"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _execute_local_job(self, job: JobSpec) -> Dict[str, Any]:
        """Execute job locally using distributed processor."""
        try:
            if self.distributed_processor:
                # Use distributed processing
                result = self.distributed_processor.process_batch_distributed(
                    file_uris=job.input_files,
                    output_dir=job.output_path,
                    processing_func=self._get_processing_function(job.job_type),
                    **job.processing_params
                )
                
                return {
                    'status': 'success',
                    'result_files': [r.get('output_file') for r in result if r.get('status') == 'completed']
                }
            else:
                # Fallback to local processing
                return self._execute_local_fallback(job)
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _execute_local_fallback(self, job: JobSpec) -> Dict[str, Any]:
        """Execute job using local fallback processing."""
        try:
            from astrogea.worker import AstrogeaWorker
            
            # Create temporary worker
            worker = AstrogeaWorker()
            
            # Process files
            result = worker.process_files(
                input_files=job.input_files,
                output_path=job.output_path,
                processing_type=job.job_type,
                **job.processing_params
            )
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _get_processing_function(self, job_type: str):
        """Get processing function for job type."""
        if job_type == "crism":
            from astrogea.core import process_crism_file
            return process_crism_file
        elif job_type == "continuum":
            from astrogea.core import continuum_removal
            return continuum_removal
        elif job_type == "mafic":
            from astrogea.core import band_parameters_mafic
            return band_parameters_mafic
        else:
            raise ValueError(f"Unknown job type: {job_type}")
    
    def _check_completed_jobs(self):
        """Check for completed jobs and clean up."""
        # This would be implemented to check job statuses
        # and clean up completed jobs
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        jobs = self.job_queue.list_jobs()
        
        stats = {
            'total_jobs': len(jobs),
            'active_jobs': len(self.active_jobs),
            'pending_jobs': len([j for j in jobs if j.status == 'pending']),
            'completed_jobs': len([j for j in jobs if j.status == 'completed']),
            'failed_jobs': len([j for j in jobs if j.status == 'failed']),
            'max_concurrent_jobs': self.max_concurrent_jobs
        }
        
        return stats

class JobAPI:
    """REST API for job management."""
    
    def __init__(self, orchestrator: JobOrchestrator):
        self.orchestrator = orchestrator
    
    def submit_job(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a new job via API."""
        try:
            job_id = self.orchestrator.submit_job(
                job_type=request_data['job_type'],
                input_files=request_data['input_files'],
                output_path=request_data['output_path'],
                processing_params=request_data.get('processing_params', {}),
                priority=request_data.get('priority', 0)
            )
            
            return {
                'job_id': job_id,
                'status': 'submitted'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status via API."""
        status = self.orchestrator.get_job_status(job_id)
        if status:
            return status
        else:
            return {'error': 'Job not found', 'status': 'error'}
    
    def list_jobs(self, status_filter: Optional[str] = None) -> Dict[str, Any]:
        """List jobs via API."""
        jobs = self.orchestrator.list_jobs(status_filter)
        return {
            'jobs': jobs,
            'count': len(jobs)
        }
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel job via API."""
        success = self.orchestrator.cancel_job(job_id)
        return {
            'cancelled': success,
            'status': 'success' if success else 'error'
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics via API."""
        return self.orchestrator.get_statistics()

def create_job_orchestrator(config: Dict[str, Any]) -> JobOrchestrator:
    """Create job orchestrator instance."""
    return JobOrchestrator(config)

def create_job_api(orchestrator: JobOrchestrator) -> JobAPI:
    """Create job API instance."""
    return JobAPI(orchestrator)
