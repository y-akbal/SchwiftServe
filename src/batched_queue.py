## inherit from async queue
from __future__ import annotations
import asyncio
import logging
from typing import Any, Optional, List, Dict, Union, defaultdict, Callable, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
from threading import RLock, Condition
from datetime import datetime
import uuid
from dataclasses import dataclass, field
import os
import random
import enum
DEBUG_MODE = os.getenv("DEBUG_MODEL", "False").lower() in ("true", "1", "t")

class dTypes(enum.Enum):
    float32 = np.float32
    float64 = np.float64
    int32 = np.int32
    int64 = np.int64
    uint8 = np.uint8
    uint16 = np.uint16
    integer = np.int64
    float = np.float64
    string = 'U10'
    str = 'U10'

## batch full exception
class BatchFullException(Exception):
    pass

class AbstractAsyncBatchQueue(ABC):
    
    @abstractmethod
    async def push(self, x: Any) -> asyncio.Future:
        raise NotImplementedError

    @abstractmethod
    async def get_batch(self) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    async def set_predictions(self, predictions: np.ndarray) -> None:
        raise NotImplementedError
    
@dataclass
class AsyncBatch(AbstractAsyncBatchQueue):
    
    name: str | None = None
    max_size: int = field(default=128)
    max_delay: float = field(default=10.0)  ## in milliseconds
    input_dtype: np.dtype | str = field(default=np.float64)
    input_shape: Tuple[int, ...] = field(default=())
    output_dtype: np.dtype | str = field(default=np.float64)
    output_shape: Tuple[int, ...] = field(default=())
   
    def __post_init__(self): 
        if self.name is None:
            self.name = f"AsyncBatch_{str(uuid.uuid4())[:8]}"
        ## Dtype casting
        if isinstance(self.input_dtype, str):
            self.input_dtype = dTypes[self.input_dtype].value
        if isinstance(self.output_dtype, str):
            self.output_dtype = dTypes[self.output_dtype].value
        
        self.lock: asyncio.Lock = asyncio.Lock() # to protect async access to the batch
        self._condition: asyncio.Condition = asyncio.Condition(asyncio.Lock()) # Single condition for both producers and consumers
        ## Internal fields that are subject to lock protection --> 
        ## 
        ## Data storage
        self._data = np.empty((self.max_size, *self.input_shape), dtype=self.input_dtype)
        
        ## Metadata 
        self._first_push_date: datetime | None = None
        self._full: bool = False
        self._ptr: int = 0
        
        ## Futures management
        self._input_futures: List[asyncio.Future] = []
        self._processing_futures: List[asyncio.Future] = []

    def __repr__(self) -> str:
        return f"AsyncBatch(name={self.name}, max_size={self.max_size}, dtype={self.input_dtype}, input_shape={self.input_shape}, output_shape={self.output_shape}, ptr={self._ptr}, full={self._full}, max_delay={self.max_delay}ms)"

    async def push(self, x: np.ndarray | List[Any]) -> asyncio.Future:
        async with self._condition: 
            while self._ptr >= self.max_size: 
                await self._condition.wait()
            
            if self._first_push_date is None:
                self._first_push_date = datetime.now()
            
            if isinstance(x, list):
                x = np.array(x, dtype=self.input_dtype)
            if x.shape != self.input_shape:
                raise ValueError(f"Input shape mismatch: expected {self.input_shape}, got {x.shape}")
            
            self._data[self._ptr] = x.astype(self.input_dtype)
            self._ptr += 1
            
            fut = asyncio.Future()
            self._input_futures.append(fut)
            
            if self._ptr == self.max_size:
                self._full = True
     
            self._condition.notify_all()
            
            return fut
    
    async def get_batch(self) -> np.ndarray:
        async with self._condition:
            while self._ptr == 0:
                await self._condition.wait()

            while not self._full:
                # Check if batch was already retrieved by another getter
                if self._first_push_date is None or self._ptr == 0:
                    break
                    
                time_elapsed_ms = (datetime.now() - self._first_push_date).total_seconds() * 1000
                remaining_time_ms = self.max_delay - time_elapsed_ms
                
                if remaining_time_ms <= 0:
                    break
                
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining_time_ms/1000.0)
                except asyncio.TimeoutError:
                    break
            
            batch = self._data[:self._ptr, ...].copy()
            
            self._processing_futures = self._input_futures[:]
            self._input_futures = []
            
            self._ptr = 0
            self._full = False
            self._first_push_date = None
            
            self._condition.notify_all()
            return batch
        
    async def set_predictions(self, predictions: np.ndarray) -> None:
        B = predictions.shape[0]
        if predictions.ndim == 1:
            predictions = predictions.reshape((B, 1))
        
        if len(self._processing_futures) != B:
            logging.error(f"Mismatch between predictions shape {B} and pending futures {len(self._processing_futures)}")
            ex = RuntimeError(f"Batch size mismatch: expected {len(self._processing_futures)}, got {B}")
            for fut in self._processing_futures:
                if not fut.done():
                    fut.set_exception(ex)
            self._processing_futures = []
            return

        for i, fut in enumerate(self._processing_futures):
            if not fut.done():
                fut.set_result(predictions[i].astype(self.output_dtype))
        
        self._processing_futures = []

    async def cancel_processing_batch(self, exception: Exception) -> None:
        for fut in self._processing_futures:
            if not fut.done():
                fut.set_exception(exception)
        self._processing_futures = []