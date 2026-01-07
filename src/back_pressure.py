import asyncio
from typing import Any, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

class BackPressureStrategy(Enum):
    SEMAPHORE = "semaphore"  # Uses semaphore to limit concurrent operations
    QUEUE = "queue"          # Uses queue with timeout for flow control
    REJECT = "reject"        # Immediately rejects when capacity is reached      

## TODO: Make this more advanced later with dynamic adjustment based on load
## TODO: Also implement different strategies like token bucket, leaky bucket, etc.

@dataclass
class BackPressureConfig:
    strategy: BackPressureStrategy = BackPressureStrategy.SEMAPHORE
    max_concurrent: int = 10
    queue_size: int = 100
    timeout: float = 1.0  # seconds, for QUEUE strategy

class AsyncBackPressure:
    """Asynchronous backpressure controller to manage load on model serving."""
    def __init__(self, config: BackPressureConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_size)
        self._shutdown = False
    
    async def __aenter__(self):
        if not await self.acquire():
            raise RuntimeError("Failed to acquire backpressure slot")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self.release()
        return False
    
    async def acquire(self) -> bool:
        if self._shutdown:
            return False
            
        if self.config.strategy == BackPressureStrategy.SEMAPHORE:
            await self.semaphore.acquire()
            return True
        elif self.config.strategy == BackPressureStrategy.QUEUE:
            try:
                await asyncio.wait_for(self.queue.put(None), timeout=self.config.timeout)
                return True
            except asyncio.TimeoutError:
                return False
        elif self.config.strategy == BackPressureStrategy.REJECT:
            if self.semaphore._value > 0:
                await self.semaphore.acquire()
                return True
            return False
        return False
    
    async def release(self) -> None:
        if self.config.strategy == BackPressureStrategy.SEMAPHORE:
            self.semaphore.release()
        elif self.config.strategy == BackPressureStrategy.REJECT:
            self.semaphore.release()
        elif self.config.strategy == BackPressureStrategy.QUEUE:
            try:
                await self.queue.get()
                self.queue.task_done()
            except:
                pass  ###  Queue might be empty, ignore exception
    
    async def execute_with_back_pressure(self, func: Callable[[], Awaitable[Any]]) -> Any | None:
        if not await self.acquire():
            return None
        try:
            result = await func()
            return result
        finally:
            await self.release()
    
    async def shutdown(self) -> None:
        self._shutdown = True

        # get rid of  any remaining items in the queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except:
                break
    
    @property
    def current_load(self) -> int:
        if self.config.strategy in (BackPressureStrategy.SEMAPHORE, BackPressureStrategy.REJECT):
            return self.config.max_concurrent - self.semaphore._value
        return 0
    
    @property
    def queue_length(self) -> int:
        return self.queue.qsize()