import asyncio 
from typing import Dict
from collections import deque

class RateLimiter:
    def __init__(self, 
                 max_requests: int, period: float):
        self.max_request = max_requests
        self.period = period
        self.request_timestamps: Dict[str, deque] = {}  # Store all request times
        self.lock = asyncio.Lock()
        self.gc_task: None | asyncio.Task = None
        
    async def start(self):
        self.gc_task = asyncio.create_task(self._gc_loop())

    async def _gc_loop(self):
        while True:
            try:
                await asyncio.sleep(30)
                async with self.lock:
                    keys_2_delete = [key for key, timestamps in self.request_timestamps.items() if not timestamps]
                    for key in keys_2_delete:
                        self.request_timestamps.pop(key)
            except asyncio.CancelledError:
                break
    async def stop(self):
        if self.gc_task:
            self.gc_task.cancel()
            try:
                await self.gc_task
            except asyncio.CancelledError:
                pass
    async def is_allowed(self, key: str) -> bool:
        async with self.lock:
            loop = asyncio.get_event_loop()
            current_time = loop.time()
            
            if key not in self.request_timestamps:
                self.request_timestamps[key] = deque()
            
            while self.request_timestamps[key] and self.request_timestamps[key][0] < current_time - self.period:
                self.request_timestamps[key].popleft()

            if len(self.request_timestamps[key]) < self.max_request:
                self.request_timestamps[key].append(current_time)
                return True
            else:
                return False
    

    


    
    