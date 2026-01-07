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

    async def is_allowed(self, key: str) -> bool:
        ## OK here is the stuff
        """
        if key is not in the timestamps dict, create a new deque for it.
        if the deque has timestamps older than the period, remove them.
        if the length of the deque is less than max_requests, append the current timestamp and return True.
        else return False.
        """

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
    

    


    
    