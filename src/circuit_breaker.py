"""
In model serving, a circuit breaker is a reliability pattern used to protect your system from cascading failures when a model or a dependency becomes slow, overloaded, or unavailable.
Think of it like an electrical circuit breaker: when something goes wrong, it trips to prevent further damage.
"""
import asyncio
from enum import Enum
from datetime import datetime

class CircuitBreakerState(Enum):
    CLOSED = "closed"      
    OPEN = "open"          
    HALF_OPEN = "half_open" 

class AsyncCircuitBreaker:  
    def __init__(self, 
                 failure_threshold: int = 5, 
                 name: str | None = "default_circuit_breaker",
                 recovery_timeout: int = 30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = asyncio.Lock()
    
    async def record_failure(self):
        current_time = datetime.now()
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = current_time
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
    
    async def record_success(self):
        async with self.lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
    
    async def can_attempt_request(self) -> bool:
        current_time = datetime.now()
        async with self.lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if self.last_failure_time and (current_time - self.last_failure_time).total_seconds() >= self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    return True
                else:
                    return False
            elif self.state == CircuitBreakerState.HALF_OPEN:
                return True
            return False
    
    @property
    def state_info(self):
        return self.state.value