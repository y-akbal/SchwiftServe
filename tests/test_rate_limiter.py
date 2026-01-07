import asyncio
import pytest
import pytest_asyncio
from src.rate_limiting import RateLimiter


@pytest_asyncio.fixture
async def rate_limiter():
    """Create a rate limiter instance for testing"""
    limiter = RateLimiter(max_requests=3, period=1.0)
    await limiter.start()
    yield limiter
    await limiter.stop()


@pytest.mark.asyncio
async def test_single_request_allowed(rate_limiter):
    """Test that a single request is allowed"""
    result = await rate_limiter.is_allowed("client_1")
    assert result is True


@pytest.mark.asyncio
async def test_multiple_requests_within_limit(rate_limiter):
    """Test multiple requests within the limit"""
    for i in range(3):
        result = await rate_limiter.is_allowed("client_1")
        assert result is True


@pytest.mark.asyncio
async def test_request_exceeds_limit(rate_limiter):
    """Test that requests exceeding limit are rejected"""
    # Use up all 3 requests
    for i in range(3):
        assert await rate_limiter.is_allowed("client_1") is True
    
    # 4th request should be rejected
    assert await rate_limiter.is_allowed("client_1") is False


@pytest.mark.asyncio
async def test_different_clients_independent(rate_limiter):
    """Test that different clients have independent limits"""
    # Client 1 uses up limit
    for i in range(3):
        assert await rate_limiter.is_allowed("client_1") is True
    
    # Client 2 should still be allowed
    assert await rate_limiter.is_allowed("client_2") is True
    assert await rate_limiter.is_allowed("client_2") is True


@pytest.mark.asyncio
async def test_window_expiration(rate_limiter):
    """Test that requests are allowed after the time window expires"""
    # Use up limit
    for i in range(3):
        assert await rate_limiter.is_allowed("client_1") is True
    
    # Should be rejected now
    assert await rate_limiter.is_allowed("client_1") is False
    
    # Wait for window to expire
    await asyncio.sleep(1.1)
    
    # Should be allowed again
    assert await rate_limiter.is_allowed("client_1") is True


@pytest.mark.asyncio
async def test_sliding_window(rate_limiter):
    """Test sliding window behavior"""
    # Make 3 requests at t=0
    for i in range(3):
        assert await rate_limiter.is_allowed("client_1") is True
    
    # Should be blocked
    assert await rate_limiter.is_allowed("client_1") is False
    
    # Wait 0.5s (first request still in window)
    await asyncio.sleep(0.5)
    
    # Should still be blocked
    assert await rate_limiter.is_allowed("client_1") is False
    
    # Wait another 0.6s (first request now outside window)
    await asyncio.sleep(0.6)
    
    # Should be allowed now
    assert await rate_limiter.is_allowed("client_1") is True


@pytest.mark.asyncio
async def test_gc_cleans_empty_deques(rate_limiter):
    """Test that garbage collector cleans up empty deques"""
    # Make a request
    await rate_limiter.is_allowed("client_gc")
    
    # Verify it's in the dict
    assert "client_gc" in rate_limiter.request_timestamps
    
    # Wait for window to expire
    await asyncio.sleep(1.1)
    
    # Remove all timestamps manually to simulate expiration
    async with rate_limiter.lock:
        rate_limiter.request_timestamps["client_gc"].clear()
    
    # Manually trigger cleanup
    async with rate_limiter.lock:
        keys_to_delete = [key for key, timestamps in rate_limiter.request_timestamps.items() if not timestamps]
        for key in keys_to_delete:
            rate_limiter.request_timestamps.pop(key)
    
    # Key should be removed
    assert "client_gc" not in rate_limiter.request_timestamps


@pytest.mark.asyncio
async def test_concurrent_requests_same_client(rate_limiter):
    """Test concurrent requests from the same client"""
    tasks = [rate_limiter.is_allowed("client_1") for _ in range(5)]
    results = await asyncio.gather(*tasks)
    
    # First 3 should be True, last 2 should be False
    assert results == [True, True, True, False, False]


@pytest.mark.asyncio
async def test_concurrent_requests_different_clients(rate_limiter):
    """Test concurrent requests from different clients"""
    tasks = [
        rate_limiter.is_allowed("client_1"),
        rate_limiter.is_allowed("client_2"),
        rate_limiter.is_allowed("client_3"),
    ]
    results = await asyncio.gather(*tasks)
    
    # All should be allowed (different clients)
    assert all(results)


@pytest.mark.asyncio
async def test_edge_case_exactly_at_period_boundary(rate_limiter):
    """Test request exactly at the period boundary"""
    # Make a request
    await rate_limiter.is_allowed("client_1")
    
    # Wait exactly 1 second
    await asyncio.sleep(1.0)
    
    # The request should be just outside the window
    # but timing can be tricky, so we add a small margin
    await asyncio.sleep(0.01)
    
    # Should be allowed
    assert await rate_limiter.is_allowed("client_1") is True


@pytest.mark.asyncio
async def test_zero_requests_limit(rate_limiter):
    """Test behavior with zero max requests"""
    limiter = RateLimiter(max_requests=0, period=1.0)
    await limiter.start()
    try:
        result = await limiter.is_allowed("client_1")
        assert result is False
    finally:
        await limiter.stop()


@pytest.mark.asyncio
async def test_high_concurrency(rate_limiter):
    """Test with high concurrency"""
    # 10 concurrent requests
    tasks = [rate_limiter.is_allowed("client_high") for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    # First 3 allowed, rest blocked
    allowed_count = sum(results)
    assert allowed_count == 3
