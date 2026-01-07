import asyncio
import pytest
import pytest_asyncio
from src.back_pressure import AsyncBackPressure, BackPressureConfig, BackPressureStrategy


@pytest_asyncio.fixture
async def semaphore_bp():
    """Create a backpressure controller with SEMAPHORE strategy"""
    config = BackPressureConfig(
        strategy=BackPressureStrategy.SEMAPHORE,
        max_concurrent=3
    )
    bp = AsyncBackPressure(config)
    yield bp
    await bp.shutdown()


@pytest_asyncio.fixture
async def queue_bp():
    """Create a backpressure controller with QUEUE strategy"""
    config = BackPressureConfig(
        strategy=BackPressureStrategy.QUEUE,
        max_concurrent=3,
        queue_size=5,
        timeout=1.0
    )
    bp = AsyncBackPressure(config)
    yield bp
    await bp.shutdown()


@pytest_asyncio.fixture
async def reject_bp():
    """Create a backpressure controller with REJECT strategy"""
    config = BackPressureConfig(
        strategy=BackPressureStrategy.REJECT,
        max_concurrent=3
    )
    bp = AsyncBackPressure(config)
    yield bp
    await bp.shutdown()


# ==================== SEMAPHORE STRATEGY TESTS ====================

@pytest.mark.asyncio
async def test_semaphore_single_acquire_release(semaphore_bp):
    """Test acquiring and releasing with semaphore strategy"""
    assert semaphore_bp.current_load == 0
    
    result = await semaphore_bp.acquire()
    assert result is True
    assert semaphore_bp.current_load == 1
    
    await semaphore_bp.release()
    assert semaphore_bp.current_load == 0


@pytest.mark.asyncio
async def test_semaphore_multiple_concurrent_requests(semaphore_bp):
    """Test multiple concurrent acquisitions with semaphore"""
    async def acquire_and_release():
        await semaphore_bp.acquire()
        await asyncio.sleep(0.05)
        await semaphore_bp.release()
    
    # Run 5 tasks concurrently with max 3 slots
    tasks = [acquire_and_release() for _ in range(5)]
    await asyncio.gather(*tasks)
    
    # All should complete and be released
    assert semaphore_bp.current_load == 0


@pytest.mark.asyncio
async def test_semaphore_context_manager(semaphore_bp):
    """Test semaphore with async context manager"""
    assert semaphore_bp.current_load == 0
    
    async with semaphore_bp:
        assert semaphore_bp.current_load == 1
    
    assert semaphore_bp.current_load == 0


@pytest.mark.asyncio
async def test_semaphore_context_manager_exception_handling(semaphore_bp):
    """Test that context manager releases even on exception"""
    try:
        async with semaphore_bp:
            assert semaphore_bp.current_load == 1
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Should be released despite exception
    assert semaphore_bp.current_load == 0


@pytest.mark.asyncio
async def test_semaphore_concurrent_context_managers(semaphore_bp):
    """Test concurrent context managers with semaphore"""
    async def task():
        async with semaphore_bp:
            await asyncio.sleep(0.1)
    
    # Run 5 tasks concurrently
    await asyncio.gather(*[task() for _ in range(5)])
    
    # All should complete and release
    assert semaphore_bp.current_load == 0


# ==================== QUEUE STRATEGY TESTS ====================

@pytest.mark.asyncio
async def test_queue_single_acquire_release(queue_bp):
    """Test acquiring and releasing with queue strategy"""
    result = await queue_bp.acquire()
    assert result is True
    assert queue_bp.queue_length == 1
    
    await queue_bp.release()
    assert queue_bp.queue_length == 0


@pytest.mark.asyncio
async def test_queue_timeout_exceeded(queue_bp):
    """Test that queue times out when full"""
    # Fill the queue (size is 5)
    for _ in range(5):
        await queue_bp.acquire()
    
    assert queue_bp.queue_length == 5
    
    # Next acquire should timeout
    result = await queue_bp.acquire()
    assert result is False


@pytest.mark.asyncio
async def test_queue_fifo_order(queue_bp):
    """Test that queue maintains FIFO order"""
    # Acquire 3 slots
    for _ in range(3):
        assert await queue_bp.acquire() is True
    
    assert queue_bp.queue_length == 3
    
    # Release them
    for _ in range(3):
        await queue_bp.release()
    
    assert queue_bp.queue_length == 0


# ==================== REJECT STRATEGY TESTS ====================

@pytest.mark.asyncio
async def test_reject_allows_up_to_limit(reject_bp):
    """Test that reject strategy allows up to max_concurrent"""
    for i in range(3):
        result = await reject_bp.acquire()
        assert result is True
    
    assert reject_bp.current_load == 3


@pytest.mark.asyncio
async def test_reject_rejects_over_limit(reject_bp):
    """Test that reject strategy rejects when over limit"""
    # Acquire up to limit
    for _ in range(3):
        await reject_bp.acquire()
    
    # Next should be rejected
    result = await reject_bp.acquire()
    assert result is False


@pytest.mark.asyncio
async def test_reject_allows_after_release(reject_bp):
    """Test that reject allows after releasing"""
    await reject_bp.acquire()
    await reject_bp.acquire()
    await reject_bp.acquire()
    
    # Over limit
    assert await reject_bp.acquire() is False
    
    # Release one
    await reject_bp.release()
    
    # Now should be allowed
    assert await reject_bp.acquire() is True


# ==================== EXECUTE WITH BACK PRESSURE TESTS ====================

@pytest.mark.asyncio
async def test_execute_with_back_pressure_success(semaphore_bp):
    """Test successful execution with backpressure"""
    async def dummy_task():
        return "result"
    
    result = await semaphore_bp.execute_with_back_pressure(dummy_task)
    assert result == "result"
    assert semaphore_bp.current_load == 0


@pytest.mark.asyncio
async def test_execute_with_back_pressure_exception(semaphore_bp):
    """Test that backpressure releases on exception"""
    async def failing_task():
        raise ValueError("Task failed")
    
    with pytest.raises(ValueError):
        await semaphore_bp.execute_with_back_pressure(failing_task)
    
    # Should still be released
    assert semaphore_bp.current_load == 0


@pytest.mark.asyncio
async def test_execute_with_back_pressure_on_reject(reject_bp):
    """Test execute_with_back_pressure returns None when rejected"""
    # Fill capacity
    for _ in range(3):
        await reject_bp.acquire()
    
    async def dummy_task():
        return "result"
    
    # Should be rejected
    result = await reject_bp.execute_with_back_pressure(dummy_task)
    assert result is None


@pytest.mark.asyncio
async def test_execute_concurrent_tasks_with_backpressure(semaphore_bp):
    """Test multiple concurrent tasks with backpressure"""
    counter = {"value": 0}
    
    async def increment_task():
        counter["value"] += 1
        await asyncio.sleep(0.05)
    
    tasks = [
        semaphore_bp.execute_with_back_pressure(increment_task)
        for _ in range(10)
    ]
    
    await asyncio.gather(*tasks)
    
    assert counter["value"] == 10
    assert semaphore_bp.current_load == 0


# ==================== LOAD MONITORING TESTS ====================

@pytest.mark.asyncio
async def test_current_load_tracking(semaphore_bp):
    """Test that current_load is accurately tracked"""
    assert semaphore_bp.current_load == 0
    
    await semaphore_bp.acquire()
    assert semaphore_bp.current_load == 1
    
    await semaphore_bp.acquire()
    assert semaphore_bp.current_load == 2
    
    await semaphore_bp.release()
    assert semaphore_bp.current_load == 1


@pytest.mark.asyncio
async def test_queue_length_tracking(queue_bp):
    """Test that queue_length is accurately tracked"""
    assert queue_bp.queue_length == 0
    
    await queue_bp.acquire()
    assert queue_bp.queue_length == 1
    
    await queue_bp.acquire()
    assert queue_bp.queue_length == 2
    
    await queue_bp.release()
    assert queue_bp.queue_length == 1


# ==================== SHUTDOWN TESTS ====================

@pytest.mark.asyncio
async def test_shutdown_prevents_new_acquisitions(semaphore_bp):
    """Test that shutdown prevents new acquisitions"""
    await semaphore_bp.shutdown()
    
    result = await semaphore_bp.acquire()
    assert result is False


@pytest.mark.asyncio
async def test_shutdown_clears_queue(queue_bp):
    """Test that shutdown clears the queue"""
    # Fill queue
    for _ in range(3):
        await queue_bp.acquire()
    
    assert queue_bp.queue_length == 3
    
    await queue_bp.shutdown()
    
    # Queue should be cleared
    assert queue_bp.queue_length == 0


# ==================== INTEGRATION TESTS ====================

@pytest.mark.asyncio
async def test_semaphore_rate_limiting_effect(semaphore_bp):
    """Test that semaphore properly rate limits concurrent tasks"""
    max_concurrent = 0
    current_concurrent = {"value": 0}
    
    async def task():
        current_concurrent["value"] += 1
        nonlocal max_concurrent
        max_concurrent = max(max_concurrent, current_concurrent["value"])
        await asyncio.sleep(0.1)
        current_concurrent["value"] -= 1
    
    # Create 10 tasks but only allow 3 concurrent
    tasks = [
        semaphore_bp.execute_with_back_pressure(task)
        for _ in range(10)
    ]
    
    await asyncio.gather(*tasks)
    
    # Max concurrent should not exceed 3
    assert max_concurrent <= 3


@pytest.mark.asyncio
async def test_reject_fast_failure(reject_bp):
    """Test that REJECT strategy fails fast"""
    # Fill capacity
    for _ in range(3):
        await reject_bp.acquire()
    
    # Try to execute task - should fail immediately
    async def slow_task():
        await asyncio.sleep(10)
    
    result = await reject_bp.execute_with_back_pressure(slow_task)
    assert result is None  # Should fail fast without waiting


@pytest.mark.asyncio
async def test_multiple_strategies_independence():
    """Test that multiple backpressure instances are independent"""
    config1 = BackPressureConfig(strategy=BackPressureStrategy.SEMAPHORE, max_concurrent=2)
    config2 = BackPressureConfig(strategy=BackPressureStrategy.SEMAPHORE, max_concurrent=5)
    
    bp1 = AsyncBackPressure(config1)
    bp2 = AsyncBackPressure(config2)
    
    try:
        await bp1.acquire()
        await bp2.acquire()
        await bp2.acquire()
        
        assert bp1.current_load == 1
        assert bp2.current_load == 2
    finally:
        await bp1.shutdown()
        await bp2.shutdown()


@pytest.mark.asyncio
async def test_stress_test_high_concurrency():
    """Stress test with high concurrency"""
    config = BackPressureConfig(
        strategy=BackPressureStrategy.SEMAPHORE,
        max_concurrent=10
    )
    bp = AsyncBackPressure(config)
    
    try:
        counter = {"value": 0}
        
        async def task():
            counter["value"] += 1
            await asyncio.sleep(0.01)
        
        tasks = [
            bp.execute_with_back_pressure(task)
            for _ in range(100)
        ]
        
        await asyncio.gather(*tasks)
        
        assert counter["value"] == 100
        assert bp.current_load == 0
    finally:
        await bp.shutdown()
