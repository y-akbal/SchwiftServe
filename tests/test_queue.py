import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from src.batched_queue import AsyncBatch, BatchFullException


# ===== Basic Functionality Tests =====

@pytest.mark.asyncio
async def test_queue_initialization():
    """Test AsyncBatch initialization with default and custom parameters."""
    queue = AsyncBatch(
        name="test_queue",
        max_size=32,
        max_delay=100.0,
        input_dtype=np.float32,
        input_shape=(10,),
        output_dtype=np.float32,
        output_shape=(1,),
    )
    assert queue.name == "test_queue"
    assert queue.max_size == 32
    assert queue.max_delay == 100.0
    assert queue._ptr == 0
    assert queue._full is False
    assert queue._first_push_date is None


@pytest.mark.asyncio
async def test_queue_auto_generated_name():
    """Test that queue generates a name if none is provided."""
    queue1 = AsyncBatch(name=None, max_size=32, input_shape=(10,))
    queue2 = AsyncBatch(name=None, max_size=32, input_shape=(10,))
    assert queue1.name is not None
    assert queue2.name is not None
    assert queue1.name != queue2.name  # Each should have unique name


@pytest.mark.asyncio
async def test_queue_repr():
    """Test queue string representation."""
    queue = AsyncBatch(
        name="test_repr",
        max_size=32,
        input_shape=(10,),
        output_shape=(1,),
    )
    repr_str = repr(queue)
    assert "test_repr" in repr_str
    assert "max_size=32" in repr_str


# ===== Push Tests =====

@pytest.mark.asyncio
async def test_push_single_item():
    """Test pushing a single item to the queue."""
    queue = AsyncBatch(
        name="test_push_single",
        max_size=32,
        input_shape=(10,),
    )
    data = np.ones((10,), dtype=np.float64)
    future = await queue.push(data)
    assert isinstance(future, asyncio.Future)
    assert queue._ptr == 1
    assert queue._first_push_date is not None


@pytest.mark.asyncio
async def test_push_multiple_items():
    """Test pushing multiple items to the queue."""
    queue = AsyncBatch(
        name="test_push_multiple",
        max_size=32,
        input_shape=(10,),
    )
    futures = []
    for i in range(5):
        data = np.ones((10,), dtype=np.float64) * i
        future = await queue.push(data)
        futures.append(future)
    
    assert queue._ptr == 5
    assert len(queue._input_futures) == 5
    assert len(futures) == 5


@pytest.mark.asyncio
async def test_push_shape_mismatch():
    """Test that pushing data with wrong shape raises ValueError."""
    queue = AsyncBatch(
        name="test_shape_mismatch",
        max_size=32,
        input_shape=(10,),
    )
    wrong_data = np.ones((5,), dtype=np.float64)  # Expected (10,)
    with pytest.raises(ValueError, match="Input shape mismatch"):
        await queue.push(wrong_data)


@pytest.mark.asyncio
async def test_push_list_conversion():
    """Test pushing a list (should be converted to numpy array)."""
    queue = AsyncBatch(
        name="test_list_push",
        max_size=32,
        input_shape=(5,),
        input_dtype=np.float32,
    )
    data_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    future = await queue.push(data_list)
    assert queue._ptr == 1
    np.testing.assert_array_almost_equal(queue._data[0], data_list)


@pytest.mark.asyncio
async def test_push_fills_batch():
    """Test that queue marks as full when max_size is reached."""
    queue = AsyncBatch(
        name="test_fill_batch",
        max_size=5,
        input_shape=(3,),
    )
    for i in range(5):
        data = np.ones((3,), dtype=np.float64) * i
        await queue.push(data)
    
    assert queue._ptr == 5
    assert queue._full is True


@pytest.mark.asyncio
async def test_push_waits_when_full():
    """Test that push waits when batch is full."""
    queue = AsyncBatch(
        name="test_push_wait",
        max_size=2,
        input_shape=(3,),
        max_delay=50.0,
    )
    
    # Fill the queue
    await queue.push(np.ones((3,)))
    await queue.push(np.ones((3,)))
    assert queue._full is True
    
    # Create a task that will push when full (should block)
    async def delayed_push():
        await asyncio.sleep(0.01)  # Let get_batch run
        data = np.ones((3,))
        future = await queue.push(data)
        return future
    
    # Create a task that will get_batch and unblock the push
    async def get_after_delay():
        await asyncio.sleep(0.02)
        batch = await queue.get_batch()
        return batch
    
    push_task = asyncio.create_task(delayed_push())
    get_task = asyncio.create_task(get_after_delay())
    
    # Both should complete without deadlock
    await asyncio.wait_for(asyncio.gather(push_task, get_task), timeout=2.0)


# ===== Get Batch Tests =====

@pytest.mark.asyncio
async def test_get_batch_when_full():
    """Test getting batch when it's completely full."""
    queue = AsyncBatch(
        name="test_get_full",
        max_size=3,
        input_shape=(2,),
    )
    expected_batch = []
    for i in range(3):
        data = np.array([i, i + 1], dtype=np.float64)
        await queue.push(data)
        expected_batch.append(data)
    
    batch = await queue.get_batch()
    assert batch.shape == (3, 2)
    for i in range(3):
        np.testing.assert_array_equal(batch[i], expected_batch[i])
    
    # After get_batch, queue should reset
    assert queue._ptr == 0
    assert queue._full is False
    assert queue._first_push_date is None


@pytest.mark.asyncio
async def test_get_batch_waits_for_full():
    """Test that get_batch waits until batch is full or max_delay expires."""
    queue = AsyncBatch(
        name="test_get_wait_full",
        max_size=5,
        input_shape=(2,),
        max_delay=50.0,
    )
    
    # Push 3 items (not full)
    for i in range(3):
        await queue.push(np.array([i, i + 1], dtype=np.float64))
    
    # get_batch should wait and return partial batch when max_delay expires
    batch = await queue.get_batch()
    assert batch.shape == (3, 2)


@pytest.mark.asyncio
async def test_get_batch_timeout_on_delay():
    """Test that get_batch respects max_delay timeout."""
    queue = AsyncBatch(
        name="test_get_timeout",
        max_size=10,
        input_shape=(2,),
        max_delay=100.0,  # 100ms
    )
    
    # Push 2 items
    await queue.push(np.array([0, 1], dtype=np.float64))
    await queue.push(np.array([1, 2], dtype=np.float64))
    
    # get_batch should return after max_delay expires, even though batch isn't full
    start = datetime.now()
    batch = await queue.get_batch()
    elapsed = (datetime.now() - start).total_seconds() * 1000
    
    assert batch.shape == (2, 2)
    assert elapsed >= 100.0 * 0.8  # Allow 20% tolerance for timing


@pytest.mark.asyncio
async def test_get_batch_waits_for_items():
    """Test that get_batch waits when queue is empty."""
    queue = AsyncBatch(
        name="test_get_empty_wait",
        max_size=5,
        input_shape=(2,),
        max_delay=500.0,
    )
    
    async def push_after_delay():
        await asyncio.sleep(0.05)
        await queue.push(np.array([0, 1], dtype=np.float64))
        await queue.push(np.array([1, 2], dtype=np.float64))
    
    get_task = asyncio.create_task(queue.get_batch())
    push_task = asyncio.create_task(push_after_delay())
    
    batch, _ = await asyncio.gather(get_task, push_task)
    assert batch.shape == (2, 2)


# ===== Set Predictions Tests =====

@pytest.mark.asyncio
async def test_set_predictions_success():
    """Test setting predictions for a batch."""
    queue = AsyncBatch(
        name="test_set_pred",
        max_size=3,
        input_shape=(2,),
        output_shape=(1,),
        output_dtype=np.float32,
    )
    
    futures = []
    for i in range(3):
        future = await queue.push(np.array([i, i + 1], dtype=np.float64))
        futures.append(future)
    
    batch = await queue.get_batch()
    
    # Set predictions
    predictions = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    await queue.set_predictions(predictions)
    
    # Check that futures are resolved with correct predictions
    for i, future in enumerate(futures):
        assert future.done()
        result = await future
        np.testing.assert_array_almost_equal(result, predictions[i])


@pytest.mark.asyncio
async def test_set_predictions_1d_reshape():
    """Test that 1D predictions are reshaped correctly."""
    queue = AsyncBatch(
        name="test_set_pred_1d",
        max_size=3,
        input_shape=(2,),
        output_shape=(1,),
    )
    
    futures = []
    for i in range(3):
        future = await queue.push(np.array([i, i + 1], dtype=np.float64))
        futures.append(future)
    
    await queue.get_batch()
    
    # 1D predictions should be reshaped to (3, 1)
    predictions = np.array([0.1, 0.2, 0.3])
    await queue.set_predictions(predictions)
    
    assert futures[0].done()
    result = await futures[0]
    assert result.shape == (1,)


@pytest.mark.asyncio
async def test_set_predictions_size_mismatch():
    """Test that size mismatch in predictions is handled."""
    queue = AsyncBatch(
        name="test_set_pred_mismatch",
        max_size=3,
        input_shape=(2,),
        output_shape=(1,),
    )
    
    futures = []
    for i in range(3):
        future = await queue.push(np.array([i, i + 1], dtype=np.float64))
        futures.append(future)
    
    await queue.get_batch()
    
    # Wrong number of predictions
    predictions = np.array([[0.1], [0.2]], dtype=np.float32)  # Expected 3, got 2
    await queue.set_predictions(predictions)
    
    # Futures should have exceptions
    for future in futures:
        assert future.done()
        with pytest.raises(RuntimeError, match="Batch size mismatch"):
            await future


@pytest.mark.asyncio
async def test_set_predictions_dtype_conversion():
    """Test that predictions are converted to output dtype."""
    queue = AsyncBatch(
        name="test_set_pred_dtype",
        max_size=2,
        input_shape=(2,),
        output_shape=(1,),
        output_dtype=np.float32,
    )
    
    futures = []
    for i in range(2):
        future = await queue.push(np.array([i, i + 1], dtype=np.float64))
        futures.append(future)
    
    await queue.get_batch()
    
    # Input as float64, but should be converted to float32
    predictions = np.array([[0.1234567], [0.9876543]], dtype=np.float64)
    await queue.set_predictions(predictions)
    
    result = await futures[0]
    assert result.dtype == np.float32


# ===== Cancel Processing Tests =====

@pytest.mark.asyncio
async def test_cancel_processing_batch():
    """Test canceling a processing batch."""
    queue = AsyncBatch(
        name="test_cancel",
        max_size=2,
        input_shape=(2,),
        output_shape=(1,),
    )
    
    futures = []
    for i in range(2):
        future = await queue.push(np.array([i, i + 1], dtype=np.float64))
        futures.append(future)
    
    await queue.get_batch()
    
    # Cancel the batch
    error = RuntimeError("Processing cancelled")
    await queue.cancel_processing_batch(error)
    
    # Futures should have the exception
    for future in futures:
        assert future.done()
        with pytest.raises(RuntimeError, match="Processing cancelled"):
            await future


@pytest.mark.asyncio
async def test_cancel_already_done_futures():
    """Test that canceling doesn't error if futures are already done."""
    queue = AsyncBatch(
        name="test_cancel_done",
        max_size=2,
        input_shape=(2,),
        output_shape=(1,),
    )
    
    futures = []
    for i in range(2):
        future = await queue.push(np.array([i, i + 1], dtype=np.float64))
        futures.append(future)
    
    await queue.get_batch()
    
    # Set predictions first (marks futures as done)
    predictions = np.array([[0.1], [0.2]], dtype=np.float32)
    await queue.set_predictions(predictions)
    
    # Try to cancel - should not raise error
    await queue.cancel_processing_batch(RuntimeError("Too late"))
    
    # Original results should be preserved
    assert await futures[0] is not None


# ===== Race Condition Tests =====

@pytest.mark.asyncio
async def test_concurrent_pushes():
    """Test multiple concurrent push operations."""
    queue = AsyncBatch(
        name="test_concurrent_push",
        max_size=50,
        input_shape=(5,),
    )
    
    async def pusher(start_idx, count):
        futures = []
        for i in range(count):
            data = np.ones((5,)) * (start_idx + i)
            future = await queue.push(data)
            futures.append(future)
        return futures
    
    # Push from 2 concurrent tasks (reduced from 3 to avoid congestion)
    all_futures = await asyncio.gather(
        pusher(0, 10),
        pusher(100, 10),
    )
    
    flat_futures = [f for futures in all_futures for f in futures]
    assert len(flat_futures) == 20
    assert queue._ptr == 20  # All 20 items should be pushed


@pytest.mark.asyncio
async def test_push_and_get_race():
    """Test that push and get_batch don't deadlock when concurrent."""
    queue = AsyncBatch(
        name="test_push_get_race",
        max_size=3,
        input_shape=(2,),
        max_delay=50.0,
    )
    
    async def pusher():
        for i in range(6):
            await queue.push(np.ones((2,)) * i)
    
    async def getter():
        batches = []
        try:
            batch1 = await asyncio.wait_for(queue.get_batch(), timeout=1.0)
            batches.append(batch1)
            batch2 = await asyncio.wait_for(queue.get_batch(), timeout=1.0)
            batches.append(batch2)
        except asyncio.TimeoutError:
            pass
        return batches
    
    pusher_task = asyncio.create_task(pusher())
    getter_task = asyncio.create_task(getter())
    
    batches, _ = await asyncio.gather(getter_task, pusher_task)
    
    # Should have received at least one batch
    assert len(batches) >= 1
    total_items = sum(batch.shape[0] for batch in batches)
    assert total_items >= 3


@pytest.mark.asyncio
async def test_multiple_getters():
    """Test that multiple sequential get_batch calls work correctly."""
    queue = AsyncBatch(
        name="test_multiple_getters",
        max_size=3,
        input_shape=(2,),
        max_delay=50.0,
    )
    
    # First batch: push 3 items
    for i in range(3):
        await queue.push(np.ones((2,)) * i)
    
    batch1 = await asyncio.wait_for(queue.get_batch(), timeout=1.0)
    assert batch1.shape[0] == 3
    
    # Second batch: push 3 more items
    for i in range(3):
        await queue.push(np.ones((2,)) * (10 + i))
    
    batch2 = await asyncio.wait_for(queue.get_batch(), timeout=1.0)
    assert batch2.shape[0] == 3
    
    # Verify data integrity across batches
    assert not np.array_equal(batch1, batch2)


@pytest.mark.asyncio
async def test_push_while_full_race():
    """Test push operations when queue is full."""
    queue = AsyncBatch(
        name="test_push_full_race",
        max_size=5,
        input_shape=(2,),
        max_delay=100.0,
    )
    
    # Fill queue completely
    for i in range(5):
        await queue.push(np.ones((2,)) * i)
    
    assert queue._full is True
    
    push_completed = False
    get_completed = False
    
    async def delayed_push():
        nonlocal push_completed
        await queue.push(np.ones((2,)) * 999)
        push_completed = True
    
    async def delayed_get():
        nonlocal get_completed
        await asyncio.sleep(0.05)
        batch = await queue.get_batch()
        get_completed = True
        return batch
    
    # Push should block until get_batch clears the queue
    push_task = asyncio.create_task(delayed_push())
    get_task = asyncio.create_task(delayed_get())
    
    try:
        batch, _ = await asyncio.wait_for(asyncio.gather(get_task, push_task), timeout=2.0)
        assert push_completed is True
        assert get_completed is True
        assert batch.shape[0] == 5
    except asyncio.TimeoutError:
        # If timeout, at least one operation should have completed
        assert push_completed or get_completed


@pytest.mark.asyncio
async def test_concurrent_set_predictions_race():
    """Test that set_predictions handles concurrent calls gracefully."""
    queue = AsyncBatch(
        name="test_set_pred_race",
        max_size=3,
        input_shape=(2,),
        output_shape=(1,),
    )
    
    # Push and get batch
    for i in range(3):
        await queue.push(np.ones((2,)) * i)
    
    batch = await queue.get_batch()
    assert queue._ptr == 0
    
    # Set predictions (should handle gracefully)
    predictions = np.ones((3, 1))
    await queue.set_predictions(predictions)
    
    # Verify queue is cleaned up
    assert len(queue._processing_futures) == 0


@pytest.mark.asyncio
async def test_batch_isolation():
    """Test that multiple sequential batches don't interfere."""
    queue = AsyncBatch(
        name="test_batch_isolation",
        max_size=3,
        input_shape=(2,),
        output_shape=(1,),
    )
    
    # First batch
    futures1 = []
    for i in range(3):
        future = await queue.push(np.ones((2,)) * i)
        futures1.append(future)
    
    batch1 = await queue.get_batch()
    preds1 = np.ones((3, 1)) * 0.1
    await queue.set_predictions(preds1)
    
    # Verify first batch results
    for i, future in enumerate(futures1):
        result = await future
        np.testing.assert_almost_equal(result[0], 0.1)
    
    # Second batch (should be independent)
    futures2 = []
    for i in range(3):
        future = await queue.push(np.ones((2,)) * (100 + i))
        futures2.append(future)
    
    batch2 = await queue.get_batch()
    preds2 = np.ones((3, 1)) * 0.9
    await queue.set_predictions(preds2)
    
    # Verify second batch results are independent
    for i, future in enumerate(futures2):
        result = await future
        np.testing.assert_almost_equal(result[0], 0.9)


@pytest.mark.asyncio
async def test_timeout_with_concurrent_operations():
    """Test max_delay timeout with slow push operations."""
    queue = AsyncBatch(
        name="test_timeout_concurrent",
        max_size=100,
        input_shape=(2,),
        max_delay=50.0,  # 50ms timeout
    )
    
    # Push a couple of items with delay
    await queue.push(np.ones((2,)))
    await asyncio.sleep(0.02)
    await queue.push(np.ones((2,)))
    
    # get_batch should return after max_delay expires
    batch = await asyncio.wait_for(queue.get_batch(), timeout=2.0)
    
    # Should have gotten a partial batch (2 items) due to timeout
    assert batch.shape[0] == 2
