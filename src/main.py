import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
import logging
import os
from typing import Any, List, Tuple, Dict
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
import numpy as np
from pydantic import BaseModel
from src.batched_queue import AsyncBatch
from src.utilities import find_value_by_key
from src.utilities import yamlUtilities, ValidateSignature, ModelLoader
from src.circuit_breaker import AsyncCircuitBreaker
from src.back_pressure import AsyncBackPressure, BackPressureConfig
from src.metrics import InferenceMetrics
from src.rate_limiting import RateLimiter

## --- ## --- ##
load_dotenv()
#######  DOMESTIC and NATIONAL Ccc TRITON cCC INFERENCE SERVER #########
PORT = int(os.getenv("PORT", 8005))
MAX_WORKERS = int(os.getenv("MAX_ML_WORKERS", 4))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 32))
LOG_DIR = os.getenv("LOG_DIRECTORY", "./logs")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "yes")
MODEL_DIR = os.getenv("MODEL_DIRECTORY", "./Models")
QUEUE_N = int(os.getenv("BATCHER_QUEUE_N", 32))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 1024))
PREDICT_TIMEOUT = int(os.getenv("PREDICT_TIMEOUT_SECONDS", 15))
RECOVERY_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", 30))
MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", 250))
REQUESTS_PERIOD = int(os.getenv("REQUESTS_PERIOD_SECONDS", 60))
####################################################################
  
logging.basicConfig(filename=os.path.join(LOG_DIR, "ml_inference_api.log"), 
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filemode='a')

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up the Model Server...")
    models, signatures = await asyncio.to_thread(load_models, MODEL_DIR)
    queues = await asyncio.to_thread(load_queues_for_all_models, signatures)
    app.state.models = models
    app.state.signatures = signatures
    app.state.model_queues = queues
    app.state.back_pressure = AsyncBackPressure(
        config=BackPressureConfig(
            strategy=BackPressureConfig.strategy.SEMAPHORE,
            max_concurrent=MAX_CONCURRENT_REQUESTS,
            queue_size=QUEUE_N
        )
    )
    #ok we need something to clear stuff and collect garbage, maybe a periodic task?
    app.state.metrics = InferenceMetrics()
    #app.state.circuit_breakers = {model_name: AsyncCircuitBreaker(name=f"circuit_breaker_{model_name}") for model_name in models.keys()}
    app.state.rate_limiter = RateLimiter(max_requests=MAX_REQUESTS, period=REQUESTS_PERIOD)
    app.state.model_forward_pass_tasks = {}
    app.state.PROCESS_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    app.state.BATCHED_POOL = ThreadPoolExecutor(max_workers=2*MAX_WORKERS)
    app.state.TASK_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    #app.state.BATCH_TASK_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS) --> no need for this

    ## Create forward pass tasks for each model
    for model_name in models.keys():
        try:
            app.state.model_forward_pass_tasks[model_name] = asyncio.create_task(model_forward_pass_worker(model_name))
            logging.debug(f"Created forward pass task for model {model_name}")
        except Exception as e:
            logging.error(f"Failed to create forward pass task for model {model_name}: {e}")
    await app.state.rate_limiter.start() ## ok start the rate limiter job for garbage collection
    ## We are ready to serve        
    logging.debug(f"Queues {app.state.model_queues}")
    logging.info(f"Model Server started with models: {list(models.keys())}")
    logging.info(f"Signatures loaded for models: {list(signatures.keys())}")
    logging.info(f"Using Process Pool with max workers: {MAX_WORKERS}, Batching Pool with max workers: {2*MAX_WORKERS}, Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS}, Log Directory: {LOG_DIR}, Debug Mode: {DEBUG_MODE}, Model Directory: {MODEL_DIR}")
    try:
        yield
    finally:
        logging.info("Shutting down, waiting for tasks to complete.")
        app.state.PROCESS_POOL.shutdown(wait=True)
        app.state.BATCHED_POOL.shutdown(wait=True)
        #app.state.back_pressure.close()
        await app.state.rate_limiter.stop()
        for task in app.state.model_forward_pass_tasks.values():
            task.cancel()
        logging.info("I am dead now.")

app = FastAPI(title="Model Server", 
              description="A basic API end point for model serving", 
              lifespan=lifespan)

async def model_forward_pass_worker(model_name: str) -> None:
    model = app.state.models[model_name]
    queue = app.state.model_queues[model_name]
    logging.debug(f"Starting forward pass worker for model {model_name} with queue {queue.name}")
    try:    
        while True:
            try:
                batch = await queue.get_batch()
                # record batch_size
                app.state.metrics.record_batch_size(model_name=model_name, batch_size=batch.shape[0], endpoint="predict_batched")
                logging.debug(f"Batch ready for model {model_name} from queue {queue.name}")
                try:
                    logging.debug(f"Model {model_name} forward pass worker found ready queue {queue.name}, of size {batch.shape}")
                    predictions = await asyncio.get_event_loop().run_in_executor(app.state.BATCHED_POOL, model.predict, batch)
                    logging.debug(f"Model {model_name} obtained predictions of shape {predictions.shape} for batch size {batch.shape}")
                    await queue.set_predictions(predictions)
                    logging.debug(f"Model {model_name} completed forward pass for batch size {batch.shape}")
                except Exception as e:
                    logging.error(f"Error during model {model_name} forward pass: {e}")
                    await queue.cancel_processing_batch(e)
            except Exception as e:
                logging.error(f"Error in worker loop for model {model_name}: {e}")
                await asyncio.sleep(1) # Prevent tight loop if get_batch fails repeatedly
    except Exception as e:
        logging.error(f"Model {model_name} forward pass worker encountered an error: {e}")

def load_queues_for_all_models(signatures: Dict[str, Dict[str, Any]])-> Dict[str, AsyncBatch]:
    model_queues:Dict[str, AsyncBatch] = {}
    for model_name, signature in signatures.items():
        model_queues[model_name] = {}
        queue_name = f"input_queue_{model_name}"
        input_shape = find_value_by_key(signature, 'model|input|shape')
        output_shape = find_value_by_key(signature, 'model|output|shape')
        input_dtype = find_value_by_key(signature, 'model|input|dtype')
        output_dtype = find_value_by_key(signature, 'model|output|dtype')
        max_delay = find_value_by_key(signature, 'model|max_batch_delay_ms')
        max_batch_size = find_value_by_key(signature, 'model|preferred_batch_size')[-1]
        logging.debug(f"Model {model_name} queue {queue_name} input shape: {input_shape}, output shape: {output_shape}, input dtype: {input_dtype}, output dtype: {output_dtype}, max delay: {max_delay}, max batch size: {max_batch_size}")
        model_queues[model_name] = AsyncBatch(name=queue_name,
                                            max_size = max_batch_size if max_batch_size is not None else MAX_BATCH_SIZE,
                                            max_delay = max_delay if max_delay is not None else 10,
                                            input_dtype= input_dtype,
                                            output_dtype= output_dtype,
                                            output_shape = tuple(output_shape[1:]),
                                            input_shape = tuple(input_shape[1:]))
    return model_queues

####  Function to load models from directory (synchronous, called in executor)
def load_models(models_dir: str)->Tuple[Dict[str, Any], Dict[str, str]]:
    models = {}
    signatures = {}

    if not os.path.exists(models_dir):
        logging.error(f"Models directory {models_dir} not found")
        raise FileNotFoundError(f"Models directory {models_dir} not found")
    
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.yaml'):
                model_path = os.path.join(root, file)
                try:
                    signature = yamlUtilities.load_yaml(model_path)
                    logging.debug(f"Loaded signature from {model_path}: {signature}")
                    ## Validate signature
                    if ValidateSignature.validate_signature(signature) is False:
                        logging.error(f"Invalid model signature in {model_path}")
                        raise ValueError(f"Invalid model signature in {model_path}")
                    logging.debug(f"Validated signature for model at {model_path}")
                    # Use directory name as key, removing .yaml extension
                    model_key = find_value_by_key(signature, 'model|name')
                    # Sanity check for duplicate model keys
                    if model_key in signatures:
                        logging.warning(f"Duplicate model signature for {model_key} found.")
                        raise ValueError(f"Duplicate model signature for {model_key} found.")
                    ## -- Let's load the model based on backend type                   
                    model = ModelLoader.load_model(signature, root)
                    logging.debug(f"Loaded model for {model_key} with signature {signature}")
                    signatures[model_key] = signature
                    models[model_key] = model
                except Exception as e:
                    logging.error(f"Error loading signature {model_path}: {e}")
                    raise e
    return models, signatures

async def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    return request.client.host if request.client else "unknown"


# Health check endpoint
@app.get("/")
async def read_root()-> Dict[str, str]:
    return {"message": "Model Server is Alive and Kicking!!!!"}

@app.get("/available_models", response_class=HTMLResponse)
async def get_available_models():
    try:
        models = app.state.models
        signatures = app.state.signatures
        
        model_items = []
        for model_name in models.keys():
            signature = signatures[model_name]
            dtype = find_value_by_key(signature, "model|input|dtype")
            input_shape = find_value_by_key(signature, "model|input|shape")
            output_shape = find_value_by_key(signature, "model|output|shape")
            backend = find_value_by_key(signature, "model|backend")
            
            model_items.append(
                f"<li><strong>{model_name}</strong><br>"
                f"Backend: {backend}<br>"
                f"Dtype: {dtype}<br>"
                f"Input Shape: {input_shape}<br>"
                f"Output Shape: {output_shape}</li>"
            )
        
        html_content = (
            "<html><body><h1>Available Models</h1><ul>"
            + "".join(model_items) +
            "</ul></body></html>"
        )
        return HTMLResponse(content=html_content)
    except Exception as e:
        logging.error(f"Error in get_available_models: {e}")
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy",
                                 "loaded_models": list(app.state.models.keys()),
                                 "max_workers": MAX_WORKERS,
                                 "debug_mode": DEBUG_MODE,
                                 "log_directory": LOG_DIR,
                                 "Semaphore_case": app.state.TASK_SEMAPHORE._value,})

@app.get("/metrics")
async def metrics():
    ## OK to let the prometheus scraper get the latest metrics
    metrics = app.state.metrics.generate_latest_metrics()
    return Response(content=metrics, media_type="text/plain; version=1.0.0; charset=utf-8")

class PredictionInput(BaseModel):
    features: Any
    # @field_validator('features')
    # def validate_features(cls, v):
    #     if not isinstance(v, list):
    #         raise ValueError("Features must be a list")
    #     return v
    
@app.post("/predict/{model_name}")
async def predict(model_name: str, input_data: PredictionInput)-> Dict[str, Any]:
    
    if model_name not in app.state.models:
        logging.error(f"Model {model_name} not found for prediction")
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = app.state.models[model_name]
    app.state.metrics.record_request(model_name=model_name, endpoint="predict")
    input_dtype = find_value_by_key(app.state.signatures[model_name], 'model|input|dtype')
    features_array = np.array(input_data.features).astype(input_dtype) ## ok we have some malloc stuff here to be controlled later!!!
    if features_array.ndim == 1:
        features_array = features_array.reshape((1, -1))
    if DEBUG_MODE and features_array.shape[0] > 1:
        logging.debug(f"Received batched prediction request of size {features_array.shape} for model {model_name}")

    async with app.state.TASK_SEMAPHORE: ## Limit concurrent predictions
        loop = asyncio.get_event_loop()
        try:
            start_time = loop.time()  # Record start time
            predictions = await loop.run_in_executor(app.state.PROCESS_POOL, model.predict, features_array)
            end_time = loop.time()  # Record end time
            if DEBUG_MODE: logging.debug(f"Inference time for {model_name}, with input shape {features_array.shape}, output shape {predictions.shape}: {(end_time - start_time) * 1000:.2f} ms")
            app.state.metrics.record_latency(model_name=model_name, latency_ms=(end_time - start_time) * 1000, endpoint="predict")
        except Exception as e:
            logging.error(f"Failed prediction for model {model_name} with input {features_array}: with error {e}")
            app.state.metrics.record_error(model_name=model_name, endpoint="predict")
            raise HTTPException(status_code=500, detail="Prediction failed")
    
    return {"model": model_name,
            "predictions": predictions.tolist()}

class BatchedPredictionInput(BaseModel):
    features: Any

@app.post("/predict_batched/{model_name}")
async def predict_batched(model_name: str, 
                          input_data: BatchedPredictionInput,
                          request: Request)-> Dict[str, Any]:
    client_ip = await get_client_ip(request)
    if model_name not in app.state.models:
        logging.error(f"Model {model_name} not found for batched prediction")
        raise HTTPException(status_code=404, detail="Model not found")
    # Rate limiting check
    is_allowed = await app.state.rate_limiter.is_allowed(client_ip)
    if not is_allowed:
        logging.warning(f"Rate limit exceeded for client {client_ip} on model {model_name}")
        raise HTTPException(status_code=429, detail="Too Many Requests - Rate limit exceeded")
    async with app.state.back_pressure:  ## Limit concurrent batched predictions
        try:
            app.state.metrics.record_request(model_name=model_name, endpoint="predict_batched")
            queue: AsyncBatch = app.state.model_queues[model_name]
            try:
                future = await queue.push(input_data.features)
            except Exception as e:
                logging.error(f"Error while pushing to batcher for model {model_name}: {e}")
                raise HTTPException(status_code=500, detail="Batched prediction failed during push")
            loop = asyncio.get_event_loop()
            loop_time = loop.time()
            try:
                result = await asyncio.wait_for(future, timeout=PREDICT_TIMEOUT)
            except asyncio.TimeoutError:
                logging.error(f"Timeout while waiting for predictions for model {model_name}")
                raise HTTPException(status_code=504, detail="Batched prediction timed out")
            except Exception as e:
                logging.error(f"Error while waiting for predictions for model {model_name}: {e}")
                raise HTTPException(status_code=500, detail="Batched prediction failed during wait")
            end_time = loop.time()  # Record end time
            if DEBUG_MODE: logging.debug(f"Forward pass completed for model {model_name} with result shape {result.shape}, time taken {(end_time - loop_time)*1000:.2f} ms")
            app.state.metrics.record_latency(model_name=model_name, latency_ms=(end_time - loop_time) * 1000, endpoint="predict_batched")
            return {"model": model_name, 
                    "predictions": result.tolist()}
        except Exception as e:
            logging.error(f"Failed batched prediction for model {model_name} with error: {e}", exc_info=True)
            app.state.metrics.record_error(model_name=model_name, endpoint="predict_batched")
            raise HTTPException(status_code=500, detail=f"Batched Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Configure uvicorn with better connection handling
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        workers=1,  
        loop="uvloop", 
        limit_concurrency=MAX_CONCURRENT_REQUESTS,  # Limit concurrent connections
        limit_max_requests=10000,  # Restart worker after N requests to prevent leaks
        timeout_keep_alive=5,  
        timeout_notify=50,
        )
        

