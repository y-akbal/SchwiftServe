## SchwiftServe

Alright we do inference but do it schwifty. 

A simple model serving for lite ML models. Batches requests, handles back pressure, circuit breaker stuff (not active). Keeps things from blowing up when too many requests come in.

## Supported Backends
- Anything pickled with joblib (including many sklearn models)
- Pure python mode (only for those that are not picklable, or some numba related stuff)
- ONNX runtime, this is the place where the whole stack shines schwiftly!
- Torch mode (only cpu based models) --> jit compile and run!

## Quick Start

1. Clone this repo
2. Put your model into models/model_name/{model_file + model_config.yaml}
2. Run with docker: `docker compose up --build`


## Config

Check the models/ folder for config files. You can tweak batch sizes and stuff there.

## To do inference

### Check Available Models
First check if your model is alive and kicking:
```bash
curl -f http://localhost:8005/available_models
```

### Inference Endpoints

There are two endpoints: `/predict` (immediate) and `/predict_batched` (batched, ~10ms delay for better throughput)

#### 1. `/predict/{model_name}` - Immediate Prediction (May be already batched)

**Request Format:**
```bash
curl -X POST http://localhost:8005/predict/{model_name} \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0]]}'
```

**Request Body:**
```json
{
  "features": [[1.0, 2.0, 3.0]]
}
```

**Response:**
```json
{
  "model": "model_name",
  "predictions": [[0.95, 0.05]]
}
```

**Example (sklearn model):**
```bash
curl -X POST http://localhost:8005/predict/my_model \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2], [7.2, 3.2, 6.0, 1.8]]}'
```

#### 2. `/predict_batched/{model_name}` - Batched Prediction (Better for throughput)

Same input format as `/predict`, but requests are batched together for better efficiency. You may wait a bit (<= 10ms) while waiting for other requests to batch together. For this endpoint no requests that are already batched are allowed! 

**Request Format:**
```bash
curl -X POST http://localhost:8005/predict_batched/{model_name} \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0]]}'
```

**Response:**
```json
{
  "model": "model_name",
  "predictions": [[0.95, 0.05]]
}
```

### Health Check & Metrics

Check server health:
```bash
curl http://localhost:8005/health
```

View Prometheus metrics:
```bash
curl http://localhost:8005/metrics
```

## Checking latency related stuff:
 - Either in DEBUG_MODE=True in .env watchout the logs in /logs
 - checkout with /metrics and use prometheus and grafana

## Testing

Run tests with `pytest`. There's a load_test.py for stress testing.

## TODO

- During the rush hours implement circuit breaker properly, and activate double queueing 
- Implement auth if needed


Thats it, keep it simple!


