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
2. Run with docker: `docker compose up`

## Config

Check the models/ folder for config files. Ya can tweak batch sizes and stuff there.

## To do inference
OK first check if your model is alive and kicking:
 - curl -f http://localhost:8005/available_models
There are two endpoints, /predict or /predict_bathed, as usual for the batched you will wait a bit (~10ms)


## Testing

Run tests with `pytest`. There's a load_test.py for stress testing.

Thats it, keep it simple! 


