from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry, disable_created_metrics

disable_created_metrics()

class InferenceMetrics:

    def __init__(self):
        self.registry = CollectorRegistry()
        self.counter = Counter(
            'inference_requests_total',
            'Total number of inference requests',
            ['model_name', 'endpoint'],
            registry=self.registry
        )
        self.gauge = Gauge(
            'inference_errors_total',
            'Total number of failed inference requests',
            ['model_name', 'endpoint'],
            registry=self.registry
        )
        self.histogram = Histogram(
            'inference_latency_ms',
            'Inference latency in milliseconds',
            ['model_name', 'endpoint'],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
            registry=self.registry
        )

        self.batch_Size = Histogram(
            'inference_batch_size',
            'Batch size for inference requests',
            ['model_name', 'endpoint'],
            buckets=tuple(range(1, 1025, 16)),
            registry=self.registry
        )
    
    def record_request(self, 
                       model_name: str, 
                       endpoint: str | None = None) -> None:
        self.counter.labels(model_name=model_name, 
                            endpoint=endpoint).inc()
    
    def record_error(self, 
                     model_name: str, 
                     endpoint: str | None = None) -> None:
        self.gauge.labels(model_name=model_name, 
                          endpoint=endpoint).inc()
    
    def record_latency(self, 
                       model_name: str, 
                       latency_ms: float, 
                       endpoint: str | None = None) -> None:
        self.histogram.labels(model_name=model_name, 
                              endpoint=endpoint).observe(latency_ms)
    def record_batch_size(self,
                          model_name: str,
                          batch_size: int,
                          endpoint: str | None = None) -> None:
        self.batch_Size.labels(model_name=model_name,
                               endpoint=endpoint).observe(batch_size)
    def generate_latest_metrics(self) -> bytes:
        return generate_latest(self.registry)