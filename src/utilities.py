from enum import Enum
import os
import yaml
from typing import Any, Dict, List
import importlib
import joblib
import onnxruntime as ort
import numpy as np

class BackendType(str, Enum):
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    TORCH = "torch"
    JAX = "jax"
    ONNX = "onnx"
    PYTHON = "python"

def find_value_by_key(data: Dict[str, Any], 
                      target_key: str) -> Any:
    """
    Instead of boring data["a"]["b"]["c"], use this function to find the value associated with target_key.
    find_value_by_key(data, "a_b_c") will return the value of data["a"]["b"]["c"].
    if there is no seperation | in target_key, just return recursively data[target_key] under the same logic.
    here | is the separator for nested keys.
    Args:
        data (Dict[str, Any]): The nested dictionary to search.
        target_key (str): The target key, possibly with underscores for nesting.
    Returns:
        Any: The value associated with the target key.
    Raises:
        KeyError: If the target key is not found in the data structure.
    """
    if not "|" in target_key:
        def search_key(d: Dict[str, Any], key: str) -> Any:
            for k, v in d.items():
                if isinstance(v, dict):
                    if result:= search_key(v, key):
                        return result
                elif k == key:
                    return v
            return None
        if result := search_key(data, target_key):
            return result
        raise KeyError(f"Key '{target_key}' not found in the data structure.")
            
    else:
        keys = [k for k in target_key.split('|') if k]
        current_data = data

        for key in keys:
            if isinstance(current_data, dict) and key in current_data:
                current_data = current_data[key]
            else:
                raise KeyError(f"Key '{key}' not found in the data structure.")

        return current_data

class yamlUtilities:
    @staticmethod
    def load_yaml(yaml_path: str) -> Dict[str, Any]:
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)

class ValidateSignature:
    @staticmethod
    def validate_python_signature(model_signature: Any) -> bool:
        ## Checkbackend 
        try:
            class_spec = find_value_by_key(model_signature, "model|model_args|class")
            kwargs = find_value_by_key(model_signature, "model|model_args|kwargs")
        except KeyError as e:
            raise ValueError(f"Missing required key in model_signature: {e}")
        if not isinstance(kwargs, dict):
            raise ValueError(f"Invalid kwargs format, expected dict but got {type(kwargs)}")
        return True
    
    @staticmethod
    def validate_sklearn_signature(model_signature: Any) -> bool:
        try:
            model_file = find_value_by_key(model_signature, "model_file")
        except KeyError as e:
            raise ValueError(f"Missing required key in model_signature: {e}")
        if not isinstance(model_file, str):
            raise ValueError(f"Invalid model_file format, expected str but got {type(model_file)}")
        if not model_file.endswith('.pkl'):
            raise ValueError(f"Invalid model_file extension, expected .pkl but got {model_file}")
        return True

    @staticmethod
    def validate_torch_signature(model_signature: Dict[str, Any]) -> bool:
        try:
            model_file = find_value_by_key(model_signature, "model_file")
        except KeyError as e:
            raise ValueError(f"Missing required key in model_signature: {e}")
        
        if not isinstance(model_file, str):
            raise ValueError(f"Invalid model_file format, expected str but got {type(model_file)}")
        
        if not (model_file.endswith('.pt') or model_file.endswith('.pth')):
            raise ValueError(f"Invalid model_file extension, expected .pt or .pth but got {model_file}")

        # Optional: validate device specification
        try:
            device = find_value_by_key(model_signature, "device")
            if device not in ["cpu", "mps"] and not device.startswith("cuda"):
                raise ValueError(f"Invalid device: {device}. Expected 'cpu', 'mps', or 'cuda' (optionally with device index like 'cuda:0')")
        except KeyError:
            pass  # device is optional, defaults to cpu

        return True

    @staticmethod
    def validate_onnx_signature(model_signature: Dict[str, Any]) -> bool:
        try:
            model_file = find_value_by_key(model_signature, "model_file")
            if not model_file.endswith('.onnx'):
                raise ValueError(f"Invalid model_file extension, expected .onnx but got {model_file}")
        except KeyError as e:
            raise ValueError(f"Missing required key in model_signature: {e}")
        try: 
            device = find_value_by_key(model_signature, "device")
            if device not in ["cpu", "cuda", "tensorrt"]:
                raise ValueError(f"Invalid device: {device}. Expected 'cpu', 'cuda', or 'tensorrt'")
        except KeyError:
            raise ValueError(f"Missing required key in model_signature: device")
        return True
    
    @staticmethod
    def validate_signature(model_signature: Dict[str, Any]) -> bool:
        for put in ["input", "output"]:
            for attr in ["dtype", "shape", "type"]:
                try:
                    _ = find_value_by_key(model_signature, f"model|{put}|{attr}")
                except KeyError as e:
                    raise ValueError(f"Missing required key in model_signature: {e}")
        batch_size = find_value_by_key(model_signature, "preferred_batch_size")
        if not isinstance(batch_size, list) and not all(isinstance(i, int) for i in batch_size):
            raise ValueError(f"Invalid preferred_batch_size format, expected list of int but got {type(batch_size)}")
        if not isinstance(find_value_by_key(model_signature, "max_batch_delay_ms"), int):
            raise ValueError(f"Invalid max_batch_delay_ms format, expected int but got {type(find_value_by_key(model_signature, 'max_batch_delay_ms'))}")
        if not isinstance(find_value_by_key(model_signature, "rush_batch_size"), int):
            raise ValueError(f"Invalid rush_batch_size format, expected int but got {type(find_value_by_key(model_signature, 'rush_batch_size'))}")
        ## backend check
        backend = find_value_by_key(model_signature, "backend")
        ## check over enums
        for backend_type in BackendType:
            if backend == backend_type.value:
                try:
                    return getattr(ValidateSignature, f"validate_{backend_type.value}_signature")(model_signature)
                except AttributeError:
                    raise NotImplementedError(f"Model validation for backend '{backend}' is not implemented yet.")
                except Exception as e:
                    raise e
        raise ValueError(f"Unsupported backend type: {backend}")
        
class ModelValidator:
    @staticmethod
    def validate_python_model(model_instance: Any) -> bool:
        if not hasattr(model_instance, "predict"):
            raise ValueError("Python model must have a 'predict' method.")
        return True
    @staticmethod
    def validate_general_model(model_instance: Any) -> bool:
        if not hasattr(model_instance, "predict"):
            raise ValueError("Model must have a 'predict' method.")
        return True

class ModelLoader:
    @staticmethod
    def load_python_model(model_signature: Dict[str, Any], root: str) -> Any:
        try:
            class_spec = find_value_by_key(model_signature, "model|model_args|class")
            kwargs = find_value_by_key(model_signature, "model|model_args|kwargs") # |kwargs needed to avoid conflict with possible nested keys
        except KeyError as e:
            raise ValueError(f"Missing required key in model_dict: {e}")
        try:
            module_path, class_name = class_spec.split(":", 1)
        except ValueError:
            raise ValueError(f"Invalid class format: '{class_spec}'. Expected 'module_path:ClassName'")
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except Exception as e:
            raise RuntimeError(f"Failed to import class '{class_name}' from module '{module_path}': {e}")
        try:
            model_instance = model_class(**kwargs)
        except TypeError as e:
            raise RuntimeError(f"Invalid kwargs for {class_name}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate {class_name}: {e}")
        ## Let's do some validation
        if ModelValidator.validate_python_model(model_instance):
            return model_instance

    @staticmethod
    def load_sklearn_model(model_signature: Dict[str, Any], root: str) -> Any:
        try:
            model_path = find_value_by_key(model_signature, "model_file")
            return joblib.load(os.path.join(root, model_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {model_path} with error: {e}, root: {root}"
                )
    
    @staticmethod
    def load_onnx_model(model_signature: Dict[str, Any], root: str) -> Any:
        model = find_value_by_key(model_signature, "model_file")
        return ONNXModelPredictor(onnx_model_path=os.path.join(root, model))

    _BACKEND_LOADERS_ = {
        BackendType.PYTHON: load_python_model,
        BackendType.SKLEARN: load_sklearn_model,
        BackendType.ONNX: load_onnx_model,
    }

    @staticmethod
    def load_model(model_signature: Dict[str, Any], root: str) -> Any:
        backend_str = find_value_by_key(model_signature, "backend")
        try:
            backend = BackendType(backend_str)
        except ValueError:
            raise ValueError(f"Unsupported backend type: {backend_str}")
        
        try: 
            loader_func = ModelLoader._BACKEND_LOADERS_[backend]
        except KeyError:
            raise NotImplementedError(f"Model loading for backend '{backend}' is not implemented yet.")
        
        return loader_func(model_signature, root)
        
class ONNXModelPredictor:
    def __init__(self, 
                onnx_model_path: str,
                providers:  List[str] | None = None, ## Later we can add more options here
                session_options: Any | None = None, # 
                warmup_iterations: int | None = None, ## if provided, run warmup
                ) -> None:
        self.session = ort.InferenceSession(onnx_model_path, sess_options=session_options, providers=providers)
        self.onnx_model_path = onnx_model_path
        self.input_names = self.session.get_inputs()[0].name
        if warmup_iterations:
            self._warmup(warmup_iterations)

    def _warmup(self, n_iterations: int = 5) -> None:
        input_shape = self.session.get_inputs()[0].shape
        input_dtype = self.session.get_inputs()[0].type
        dtype_map = {
            'tensor(float)': np.float32,
            'tensor(double)': np.float64,
        }
        np_dtype = dtype_map.get(input_dtype, np.float32)
        dummy_input = np.random.rand(*[dim if isinstance(dim, int) else 1 for dim in input_shape]).astype(np_dtype)
        for _ in range(n_iterations):
            self.predict(dummy_input)

    def predict(self, 
                input_data: np.ndarray) -> Any:
        return self.session.run(None, {self.input_names: input_data})[0]

    @property
    def providers(self) -> List[str]:
        return self.session.get_providers()
    
    def __repr__(self) -> str:
        return f"ONNXModelPredictor(onnx_model_path={self.onnx_model_path}, providers={self.providers})"