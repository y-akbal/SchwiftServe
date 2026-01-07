import pytest
import os
import tempfile
import yaml
from unittest.mock import patch, MagicMock
from src.utilities import (
    find_value_by_key,
    yamlUtilities,
    ValidateSignature,
    ModelValidator,
    ModelLoader,
    ONNXModelPredictor,
    BackendType
)


# Tests for find_value_by_key
def test_find_value_by_key_with_separator():
    data = {'a': {'b': {'c': 42}}}
    assert find_value_by_key(data, 'a|b|c') == 42


def test_find_value_by_key_separator_key_not_found():
    data = {'a': {'b': {'c': 42}}}
    with pytest.raises(KeyError):
        find_value_by_key(data, 'a|b|d')


def test_find_value_by_key_recursive_search_top_level():
    data = {'x': 1}
    assert find_value_by_key(data, 'x') == 1


def test_find_value_by_key_recursive_search_nested():
    data = {'a': {'x': 2}}
    assert find_value_by_key(data, 'x') == 2


def test_find_value_by_key_recursive_not_found():
    data = {'a': {'b': 1}}
    with pytest.raises(KeyError):
        find_value_by_key(data, 'z')


def test_find_value_by_key_empty_dict():
    data = {}
    with pytest.raises(KeyError):
        find_value_by_key(data, 'x')


def test_find_value_by_key_deep_separator():
    data = {'a': {'b': {'c': {'d': 100}}}}
    assert find_value_by_key(data, 'a|b|c|d') == 100


def test_find_value_by_key_separator_intermediate_not_dict():
    data = {'a': {'b': 42}}
    with pytest.raises(KeyError):
        find_value_by_key(data, 'a|b|c')


def test_find_value_by_key_empty_key():
    data = {'': 42}
    assert find_value_by_key(data, '') == 42


def test_find_value_by_key_multiple_same_keys():
    data = {'x': 1, 'a': {'x': 2}}
    # Recursive search returns first found
    assert find_value_by_key(data, 'x') == 1


def test_find_value_by_key_non_dict_in_nested():
    data = {'a': {'b': [1, 2, 3]}}
    with pytest.raises(KeyError):
        find_value_by_key(data, 'c')


# Tests for yamlUtilities
def test_yaml_utilities_load_yaml():
    data = {'key': 'value', 'num': 42}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        loaded = yamlUtilities.load_yaml(f.name)
    os.unlink(f.name)
    assert loaded == data


def test_yaml_utilities_load_yaml_file_not_found():
    with pytest.raises(FileNotFoundError):
        yamlUtilities.load_yaml('nonexistent.yaml')


# Tests for ValidateSignature
def test_validate_python_signature_valid():
    sig = {
        'model': {
            'model_args': {
                'class': 'module:Class',
                'kwargs': {'param': 'value'}
            }
        }
    }
    assert ValidateSignature.validate_python_signature(sig) is True


def test_validate_python_signature_missing_class():
    sig = {'model': {'model_args': {'kwargs': {}}}}
    with pytest.raises(ValueError, match="Missing required key"):
        ValidateSignature.validate_python_signature(sig)


def test_validate_python_signature_invalid_kwargs():
    sig = {
        'model': {
            'model_args': {
                'class': 'module:Class',
                'kwargs': 'not_dict'
            }
        }
    }
    with pytest.raises(ValueError, match="Invalid kwargs format"):
        ValidateSignature.validate_python_signature(sig)


def test_validate_sklearn_signature_valid():
    sig = {'model_file': 'model.pkl'}
    assert ValidateSignature.validate_sklearn_signature(sig) is True


def test_validate_sklearn_signature_missing_file():
    sig = {}
    with pytest.raises(ValueError, match="Missing required key"):
        ValidateSignature.validate_sklearn_signature(sig)


def test_validate_sklearn_signature_invalid_extension():
    sig = {'model_file': 'model.txt'}
    with pytest.raises(ValueError, match="Invalid model_file extension"):
        ValidateSignature.validate_sklearn_signature(sig)


def test_validate_torch_signature_valid():
    sig = {'model_file': 'model.pt'}
    assert ValidateSignature.validate_torch_signature(sig) is True


def test_validate_torch_signature_missing_file():
    sig = {}
    with pytest.raises(ValueError, match="Missing required key"):
        ValidateSignature.validate_torch_signature(sig)


def test_validate_torch_signature_invalid_extension():
    sig = {'model_file': 'model.txt'}
    with pytest.raises(ValueError, match="Invalid model_file extension"):
        ValidateSignature.validate_torch_signature(sig)


def test_validate_torch_signature_invalid_device():
    sig = {'model_file': 'model.pt', 'device': 'invalid'}
    with pytest.raises(ValueError, match="Invalid device"):
        ValidateSignature.validate_torch_signature(sig)


def test_validate_onnx_signature_valid():
    sig = {'model_file': 'model.onnx', 'device': 'cpu'}
    assert ValidateSignature.validate_onnx_signature(sig) is True


def test_validate_onnx_signature_missing_file():
    sig = {'device': 'cpu'}
    with pytest.raises(ValueError, match="Missing required key"):
        ValidateSignature.validate_onnx_signature(sig)


def test_validate_onnx_signature_invalid_extension():
    sig = {'model_file': 'model.txt', 'device': 'cpu'}
    with pytest.raises(ValueError, match="Invalid model_file extension"):
        ValidateSignature.validate_onnx_signature(sig)


def test_validate_onnx_signature_missing_device():
    sig = {'model_file': 'model.onnx'}
    with pytest.raises(ValueError, match="Missing required key"):
        ValidateSignature.validate_onnx_signature(sig)


def test_validate_onnx_signature_invalid_device():
    sig = {'model_file': 'model.onnx', 'device': 'invalid'}
    with pytest.raises(ValueError, match="Invalid device"):
        ValidateSignature.validate_onnx_signature(sig)


def test_validate_signature_valid_python():
    sig = {
        'model': {
            'input': {'dtype': 'float32', 'shape': [10], 'type': 'tensor'},
            'output': {'dtype': 'float32', 'shape': [1], 'type': 'tensor'},
            'model_args': {
                'class': 'module:Class',
                'kwargs': {}
            }
        },
        'preferred_batch_size': [1, 2, 4],
        'max_batch_delay_ms': 100,
        'rush_batch_size': 1,
        'backend': 'python'
    }
    assert ValidateSignature.validate_signature(sig) is True


def test_validate_signature_missing_input_attr():
    sig = {
        'model': {
            'input': {'dtype': 'float32', 'shape': [10]},
            'output': {'dtype': 'float32', 'shape': [1], 'type': 'tensor'}
        },
        'preferred_batch_size': [1],
        'max_batch_delay_ms': 100,
        'rush_batch_size': 1,
        'backend': 'python'
    }
    with pytest.raises(ValueError, match="Missing required key"):
        ValidateSignature.validate_signature(sig)


def test_validate_signature_invalid_batch_size():
    sig = {
        'model': {
            'input': {'dtype': 'float32', 'shape': [10], 'type': 'tensor'},
            'output': {'dtype': 'float32', 'shape': [1], 'type': 'tensor'}
        },
        'preferred_batch_size': 'invalid',
        'max_batch_delay_ms': 100,
        'rush_batch_size': 1,
        'backend': 'python'
    }
    with pytest.raises(ValueError, match="Invalid preferred_batch_size format"):
        ValidateSignature.validate_signature(sig)


def test_validate_signature_unsupported_backend():
    sig = {
        'model': {
            'input': {'dtype': 'float32', 'shape': [10], 'type': 'tensor'},
            'output': {'dtype': 'float32', 'shape': [1], 'type': 'tensor'}
        },
        'preferred_batch_size': [1],
        'max_batch_delay_ms': 100,
        'rush_batch_size': 1,
        'backend': 'unsupported'
    }
    with pytest.raises(ValueError, match="Unsupported backend type"):
        ValidateSignature.validate_signature(sig)


# Tests for ModelValidator
def test_validate_python_model_valid():
    class MockModel:
        def predict(self, x):
            return x
    assert ModelValidator.validate_python_model(MockModel()) is True


def test_validate_python_model_missing_predict():
    class MockModel:
        pass
    with pytest.raises(ValueError, match="must have a 'predict' method"):
        ModelValidator.validate_python_model(MockModel())


def test_validate_general_model_valid():
    class MockModel:
        def predict(self, x):
            return x
    assert ModelValidator.validate_general_model(MockModel()) is True


def test_validate_general_model_missing_predict():
    class MockModel:
        pass
    with pytest.raises(ValueError, match="must have a 'predict' method"):
        ModelValidator.validate_general_model(MockModel())


# Tests for ModelLoader
@patch('src.utilities.importlib.import_module')
@patch('src.utilities.ModelValidator.validate_python_model')
def test_load_python_model(mock_validate, mock_import):
    mock_validate.return_value = True
    mock_module = MagicMock()
    mock_class = MagicMock()
    mock_module.ClassName = mock_class
    mock_import.return_value = mock_module

    sig = {
        'model': {
            'model_args': {
                'class': 'module.path:ClassName',
                'kwargs': {'param': 'value'}
            }
        }
    }
    model = ModelLoader.load_python_model(sig, '/root')
    mock_import.assert_called_with('module.path')
    mock_class.assert_called_with(param='value')
    assert model is not None


@patch('src.utilities.joblib.load')
def test_load_sklearn_model(mock_joblib):
    mock_joblib.return_value = 'mock_model'
    sig = {'model_file': 'model.pkl'}
    model = ModelLoader.load_sklearn_model(sig, '/root')
    mock_joblib.assert_called_with(os.path.join('/root', 'model.pkl'))
    assert model == 'mock_model'


@patch('src.utilities.ONNXModelPredictor')
def test_load_onnx_model(mock_onnx):
    mock_predictor = MagicMock()
    mock_onnx.return_value = mock_predictor
    sig = {'model_file': 'model.onnx'}
    model = ModelLoader.load_onnx_model(sig, '/root')
    mock_onnx.assert_called_with(onnx_model_path=os.path.join('/root', 'model.onnx'))
    assert model == mock_predictor


def test_load_model_python():
    sig = {
        'backend': 'python',
        'model': {
            'model_args': {
                'class': 'module.path:ClassName',
                'kwargs': {'param': 'value'}
            }
        }
    }
    with patch('src.utilities.importlib.import_module') as mock_import, \
         patch('src.utilities.ModelValidator.validate_python_model') as mock_validate:
        mock_validate.return_value = True
        mock_module = MagicMock()
        mock_class = MagicMock()
        mock_module.ClassName = mock_class
        mock_import.return_value = mock_module

        model = ModelLoader.load_model(sig, '/root')
        mock_import.assert_called_with('module.path')
        mock_class.assert_called_with(param='value')
        assert model is not None


def test_load_model_sklearn():
    sig = {'backend': 'sklearn', 'model_file': 'model.pkl'}
    with patch('src.utilities.joblib.load') as mock_joblib:
        mock_joblib.return_value = 'sklearn_model'
        model = ModelLoader.load_model(sig, '/root')
        mock_joblib.assert_called_with(os.path.join('/root', 'model.pkl'))
        assert model == 'sklearn_model'


def test_load_model_onnx():
    sig = {'backend': 'onnx', 'model_file': 'model.onnx'}
    with patch('src.utilities.ONNXModelPredictor') as mock_onnx:
        mock_predictor = MagicMock()
        mock_onnx.return_value = mock_predictor
        model = ModelLoader.load_model(sig, '/root')
        mock_onnx.assert_called_with(onnx_model_path=os.path.join('/root', 'model.onnx'))
        assert model == mock_predictor


def test_load_model_unsupported_backend():
    sig = {'backend': 'unsupported'}
    with pytest.raises(ValueError, match="Unsupported backend type"):
        ModelLoader.load_model(sig, '/root')


def test_load_model_not_implemented_backend():
    sig = {'backend': 'tensorflow'}  # Assuming not implemented
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        ModelLoader.load_model(sig, '/root')


# Tests for ONNXModelPredictor
@patch('src.utilities.ort.InferenceSession')
def test_onnx_predictor_init(mock_session):
    mock_sess = MagicMock()
    mock_input = MagicMock()
    mock_input.name = 'input'
    mock_sess.get_inputs.return_value = [mock_input]
    mock_session.return_value = mock_sess

    predictor = ONNXModelPredictor('path/model.onnx')
    mock_session.assert_called_with('path/model.onnx', sess_options=None, providers=None)
    assert predictor.session == mock_sess
    assert predictor.input_names == 'input'


@patch('src.utilities.ort.InferenceSession')
def test_onnx_predictor_init_with_warmup(mock_session):
    mock_sess = MagicMock()
    mock_input = MagicMock()
    mock_input.name = 'input'
    mock_input.shape = [1, 10]
    mock_input.type = 'tensor(float)'
    mock_sess.get_inputs.return_value = [mock_input]
    mock_sess.run.return_value = [['output']]
    mock_session.return_value = mock_sess

    with patch('src.utilities.np.random.rand') as mock_rand, \
         patch('src.utilities.np.float32') as mock_dtype:
        mock_rand.return_value = MagicMock()
        mock_rand.return_value.astype.return_value = 'dummy_input'

        predictor = ONNXModelPredictor('path/model.onnx', warmup_iterations=3)
        assert mock_sess.run.call_count == 3


@patch('src.utilities.ort.InferenceSession')
def test_onnx_predictor_predict(mock_session):
    mock_sess = MagicMock()
    mock_input = MagicMock()
    mock_input.name = 'input'
    mock_sess.get_inputs.return_value = [mock_input]
    mock_sess.run.return_value = ['result']
    mock_session.return_value = mock_sess

    predictor = ONNXModelPredictor('path/model.onnx')
    result = predictor.predict('input_data')
    mock_sess.run.assert_called_with(None, {'input': 'input_data'})
    assert result == 'result'


@patch('src.utilities.ort.InferenceSession')
def test_onnx_predictor_providers(mock_session):
    mock_sess = MagicMock()
    mock_sess.get_providers.return_value = ['CPUExecutionProvider']
    mock_session.return_value = mock_sess
    mock_input = MagicMock()
    mock_input.name = 'input'
    mock_sess.get_inputs.return_value = [mock_input]

    predictor = ONNXModelPredictor('path/model.onnx')
    assert predictor.providers == ['CPUExecutionProvider']


@patch('src.utilities.ort.InferenceSession')
def test_onnx_predictor_repr(mock_session):
    mock_sess = MagicMock()
    mock_sess.get_providers.return_value = ['CPU']
    mock_session.return_value = mock_sess
    mock_input = MagicMock()
    mock_input.name = 'input'
    mock_sess.get_inputs.return_value = [mock_input]

    predictor = ONNXModelPredictor('path/model.onnx')
    assert 'ONNXModelPredictor' in repr(predictor)