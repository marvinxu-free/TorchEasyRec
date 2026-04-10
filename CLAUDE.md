# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TorchEasyRec is a PyTorch-based recommendation system framework implementing state-of-the-art deep learning models for matching, ranking, multi-task learning, and generative recommendation. It uses Protobuf for configuration and supports distributed training via TorchRec.

## Development Commands

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (required for development)
pre-commit install
```

### Testing

```bash
# Run all tests
python -m unittest discover -s tzrec -p "*_test.py"

# Run single test file
python -m tzrec.modules.fm_test

# Run single test case
python -m tzrec.modules.fm_test FactorizationMachineTest.test_fm_0

# Run tests for specific module
python -m unittest discover -s tzrec/datasets -p "*_test.py"
```

### Code Quality

```bash
# Run pre-commit checks on all files
pre-commit run -a

# Type checking with pyre
python scripts/pyre_check.py

# Format code (automatically done by pre-commit)
ruff format .
ruff check --fix .
```

### Protobuf

```bash
# Generate Python code from .proto files
bash scripts/gen_proto.sh
```

**Important**: After modifying any `.proto` file in `tzrec/protos/`, you MUST run this command to regenerate the Python code.

### Training & Evaluation

```bash
# Train model
python -m tzrec.main --pipeline_config_path=examples/model.config

# Evaluate model
python -m tzrec.eval --pipeline_config_path=examples/model.config

# Export model
python -m tzrec.export --pipeline_config_path=examples/model.config --export_dir=export/

# Predict
python -m tzrec.predict \
    --predict_input_path=input_data \
    --predict_output_path=output_data \
    --scripted_model_path=export/
```

### Build & Deploy

```bash
# Build wheel package
bash scripts/build_wheel.sh nightly  # or 'release'

# Build docker image
bash scripts/build_docker.sh

# Build documentation
bash scripts/doc/build_docs.sh
```

## Architecture Concepts

### Configuration-Driven Workflow

TorchEasyRec uses Protobuf text format for all configurations. A typical pipeline config includes:

1. **data_config**: Dataset type, batch size, input paths
2. **feature_configs**: Feature definitions (IdFeature, RawFeature, etc.)
3. **model_config**: Model architecture and hyperparameters
4. **train_config**: Optimizers, learning rate schedulers
5. **eval_config**: Evaluation metrics and frequency

Example structure:
```
train_input_path: "path/to/train"
model_dir: "experiments/model"
train_config { ... }
data_config { ... }
feature_configs { ... }
model_config { ... }
```

### Feature System

Features are defined in `feature_configs` and parsed by `BaseFeature` subclasses:

- **IdFeature**: Categorical features with embedding
- **RawFeature**: Numerical features
- **ComboFeature**: Feature combinations
- **LookupFeature**: Lookup-based features
- **SequenceFeature**: Sequential features
- **CustomFeature**: User-defined features

Feature groups in model config group features for processing:
```
feature_groups {
    group_name: "user"
    feature_names: "user_id"
    feature_names: "age"
}
```

### Model Architecture

All models inherit from `BaseModel` (`tzrec/models/model.py`) and must implement:

- `predict(batch) -> Dict[str, torch.Tensor]`: Forward pass returning predictions
- `loss(predictions, batch) -> Dict[str, torch.Tensor]`: Compute loss
- `init_metric()`: Initialize metrics
- `update_metric(predictions, batch)`: Update metrics

**Model Registration**: Models are auto-registered via `@BaseModel.register()` decorator:

```python
@BaseModel.register()
class DeepFM(BaseModel):
    ...
```

Model types:
- **Matching**: DSSM, MIND, TDM, DAT (two-tower or tree-based)
- **Ranking**: DeepFM, WideAndDeep, DIN, DLRM, DCN, etc.
- **Multi-task**: MMoE, PLE, DBMTL, PEPNet
- **Generative**: DLRM-HSTU

### Training Pipeline

`tzrec/main.py` orchestrates the training flow:

1. Parse config and create features
2. Create dataset and dataloader
3. Initialize model with DistributedModelParallel (TorchRec)
4. Create optimizers (separate for sparse/dense parameters)
5. Training loop with gradient scaling (mixed precision)
6. Periodic evaluation and checkpointing

### Distributed Training

Uses PyTorch Distributed + TorchRec for:
- **Data parallelism**: Across multiple GPUs/nodes
- **Model parallelism**: For large embedding tables (row-wise, column-wise, table-wise sharding)
- **Mixed precision**: FP16/BF16 training

Key components:
- `DistributedModelParallel`: Wraps model for distributed execution
- `create_train_pipeline()`: Sets up distributed training pipeline
- `EmbeddingGroup`: Manages sharded embeddings

## Extending the Framework

### Adding a New Model

1. Create model file in `tzrec/models/your_model.py`
2. Inherit from `BaseModel` (or `RankModel`, `MatchModel`)
3. Implement required methods:
   ```python
   @BaseModel.register()
   class YourModel(BaseModel):
       def __init__(self, model_config, features, labels, **kwargs):
           super().__init__(model_config, features, labels, **kwargs)
           # Define layers

       def predict(self, batch):
           # Forward pass
           return {"prediction": output}

       def loss(self, predictions, batch):
           # Compute loss
           return {"loss": loss_value}
   ```

4. Add proto definition in `tzrec/protos/models/your_model.proto`:
   ```protobuf
   message YourModelConfig {
       // hyperparameters
   }
   ```

5. Update `tzrec/protos/model.proto` to include your config
6. Run `bash scripts/gen_proto.sh` to regenerate proto Python files
7. Add tests in `tzrec/models/your_model_test.py`

### Adding a New Feature

1. Create feature file in `tzrec/features/your_feature.py`
2. Inherit from `BaseFeature`:
   ```python
   @BaseFeature.register()
   class YourFeature(BaseFeature):
       def forward(self, batch):
           # Process feature
           return embeddings
   ```

3. Add proto definition in `tzrec/protos/feature.proto`
4. Run `bash scripts/gen_proto.sh`
5. Add tests

### Adding a New Dataset

1. Create dataset file in `tzrec/datasets/your_dataset.py`
2. Inherit from `BaseDataset`:
   ```python
   @BaseDataset.register()
   class YourDataset(BaseDataset):
       def _read_stream(self):
           # Yield data records
           pass
   ```

3. Add proto definition in `tzrec/protos/data.proto`
4. Run `bash scripts/gen_proto.sh`
5. Add tests

## Key Files

| File | Purpose |
|------|---------|
| `tzrec/main.py` | Main entry point for train/eval/export/predict |
| `tzrec/models/model.py` | BaseModel class and model registration |
| `tzrec/features/feature.py` | BaseFeature class and feature registration |
| `tzrec/datasets/dataset.py` | BaseDataset and dataloader creation |
| `tzrec/modules/embedding.py` | Embedding layers and groups |
| `tzrec/utils/config_util.py` | Config parsing and validation |
| `tzrec/utils/load_class.py` | Dynamic class loading and registration |

## Important Patterns

### Proto Configuration Pattern

All components use Protobuf for configuration:
1. Define config in `.proto` file
2. Compile to Python with `gen_proto.sh`
3. Parse in code using `config_util.parse_pipeline_config()`
4. Access via generated `_pb2.py` classes

### Registration Pattern

Components are auto-registered using decorators:
- `@BaseModel.register()` for models
- `@BaseFeature.register()` for features
- `@BaseDataset.register()` for datasets

Registration adds class to `_CLASS_MAP` dictionary for dynamic instantiation.

### Sparse/Dense Parameter Split

The framework separates sparse (embedding) and dense parameters:
- Different optimizers (e.g., Adagrad for sparse, Adam for dense)
- Different learning rate schedules
- Handled via `CombinedOptimizer` from TorchRec

### TorchRec Integration

Embedding tables use TorchRec primitives:
- `EmbeddingBagCollection`: For pooling embeddings
- `EmbeddingCollection`: For sequence features
- `DistributedModelParallel`: Automatic sharding

## Testing Strategy

- **Unit tests**: Use `unittest` framework, files named `*_test.py`
- **Test structure**: One test class per module, test methods named `test_<scenario>`
- **Fixtures**: Test data in `tzrec/tests/test_data/` or generated in tests
- **Coverage**: Aim for comprehensive coverage of edge cases

## Code Style

- **Formatting**: Ruff (configured in `.ruff.toml`)
- **Type hints**: Use pyre for static type checking
- **Docstrings**: Google style docstrings
- **License header**: Apache 2.0 header required (auto-added by pre-commit)
- **Imports**: Grouped as: stdlib, third-party, local (separated by blank lines)

## Common Issues

### Proto Compilation

If you see `ImportError` for `_pb2` modules, run:
```bash
bash scripts/gen_proto.sh
```

### Distributed Training

For multi-GPU training, ensure:
- `torchrun` or `mpirun` is used to launch
- Environment variables (`RANK`, `WORLD_SIZE`, etc.) are set
- Data is properly sharded across workers

### Feature Not Found

If features aren't found:
1. Check `feature_configs` spelling matches data columns
2. Verify `fg_mode` in `data_config` (FG_NONE vs FG_SIMPLE)
3. Ensure feature is in correct feature group in model config

## Resources

- [README.md](README.md) - Project overview and features
- [docs/source/develop.md](docs/source/develop.md) - Development guide
- [docs/source/models/](docs/source/models/) - Model documentation
- [examples/](examples/) - Example configurations
