import argparse
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, Optional, Self


@dataclass
class RunConfig(ABC):
    run_dir: Path
    runs_base_dir: Path
    batch_size: int
    include_elements: Optional[list[int]]
    exclude_elements: Optional[list[int]]
    num_proc_test: int
    world_size: int
    global_rank: int
    local_rank: int

    @classmethod
    @abstractmethod
    def _add_args(cls, parser: argparse.ArgumentParser):
        pass

    @classmethod
    def get_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--run_dir",
            type=Path,
            help="Directory for training or testing. Created automatically during training if not specified.",
        )
        parser.add_argument(
            "--runs_base_dir",
            type=Path,
            default=Path("runs"),
            help="Directory where automatically created directory for training dataset is placed.",
        )
        parser.add_argument("--batch_size", type=int, default=1, help="Number of samples in a batch per GPU.")
        parser.add_argument(
            "--include_elements",
            type=int,
            default=None,
            nargs="*",
            help="Only take samples that include at least one of the elements with listed atomic numbers.",
        )
        parser.add_argument(
            "--exclude_elements",
            type=int,
            default=None,
            nargs="*",
            help="Only take samples that do not include any of the elements with listed atomic numbers.",
        )
        parser.add_argument(
            "--num_proc_test", type=int, default=1, help="Number of parallel processes for computing test statistics."
        )
        cls._add_args(parser)
        return parser.parse_args()

    @classmethod
    def from_cmd_args(cls) -> Self:
        args = vars(cls.get_args())
        args["world_size"] = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        args["global_rank"] = int(os.environ["RANK"]) if "RANK" in os.environ else 0
        args["local_rank"] = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        return cls(**args)

    @classmethod
    def from_dict(cls, config_dict: dict) -> Self:
        field_names = [field.name for field in fields(cls)]
        for field in field_names:
            if field not in config_dict:
                raise ValueError(f"Field {field} not found in config dict.")
        for key in list(config_dict.keys()):
            if key not in field_names:
                del config_dict[key]
        for field in fields(cls):
            if field.type == Path and config_dict[field.name] is not None:
                config_dict[field.name] = Path(config_dict[field.name])
        return cls(**config_dict)

    def into_json_dict(self) -> dict:
        config_dict = asdict(self)
        # These are not JSON serializable, so convert them to str
        for key in config_dict:
            if isinstance(config_dict[key], Path):
                config_dict[key] = str(config_dict[key])
        return config_dict


@dataclass
class TrainConfig(RunConfig):
    base_model: Path
    dataset: Path
    testset: Path
    train_samples: Optional[int]
    test_samples: Optional[int]
    num_epochs: int
    test_interval: int
    lr_scheduler: Literal["warmup-decay", "cosine"]
    lr: float
    lr_warm: int
    lr_decay: float
    lr_mult: int
    irreps_hidden: str
    correlation_order: int
    num_layers: int
    finetune_method: str

    @classmethod
    def _add_args(cls, parser):
        parser.description = "Train model"
        parser.add_argument("--base_model", type=Path, help="Directory of model used as starting point for fine tuning")
        parser.add_argument("--dataset", type=Path, help="Path to training dataset")
        parser.add_argument("--testset", type=Path, help="Path to test dataset")
        parser.add_argument(
            "--train_samples", type=int, default=None, help="Number of samples to take from the training set."
        )
        parser.add_argument(
            "--test_samples", type=int, default=None, help="Number of samples to take from the test set."
        )
        parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the model.")
        parser.add_argument("--test_interval", type=int, default=1, help="Number of epochs between test evaluations.")
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="warmup-decay",
            help="Type of learning rate scheduler to use. Either 'warmup-decay' or 'cosine'.",
        )
        parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate for optimization.")
        parser.add_argument("--lr_warm", type=int, default=4000, help="Number of steps for learning rate warmup.")
        parser.add_argument(
            "--lr_decay",
            type=float,
            default=10000,
            help="Number of batches for learning rate decay or epochs between cosine anneal restarts.",
        )
        parser.add_argument("--lr_mult", type=int, default=1, help="Multiplication factor for cosine annealing restart steps.")
        parser.add_argument(
            "--irreps_hidden", type=str, default="128-128-128-128", help="Number of irreps in equivariant layers."
        )
        parser.add_argument("--correlation_order", type=int, default=3, help="MACE layer correlation order.")
        parser.add_argument("--num_layers", type=int, default=3, help="Number of convolution layer.")
        parser.add_argument("--finetune_method", type=str, default="restart-all", help="Type of finetuning to perform.")


@dataclass
class TestConfig(RunConfig):
    testset: Path
    test_samples: Optional[int]
    weights_epoch: Optional[int]

    @classmethod
    def _add_args(cls, parser):
        parser.description = "Test model"
        parser.add_argument("--testset", type=Path, help="Path to test dataset")
        parser.add_argument(
            "--test_samples", type=int, default=None, help="Number of samples to take from the test set."
        )
        parser.add_argument("--weights_epoch", type=int, default=None, help="Epoch to load weights from.")
        return parser.parse_args()
