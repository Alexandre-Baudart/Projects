import joblib
import os
import shutil
import json
import argparse
from typing import final, Literal
from pathlib import Path
from datetime import datetime

from ..project_dataset import ProjectTrainDataset, ProjectTestDataset

from .nncore.nno import NNO
from .nncore.nnp import NNP
from .utils import criterion_config, optimizer_config, select_device, random_init
from .callbacks import callbacks_config

from data.dataset import Dataset_
from .calibration import CalibratedModel

class DLSession:
    def __init__(self,
                 config: dict,
                 dataset: Dataset_,
                 target: str,
                 preprocess_options: dict | None = None,
                 metadata: dict | None = None):

        if metadata is None:
            metadata = {
                "name": "sandbox",
                "paths": {
                    "metadata": "sandbox_metadata_dl.json",
                    "runs": "runs/sandbox/dl"
                },
                "default": {
                    "format": "pt",
                },
                "last_run": 0
        }

        self.metadata = metadata

        self.root = Path(self.metadata["paths"]["runs"])
        self.metadata_path = self.root / self.metadata["paths"]["metadata"]

        self._load_metadata()

        self.active_run = None
        self.run_metadata = None

        self.dataset = dataset
        self.target = target

        preprocess_options = preprocess_options or {}
        self.dataset.build_preprocessing(target=self.target, **preprocess_options)

        self.config = config
        random_init(seed=self.config.get("random_seed", 42))

        self.raw_model = self.config["raw_model"]

        self.nnp = None

        self.mode = self.config.get("mode", "clf_binary")
        self.device = select_device(type_=self.config.get("device_type", "cpu"))

        self.preprocess = self.dataset.get_preprocess()
        self.preprocess_ = None

    @final
    def _save_run_json(self, path: Path, results: dict):
        filename = path.name
        already_exists = os.path.exists(path)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

            if not already_exists:
                print(f"\nNote: {filename} has been written.")
            else:
                print(f"\nNote: {filename} has been rewritten.")

    @final
    def _save_metadata(self, type_: Literal["global", "run"] = "global"):
        if type_ == "global":
            metadata = self.metadata.copy()
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.metadata_path
        else:
            metadata = self.run_metadata.copy()
            path = self.active_run / "metadata.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                metadata,
                f,
                indent=4,
            )

    @final
    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

                required = {
                    "name",
                    "paths",
                    "default",
                    "last_run"
                }

                missing = required - self.metadata.keys()

                if missing:
                    raise ValueError(f"Invalid session metadata!" f"Missing keys: {missing}")

    @final
    def new_run(self):
        run_id = self.metadata["last_run"] + 1
        self.metadata["last_run"] = run_id

        self._save_metadata()

        run_dir = (
                self.root / f"run_{run_id:03d}"
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.active_run = run_dir

        self.run_metadata = self._create_run_metadata(run_id)

        self._save_metadata(type_="run")

        return run_dir

    @final
    def clean_sandbox(self):
        if self.metadata["name"] == "sandbox":
            sandbox_root = self.root

            if sandbox_root.exists():
                shutil.rmtree(sandbox_root)

            sandbox_root.mkdir(
                parents=True,
                exist_ok=True
            )

            self.active_run = None
            self.run_metadata = None

            self.metadata["last_run"] = 0

            self._save_metadata()

            print("\nNote: sandbox has been cleaned with success!")

    @final
    def _create_run_metadata(self, run_id: int):
        return {
            "run_id": run_id,
            "model": self.raw_model.__name__,
            "target": self.target,
            "created_at": datetime.now().isoformat(),
            "status": "created"
        }

    @final
    def _check_model_path_exists(self, model_path: Path | str):
        model_path = Path(model_path)
        return model_path.exists()

    @final
    def load(self, model_path: Path | str, dataset= None):
        if not self._check_model_path_exists(model_path):
            raise FileNotFoundError(f"No model at: {model_path}")

        model_config = self.config["train"]["params"]
        model_config["n_classes"] = dataset.get_n_classes()
        model_config["input_size"] = dataset.get_input_size()

        model = self.raw_model(model_config)
        model.to(self.device)

        criterion = criterion_config(
            self.mode,
            criterion_info=self.config.get("criterion_info", {}),
        )

        self.nnp = NNP.load_model(
            model=model,
            model_path=str(model_path),
            criterion=criterion,
            dataloader_params=self.config.get("dataloader_params", {}),
            metric_info=self.config.get("metric_info", {}),
            device=self.device
        )

    @final
    def optimize(self):
        optim_info = self.config["optim_info"]

        dataset_optim = ProjectTrainDataset(
            dataset=self.dataset,
            target=self.target,
            preprocess_path=str(self.active_run),
            task="optim",
            use_stratified_split=optim_info.get("use_stratified_split", False)
        )

        criterion = criterion_config(
            self.mode,
            criterion_info=self.config.get("criterion_info", {}),
        )

        nno = NNO(
            self.raw_model,
            dataset_train=dataset_optim,
            n_epochs=self.config.get("n_epochs", 30),
            batch_size=self.config.get("batch_size", 32),
            criterion=criterion,
            metric_info=self.config.get("metric_info", {}),
            dataloader_params=self.config.get("dataloader_params", {}),
            device=self.device
        )

        search_space = self.config["optim"]["search_space"]
        search_space["n_classes"] = dataset_optim.get_n_classes()
        search_space["input_size"] = dataset_optim.get_input_size()

        if self.config.get("get_transformer_stuff", False):
            n_num, n_cat, cat_idx, cat_cardinalities = dataset_optim.get_tab_transformer_stuff()
            dataset_optim.shift_cat_indices(cat_idx)

            transformer_stuff = {
                "n_num": n_num,
                "n_cat": n_cat,
                "cat_idx": cat_idx,
                "cat_cardinalities": cat_cardinalities
            }

            search_space = {
                **search_space,
                **transformer_stuff
            }

        results = nno.optimize(
            search_space=search_space,
            optimizer_info=self.config.get("optimizer_info", {}),
            grace_period=optim_info.get("grace_period", 3),
            reduction_factor=optim_info.get("reduction_factor", 3),
            n_samples=optim_info.get("n_samples", 5),
            max_concurrent_trials=optim_info.get("max_concurrent_trials", 1),
            resources_per_trial=optim_info.get("resources_per_trial", None),
            verbose=optim_info.get("verbose", 2)
        )

        if self.active_run is not None:
            self._save_run_json(
                self.active_run /"optimization.json",
                results
            )

    @final
    def train(self, logits_scaling: bool = False, save_model: bool = False):
        train_info = self.config["train_info"]

        dataset_train = ProjectTrainDataset(
                dataset=self.dataset,
                target=self.target,
                preprocess_path=str(self.active_run),
                task="train",
                use_stratified_split=train_info.get("use_stratified_split", False)
        )

        self.preprocess_ = dataset_train.get_preprocess_()

        optimizer_info = train_info.get("optimizer_info", {})
        optimizer_info["weight_decay"] = self.config["train"].get("weight_decay", 0.0)

        criterion = criterion_config(
            self.mode,
            criterion_info=self.config.get("criterion_info", {}),
        )

        model_config = self.config["train"]["params"]
        model_config["n_classes"] = dataset_train.get_n_classes()
        model_config["input_size"] = dataset_train.get_input_size()

        if not logits_scaling:
            model_config["scaling_info"] = None

        if self.config.get("get_transformer_stuff", False):
            n_num, n_cat, cat_idx, cat_cardinalities = dataset_train.get_tab_transformer_stuff()
            dataset_train.shift_cat_indices(cat_idx)

            transformer_stuff = {
                "n_num": n_num,
                "n_cat": n_cat,
                "cat_idx": cat_idx,
                "cat_cardinalities": cat_cardinalities
            }

            model_config = {
                **model_config,
                **transformer_stuff
            }

        model = self.raw_model(model_config)
        model.to(self.device)

        if self.nnp is None or (self.nnp is not None and not self.nnp.has_optimizer()):
            optimizer = optimizer_config(
                model_or_params=model,
                optimizer_info=optimizer_info
            )

            scheduler_info = train_info.get("scheduler_info", None)
        else:
            optimizer = self.nnp.optimizer
            scheduler_info = self.nnp.callbacks["scheduler"]

        callbacks = callbacks_config(
            optimizer=optimizer,
            scheduler_info=scheduler_info,
            **self.config.get("callbacks_info", {})
        )

        self.nnp = NNP(
            model=model,
            mode=self.mode,
            batch_size=self.config.get("batch_size", 32),
            precision=train_info.get("precision", 32),
            memory_format=train_info.get("memory_format", None),
            criterion=criterion,
            metric_info=train_info.get("metric_info", {}),
            dataloader_params=self.config.get("dataloader_params", {}),
            device=self.device
        )

        if save_model and self.active_run is None:
            self.new_run()

        results = self.nnp.train(
            dataset=dataset_train,
            n_epochs=train_info.get("n_epochs", 30),
            lr=self.config.get("lr", 1e-3),
            optimizer=optimizer,
            clip_grad_norm=train_info.get("clip_grad_norm", False),
            batch_proc_fn=train_info.get("batch_proc_fn", None),
            use_ema=train_info.get("use_ema", False),
            ema_decay=train_info.get("ema_decay", 0.999),
            accumulate_grad_batches=train_info.get("accumulate_grad_batches", 1),
            unfreezing_schedule=train_info.get("unfreezing_schedule", None),
            callbacks=callbacks,
            save_root=str(self.active_run),
            save_best_model=save_model,
            enable_checkpoints=train_info.get("enable_checkpoints", False)
        )

        if save_model:
            if self.preprocess_ is not None:
                joblib.dump(self.preprocess_, (self.active_run / "preprocess_.joblib"))
                print("\nNote: preprocess_ saved with success!")

            self.run_metadata["status"] = "trained"

            if logits_scaling:
                self.run_metadata["logits_scaling"] = model_config["scaling_info"]["method"]

            self._save_metadata(type_="run")

            self._save_run_json(
                self.active_run /"train.json",
                results
            )

    def calibrate(self, model_path: str | None = None, save_model: bool = False):
        calib_info = self.config.get("calib_info", None)

        if calib_info is not None:
            if model_path is not None:
                model_path = Path(model_path)
                if self._check_model_path_exists(model_path):
                    self.active_run = model_path.parent

                    dataset_calib = ProjectTrainDataset(
                        dataset=self.dataset,
                        target=self.target,
                        preprocess_path=str(self.active_run),
                        task="calib",
                        use_stratified_split=calib_info.get("use_stratified_split", False)
                    )

                    self.load(model_path, dataset_calib)
                else:
                    raise FileNotFoundError(f"No model at: {model_path}")
            else:
                dataset_calib = ProjectTrainDataset(
                    dataset=self.dataset,
                    target=self.target,
                    preprocess_path=str(self.active_run),
                    task="calib",
                    use_stratified_split=calib_info.get("use_stratified_split", False)
                )

            calib_info["n_classes"] = dataset_calib.get_n_classes()

            if save_model and self.active_run is None:
                self.new_run()

            train_info = self.config["train_info"]

            optimizer_info = calib_info.get("optimizer_info", {})
            optimizer_info["weight_decay"] = calib_info.get("weight_decay", 0.0)

            criterion = criterion_config(
                self.mode,
                criterion_info=self.config.get("criterion_info", {}),
            )

            calibrated_model = CalibratedModel(trained_model=self.nnp.model, config=calib_info)

            optimizer = optimizer_config(
                model_or_params=calibrated_model,
                optimizer_info=optimizer_info
            )

            callbacks = callbacks_config(
                optimizer=optimizer,
                scheduler_info=calib_info.get("scheduler_info", None),
                **self.config.get("callbacks_info", {})
            )

            self.nnp = NNP(
                model=calibrated_model,
                mode=self.mode,
                batch_size=self.config.get("batch_size", 32),
                precision=train_info.get("precision", 32),
                memory_format=train_info.get("memory_format", None),
                criterion=criterion,
                metric_info=calib_info.get("metric_info", {}),
                dataloader_params=self.config.get("dataloader_params", {}),
                device=self.device
            )

            results = self.nnp.train(
                dataset=dataset_calib,
                n_epochs=train_info.get("n_epochs", 30),
                lr=self.config.get("lr", 1e-3),
                optimizer=optimizer,
                clip_grad_norm=train_info.get("clip_grad_norm", False),
                batch_proc_fn=train_info.get("batch_proc_fn", None),
                use_ema=train_info.get("use_ema", False),
                ema_decay=train_info.get("ema_decay", 0.999),
                accumulate_grad_batches=train_info.get("accumulate_grad_batches", 1),
                unfreezing_schedule=train_info.get("unfreezing_schedule", None),
                callbacks=callbacks,
                save_root=str(self.active_run),
                save_best_model=save_model,
                enable_checkpoints=train_info.get("enable_checkpoints", False)
            )

            if save_model:
                if self.preprocess_ is not None:
                    joblib.dump(self.preprocess_, (self.active_run / "preprocess_.joblib"))
                    print("\nNote: preprocess_ saved with success!")

                self.run_metadata["status"] = "calibrated"
                self.run_metadata["calibration"] = "sigmoid"

                self._save_metadata(type_="run")

    def test(self, model_path: str | None = None, conf_matrix: bool = False):
        if model_path is not None:
            model_path = Path(model_path)
            if self._check_model_path_exists(model_path):
                self.active_run = model_path.parent

                dataset_test = ProjectTestDataset(
                    dataset=self.dataset,
                    target=self.target,
                    preprocess_=self.preprocess_,
                )

                self.load(model_path, dataset_test)
            else:
                raise FileNotFoundError(f"No model at: {model_path}")
        else:
            dataset_test = ProjectTestDataset(
                dataset=self.dataset,
                target=self.target,
                preprocess_=self.preprocess_,
            )

        if self.preprocess_ is None:
            dataset_test.load_preprocess_(path=str(self.active_run))

        if self.config.get("get_transformer_stuff", False):
            n_num, n_cat, cat_idx, cat_cardinalities = dataset_test.get_tab_transformer_stuff()
            dataset_test.shift_cat_indices(cat_idx)

        metrics = self.config["test_info"].get("metrics", None)

        eval_path = self.active_run

        if self.active_run is not None and conf_matrix:
            eval_path = eval_path / "evaluation"
            os.mkdir(path=eval_path)

        results = self.nnp.evaluate(
            dataset=dataset_test,
            metrics=metrics,
            calib_eval=self.config["test_info"].get("calib_eval", False),
            conf_matrix=conf_matrix,
            save_root=eval_path
        )

        if self.active_run is not None:
            self._save_run_json(
                eval_path / "evaluation.json",
                results
            )

    @final
    def check_high_confidence_bias(self, model_path: str | None = None):
        if model_path is not None:
            model_path = self.root / model_path
            if self._check_model_path_exists(model_path):
                self.active_run = model_path.parent

                dataset_check = ProjectTestDataset(
                    dataset=self.dataset,
                    target=self.target,
                    preprocess_=self.preprocess_,
                )

                self.load(model_path, dataset_check)
            else:
                raise FileNotFoundError(f"No model at: {model_path}")
        else:
            dataset_check = ProjectTestDataset(
                dataset=self.dataset,
                target=self.target,
                preprocess_=self.preprocess_,
            )

        if self.preprocess_ is None:
            dataset_check.load_preprocess_(path=str(self.active_run))

        if self.config.get("get_transformer_stuff", False):
            n_num, n_cat, cat_idx, cat_cardinalities = dataset_check.get_tab_transformer_stuff()
            dataset_check.shift_cat_indices(cat_idx)

        res, gap = self.nnp.check_high_confidence_bias(dataset_check)

        if res:
            print(f"\nThe model is over-confident (overconfidence gap : {gap:.2f}) and needs to be recalibrated.")
        else:
            print(
                f"\nThe model is honest in its predictions (overconfidence gap : {gap:.2f}) and does not need to be recalibrated.")

    @final
    def benchmark(self, model_path: str | None = None, n_iterations: int = 100):
        if model_path is not None:
            model_path = Path(model_path)
            if self._check_model_path_exists(model_path):
                self.active_run = model_path.parent

                dataset_benchmark = ProjectTestDataset(
                    dataset=self.dataset,
                    target=self.target,
                    preprocess_=self.preprocess_,
                )

                self.load(model_path, dataset_benchmark)
            else:
                raise FileNotFoundError(f"No model at: {model_path}")
        else:
            dataset_benchmark = ProjectTestDataset(
                dataset=self.dataset,
                target=self.target,
                preprocess_=self.preprocess_,
            )

        if self.preprocess_ is None:
            dataset_benchmark.load_preprocess_(path=str(self.active_run))

        if self.config.get("get_transformer_stuff", False):
            n_num, n_cat, cat_idx, cat_cardinalities = dataset_benchmark.get_tab_transformer_stuff()
            dataset_benchmark.shift_cat_indices(cat_idx)

        results = self.nnp.benchmark(dataset_benchmark, n_iterations=n_iterations)

        if self.active_run is not None:
            self._save_run_json(
                self.active_run / "benchmark.json",
                results
            )
            
class DLOrchestrator:
    def __init__(self, session_args):
        self.session_args = session_args
        self.sess_models = self.session_args["model_config"].keys()

        self.ACTION_ORDER = ["optimize", "train", "calibrate", "test", "check_high_confidence_bias", "benchmark"]

        self.args = None
        self._from_cli()

        self.session = None
        self._build_session()

        if self.args.load is not None:
            self.model_path = os.path.join(self.args.root, self.args.load)
        else:
            self.model_path = None

        actions = self.args.actions
        if self.args.calibrate: actions.append("calibrate")

        self.actions = []
        self._resolve_actions(self.args.actions)

    def _from_cli(self):
        parser = argparse.ArgumentParser(
            description="..."
        )
        parser.add_argument(
            "--model", "-m", choices=self.sess_models,
            help="Model used"
        )
        parser.add_argument(
            "--actions", "-a", nargs='+',
            choices=["optimize", "train", "test", "check_high_confidence_bias", "benchmark"],
            help="Action"
        )
        parser.add_argument(
            "--logits_scaling", "-lsc", default=False, action="store_true", help="Scale logits"
        )
        parser.add_argument(
            "--calibrate", "-c", default=False, action="store_true", help="Calibrate model"
        )
        parser.add_argument(
            "--save", "-s", default=False, action="store_true", help="Save model"
        )
        parser.add_argument(
            "--root", "-r", default="runs/sandbox", help="Root"
        )
        parser.add_argument(
            "--save_name", "-sn", default=None, help="Save filename"
        )
        parser.add_argument(
            "--save_format", "-sf", default="joblib", help="Save format"
        )
        parser.add_argument(
            "--load", "-l", default=None, help="Load model"
        )
        parser.add_argument(
            "--conf_matrix", "-cm", default=False, action="store_true", help="Confusion matrix"
        )
        parser.add_argument(
            "--clean_sandbox", "-cs", default=False, action="store_true", help="Clean sandbox"
        )
        parser.add_argument(
            "--device", "-d", choices=["cpu", "gpu"], default="cpu",
        )

        self.args = parser.parse_args()

    def _build_session(self):
        config = {
            **self.session_args["config"],
            **self.session_args["model_config"][self.args.model],
            "device_type": self.args.device
        }

        self.session = DLSession(
            config=config,
            dataset=self.session_args["dataset"],
            target=self.session_args["target"],
            preprocess_options=config["preprocess"],
            metadata=self.session_args.get("metadata", None)
        )

    @final
    def _resolve_actions(self, requested_actions: list | None = None):
        if requested_actions is None:
            return

        actions = set()

        def add_action(action_):
            if action_ in actions:
                return

            if (
                    action_ in ["calibrate", "test", "check_high_confidence_bias", "benchmark"]
                    and "train" not in actions
                    and not self.model_path
            ):
                raise RuntimeError(f"A trained model is required for the following action: {action_}")

            actions.add(action_)

        for action in requested_actions:
            add_action(action)

        for action in self.ACTION_ORDER:
            if action in actions:
                self.actions.append(action)

    @final
    def run(self):
        if self.args.clean_sandbox:
            self.session.clean_sandbox()

        for action in self.actions:
            if action == "optimize":
                self.session.optimize()

            elif action == "train":
                self.session.train(
                    logits_scaling=self.args.logits_scaling,
                    save_model=self.args.save
                )

            elif action == "calibrate":
                self.session.calibrate(
                    model_path=self.args.load,
                    save_model=self.args.save
                )

            elif action == "test":
                self.session.test(
                    model_path=self.args.load,
                    conf_matrix=self.args.conf_matrix
                )

            elif action == "check_high_confidence_bias":
                self.session.check_high_confidence_bias(model_path=self.args.load)

            elif action == "benchmark":
                self.session.benchmark(model_path=self.args.load)