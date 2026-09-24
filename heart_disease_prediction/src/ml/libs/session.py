import os
import shutil
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import final, Literal

from data.dataset import Dataset_
from .utils import random_init

class MLSession:
    def __init__(self,
                 config: dict,
                 dataset: Dataset_ | None = None,
                 target: str | None = None,
                 preprocess_options: dict | None = None,
                 metadata: dict | None = None
     ) :

        if metadata is None:
            metadata = {
                "name": "sandbox",
                "paths": {
                    "metadata": "sandbox_metadata_ml.json",
                    "runs": "runs/sandbox/ml"
                },
                "default": {
                    "format": "joblib",
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

        if self.dataset is not None :
            preprocess_options = preprocess_options or {}
            self.dataset.build_preprocessing(target=self.target, **preprocess_options)

        self.config = config
        random_init(seed=self.config.get("random_seed", 42))

        self.mode = self.config.get("mode", "clf_binary")

        raw_model = self.config["raw_model"]
        self.model_ = raw_model(self.mode, self.config.get("params", None))

        if self.dataset is not None:
            self.model_.set_preprocess(self.dataset.get_preprocess())

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

    def _create_run_metadata(self, run_id: int):
        return {
            "run_id": run_id,
            "model": self.model_.__class__.__name__,
            "target": self.target,
            "created_at": datetime.now().isoformat(),
            "status": "created"
        }

    @final
    def load(self, model_path: Path | str):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"No model found at: {model_path}")

        self.active_run = model_path.parent
        self.model_ = self.model_.load(model_path=str(model_path))

    @final
    def save(self, filename: str | None = None, format_: str | None = None) :
        if self.active_run is None:
            self.new_run()

        if format_ is None:
            format_ = self.metadata["default"]["format"]

        extensions = {
            "joblib": ".joblib",
            "skops": ".skops",
        }

        if format_ not in extensions:
            raise ValueError(f"Unknown format: {format_}")

        self.root.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = self.model_.__class__.__name__.lower()

        path = self.active_run / f"{filename}{extensions[format_]}"

        self.model_.save(
            path=path,
            format_=format_,
        )

    @final
    def optimize(self, **kwargs) :
        if self.dataset is None:
            raise ValueError("A dataset must be provided.")

        X_search, y_search = self.dataset.get_optim_set(
            target=self.target,
            search_set_size=self.config["optim"].get("search_set_size", 0.3)
        )

        results = self.model_.optimize(X_search, y_search, self.config["optim"].get("metric", "acc"), **kwargs)

        if self.active_run is not None:
            self._save_run_json(
                self.active_run / "optimization.json",
                results
            )

    @final
    def train(self, calibrate: bool = True, save_model: bool = False, get_features_importance: bool = False, **kwargs) :
        if self.dataset is None:
            raise ValueError("A dataset must be provided.")

        X_train, y_train = self.dataset.get_train_set(target=self.target)

        if calibrate:
            X_calib, y_calib = self.dataset.get_calib_set(target=self.target)
        else:
            X_calib, y_calib = None, None

        results = self.model_.fit(
            X=X_train, y=y_train,
            metric=self.config["train"].get("metric", "acc"),
            calibrate=calibrate,
            calib_set=(X_calib, y_calib),
            get_features_importance=get_features_importance
        )

        if save_model:
            self.save(
                filename=kwargs.get("filename", None),
                format_=kwargs.get("format_", "joblib")
            )

            self.run_metadata["status"] = "trained"

            if calibrate:
                self.run_metadata["calibration"] = "sigmoid"
                self.run_metadata["status"] = "calibrated"

            self._save_metadata(type_="run")

            self._save_run_json(
                self.active_run /"train.json",
                results
            )

    @final
    def cross_validate(self, calibrate: bool = False, **kwargs) :
        if self.dataset is None:
            raise ValueError("A dataset must be provided.")

        X_train, y_train = self.dataset.get_train_set(target=self.target)

        if calibrate:
            X_calib, y_calib = self.dataset.get_calib_set(target=self.target)
        else:
            X_calib, y_calib = None, None

        self.model_.cross_validate(
            X=X_train,
            y=y_train,
            calibrate=calibrate,
            calib_set=(X_calib, y_calib),
            **self.config["cv"],
            **kwargs
        )

    @final
    def predict(self, X, return_probs: bool = False):
        return self.model_.predict(X, return_probs=return_probs)

    @final
    def test(self, conf_matrix: bool = False) :
        if self.dataset is None:
            raise ValueError("A dataset must be provided.")

        metrics = self.config["test"].get("metrics", None)
        if metrics is None:
            metrics = ["acc", "precision", "recall", "f1-score", "auc", "pr_auc"]

        X_test, y_test = self.dataset.get_test_set(target=self.target)

        results = self.model_.test(
            X_test, y_test,
            metrics=metrics,
            conf_matrix=conf_matrix,
            calib_eval=self.config["test"].get("calib_eval", False),
            decision_threshold=self.config.get("decision_threshold", 0.5),
        )

        if self.active_run is not None:
            self._save_run_json(
                self.active_run /"evaluation.json",
                results
            )

    @final
    def check_high_confidence_bias(self, **kwargs) :
        if self.dataset is None:
            raise ValueError("A dataset must be provided.")

        X_test, y_test = self.dataset.get_test_set(target=self.target)

        res, gap = self.model_.check_high_confidence_bias(X_test, y_test, **kwargs)

        if res :
            print(f"\nThe model is over-confident (overconfidence gap : {gap:.2f}) and needs to be recalibrated.")
        else :
            print(f"\nThe model is honest in its predictions (overconfidence gap : {gap:.2f}) and does not need to be recalibrated.")

    @final
    def benchmark(self):
        if self.dataset is None:
            raise ValueError("A dataset must be provided.")

        X, _ = self.dataset.get_test_set(target=self.target)

        results = self.model_.benchmark(X, n_iterations=self.config["benchmark"].get("n_iterations", 100))

        if self.active_run is not None:
            self._save_run_json(
                self.active_run /"benchmark.json",
                results
            )
            
class MLOrchestrator:
    def __init__(self, session_args):
        self.session_args = session_args
        self.sess_models = self.session_args["model_config"].keys()

        self.ACTION_ORDER = ["optimize", "train", "test", "cross_validate", "check_high_confidence_bias", "benchmark"]
        self.ACTION_DEPENDENCIES = {
            "test": ["train"],
            "check_high_confidence_bias": ["train"],
            "benchmark": ["train"],
            "optimize": [],
            "cross_validate": [],
            "train": []
        }

        self.args = None
        self._from_cli()

        self.session = None
        self._build_session()

        if self.args.load is not None:
            self.model_path = os.path.join(self.args.root, self.args.load)
        else:
            self.model_path = None

        self.actions = []

        self.model_path = self.args.load

        if self.model_path is not None:
            self.actions = ["load"]

        self._resolve_actions(self.args.actions)

    @final
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
            choices=["optimize", "cross_validate", "train", "test", "check_high_confidence_bias", "benchmark"],
            help="Action"
        )
        parser.add_argument(
            "--calibrate", "-c", default=False, action="store_true", help="Calibrate model"
        )
        parser.add_argument(
            "--save", "-s", default=False, action="store_true", help="Save activation"
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
            "--get_features_importance", "-gfi", default=False, action="store_true", help="Get features importance"
        )
        parser.add_argument(
            "--clean_sandbox", "-cs", default=False, action="store_true", help="Clean sandbox"
        )
        parser.add_argument(
            "--HELP", "-H", default=False, action="store_true", help="Show help"
        )

        # args = dict(vars(parser.parse_args()))
        self.args = parser.parse_args()

    @final
    def _resolve_actions(self, requested_actions: list | None = None):
        if requested_actions is None:
            return

        actions = set()

        def add_action(action_):
            if action_ in actions:
                return

            if (
                    action_ in ["test", "check_high_confidence_bias", "benchmark"]
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

    def _build_session(self):
        config = {
            **self.session_args["config"],
            **self.session_args["model_config"][self.args.model]
        }

        self.session = MLSession(
            config=config,
            dataset=self.session_args["dataset"],
            target=self.session_args["target"],
            preprocess_options=config["preprocess"],
            metadata=self.session_args.get("metadata", None)
        )

    @final
    def run(self):
        if self.args.clean_sandbox:
            self.session.clean_sandbox()

        if self.args.HELP:
            self.help()
            return

        for action in self.actions:
            if action == "load":
                self.session.load(model_path=self.model_path)

            elif action == "optimize":
                self.session.optimize()

            elif action == "cross_validate":
                self.session.cross_validate(calibrate=self.args.calibrate)

            elif action == "train":
                self.session.train(
                    calibrate=self.args.calibrate,
                    save_model=self.args.save,
                    get_features_importance=self.args.get_features_importance
                )

            elif action == "test":
                self.session.test(conf_matrix=self.args.conf_matrix)

            elif action == "check_high_confidence_bias":
                self.session.check_high_confidence_bias()

            elif action == "benchmark":
                self.session.benchmark()

    @staticmethod
    def help():
        print("""
ML Session Helper

==================

Usage:
    python -m src.ml.ml_main [OPTIONS]
    
Available models:
    logreg -- Logistic Regression
    svm -- SVM
    random_forest -- Random Forest
    xgb -- XGBoost
    catb -- CatBoost
    
Available actions:
    optimize
    train
    cross-validate
    test -- requires train
    check_high_confidence_bias -- requires train
    benchmark -- requires train
    
Available formats:
    joblib
    skops
        
Options:
    -m, --model <model>:
        to select a model for the session
    
    -a, --actions <actions>:
        to resolve the provided actions
    
    -c, --calibrate:
        to calibrate the model
    
    -s, --save:
        to save the session results (metadata, train, test,...) 
        and the model 
    
    -r, --root <new_root>:
        to specify a new root for backup (default: runs/sandbox)
    
    -sn, --save_name <save_name>:
        to specify a save filename (optional)
        
    -sf, --save_format <save_format>:
        to specify a save format for model (default: joblib)
        
    -l, --load <model_path>:
        to load the specified model
    
    -cm, --conf_matrix:
        to enable the building of confusion matrix
    
    -gfi, --get_features_importance:
        to recover features importance
    
    -cs, --clean_sandbox:
        to clean sandbox
    """)