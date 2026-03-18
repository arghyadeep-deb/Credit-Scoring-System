from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


class CreditScoreNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)


@dataclass
class HybridCreditScorer:
    label_encoder: Any
    feature_columns: list[str]
    categorical_cols: list[str]
    scaler: Any
    blend_weights: tuple[float, float]
    calibrated_ensemble: Any
    nn_model: CreditScoreNN
    rejection_label: str
    one_hot_mapping: dict[str, list[str]]
    numeric_input_fields: list[str]
    shap_explainer: Any | None

    @classmethod
    def load(cls, artifacts_dir: Path) -> "HybridCreditScorer":
        with (artifacts_dir / "hybrid_preprocessing.pkl").open("rb") as f:
            bundle = cloudpickle.load(f)

        label_encoder = bundle["label_encoder"]
        feature_columns = bundle["feature_columns"]
        categorical_cols = bundle.get("categorical_cols", [])
        scaler = bundle["scaler"]
        blend_weights = bundle.get("blend_weights", (0.6, 0.4))
        calibrated_ensemble = bundle["calibrated_ensemble"]

        nn_model = CreditScoreNN(input_dim=len(feature_columns), num_classes=len(label_encoder.classes_))
        state_dict = torch.load(artifacts_dir / "nn_40_state_dict.pt", map_location="cpu")
        nn_model.load_state_dict(state_dict)
        nn_model.eval()

        rejection_label = label_encoder.classes_[-1]
        one_hot_mapping = cls._build_one_hot_mapping(feature_columns, categorical_cols)
        one_hot_cols = {item for values in one_hot_mapping.values() for item in values}
        numeric_input_fields = [col for col in feature_columns if col not in one_hot_cols]

        shap_explainer = None
        if shap is not None:
            try:
                fitted_xgb = calibrated_ensemble.estimator.named_estimators_["xgb"]
                shap_explainer = shap.TreeExplainer(fitted_xgb)
            except Exception:
                shap_explainer = None

        return cls(
            label_encoder=label_encoder,
            feature_columns=feature_columns,
            categorical_cols=categorical_cols,
            scaler=scaler,
            blend_weights=blend_weights,
            calibrated_ensemble=calibrated_ensemble,
            nn_model=nn_model,
            rejection_label=rejection_label,
            one_hot_mapping=one_hot_mapping,
            numeric_input_fields=numeric_input_fields,
            shap_explainer=shap_explainer,
        )

    @staticmethod
    def _build_one_hot_mapping(
        feature_columns: list[str], categorical_cols: list[str]
    ) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for cat_col in categorical_cols:
            prefix = f"{cat_col}_"
            cols = [col for col in feature_columns if col.startswith(prefix)]
            mapping[cat_col] = cols
        return mapping

    def _validate_and_vectorize(self, raw_features: dict[str, Any]) -> pd.DataFrame:
        vec = {col: 0.0 for col in self.feature_columns}
        unknown = []

        for key, value in raw_features.items():
            if key not in vec:
                unknown.append(key)
                continue
            try:
                vec[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature '{key}' must be numeric.") from exc

        if unknown:
            raise ValueError(f"Unknown feature(s): {', '.join(sorted(unknown))}")

        return pd.DataFrame([vec], columns=self.feature_columns, dtype=float)

    def raw_to_encoded(self, raw_payload: dict[str, Any]) -> dict[str, float]:
        encoded = {col: 0.0 for col in self.feature_columns}

        for col in self.numeric_input_fields:
            if col in raw_payload and raw_payload[col] is not None:
                try:
                    encoded[col] = float(raw_payload[col])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Numeric field '{col}' must be numeric.") from exc

        for cat_col, dummy_cols in self.one_hot_mapping.items():
            raw_val = raw_payload.get(cat_col)
            if raw_val is None:
                continue

            val = str(raw_val).strip()
            if val in ("", "__baseline__"):
                continue

            target_col = f"{cat_col}_{val}"
            if target_col in dummy_cols:
                encoded[target_col] = 1.0
                continue

            allowed = [c.replace(f"{cat_col}_", "", 1) for c in dummy_cols]
            raise ValueError(
                f"Invalid category value '{val}' for '{cat_col}'. Allowed: {allowed} plus __baseline__."
            )

        return encoded

    def get_raw_schema(self) -> dict[str, Any]:
        categorical_fields: dict[str, list[str]] = {}
        for cat_col, dummy_cols in self.one_hot_mapping.items():
            options = ["__baseline__"] + [c.replace(f"{cat_col}_", "", 1) for c in dummy_cols]
            categorical_fields[cat_col] = options

        return {
            "numeric_fields": self.numeric_input_fields,
            "categorical_fields": categorical_fields,
        }

    def _extract_top_shap_drivers(
        self, x_df: pd.DataFrame, pred_idx: int, top_k: int = 3
    ) -> list[str] | None:
        if self.shap_explainer is None:
            return None

        try:
            shap_obj = self.shap_explainer.shap_values(x_df)
            arr = np.asarray(shap_obj)

            if isinstance(shap_obj, list):
                class_values = np.asarray(shap_obj[pred_idx])
                row_values = class_values[0] if class_values.ndim == 2 else class_values
            elif arr.ndim == 3:
                if arr.shape[2] > pred_idx:
                    row_values = arr[0, :, pred_idx]
                else:
                    row_values = arr[pred_idx, 0, :]
            elif arr.ndim == 2:
                row_values = arr[0]
            else:
                return None

            contrib = pd.Series(np.abs(row_values), index=x_df.columns)
            return contrib.sort_values(ascending=False).head(top_k).index.tolist()
        except Exception:
            return None

    def _top_local_reasons(self, x_df: pd.DataFrame, top_k: int = 3) -> list[str]:
        x = x_df.iloc[0].values
        mean = getattr(self.scaler, "mean_", np.zeros_like(x))
        scale = getattr(self.scaler, "scale_", np.ones_like(x))
        safe_scale = np.where(scale == 0, 1.0, scale)
        z = np.abs((x - mean) / safe_scale)

        pairs = sorted(zip(self.feature_columns, z), key=lambda t: t[1], reverse=True)
        selected = [name for name, score in pairs if score > 0][:top_k]

        if not selected:
            selected = self.feature_columns[:top_k]

        return selected

    @staticmethod
    def _humanize_feature_name(name: str) -> str:
        mapping = {
            "enq_L3m": "recent loan inquiries",
            "num_times_delinquent": "past delinquencies",
            "time_since_recent_payment": "recent repayment behavior",
            "time_since_recent_enq": "time since last inquiry",
            "NETMONTHLYINCOME": "monthly income",
            "AGE": "age",
            "Age_Oldest_TL": "age of oldest credit line",
            "Age_Newest_TL": "age of newest credit line",
            "Tot_Missed_Pmnt": "missed payments",
        }
        if name in mapping:
            return mapping[name]

        cleaned = name.replace("_", " ").replace("pct", "percentage").strip()
        return cleaned.lower()

    def _build_human_reasoning(
        self,
        decision: str,
        confidence: float,
        reasons: list[str],
    ) -> str:
        readable = [self._humanize_feature_name(item) for item in reasons]

        if len(readable) >= 3:
            reason_part = f"{readable[0]}, {readable[1]}, and {readable[2]}"
        elif len(readable) == 2:
            reason_part = f"{readable[0]} and {readable[1]}"
        elif len(readable) == 1:
            reason_part = readable[0]
        else:
            reason_part = "the provided financial profile"

        if decision == "Accepted":
            return (
                "The application is accepted. "
                f"The most important factors for this result were {reason_part}. "
                "In simple terms, these factors matched patterns that are usually seen in lower-risk applicants. "
                f"Model confidence is {confidence:.1%}, which means the model is fairly sure about this prediction, but it is not a guarantee."
            )

        return (
            "The application is rejected. "
            f"The most important factors for this result were {reason_part}. "
            "In simple terms, these factors matched patterns that are usually seen in higher-risk applicants. "
            f"Model confidence is {confidence:.1%}, which means the model is fairly sure about this prediction, but it is not a guarantee."
        )

    def _predict_from_vector(self, x_df: pd.DataFrame) -> dict[str, Any]:
        ensemble_probs = self.calibrated_ensemble.predict_proba(x_df)

        nn_input = torch.FloatTensor(self.scaler.transform(x_df))
        with torch.no_grad():
            nn_probs = torch.softmax(self.nn_model(nn_input), dim=1).numpy()

        w_ens, w_nn = self.blend_weights
        final_probs = (w_ens * ensemble_probs) + (w_nn * nn_probs)

        pred_idx = int(np.argmax(final_probs[0]))
        pred_label = str(self.label_encoder.inverse_transform([pred_idx])[0])
        confidence = float(np.max(final_probs[0]))
        decision = "Rejected" if pred_label == self.rejection_label else "Accepted"

        reasons = self._extract_top_shap_drivers(x_df, pred_idx=pred_idx, top_k=3)
        if not reasons:
            reasons = self._top_local_reasons(x_df)

        reason_text = self._build_human_reasoning(
            decision=decision,
            confidence=confidence,
            reasons=reasons,
        )

        class_probs = {
            str(cls_name): float(prob)
            for cls_name, prob in zip(self.label_encoder.classes_, final_probs[0])
        }

        return {
            "decision": decision,
            "predicted_label": pred_label,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "top_drivers": reasons,
            "reasoning": reason_text,
            "rejection_label": str(self.rejection_label),
        }

    def predict_one(self, raw_features: dict[str, Any]) -> dict[str, Any]:
        x_df = self._validate_and_vectorize(raw_features)
        return self._predict_from_vector(x_df)

    def predict_one_raw(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        encoded = self.raw_to_encoded(raw_payload)
        return self.predict_one(encoded)

    def predict_batch(
        self, records: list[dict[str, Any]], input_type: str = "encoded"
    ) -> list[dict[str, Any]]:
        output = []
        for idx, record in enumerate(records):
            try:
                if input_type == "raw":
                    pred = self.predict_one_raw(record)
                else:
                    pred = self.predict_one(record)
            except Exception as exc:
                pred = {
                    "decision": "Error",
                    "predicted_label": "N/A",
                    "confidence": 0.0,
                    "class_probabilities": {},
                    "top_drivers": [],
                    "reasoning": str(exc),
                    "rejection_label": str(self.rejection_label),
                }

            pred["row_index"] = idx
            output.append(pred)

        return output
