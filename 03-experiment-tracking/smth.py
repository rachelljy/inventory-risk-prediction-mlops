import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.utils.class_weight import compute_class_weight

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, SMOTENC

from xgboost import XGBClassifier


def build_logit_pipeline(numerical_features, categorical_features):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ])

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ])
    return pipeline


def build_rf_pipeline(X_train, numerical_features, categorical_features):
    X_train_rf = X_train.copy()

    for col in categorical_features:
        X_train_rf[col] = X_train_rf[col].astype("object")

    cat_indices = [X_train_rf.columns.get_loc(col) for col in categorical_features]

    smote_nc = SMOTENC(
        categorical_features=cat_indices,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    rf_pipeline = ImbPipeline(steps=[
        ("smote", smote_nc),
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ))
    ])

    return rf_pipeline


def build_xgb_pipeline(numerical_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    xgb_pipeline = SkPipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss"
        ))
    ])

    return xgb_pipeline


def get_xgb_sample_weights(y_train):
    classes = np.unique(y_train)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([class_weight_dict[y] for y in y_train])

    return sample_weights, class_weight_dict