import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class DelayModel:
    def __init__(self):
        """Initialize the model and load necessary resources."""
        self._model = None
        self._feature_names = None
        self._scaler = None
        self.X_test = None
        self.y_test = None

        # Define paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "model.joblib")
        self.feature_columns_path = os.path.join(base_dir, "feature_columns.joblib")

        # Try to load model and feature columns if available
        try:
            self._model = joblib.load(self.model_path)
            self._feature_names = joblib.load(self.feature_columns_path)
            print("Model and feature columns loaded.")
        except FileNotFoundError:
            print("Model or feature columns not found. Please train the model first.")

    def preprocess(self, data, target_column=None):
        """Preprocess the input data for training or serving predictions."""
        data = data.copy()

        data['high_season'] = data['Fecha-I'].apply(self._is_high_season)
        data['min_diff'] = (pd.to_datetime(data['Fecha-O']) - pd.to_datetime(data['Fecha-I'])).dt.total_seconds() / 60
        data['period_day'] = data['Fecha-I'].apply(self._get_period_day)
        if target_column:
            data['delay'] = (data['min_diff'] > 15).astype(int)

        # Convert MES to a categorical column
        # data['MES'] = data['MES'].astype(str)
        data['MES'] = data['MES'].astype(str)
        data['OPERA'] = data['OPERA'].astype(str)
        data['SIGLADES'] = data['SIGLADES'].astype(str)


        # Define features and target
        features = data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM']]
        target = data[[target_column]] if target_column else None  # Asegurarse de devolver un DataFrame

        # One-hot encoding for categorical variables
        features = pd.get_dummies(features, drop_first=True)

        # Define expected feature columns (FEATURES_COLS)
        FEATURES_COLS = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]

        # Agregar columnas faltantes con valores de 0
        for col in FEATURES_COLS:
            if col not in features.columns:
                features[col] = 0

        # Eliminar columnas adicionales
        features = features[FEATURES_COLS]

        # Escalar características
        if self._scaler is None:
            self._scaler = StandardScaler()
            features = self._scaler.fit_transform(features)
        else:
            features = self._scaler.transform(features)

        if target is not None:
            return pd.DataFrame(features, columns=FEATURES_COLS), target
        else:
            return pd.DataFrame(features, columns=FEATURES_COLS)


    def fit(self, features, target):
        """Train the model and save necessary resources."""
        X_train, self.X_test, y_train, self.y_test = train_test_split(features, target, test_size=0.33, random_state=42)

        # Handle class imbalance with SMOTE
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        print("Clase 0 después de SMOTE:", len(y_train[y_train == 0]))
        print("Clase 1 después de SMOTE:", len(y_train[y_train == 1]))

        # Define XGBoost parameters
        xgb_params = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 200],
            'scale_pos_weight': [5, 10, 15]
            # 'scale_pos_weight': [len(y_train[y_train == 0]) / len(y_train[y_train == 1])]
        }

        # Perform GridSearchCV
        grid = GridSearchCV(XGBClassifier(random_state=1), xgb_params, scoring='recall', cv=5, verbose=1)
        grid.fit(X_train, y_train)

        # Save the best model
        self._model = grid.best_estimator_

        # Save model and feature columns
        joblib.dump(self._model, self.model_path)
        joblib.dump(self._feature_names, self.feature_columns_path)
        print(f"Saving model to: {self.model_path}")
        print(f"Saving feature columns to: {self.feature_columns_path}")
        print("Model and feature columns saved.")

        # Plot feature importance
        self._plot_feature_importance(self._model)

    def predict(self, data=None, features=None):
        """Predict using the trained model."""
        if self._model is None:
            raise ValueError("Model not loaded. Train the model first.")

        if features is None:
            features = self.preprocess(data)

        predictions = self._model.predict(features)
        return predictions.tolist()

    def evaluate(self):
        """Evaluate the model on test data."""
        if self._model is None or self.X_test is None or self.y_test is None:
            raise ValueError("Model or test data not available. Train the model first.")

        predictions = self._model.predict(self.X_test)
        print("Classification Report:")
        print(classification_report(self.y_test, predictions, zero_division=0))  # Añade zero_division=0
        print("Confusion Matrix:")
        cm = confusion_matrix(self.y_test, predictions)
        print(cm)

        # Plot confusion matrix
        self._plot_confusion_matrix(cm)

        # Plot ROC curve
        y_proba = self._model.predict_proba(self.X_test)[:, 1]
        self._plot_roc_curve(self.y_test, y_proba)

    def _plot_feature_importance(self, model):
        importance = model.feature_importances_
        feature_names = self._feature_names

        # Validar que las longitudes coincidan
        if len(feature_names) != len(importance):
            print("Advertencia: Mismatch en longitudes de features.")
            feature_names = feature_names[:len(importance)]  # Ajustar al tamaño de importance

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importance, color='purple', alpha=0.7)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.show()


    def _plot_confusion_matrix(self, cm):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def _plot_roc_curve(self, y_test, y_proba):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

    @staticmethod
    def _is_high_season(date_str):
        date = pd.to_datetime(date_str)
        return int(
            (date.month == 12 and date.day >= 15) or
            (date.month == 3 and date.day <= 3) or
            (date.month == 7 and date.day <= 31) or
            (date.month == 9 and date.day <= 30)
        )

    @staticmethod
    def _get_period_day(date_str):
        hour = pd.to_datetime(date_str).hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 19:
            return 'afternoon'
        else:
            return 'night'

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../data/data.csv")
    data = pd.read_csv(data_path, low_memory=False)

    model = DelayModel()

    features, target = model.preprocess(data, target_column="delay")
    model.fit(features, target)
    model.evaluate()

    # Test predictions on new data
    new_data = data.iloc[:5]
    features = model.preprocess(new_data)
    predictions = model.predict(features=features)
    print("Predictions:", predictions)
