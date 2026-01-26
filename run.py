from sklearn.pipeline import Pipeline
from models.random_forest import build_model_RF
from common.preprocess import build_preprocessor


pipeline = Pipeline([("preprocess", build_preprocessor()), 
                    ("model",build_model_RF)
                    ])


pipeline.fit(X_train, y_train)