from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib
import dvc.api

# Step 2: Load a simple dataset
X, y = load_iris(return_X_y=True)

# Step 3: Train a simple logistic regression model (our slim model)
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Step 4: Save the trained model to disk
joblib.dump(model, 'model.joblib')

from dvclive import Live
with Live() as live:
    live.log_artifact(
        str("model.joblib"),
        type="model",
        name="test-model",
        desc="This is a Computer Vision (CV) model that's segmenting out swimming pools from satellite images.",
        labels=["cv", "segmentation"],
    )

