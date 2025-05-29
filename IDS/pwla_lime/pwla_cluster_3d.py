import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from lime.lime_tabular import LimeTabularExplainer, TableDomainMapper
from lime.lime_base import LimeBase
from lime import explanation
import scipy as sp
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MultiRegionLimeBase3D(LimeBase):
    """Modified LIME base class that supports multiple local linear models"""

    def __init__(self, kernel_fn, verbose=False, random_state=None, n_regions=3):
        super().__init__(kernel_fn, verbose, random_state)
        self.n_regions = n_regions

    def divide_regions(self, data, predictions, distances, original_instance):
        """Divide the perturbed instances into regions"""
        # Calculate gradients to identify decision boundary
        pred_diffs = np.abs(predictions[:, 1] - 0.5)  # Distance from decision boundary

        # Combine features for clustering
        clustering_features = np.column_stack(
            [data, pred_diffs.reshape(-1, 1), distances.reshape(-1, 1)]
        )

        # Scale features for clustering
        clustering_features = (
            clustering_features - clustering_features.mean(axis=0)
        ) / clustering_features.std(axis=0)

        # Perform clustering
        kmeans = KMeans(n_clusters=self.n_regions, random_state=self.random_state)
        regions = kmeans.fit_predict(clustering_features)

        return regions

    def explain_instance_with_data(
        self,
        neighborhood_data,
        neighborhood_labels,
        distances,
        label,
        num_features,
        feature_selection="auto",
        model_regressor=None,
    ):
        """Modified explanation method using multiple local models"""
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]

        # Get regions considering decision boundary
        regions = self.divide_regions(
            neighborhood_data, neighborhood_labels, distances, neighborhood_data[0]
        )

        best_score = float("-inf")
        best_explanation = None

        # Loop through regions and plot the perturbed points for each region
        for region_idx in range(self.n_regions):
            region_mask = regions == region_idx
            if np.sum(region_mask) < max(
                10, num_features + 1
            ):  # Skip if too few points
                continue

            # Get region-specific data
            region_data = neighborhood_data[region_mask]
            region_labels = labels_column[region_mask]
            region_weights = weights[region_mask]

            # Train model
            if model_regressor is None:
                model_regressor = Ridge(
                    alpha=1, fit_intercept=True, random_state=self.random_state
                )

            easy_model = model_regressor
            easy_model.fit(
                region_data[:, :num_features],
                region_labels,
                sample_weight=region_weights,
            )

            # Calculate score
            score = easy_model.score(
                region_data[:, :num_features],
                region_labels,
                sample_weight=region_weights,
            )

            # Predict on original instance
            local_pred = easy_model.predict(
                neighborhood_data[0, :num_features].reshape(1, -1)
            )

            print(f"Cluster {region_idx}:")
            print(f"  Samples in cluster: {len(region_data)}")
            print(f"  Local score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_explanation = (
                    easy_model.intercept_,
                    sorted(
                        zip(range(num_features), easy_model.coef_),
                        key=lambda x: np.abs(x[1]),
                        reverse=True,
                    ),
                    score,
                    local_pred,
                )

        if best_explanation is None:
            print(f"Upao u original lime")
            # Fallback to original LIME if no good regions found
            return super().explain_instance_with_data(
                neighborhood_data,
                neighborhood_labels,
                distances,
                label,
                num_features,
                feature_selection,
                model_regressor,
            )

        return best_explanation


class MultiRegionLimeTabularExplainer3D(LimeTabularExplainer):
    """LIME tabular explainer with multiple local linear models"""

    def __init__(self, training_data, **kwargs):
        n_regions = kwargs.pop("n_regions", 3)
        super().__init__(training_data, **kwargs)

        # Override base explainer with multi-region version
        if self.base.kernel_fn is not None:
            self.base = MultiRegionLimeBase3D(
                self.base.kernel_fn,
                verbose=self.base.verbose,
                random_state=self.random_state,
                n_regions=n_regions,
            )


def plot_decision_boundary_with_explanation3D(
    X, y, classifier, test_instance, pwla_explanation, lime_explanation
):

    # Plot the dataset with the test instance
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm, s=10)
    ax.scatter(
        test_instance[0],
        test_instance[1],
        test_instance[2],
        color="red",
        s=100,
        label="Test Instance",
    )
    ax.set_title("3D Swiss Roll Dataset with Test Instance")
    ax.legend()
    # plt.show()

    # Function to extract and plot explanation data
    def plot_explanation(explanation, ax, title):
        # Extract feature labels and their weights
        exp = explanation.as_list()
        features = [item[0] for item in exp]
        weights = [item[1] for item in exp]

        # Create a horizontal bar chart
        ax.barh(features, weights, color=["red" if w < 0 else "green" for w in weights])
        ax.set_title(title)
        ax.invert_yaxis()  # Ensure the top feature is at the top
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel("Feature Importance", fontsize=12)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # PWLA Explanation
    plot_explanation(pwla_explanation, axes[0], "PWLA Explanation")

    # LIME Explanation
    plot_explanation(lime_explanation, axes[1], "LIME Explanation")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    return plt


def demonstrate_multi_region_lime():
    """Demonstrate the multi-region LIME implementation"""
    # Generate moon dataset
    # Generate Swiss Roll dataset
    X, y = make_swiss_roll(n_samples=300, noise=0.1, random_state=42)
    y = (y > 10).astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    feature_names = ["X1", "X2", "X3"]
    # Initialize multi-region LIME explainer
    pwla_explainer = MultiRegionLimeTabularExplainer3D(
        X_train,
        mode="classification",
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        n_regions=3,
        discretize_continuous=True,
        verbose=True,
    )

    # Select an instance near decision boundary
    test_idx = 18
    test_instance =  np.array([12, 5, 8])
    print(test_instance)
    print("Prediction: ",clf.predict_proba(test_instance.reshape(1, -1)  ))

    predict_proba = clf.predict_proba

    # Generate explanation
    pwla_explanation = pwla_explainer.explain_instance(
        test_instance, predict_proba, num_features=3, num_samples=5000, labels=[1]
    )

    # Original lime

    lime_explainer = LimeTabularExplainer(
        X_train,
        mode="classification",
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        discretize_continuous=True,
    )

    lime_explanation = lime_explainer.explain_instance(
        test_instance, predict_proba, num_features=3, labels=[1]
    )

    plot_decision_boundary_with_explanation3D(
        X, y, clf, test_instance, pwla_explanation, lime_explanation
    )
    # plot_decision_boundary_with_explanation3D(pwla_explanation, lime_explanation)
    plt.show()

    return pwla_explainer, pwla_explanation, clf


if __name__ == "__main__":
    pwla_explainer, pwla_explanation, clf = demonstrate_multi_region_lime()
