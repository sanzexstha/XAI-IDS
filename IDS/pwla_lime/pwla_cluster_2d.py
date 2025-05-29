import numpy as np
from sklearn.datasets import make_moons
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


class MultiRegionLimeBase(LimeBase):
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

            # plot_decision_boundary_local_model(
            #     neighborhood_data, labels_column, region_data, region_labels, region_idx
            # )

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

    def find_optimal_clusters(self, clustering_features, max_clusters=10):
        inertias = []
        
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(clustering_features)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        # You could implement elbow detection here
        # For now, you would need to plot and inspect:
        plt.plot(range(1, max_clusters + 1), inertias)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.show()
        
        return optimal_k  # You would need to determine this


class MultiRegionLimeTabularExplainer(LimeTabularExplainer):
    """LIME tabular explainer with multiple local linear models"""

    def __init__(self, training_data, **kwargs):
        n_regions = kwargs.pop("n_regions", 3)
        super().__init__(training_data, **kwargs)

        # Override base explainer with multi-region version
        if self.base.kernel_fn is not None:
            self.base = MultiRegionLimeBase(
                self.base.kernel_fn,
                verbose=self.base.verbose,
                random_state=self.random_state,
                n_regions=n_regions,
            )


def plot_decision_boundary_with_explanation(
    X, y, classifier, explainer, test_instance, pwla_explanation, lime_explanation
):
    """Plot the decision boundary and LIME explanation."""
    plt.figure(figsize=(15, 5))

    # Plot 1: Decision boundary
    plt.subplot(121)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.scatter(test_instance[0], test_instance[1], color="red", s=200, marker="*")
    plt.title("Decision Boundary with Test Instance")

    plt.show()

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

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


def plot_decision_boundary_local_model(X, y, Xp, yp, region_number):
    """Plot the decision boundary and LIME explanation."""
    plt.figure(figsize=(15, 5))

    # Plot 1: Decision boundary
    plt.subplot(121)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xp_min, xp_max = Xp[:, 0].min() - 0.5, Xp[:, 0].max() + 0.5
    yp_min, yp_max = Xp[:, 1].min() - 0.5, Xp[:, 1].max() + 0.5
    if xp_min < x_min:
        x_min = xp_min
    if xp_max < x_max:
        x_max = xp_max
    if yp_min < y_min:
        y_min = yp_min
    if yp_max < y_max:
        y_max = yp_max
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    plt.scatter(X[:, 0], X[:, 1], alpha=0.8)
    plt.scatter(Xp[:, 0], Xp[:, 1], alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Perturbed samples in region number {region_number}")

    plt.show()


def demonstrate_multi_region_lime():
    """Demonstrate the multi-region LIME implementation"""
    # Generate moon dataset
    X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Select an instance near decision boundary
    test_idx = 9 # 6
    test_instance = np.array([0.32, 0.10])
    print("prediction for this data point:", X_test[test_idx])
    print("Prediction: ",clf.predict_proba(test_instance.reshape(1, -1) ))

    feature_names = ["Feature 1", "Feature 2"]
    # Initialize multi-region LIME explainer
    explainer = MultiRegionLimeTabularExplainer(
        X_train,
        mode="classification",
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        n_regions=3,
        discretize_continuous=False,
        verbose=True,
    )

    predict_proba = clf.predict_proba

    # Original lime
    lime_explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        discretize_continuous=False,
    )

    lime_explanation = lime_explainer.explain_instance(
        test_instance, predict_proba, num_features=2, labels=[1]
    )

    # Generate explanation
    pwla_explanation = explainer.explain_instance(
        test_instance, predict_proba, num_features=2, num_samples=5000, labels=[1]
    )
    print(f"Explanation is for class 1")

    plot_decision_boundary_with_explanation(
        X, y, clf, explainer, test_instance, pwla_explanation, lime_explanation
    )
    plt.show()

    return explainer, exp, clf


if __name__ == "__main__":
    explainer, exp, clf = demonstrate_multi_region_lime()
