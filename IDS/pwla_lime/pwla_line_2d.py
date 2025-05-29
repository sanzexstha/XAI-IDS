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
import io


class MultiRegionLimeBase(LimeBase):
    def __init__(self, kernel_fn, verbose=False, random_state=None, n_regions=3):
        super().__init__(kernel_fn, verbose, random_state)
        self.n_regions = n_regions
        self.predict_fn = None

    def divide_regions(self, data, predictions, distances, original_instance):
        """Divide the perturbed instances into regions using straight lines normal to the decision boundary."""
        pred_diffs = predictions[:, 1] - 0.5
        gradients = np.gradient(pred_diffs, axis=0)
        projections = np.dot(data - original_instance, gradients[:2])
        region_size = len(projections) // self.n_regions
        sorted_indices = np.argsort(projections)
        regions = np.zeros_like(projections, dtype=int)
        for i in range(self.n_regions):
            start_idx = i * region_size
            end_idx = (
                (i + 1) * region_size if i < self.n_regions - 1 else len(projections)
            )
            regions[sorted_indices[start_idx:end_idx]] = i

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
        """Modified explanation method using straight-line regions"""
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]

        # Get regions using straight-line division
        regions = self.divide_regions(
            neighborhood_data, neighborhood_labels, distances, neighborhood_data[0]
        )

        best_score = float("-inf")
        best_explanation = None

        # Train model for each region
        for region_idx in range(self.n_regions):
            region_mask = regions == region_idx
            if np.sum(region_mask) < max(
                10, num_features + 1
            ):  # Skip if too few points
                print(f"Skipping region {region_idx} with {np.sum(region_mask)} points")
                continue

            # Get region-specific data
            region_data = neighborhood_data[region_mask]
            region_labels = labels_column[region_mask]
            region_weights = weights[region_mask]

            # plot_decision_boundary_local_model(
            #     neighborhood_data, labels_column, region_data, region_labels, region_idx
            # )

            print(f"Number of points in each region: {len(region_data)}")
            print(f"Number of unique regions: {self.n_regions}")

            # Select features for this region
            used_features = self.feature_selection(
                region_data,
                region_labels,
                region_weights,
                num_features,
                feature_selection,
            )

            # Train model
            if model_regressor is None:
                model_regressor = Ridge(
                    alpha=1, fit_intercept=True, random_state=self.random_state
                )

            easy_model = model_regressor
            easy_model.fit(
                region_data[:, used_features],
                region_labels,
                sample_weight=region_weights,
            )

            # Calculate score
            score = easy_model.score(
                region_data[:, used_features],
                region_labels,
                sample_weight=region_weights,
            )

            # Predict on original instance
            local_pred = easy_model.predict(
                neighborhood_data[0, used_features].reshape(1, -1)
            )

            print(f"Region {region_idx} score: {score}")

            if score > best_score:
                best_score = score
                best_explanation = (
                    easy_model.intercept_,
                    sorted(
                        zip(used_features, easy_model.coef_),
                        key=lambda x: np.abs(x[1]),
                        reverse=True,
                    ),
                    score,
                    local_pred,
                )

        if best_explanation is None:
            print("Falling back to original LIME - no good regions found")
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

    def __data_inverse(self, data_row, num_samples, sampling_method):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model
            sampling_method: 'gaussian' or 'lhs'
        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        is_sparse = sp.sparse.issparse(data_row)
        if is_sparse:
            num_cols = data_row.shape[1]
            data = sp.sparse.csr_matrix((num_samples, num_cols), dtype=data_row.dtype)
        else:
            num_cols = data_row.shape[0]
            data = np.zeros((num_samples, num_cols))
        categorical_features = range(num_cols)
        if self.discretizer is None:
            instance_sample = data_row
            scale = self.scaler.scale_
            mean = self.scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                mean = mean[non_zero_indexes]

            if sampling_method == "gaussian":
                data = self.random_state.normal(0, 1, num_samples * num_cols).reshape(
                    num_samples, num_cols
                )
                data = np.array(data)
            elif sampling_method == "lhs":
                data = lhs(num_cols, samples=num_samples).reshape(num_samples, num_cols)
                means = np.zeros(num_cols)
                stdvs = np.array([1] * num_cols)
                for i in range(num_cols):
                    data[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(data[:, i])
                data = np.array(data)
            else:
                warnings.warn(
                    """Invalid input for sampling_method.
                                 Defaulting to Gaussian sampling.""",
                    UserWarning,
                )
                data = self.random_state.normal(0, 1, num_samples * num_cols).reshape(
                    num_samples, num_cols
                )
                data = np.array(data)

            if self.sample_around_instance:
                data = data * scale + instance_sample
            else:
                data = data * scale + mean
            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix(
                        (num_samples, data_row.shape[1]), dtype=data_row.dtype
                    )
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(
                        range(
                            0,
                            len(non_zero_indexes) * (num_samples + 1),
                            len(non_zero_indexes),
                        )
                    )
                    data_1d_shape = data.shape[0] * data.shape[1]
                    data_1d = data.reshape(data_1d_shape)
                    data = sp.sparse.csr_matrix(
                        (data_1d, indexes, indptr),
                        shape=(num_samples, data_row.shape[1]),
                    )
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(
                values, size=num_samples, replace=True, p=freqs
            )
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row

        return data, inverse


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


def plot_decision_boundary_with_explanation(pwla_explanation, lime_explanation):
    # def plot_decision_boundary_with_explanation(pwla_explanation):
    """Plot the decision boundary and LIME explanation."""
    # plt.figure(figsize=(15, 5))

    # plt.subplot(122)
    exp_plot_pwla = pwla_explanation.as_pyplot_figure()
    # plt.title("PWLA Explanation")

    # plt.subplot(122)
    exp_plot_lime = lime_explanation.as_pyplot_figure()
    # plt.title("LIME Explanation")

    plt.show()
    return plt


def plot_decision_boundary_with_explanation(
    X, y, classifier, test_instance, pwla_explanation, lime_explanation
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

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

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
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    print(pwla_explanation.local_exp)
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
    X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    feature_names = ["Feature 1", "Feature 2"]
    # Initialize multi-region LIME explainer
    pwla_explainer = MultiRegionLimeTabularExplainer(
        X_train,
        mode="classification",
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        n_regions=3,
        discretize_continuous=True,
        verbose=True,
    )

    # Select an instance near decision boundary
    test_idx = 9 # 6
    test_instance = np.array([0.42, 0.15])
    print("prediction for this data point:", test_instance)
    print("Prediction: ",clf.predict_proba(test_instance.reshape(1, -1) ))

    predict_proba = clf.predict_proba

    # Generate explanation
    pwla_explanation = pwla_explainer.explain_instance(
        test_instance, predict_proba, num_features=2, num_samples=5000, labels=[1]
    )

    # Original lime

    lime_explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        discretize_continuous=True,
    )

    lime_explanation = lime_explainer.explain_instance(
        test_instance, predict_proba, num_features=2, labels=[1]
    )

    plot_decision_boundary_with_explanation(
        X, y, clf, test_instance, pwla_explanation, lime_explanation
    )
    # plot_decision_boundary_with_explanation(pwla_explanation)
    plt.show()

    return pwla_explainer, pwla_explanation, clf


if __name__ == "__main__":
    pwla_explainer, pwla_explanation, clf = demonstrate_multi_region_lime()
