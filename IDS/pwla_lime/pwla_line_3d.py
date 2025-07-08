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
    def __init__(self, kernel_fn, verbose=False, random_state=None, n_regions=3):
        super().__init__(kernel_fn, verbose, random_state)
        self.n_regions = n_regions
        self.predict_fn = None
        self.best_model = None

    def divide_regions(self, data, predictions, distances, original_instance, num_features):
        """Divide the perturbed instances into regions using straight lines normal to the decision boundary."""
        pred_diffs = predictions[:, 1] - 0.5
        gradients = np.gradient(pred_diffs, axis=0)
        projections = np.dot(data - original_instance, gradients[:num_features])
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
            neighborhood_data, neighborhood_labels, distances, neighborhood_data[0], num_features=num_features
        )

        best_score = float("-inf")
        best_explanation = None
        best_model = None

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
                best_model = easy_model

                # all_coefs = easy_model.coef_[0]                                  # :contentReference[oaicite:0]{index=0}

                # # 3. Get the list of basis-function names (first one is the constant “1”)
                # basis_names = [str(bf) for bf in easy_model.basis_]             # :contentReference[oaicite:1]{index=1}

                # # 4. The intercept is the coefficient on the constant term
                # intercept = all_coefs[0]

                # # 5. The remaining features are the other basis functions
                # used_basis = basis_names[1:]
                # used_coefs  = all_coefs[1:]

                # # 6. Pair them up and sort by absolute importance
                # feat_coef_pairs = sorted(
                #     zip(used_basis, used_coefs),
                #     key=lambda x: abs(x[1]),
                #     reverse=True
                # )

                # # 7. Now you have exactly the same pieces you used before:
                # best_explanation = (
                #     intercept,
                #     feat_coef_pairs,
                #     score,
                #     local_pred
                # )
                best_explanation = (
                    easy_model.intercept_,
                    # easy_model.tree_.value[0][0][0],


                    sorted(
                        zip(used_features, easy_model.coef_ ),  #easy_model.coef_  easy_model.feature_importances_
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
        self.best_model = best_model

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


# def plot_decision_boundary_with_explanation3D(pwla_explanation, lime_explanation):
#     # def plot_decision_boundary_with_explanation(pwla_explanation):
#     """Plot the decision boundary and LIME explanation."""
#     # plt.figure(figsize=(15, 5))

#     # plt.subplot(122)
#     exp_plot_pwla = pwla_explanation.as_pyplot_figure()
#     # plt.title("PWLA Explanation")

#     # plt.subplot(122)
#     # exp_plot_lime = lime_explanation.as_pyplot_figure()
#     # plt.title("LIME Explanation")

#     plt.show()
#     return plt
from sklearn.metrics import mean_squared_error


def plot_decision_boundary_with_explanation3D(
    X, y, classifier, test_instance, pwla_explanation, lime_explanation, X_test
):

    i = 0
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
    ax.set_title(f"3D Swiss Roll Dataset")
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
# 2) Define the Local Approximation Accuracy function
def LocalApproxAccuracy(y_true, gm_model, perturbed_data):
    """
    Compute the RMSE between the classifier’s predictions (y_true)
    and each of the two GMM‐component probability streams, then
    return the smaller RMSE as the local approximation fidelity.
    """
    # predict_proba returns an array of shape (n_samples, n_components)
    gm_probs = gm_model.predict_proba(perturbed_data)

    # RMSE against component‑1 probabilities
    rmse_comp1 = mean_squared_error(y_true, gm_probs[:, 1])
    # RMSE against component‑0 probabilities
    rmse_comp0 = mean_squared_error(y_true, gm_probs[:, 0])

    return min(rmse_comp1, rmse_comp0)


def demonstrate_multi_region_lime():
    """Demonstrate the multi-region LIME implementation"""
    # Generate moon dataset
    # Generate Swiss Roll dataset
    # X, y = make_swiss_roll(n_samples=300, noise=0.1, random_state=42)
    # y = (y > 10).astype(int)

    # # Train/test split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )
    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf.fit(X_train, y_train)
    # print(X_train.shape)

    from sklearn.datasets import make_swiss_roll
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.mixture import GaussianMixture
    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.metrics import mean_squared_error, precision_score, recall_score, confusion_matrix, roc_curve
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from lemna import data_inverse, fidelity_test1, fidelity_test2, fidelity_test3

    # 1) Generate Swiss‑roll & binarize
    X, y = make_swiss_roll(n_samples=300, noise=0.1, random_state=42)
    y = (y > 10).astype(int)            # labels: 0 or 1

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) (Optional) wrap in DataFrame to reuse .values, feature_names, etc.
    feature_names = [f"X{i+1}" for i in range(X.shape[1])]
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['class'] = y_train

    # 4) Compute per‑feature means & maxs for your perturbation helper
    feature_means = df_train[feature_names].mean().tolist()
    feature_maxs  = df_train[feature_names].max().tolist()

    # 5) Create LIME explainer on raw NumPy matrix
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["0","1"],
        discretize_continuous=True,
        random_state=42
    )

    # 6) Train the RF classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # — now you can call your existing data_inverse, fidelity_test1/2/3, LocalApproxAccuracy, etc. —
    # e.g.:
    data, inverse = data_inverse(explainer, X_train[0], num_samples=1000)

    # Build and fit a GMM on the LIME‐perturbed neighborhood
    gm = GaussianMixture(n_components=2, covariance_type='diag', random_state=42)
    p_data_train = inverse
    p_data_test  = (clf.predict(inverse) >= 0.5).astype(int)
    gm.fit(p_data_train, p_data_test)

    # Compute feature‐counts for fidelity tests
    # (this mirrors your original loop over gm.predict == target)
    target = gm.predict(X_train[0].reshape(1,-1))[0]
    feature_count = np.zeros(X_train.shape[1], dtype=int)
    for i, row in enumerate(p_data_train):
        if gm.predict(row.reshape(1,-1))[0] == target:
            feature_count += (row != 0).astype(int)

    from sklearn.metrics import mean_squared_error

    # … after you’ve built p_data_train and fit gm …

    # 1) Get the classifier’s “true” predictions on the perturbed data
    y_pred = clf.predict(p_data_train)


    # 3) Call it and print
    local_rmse = LocalApproxAccuracy(y_pred, gm, p_data_train)
    print(f"Local Approximation Accuracy RMSE: {local_rmse:.4f}")



    # # Finally, run your fidelity and RMSE evaluations
    # o1, n1, n_prob = fidelity_test1(clf, df_train[feature_names], feature_count)
    # print("Fidelity Test 1 precision:", precision_score(o1, n1))
    # print("Fidelity Test 1 recall:   ", recall_score(o1, n1))

    # rmse_loc = LocalApproxAccuracy()
    # print("Local Approx RMSE:", rmse_loc)

# …and so on for fidelity_test2 / fidelity_test3 / ROC curves etc.


    feature_names = ["X1", "X2", "X3"]
    # Initialize multi-region LIME explainer
    pwla_explainer = MultiRegionLimeTabularExplainer3D(
        X_train,
        mode="classification",
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        n_regions=1,
        discretize_continuous=True,
        verbose=True,
    )

    # Select an instance near decision boundary
    # 15 16
    test_idx = 18
    test_instance =  np.array([12,5,8])
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

    N = 10
    P_true = clf.predict(X_test[:N])
    print("True prediction: ", P_true)
    P_pwla, P_lime = [], []

    for x in X_test[:N]:
        exp_pwla = pwla_explainer.explain_instance(
            x, clf.predict_proba, num_features=3, num_samples=5000
        )
        exp_lime = lime_explainer.explain_instance(
            x, clf.predict_proba, num_features=3, num_samples=5000
        )
        # print(exp_pwla.local_pred)
        P_pwla.append(exp_pwla.local_pred[1])
        P_lime.append(exp_lime.local_pred[1])

    P_pwla = np.array(P_pwla)
    P_lime = np.array(P_lime)

    rmse_pwla = np.sqrt(mean_squared_error(P_true, P_pwla))
    rmse_lime = np.sqrt(mean_squared_error(P_true, P_lime))

    print(f"Over {N} samples PWLA RMSE: {rmse_pwla:.4f}, LIME RMSE: {rmse_lime:.4f}")


    plot_decision_boundary_with_explanation3D(
        X, y, clf, test_instance, pwla_explanation, lime_explanation, X_test
    )
    # plot_decision_boundary_with_explanation3D(pwla_explanation, lime_explanation)
    plt.show()

    return pwla_explainer, pwla_explanation, clf


if __name__ == "__main__":
    pwla_explainer, pwla_explanation, clf = demonstrate_multi_region_lime()
