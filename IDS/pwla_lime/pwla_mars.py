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
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(
            neighborhood_data, labels_column, weights, num_features, feature_selection
        )
        if model_regressor is None:
            model_regressor = Ridge(
                alpha=1, fit_intercept=True, random_state=self.random_state
            )
        easy_model = model_regressor
        easy_model.fit(
            neighborhood_data[:, used_features], labels_column, sample_weight=weights
        )
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features], labels_column, sample_weight=weights
        )

        local_pred = easy_model.predict(
            neighborhood_data[0, used_features].reshape(1, -1)
        )

        print(f"LIME original local pred of explanation is: {local_pred}")
        print(f"LIME original local score of explanation is: {prediction_score}")

        if self.verbose:
            print("Intercept", easy_model.intercept_)
            print(
                "Prediction_local",
                local_pred,
            )
            print("Right:", neighborhood_labels[0, label])

        all_coefs = easy_model.coef_[0]

        # 3. Get the list of basis-function names (first one is the constant “1”)
        basis_names = [str(bf) for bf in easy_model.basis_]

        # 4. The intercept is the coefficient on the constant term
        intercept = all_coefs[0]

        # 5. The remaining features are the other basis functions
        used_basis = basis_names[1:]
        used_coefs  = all_coefs[1:]

        # 6. Pair them up and sort by absolute importance
        feat_coef_pairs = sorted(
            zip(used_basis, used_coefs),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # 7. Now you have exactly the same pieces you used before:
        return (
            intercept,
            feat_coef_pairs,
            prediction_score,
            local_pred
        )



class MultiRegionLimeTabularExplainer3D(LimeTabularExplainer):
    """LIME tabular explainer with multiple local linear models"""

    def __init__(self, training_data, **kwargs):
        super().__init__(training_data, **kwargs)

        # Override base explainer with multi-region version
        if self.base.kernel_fn is not None:
            self.base = MultiRegionLimeBase3D(
                self.base.kernel_fn,
                verbose=self.base.verbose,
                random_state=self.random_state,
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





