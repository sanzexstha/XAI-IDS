# %%
from enum import Enum

from pyearth import Earth

import numpy as np
from sklearn.utils import Bunch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn import metrics
import math
import lime
import lime.lime_tabular
from sklearn.linear_model import Ridge
import random
from scipy.special import gammainccinv, erfinv
from sklearn.metrics import pairwise_distances
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.stats import chi2, norm
from numpy.random import default_rng
from types import LambdaType


class Dataset(Enum):
  GENERATED = 'Generated'
  BLOBS = 'Blobs'
  IRIS2D = 'Iris (2D)'
  IRIS = 'Iris'
  BC = 'Breast cancer'
  PIMA = 'Pima Indian'
  WINE = 'Wine'

  @property
  def instance(self):
    if self == Dataset.GENERATED:
      amount = 2500
      data = np.random.uniform(0,[3,10],(amount,2))
      return Bunch(
        data=data,
        feature_names=['feature 1', 'feature 2'],
        target_names=['yellow', 'blue'],
        target=(np.logical_and(data[:,0]>1, data[:,1]>2)).astype(int)
      )

    if self == Dataset.BLOBS:
      from sklearn.datasets import make_blobs
      amount = 250
      X,y = make_blobs(amount, n_features = 3, centers = 4)
      return Bunch(
        data=X,
        feature_names=['feature 1', 'feature 2', 'feature 3'],
        target_names=['yellow', 'blue'],
        target=(y>1).astype(int)
      )

    if self == Dataset.IRIS2D:
      from sklearn.datasets import load_iris
      dataset = load_iris()
      indexes = np.array([0,1])
      dataset.data = dataset.data[:, indexes]
      dataset.feature_names = np.array(dataset.feature_names)[indexes]
      return dataset

    if self == Dataset.IRIS:
      from sklearn.datasets import load_iris
      return load_iris()

    if self == Dataset.BC:
      from sklearn.datasets import load_breast_cancer
      dataset = load_breast_cancer()
      return dataset

    if self == Dataset.WINE:
      from sklearn.datasets import load_wine
      dataset = load_wine()
      return dataset

    if self == Dataset.PIMA:
      from sklearn.datasets import load_diabetes
      dataset = load_diabetes()
      dataset.target = (dataset.target > 200).astype(int)
      return dataset


class Classifier(Enum):
  DT = 'Decision tree'
  RF = 'Random Forest'
  MLP = 'Perceptron'
  GNB = 'Naive Bayes'
  SVM = 'Support vector machine'
  QDA = 'Quadratic Discriminant Analysis'

  @property
  def instance(self):
    if self == Classifier.DT:
      from sklearn.tree import DecisionTreeClassifier
      return DecisionTreeClassifier(random_state=1)
    if self == Classifier.RF:
      from sklearn.ensemble import RandomForestClassifier
      return RandomForestClassifier(random_state=1, n_estimators=200)
    elif self == Classifier.MLP:
      from sklearn.neural_network import MLPClassifier
      return MLPClassifier(alpha=0.1, hidden_layer_sizes=(100,100,100))
    elif self == Classifier.GNB:
      from sklearn.naive_bayes import GaussianNB
      return GaussianNB()
    elif self == Classifier.SVM:
      from sklearn.svm import SVC
      return SVC(probability=True)
    elif self == Classifier.QDA:
      from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
      return QuadraticDiscriminantAnalysis()

# %%
from sklearn.preprocessing import normalize
from bisect import bisect
from scipy.optimize import newton
from functools import partial
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class LemonExplainer(object):
  """
  Intantiates the explainer.
  """
  def __init__(self, training_data, distance_kernel=None, sample_size = 5000, radius_max=1, random_state=None):
    self.random_state = check_random_state(random_state)
    np.random.seed(random_state)

    self.training_data = training_data
    self.scaler = StandardScaler(with_mean=False)
    self.scaler.fit(training_data)

    # Create hypersphere samples. The sphere is only computed once for performance and stability,
    # but it would be better to resample the sphere every time `explain_instance` is called.
    # I checked, this does not affect the results in any way.
    dimensions = training_data.shape[1]

    if distance_kernel is None:
      self.distance_kernel = np.vectorize(lambda x: x ** (1 / dimensions))
    else:
      self.distance_kernel = np.vectorize(self._transform(distance_kernel, dimensions, radius_max=radius_max))

    sphere = np.random.normal(size=(sample_size, dimensions))
    sphere = normalize(sphere)
    sphere *= self.distance_kernel(np.random.uniform(size=sample_size)).reshape(-1,1)

    self.sphere = sphere

  @property
  def surrogate(self):
    try:
      return self._surrgate
    except AttributeError:
      self._surrogate = Ridge(alpha=0, fit_intercept=True, normalize=True, random_state=self.random_state)
      return self._surrogate

  def explain_instance(self, instance, predict_fn, labels=(1,), surrogate=None):
    surrogate = surrogate or self.surrogate

    # Create transfer dataset by perturbing the original instance with the hypersphere samples
    X_transfer = self.scaler.inverse_transform(self.sphere) + np.array([instance])
    y_transfer = predict_fn(X_transfer)

    def explain_label(label):
      surrogate.fit(X_transfer, y_transfer[:,label])
      score = surrogate.score(X_transfer, y_transfer[:,label])
      return (surrogate.coef_[1:], score)   #surrogate.tree_.value[0][0][0]

    return [explain_label(label) for label in labels]

  def _transform(self, kernel, dimensions, sample_size = 1000, radius_max=1):
    """
    Inverse transform sampling
    """
    cdf_samples = np.array([kernel(x)*(x**(dimensions-1)) for x in np.linspace(0, radius_max, sample_size)])
    cdf_samples = np.cumsum(cdf_samples)
    cdf_samples /= cdf_samples[-1]
    return lambda y: radius_max * (bisect(cdf_samples, y) / sample_size)


def uniform_kernel(x):
    return 1


def gaussian_kernel(x, kernel_width):
    # https://github.com/marcotcr/lime/blob/fd7eb2e6f760619c29fca0187c07b82157601b32/lime/lime_tabular.py#L251
    return np.sqrt(np.exp(-(x ** 2) / kernel_width ** 2))


def sqcos_kernel(x):
    return np.cos(x)**2


def trapezoid_kernel(x, a, b):
    if 0 <= x and  x <= a:
        return (2 / (a + b))
    elif a <= x and x <= b:
        return (2 / (a + b)) * ((b - x) / (b - a))
    else:
        return 0

# %%
# For the `plot_sample_data` debug feature in the next cell, you need to patch LIME as such:

# Change return statement from `explain_instance_with_data` (lime_base.py:204) to

#         return (easy_model.intercept_,
#                 sorted(zip(used_features, easy_model.coef_),
#                        key=lambda x: np.abs(x[1]), reverse=True),
#                 prediction_score, local_pred, neighborhood_data[:, used_features], weights)

# And `ret_exp` in `explain_instance` (lime_tabular.py:450) to

#             (ret_exp.intercept[label],
#              ret_exp.local_exp[label],
#              ret_exp.score, ret_exp.local_pred, ret_exp.data, ret_exp.weights) = self.base.explain_instance_with_data(
#                     scaled_data,
#                     yss,
#                     distances,
#                     label,
#                     num_features,
#                     model_regressor=model_regressor,
#                     feature_selection=self.feature_selection)

#print sklearn.__version__
import sklearn


print("sklearn version:", sklearn.__version__)

# %%

import warnings
warnings.filterwarnings('ignore')
# MARS surrogate
# from pyearth import Earth
  # decision tree of maximum depth 3
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from pwla_lime.pwla_line_3d import MultiRegionLimeTabularExplainer3D

def do_experiment(Xtr, classifier, KERNEL_SIZE, plot_sample_data=False):
  from scipy.special import gammainccinv, erfinv
  sample_size_lime = 5000
  sample_size_lemon = 5000

  DIMENSIONS = Xtr.shape[1]
  p = 0.999

  if type(KERNEL_SIZE) is LambdaType:
    KERNEL_SIZE = KERNEL_SIZE(DIMENSIONS)

  kernel_width_lime = KERNEL_SIZE
  radius = KERNEL_SIZE * np.sqrt(2*gammainccinv(DIMENSIONS/2, (1-p)))
  kernel_width_lemon = KERNEL_SIZE
  kernel_width_pwla = KERNEL_SIZE
  diameter = radius * 2

  # explainers
  surrogate_lime = Ridge(fit_intercept=True, random_state=123)
#   surrogate_lemon = Earth(max_degree=1)
  # surrogate_pwla = Earth(max_degree=1)          # piecewise-linear

  explainer_lime = lime.lime_tabular.LimeTabularExplainer(
      Xtr,
      kernel_width=kernel_width_lime,
      feature_selection='none',
      sample_around_instance=True, # Important! Essentially: True = synthetic, False = observation-based sampling
      discretize_continuous=False
  )

  # surrogate_lemon = DecisionTreeRegressor(max_depth=3, random_state=123)
  surrogate_pwla = Ridge(fit_intercept=True, random_state=123)
  surrogate_lemon = Ridge(fit_intercept=True, random_state=123)



  # add to surrogate_lemon
  # surrogate_pwla = DecisionTreeRegressor(max_depth=3, random_state=123)
  # surrogate_pwla = DecisionTreeClassifier(max_depth=3, random_state=123)
  # plot decision tree after training
  # sdsd

  # # Initialize multi-region LIME explainer
  pwla_explainer = MultiRegionLimeTabularExplainer3D(
      Xtr,
      kernel_width=kernel_width_pwla,
      feature_selection='none',
      sample_around_instance=True, # Important! Essentially: True = synthetic, False = observation-based sampling
      discretize_continuous=False,
  )



  explainer_lemon = LemonExplainer(
      Xtr,
      sample_size=sample_size_lemon,
      distance_kernel=partial(gaussian_kernel, kernel_width=kernel_width_lemon),
      radius_max=radius
  )

  Xeval = Xtr
  yeval = classifier.predict_proba(Xeval)
  ceval = classifier.predict(Xeval)
  scaled_data = explainer_lime.scaler.transform(Xeval)

  points = []
  points2 = []

  # for many test instances
  for i in range(0, Xeval.shape[0]):
      testInstance = (Xeval[i,:], classifier.predict(Xeval[i,:].reshape(1, -1))[0])
      # print("Test instance:", i, testInstance)

      explanation_lime = explainer_lime.explain_instance(
          testInstance[0],
          classifier.predict_proba,
          num_samples=sample_size_lime,
          labels=(testInstance[1],),
          model_regressor=surrogate_lime
      )


      #     # Generate explanation
      pwla_explanation = pwla_explainer.explain_instance(
          testInstance[0], classifier.predict_proba, num_samples=sample_size_lime, num_features=DIMENSIONS, labels=(testInstance[1],),model_regressor=surrogate_pwla
      )

      explanation_lemon = explainer_lemon.explain_instance(
          testInstance[0],
          classifier.predict_proba,
          labels=(testInstance[1],),
          surrogate=surrogate_lemon
      )

      ## Misleading: self-reported scores from LIME and LEMON
      # lime_fidelity = explanation_lime.score
      # lemon_fidelity = explanation_lemon[0][1]

      ## Generate 50000 new samples in the neighborhood to compute R^2 between reference and surrogate models.
      sample_size = 50000
      dimensions = DIMENSIONS
      sphere = np.random.normal(size=(sample_size, dimensions))
      sphere = normalize(sphere)
      sphere *= explainer_lemon.distance_kernel(np.random.uniform(size=sample_size)).reshape(-1,1)
      X_transfer = explainer_lemon.scaler.inverse_transform(sphere) + np.array([testInstance[0]])
      y_transfer = classifier.predict_proba(X_transfer)
      scaled_lime = (X_transfer - explainer_lime.scaler.mean_) / explainer_lime.scaler.scale_
      # scaled_lemon = (X_transfer - explainer_lime.scaler.mean_) / explainer_lime.scaler.scale_

      scaled_pwla = (X_transfer - pwla_explainer.scaler.mean_) / pwla_explainer.scaler.scale_


      # from sklearn.metrics import mean_squared_error
      from sklearn.metrics import root_mean_squared_error
      pwla_fidelity = root_mean_squared_error(
        y_transfer[:,testInstance[1]],
        surrogate_pwla.predict(scaled_pwla), # X_transfer
        # squared=False
      )
      scaled_lime = (X_transfer - explainer_lime.scaler.mean_) / explainer_lime.scaler.scale_

      lime_fidelity = root_mean_squared_error(
        y_transfer[:,testInstance[1]],
        surrogate_lime.predict(scaled_lime),
        # squared=False
      )

      points.append([i, pwla_fidelity])
      points2.append([i, lime_fidelity])

  return np.array(points), np.array(points2)


## Sanity check (This requires patching LIME! See previous cell.)
# data = Dataset.IRIS.instance
# default_rng(seed=2).shuffle(data.data)
# Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.6, random_state=123)
# classifier = Classifier.GNB.instance
# classifier.fit(Xtr, ytr)
# do_experiment(Xtr, classifier, 0.2, plot_sample_data=True)

# data = Dataset.BC.instance
# default_rng(seed=2).shuffle(data.data)
# Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.6, random_state=123)
# classifier = Classifier.MLP.instance
# classifier.fit(Xtr, ytr)
# print(Xtr.shape[0])
# points, points2 = do_experiment(Xtr, classifier, 0.5, plot_sample_data=False)
# score_pwla = np.mean(points[:,1])
# score_lime = np.mean(points2[:,1])
# print("PWLA score:", score_pwla)
# print("LIME score:", score_lime)

#!/usr/bin/env python3
# faithfulness_experiment.py

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from itertools import product
import contextlib
import joblib
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

# Import your dataset and classifier definitions, and the experiment function

# Define datasets, classifiers, and kernel widths
datasets = [
    # Dataset.IRIS2D,
    # Dataset.WINE,
    Dataset.PIMA,
    Dataset.BC
]

classifiers = [
    # Classifier.GNB,
    Classifier.MLP,
    Classifier.RF
]

kernel_widths = [
    # 0.1,
    # # 0.2,
    # 0.3,
    # # 0.4,
    0.5,
    # # 0.6,
    # # 0.7,
    # # 0.8,
    # # 0.9,
    1.0,
    # 1.5,
    # 2.0,
    # 2.5,
    # 3.0,
    # 3.5,
    lambda n: np.sqrt(n) * 0.75,
    # 4.0
]

# Context manager to integrate joblib progress with tqdm
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib to report into tqdm progress bar"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# Function to compute faithfulness scores
def faithfulness(dataset, classifier, kernel_width):
    data = dataset.instance
    X = data.data
    y = data.target

    clf = classifier.instance
    clf.fit(X, y)

    points_lemon, points_lime = do_experiment(X, clf, kernel_width, plot_sample_data=False)
    score_lemon = np.mean(points_lemon[:, 1])
    score_lime = np.mean(points_lime[:, 1])
    return score_lime, score_lemon

# Main execution
if __name__ == '__main__':
    parameters = list(product(datasets, classifiers, kernel_widths))
    total_runs = len(parameters)

    print(f"Running faithfulness experiments on {total_runs} parameter combinations...")

    with tqdm(total=total_runs) as progress_bar:
        with tqdm_joblib(progress_bar):
            results = Parallel(n_jobs=cpu_count())(
                delayed(faithfulness)(dataset, clf, kw) for dataset, clf, kw in parameters
            )

    # Collect and display results
    df = pd.DataFrame(results, index=[f"{d.name}-{c.name}-{(kw.__name__ if hasattr(kw, '__name__') else kw)}" for d, c, kw in parameters],
                      columns=['LIME', 'LEMON'])
    # Optionally, save to CSV
    df.to_csv('faithfulness_results.csv')
    print("Results saved to faithfulness_results.csv")



