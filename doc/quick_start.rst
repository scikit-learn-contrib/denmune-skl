.. _quick_start:

###############
Getting Started
###############

This guide provides the essential steps to install ``denmune-skl`` and run your first clustering analysis.

Installation
============

The package can be installed from PyPI using ``pip``.

.. prompt:: bash

   pip install denmune-skl

.. note::
   Until the project is accepted into ``scikit-learn-contrib`` and published, you must install it directly from the source repository.

From Source
-----------

To install the latest development version, clone the repository and install it locally:

.. prompt:: bash

   git clone https://github.com/scikit-learn-contrib/denmune-skl.git
   cd denmune-skl
   pip install .


Basic Usage
===========

The following example demonstrates how to use ``DenMune`` to cluster a simple non-convex dataset.

.. code-block:: python

   from sklearn.datasets import make_moons
   from sklearn.preprocessing import StandardScaler
   from denmune_skl import DenMune
   import matplotlib.pyplot as plt

   # 1. Generate and prepare data
   X, y = make_moons(n_samples=250, noise=0.07, random_state=42)
   X = StandardScaler().fit_transform(X)

   # 2. Initialize and fit the model
   # The k_nearest parameter is the main hyperparameter to tune.
   model = DenMune(k_nearest=20, random_state=42)
   labels = model.fit_predict(X)

   # 3. Visualize the results
   n_clusters = model.n_clusters_
   print(f"Estimated number of clusters: {n_clusters}")

   plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
   plt.title(f"DenMune Clustering (k={model.k_nearest})")
   plt.xlabel("Feature 1")
   plt.ylabel("Feature 2")
   plt.show()


Where to Go Next
================

*   To learn about the algorithm's theory and parameter tuning, see the :doc:`user_guide`.
*   For more detailed code examples, browse the :doc:`auto_examples/index`.
*   For detailed information on the class and its methods, consult the :doc:`api`.
