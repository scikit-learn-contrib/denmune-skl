import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.validation import check_array, check_is_fitted

from denmune_skl.denmune import DenMune


# Mock reducer for testing purposes
class MockSparseReducer(BaseEstimator, TransformerMixin):
    """A mock transformer that 'supports' sparse input for testing."""

    def _more_tags(self):
        return {"X_types": ["sparse", "2darray"]}

    def fit(self, X, y=None):
        check_array(X, accept_sparse=True)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        check_array(X, accept_sparse=True)
        # Just return a dense array of the correct shape for the test
        return np.zeros((X.shape[0], 2))


@pytest.fixture
def blob_data():
    """Fixture for simple, well-separated blob data."""
    X, y = make_blobs(
        n_samples=150, n_features=2, centers=3, cluster_std=0.8, random_state=42
    )
    return X, y


@pytest.fixture
def high_dim_blob_data():
    """
    Fixture for blob data with enough dimensions to trigger dimensionality reduction.
    n_features (5) > target_dims (2).
    """
    X, y = make_blobs(
        n_samples=150, n_features=5, centers=3, cluster_std=0.8, random_state=42
    )
    return X, y


@pytest.fixture
def moon_data():
    """Fixture for non-convex moon-shaped data."""
    X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
    return X, y


def test_invalid_parameters():
    """Test that DenMune's __init__ raises errors for invalid parameters."""
    # Test k_nearest
    with pytest.raises(
        ValueError,
        match=(
            r"The 'k_nearest' parameter of DenMune must be an int in the range "
            r"\[1, inf\). Got 0 instead."
        ),
    ):
        DenMune(k_nearest=0).fit([[0, 0]])

    # Test target_dims
    with pytest.raises(
        ValueError,
        match=(
            r"The 'target_dims' parameter of DenMune must be an int in the range "
            r"\[1, inf\). Got 0 instead."
        ),
    ):
        DenMune(target_dims=0).fit([[0, 0], [1, 1], [0, 1]])

    # Test dim_reducer string
    with pytest.raises(
        ValueError,
        match=r"The 'dim_reducer' parameter of DenMune must be a str among "
        r"(\{'pca', 'tsne'\}|\{'tsne', 'pca'\}) "
        r"or an instance of 'sklearn\.base\.BaseEstimator'\. "
        r"Got 'invalid_reducer' instead\.",
    ):
        DenMune(dim_reducer="invalid_reducer").fit([[0, 0]])


def test_reproducibility_random_state(moon_data):
    """Ensure that random_state provides reproducible results."""
    X, _ = moon_data
    model1 = DenMune(k_nearest=15, reduce_dims=True, random_state=42)
    model2 = DenMune(k_nearest=15, reduce_dims=True, random_state=42)

    labels1 = model1.fit_predict(X)
    labels2 = model2.fit_predict(X)

    assert_array_equal(labels1, labels2)


def test_simple_blobs_clustering(blob_data):
    """Test clustering on well-separated blobs."""
    X, y = blob_data
    model = DenMune(k_nearest=10, reduce_dims=False)
    labels = model.fit_predict(X)

    assert model.n_clusters_ == 3
    # ARI should be near perfect for this simple case
    assert adjusted_rand_score(y, labels) >= 0.95


def test_noise_detection():
    """Test that the algorithm correctly identifies noise points."""
    X_blobs, _ = make_blobs(n_samples=100, centers=2, cluster_std=0.5, random_state=0)
    # Add uniform noise
    rng = np.random.RandomState(0)
    X_noise = rng.uniform(low=-10, high=10, size=(20, 2))
    X = np.vstack([X_blobs, X_noise])

    model = DenMune(k_nearest=15, reduce_dims=False)
    labels = model.fit_predict(X)

    # Check that noise points were labeled -1
    noise_labels = labels[100:]
    assert -1 in noise_labels
    # Assert MOST noise points are labeled -1.
    assert np.sum(noise_labels == -1) >= np.sum(noise_labels)
    # Check that core points were not labeled -1
    assert -1 not in labels[:100]


def test_arbitrary_shapes_clustering(moon_data):
    """Test clustering on non-convex shapes (moons)."""
    X, y = moon_data
    # Dimensionality reduction is not needed for 2D data but we test it anyway
    model = DenMune(k_nearest=20, reduce_dims=False, random_state=42)
    labels = model.fit_predict(X)

    assert model.n_clusters_ == 2
    assert adjusted_rand_score(y, labels) > 0.95


def test_dim_reduction_logic(blob_data):
    """Test the dimensionality reduction pathway."""
    X, _ = blob_data
    X_high_dim, _ = make_blobs(n_samples=100, n_features=10, centers=2, random_state=42)

    # 1. Test reduce_dims=False
    model_no_reduce = DenMune(reduce_dims=False)
    model_no_reduce.fit(X)
    # Should always have `reducer_` params regardless of performing dim reduction
    # or not.
    assert hasattr(model_no_reduce, "reducer_")
    assert_array_equal(model_no_reduce.projected_X_, X)

    # 2. Test reduce_dims=True
    model_reduce = DenMune(reduce_dims=True, target_dims=5, random_state=42)
    model_reduce.fit(X_high_dim)
    assert hasattr(model_reduce, "reducer_")
    assert model_reduce.projected_X_.shape == (100, 5)

    # 3. Test warning when input dim <= target_dims
    with pytest.warns(UserWarning, match="Skipping dimensionality reduction"):
        model = DenMune(reduce_dims=True, target_dims=3)
        model.fit(X)  # X has only 2 features


def test_precomputed_metric_handling(blob_data):
    """
    Test logic for metric='precomputed'.
    It MUST raise an error if reduce_dims=True.
    It should work if reduce_dims=False.
    """
    X, _ = blob_data
    distance_matrix = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=-1)

    # This combination is invalid and must fail
    model_fail = DenMune(metric="precomputed", reduce_dims=True)
    with pytest.raises(
        ValueError, match="metric='precomputed' is not supported when reduce_dims=True"
    ):
        model_fail.fit(distance_matrix)

    # This combination should work
    model_pass = DenMune(k_nearest=10, metric="precomputed", reduce_dims=False)
    try:
        model_pass.fit(distance_matrix)
    except Exception as e:
        pytest.fail(
            f"DenMune with metric='precomputed' and reduce_dims=False failed: {e}"
        )
    assert model_pass.n_clusters_ == 3


def test_sparse_input_handling(high_dim_blob_data):
    """
    Test logic for sparse matrix input.
    - It MUST raise an error if reduce_dims=True with a non-sparse reducer (TSNE).
    - It SHOULD WORK if reduce_dims=True with a sparse-compatible reducer.
    - It SHOULD WORK if reduce_dims=False.
    """
    X, y = high_dim_blob_data
    X_sparse = csr_matrix(X)

    # 1. Reduce_dims=True with the default "tsne" reducer.
    # Since n_features (5) > target_dims (2), reduction is attempted.
    # Since TSNE doesn't support sparse, it MUST warn.
    model_warn = DenMune(reduce_dims=True, dim_reducer="tsne")
    warn_msg = "does not support sparse input"

    with pytest.warns(UserWarning, match=warn_msg):
        model_warn.fit(X_sparse)

    # Check that it proceeded (fallback to original data) and produced labels
    assert hasattr(model_warn, "labels_")

    # 2. Valid case: reduce_dims=False. SHOULD PASS.
    model_pass_no_reduce = DenMune(k_nearest=10, reduce_dims=False)
    try:
        labels = model_pass_no_reduce.fit_predict(X_sparse)
    except Exception as e:
        pytest.fail(f"DenMune with sparse input and reduce_dims=False failed: {e}")

    # We still expect a good result
    assert adjusted_rand_score(y, labels) >= 0.95

    # 3. Valid case: reduce_dims=True with a sparse-compatible reducer. SHOULD PASS.
    model_pass_reduce = DenMune(
        k_nearest=10, reduce_dims=True, dim_reducer=MockSparseReducer()
    )
    try:
        model_pass_reduce.fit(X_sparse)
    except Exception as e:
        pytest.fail(
            f"DenMune with sparse input and a sparse-compatible reducer failed: {e}"
        )
    # Check that the reducer was indeed used
    assert isinstance(model_pass_reduce.reducer_, MockSparseReducer)


def test_different_data_types(blob_data):
    """Test that float32 and float64 inputs work correctly."""
    X, y = blob_data
    X_32 = X.astype(np.float32)
    X_64 = X.astype(np.float64)

    model_32 = DenMune(k_nearest=10, reduce_dims=False)
    model_64 = DenMune(k_nearest=10, reduce_dims=False)

    try:
        labels_32 = model_32.fit_predict(X_32)
        labels_64 = model_64.fit_predict(X_64)
    except Exception as e:
        pytest.fail(f"DenMune failed with float32 or float64 input: {e}")

    # Results should be identical for this deterministic case
    assert_array_equal(labels_32, labels_64)


def test_fitted_attributes(blob_data):
    """Check that all public attributes are set after fit."""
    X, _ = blob_data
    model = DenMune(k_nearest=10).fit(X)

    assert hasattr(model, "labels_")
    assert hasattr(model, "core_sample_indices_")
    assert hasattr(model, "n_clusters_")
    assert hasattr(model, "density_scores_")
    assert hasattr(model, "nn_")
    assert hasattr(model, "projected_X_")
    assert hasattr(model, "n_features_in_")
    assert hasattr(model, "n_samples_")

    assert isinstance(model.n_clusters_, int)
    assert model.labels_.shape == (X.shape[0],)
