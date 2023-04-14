import logging
import numpy as np
from scipy.linalg import svd
from modules.base import BaseEstimator
from modules.categorical_scatter import categorical_scatter_2d
from sklearn.linear_model import LinearRegression

np.random.seed(1000)

class PCA(BaseEstimator):
    y_required = False

    def __init__(self, n_components, solver="svd"):
        """Principal component analysis (PCA) implementation.
        Transforms a dataset of possibly correlated values into n linearly
        uncorrelated components. The components are ordered such that the first
        has the largest possible variance and each following component as the
        largest possible variance given the previous components. This causes
        the early components to contain most of the variability in the dataset.
        Parameters
        ----------
        n_components : int
        solver : str, default 'svd'
            {'svd', 'eigen'}
        """
        self.solver = solver
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self._decompose(X)

    def _decompose(self, X):
        # Mean centering
        X = X.copy()
        X -= self.mean

        if self.solver == "svd":
            _, s, Vh = svd(X, full_matrices=True)
        elif self.solver == "eigen":
            s, Vh = np.linalg.eig(np.cov(X.T))
            Vh = Vh.T

        s_squared = s ** 2
        variance_ratio = s_squared / s_squared.sum()
        logging.info("Explained variance ratio: %s" % (variance_ratio[0: self.n_components]))
        self.components = Vh[0: self.n_components]

    #def transform(self, X):
    #    X = X.copy()
    #    X -= self.mean
    #    return np.dot(X, self.components.T)
    def transform(self, X):
        self.fit(X)
        X_center=X-self.mean
        X_pca = X_center @ self.components.T
        print(X_pca.shape)
        transform_x = (X_pca @ self.components) + self.mean
        return transform_x

    
    def _predict(self, X=None):
        return self.transform(X)
    
class SVD:
    def __init__(self, n_components):
        self.n_components = n_components

    def transform(self,X):
        print(X.shape)
        u,s,vh = np.linalg.svd(X)
        s = np.diag(s)
        X_transformed = np.matmul(u[:,:self.n_components],
                                  s[:self.n_components, :self.n_components])
        X_transformed = np.matmul(X_transformed, vh[:self.n_components, :])
        return X_transformed


class tsne:
    def __init__(self,flag=True):
        self.flag = flag
        self.components = None
        self.mean = None
        self.NUM_POINTS = 200            # Number of samples from MNIST
        self.CLASSES_TO_USE = [0, 1, 8]  # MNIST classes to use
        self.PERPLEXITY = 20
        self.SEED = 1                    # Random seed
        self.MOMENTUM = 0.9
        self.LEARNING_RATE = 10.
        self.NUM_ITERS = 500             # Num iterations to train for
        self.TSNE = True               # If False, Symmetric SNE
        self.NUM_PLOTS = 5               # Num. times to plot in training
        self.rng = np.random.RandomState(self.SEED)

    def neg_squared_euc_dists(self,X):
        """Compute matrix containing negative squared euclidean
        distance for all pairs of points in input matrix X

        # Arguments:
            X: matrix of size NxD
        # Returns:
            NxN matrix D, with entry D_ij = negative squared
            euclidean distance between rows X_i and X_j
        """
        # Math? See https://stackoverflow.com/questions/37009647
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return -D
    
    def softmax(self, X, diag_zero=True):
        """Take softmax of each row of matrix X."""

        # Subtract max for numerical stability
        e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))

        # We usually want diagonal probailities to be 0.
        if diag_zero:
            np.fill_diagonal(e_x, 0.)

        # Add a tiny constant for stability of log we take later
        e_x = e_x + 1e-8  # numerical stability

        return e_x / e_x.sum(axis=1).reshape([-1, 1])
    
    def calc_prob_matrix(self, distances, sigmas=None):
        """Convert a distances matrix to a matrix of probabilities."""
        if sigmas is not None:
            two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
            return self.softmax(distances / two_sig_sq)
        else:
            return self.softmax(distances)

    def binary_search(self, eval_fn, target, tol=1e-10, max_iter=10000, lower=1e-20, upper=1000.):
        """Perform a binary search over input values to eval_fn.
        # Arguments
            eval_fn: Function that we are optimising over.
            target: Target value we want the function to output.
            tol: Float, once our guess is this close to target, stop.
            max_iter: Integer, maximum num. iterations to search for.
            lower: Float, lower bound of search range.
            upper: Float, upper bound of search range.
        # Returns:
            Float, best input value to function found during search.
        """
        for i in range(max_iter):
            guess = (lower + upper) / 2.
            val = eval_fn(guess)
            if val > target:
                upper = guess
            else:
                lower = guess
            if np.abs(val - target) <= tol:
                break
        return guess
    
    def calc_perplexity(self, prob_matrix):
        """Calculate the perplexity of each row 
        of a matrix of probabilities."""
        entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
        perplexity = 2 ** entropy
        return perplexity


    def perplexity(self, distances, sigmas):
        """Wrapper function for quick calculation of 
        perplexity over a distance matrix."""
        return self.calc_perplexity(self.calc_prob_matrix(distances, sigmas))


    def find_optimal_sigmas(self, distances, target_perplexity):
        """For each row of distances matrix, find sigma that results
        in target perplexity for that role."""
        sigmas = [] 
        # For each row of the matrix (each point in our dataset)
        for i in range(distances.shape[0]):
            # Make fn that returns perplexity of this row given sigma
            eval_fn = lambda sigma: \
                self.perplexity(distances[i:i+1, :], np.array(sigma))
            # Binary search over sigmas to achieve target perplexity
            correct_sigma = self.binary_search(eval_fn, target_perplexity)
            # Append the resulting sigma to our output array
            sigmas.append(correct_sigma)
        return np.array(sigmas)

    def q_joint(self,Y):
        """Given low-dimensional representations Y, compute
        matrix of joint probabilities with entries q_ij."""
        # Get the distances from every point to every other
        distances = self.neg_squared_euc_dists(Y)
        # Take the elementwise exponent
        exp_distances = np.exp(distances)
        # Fill diagonal with zeroes so q_ii = 0
        np.fill_diagonal(exp_distances, 0.)
        # Divide by the sum of the entire exponentiated matrix
        return exp_distances / np.sum(exp_distances), None


    def p_conditional_to_joint(self, P):
        """Given conditional probabilities matrix P, return
        approximation of joint distribution probabilities."""
        return (P + P.T) / (2. * P.shape[0])
    
    def p_joint(self, X, target_perplexity):
        """Given a data matrix X, gives joint probabilities matrix.

        # Arguments
            X: Input data matrix.
        # Returns:
            P: Matrix with entries p_ij = joint probabilities.
        """
        # Get the negative euclidian distances matrix for our data
        distances = self.neg_squared_euc_dists(X)
        # Find optimal sigma for each row of this distances matrix
        sigmas = self.find_optimal_sigmas(distances, target_perplexity)
        # Calculate the probabilities based on these optimal sigmas
        p_conditional = self.calc_prob_matrix(distances, sigmas)
        # Go from conditional to joint probabilities matrix
        P = self.p_conditional_to_joint(p_conditional)
        return P
    
    def symmetric_sne_grad(self, P, Q, Y, _):
        """Estimate the gradient of the cost with respect to Y"""
        pq_diff = P - Q  # NxN matrix
        pq_expanded = np.expand_dims(pq_diff, 2)  #NxNx1
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  #NxNx2
        grad = 4. * (pq_expanded * y_diffs).sum(1)  #Nx2
        return grad

    def fit_tsne(self,X, y):
        """Estimates a SNE model.
        # Arguments
            X: Input data matrix.
            y: Class labels for that matrix.
            P: Matrix of joint probabilities.
            rng: np.random.RandomState().
            num_iters: Iterations to train for.
            q_fn: Function that takes Y and gives Q prob matrix.
            plot: How many times to plot during training.
        # Returns:
            Y: Matrix, low-dimensional representation of X.
        """
        P = self.p_joint(X, self.PERPLEXITY)
        # Initialise our 2D representation
        Y = self.rng.normal(0., 0.0001, [X.shape[0], 2])

        # Initialise past values (used for momentum)
        if self.MOMENTUM:
            Y_m2 = Y.copy()
            Y_m1 = Y.copy()

        # Start gradient descent loop
        for i in range(self.NUM_ITERS):

            # Get Q and distances (distances only used for t-SNE)
            Q, distances = self.q_tsne(Y)
            # Estimate gradients with respect to Y
            grads = self.tsne_grad(P, Q, Y, distances)

            # Update Y
            Y = Y - self.LEARNING_RATE * grads
            if self.MOMENTUM:  # Add momentum
                Y += self.MOMENTUM * (Y_m1 - Y_m2)
                # Update previous Y's for momentum
                Y_m2 = Y_m1.copy()
                Y_m1 = Y.copy()

            # Plot sometimes
            if self.NUM_PLOTS and i % (self.NUM_ITERS / self.NUM_PLOTS) == 0:
                categorical_scatter_2d(Y, y, alpha=1.0, ms=6,
                                    show=True, figsize=(9, 6))
                
        #return Y
        print(self.flag)
        if self.flag:
            model_recons = LinearRegression()
            model_recons.fit(Y,X)
            recontruc_Y = model_recons.predict(Y)
            Y=recontruc_Y
        return Y

    def q_tsne(self, Y):
        """t-SNE: Given low-dimensional representations Y, compute
        matrix of joint probabilities with entries q_ij."""
        distances = self.neg_squared_euc_dists(Y)
        inv_distances = np.power(1. - distances, -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances), inv_distances
    
    def tsne_grad(self, P, Q, Y, inv_distances):
        """Estimate the gradient of t-SNE cost with respect to Y."""
        pq_diff = P - Q
        pq_expanded = np.expand_dims(pq_diff, 2)
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

        # Expand our inv_distances matrix so can multiply by y_diffs
        distances_expanded = np.expand_dims(inv_distances, 2)

        # Multiply this by inverse distances matrix
        y_diffs_wt = y_diffs * distances_expanded

        # Multiply then sum over j's
        grad = 4. * (pq_expanded * y_diffs_wt).sum(1)
        return grad