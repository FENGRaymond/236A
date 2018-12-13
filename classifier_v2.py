'''Libraries for Prototype selection'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import cvxpy as cvx
import math as mt
from sklearn.model_selection import KFold
import sklearn.metrics
from scipy import stats


# lambda = 1/n in original paper

class classifier():
    """Contains functions for prototype selection"""

    def __init__(self, X, y, epsilon_, lambda_):
        """
        Store data points as unique indexes, and initialize
        the required member variables eg. epsilon, lambda,
        interpoint distances, points in neighborhood

        :param X: training data
        :param y: training class
        :param epsilon_: radius, constant
        :param lambda_: weight, constant
        """

        self.epsilon_ = epsilon_
        self.lambda_ = lambda_
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.nbr_mask = None
        self.alpha = np.empty(len(X))
        self.xi = np.empty(len(X))

        # For testing
        self.probe = None

        # How many dimensions are there in train data?
        self.dim_num = len(self.X_train.shape)

        # Calculate interpoint distances
        self.dist = self.compute_dist(self.X_train)

        # Create points in neighborhood
        self.nbr_mask = self.dist < self.epsilon_

    """Implement modules which will be useful in the train_lp() function
    for example
    1) operations such as intersection, union etc of sets of datapoints
    2) check for feasibility for the randomized algorithm approach
    3) compute the objective value with current prototype set
    4) fill in the train_lp() module which will make 
    use of the above modules and member variables defined by you
    5) any other module that you deem fit and useful for the purpose."""

    def train_lp(self, verbose=False):
        """
        Implement the linear programming formulation and solve using cvxpy for prototype selection
        Input:
            verbose:
        """

        def _union_size(self, train_sets, idx):
            count = 0
            idx_list = [tmp for tmp, x in enumerate(self.nbr_mask[idx] == True) if x]
            for j in idx_list:
                if not (j in train_sets):
                    count += 1
            return count, idx_list

        # Separate into L prize-collecting set cover problems
        train_sets = {}
        cls_count = np.unique(self.y_train)
        # cls_num = len(cls_count)
        for cls in cls_count:
            # keys in train_sets are indeices of the training data of the same class
            train_sets[cls] = [i for i, x in enumerate(self.y_train == cls) if x]

            # Initialize variables/constants
            alpha_ = cvx.Variable(self.dim_num)
            xi_ = cvx.Variable(self.dim_num)
            Cl = np.zeros(alpha_.shape)

            # Set up obj and constraints
            cons = [0 <= alpha_, alpha_ <= 1, 0 <= xi_]

            for true_idx, ref_idx in enumerate(train_sets[cls]):
                tmp, circle_idx = _union_size(self, train_sets[cls], ref_idx)
                Cl[ref_idx] = self.lambda_ + tmp

                nbr_list = np.zeros(alpha_.shape)

                nbr_list[circle_idx] = 1

                cons += [1 - xi_[ref_idx] <= sum(np.asarray(nbr_list) * alpha_)]

            obj = cvx.Minimize(sum(Cl * alpha_) + sum(xi_))

            # Solve for class "cls"
            prob_cls = cvx.Problem(obj, cons)
            prob_cls.solve()

            alpha_.value[alpha_.value > 1] = 1
            xi_.value[xi_.value < 0] = 0

            # Bernoulli rounding
            A_l = np.zeros(alpha_.shape)
            S_i = np.zeros(xi_.shape)

            flag = False
            while not flag:
                for t in range(int(2 * np.log(len(train_sets[cls])))):
                    A_tilt = np.random.binomial(1, alpha_.value)
                    S_tilt = np.random.binomial(1, xi_.value)
                    A_l = np.maximum(A_l, A_tilt)
                    S_i = np.maximum(S_i, S_tilt)
                flag = self._feasibility(train_sets[cls], prob_cls.value, A_l, S_i, nbr_list, Cl)

            for ref_idx, true_idx in enumerate(train_sets[cls]):
                self.alpha[true_idx] = A_l[ref_idx]
                self.xi[true_idx] = S_i[ref_idx]
            del alpha_, xi_, A_l, S_i

    def _feasibility(self, train_sets, opt, A_l, S_i, nbr_list, Cl):
        for true_idx, ref_idx in enumerate(train_sets):
            if ((1 - S_i[true_idx]) > np.sum(nbr_list * A_l)):
                return False

        if np.sum(S_i) + np.sum(Cl * A_l) > 2 * np.log(len(train_sets)) * opt:
            return False
        else:
            return True

    def objective_value(self):
        """
        Implement a function to compute the objective value of the integer optimization
        problem after the training phase

		may not be equal to the optimal value of LP

        """
        self.obj_val = np.sum(self.xi) + self.lambda_ * np.sum(self.alpha)
        return self.obj_val

    def compute_dist(self, instances):
        num_train = self.X_train.shape[0]
        num_test = instances.shape[0]

        mat_mul = np.matmul(instances, self.X_train.transpose())
        X_norm = ((np.linalg.norm(instances, axis=1)) ** 2)[np.newaxis].transpose()
        Xtr_norm = ((np.linalg.norm(self.X_train.transpose(), axis=0)) ** 2)[np.newaxis]

        dists = -2 * mat_mul + X_norm
        dists += Xtr_norm
        dists[dists < 0] = 0
        dists = np.sqrt(dists)

        return dists

    def predict(self, instances):
        """
        Predicts the label for an array of instances using the framework learnt
        Input:
            instances:
        """
        dist_matrix = self.compute_dist(instances)

        tmp = dist_matrix * (dist_matrix < self.epsilon_) * (self.alpha[np.newaxis].transpose())
        pred = []
        for i in range(len(instances)):
            indices = [j for j, x in enumerate(tmp[i]) if x]
            proto_list = self.y_train[indices]
            print(proto_list)
            pred.append(stats.mode(proto_list))

        return pred


def cross_val(data, target, epsilon_, lambda_, k, verbose):
    """
    Implement a function which will perform k fold cross validation
    for the given epsilon and lambda and returns the average test error and number of prototypes
    Input:
        data:
        target:
        epsilon_:
        lambda_:
        k:
        verbose:
    Output:
         average test error
         average number of prototypes
         average objective value
    """
    kf = KFold(n_splits=k, random_state=42)
    score = 0
    prots = 0
    obj_val = 0
    for train_index, test_index in kf.split(data):
        ps = classifier(data[train_index], target[train_index], epsilon_, lambda_)
        ps.train_lp(verbose)
        obj_val += ps.objective_value()
        score += sklearn.metrics.accuracy_score(target[test_index], ps.predict(data[test_index]))
        '''implement code to count the total number of prototypes learnt and store it in prots'''
    score /= k
    prots /= k
    obj_val /= k
    return score, prots, obj_val


if __name__ == '__main__':
    X, y = load_iris(True)
    test = classifier(X, y, 0.5, 1 / len(X))
    test.train_lp()
    pred = test.predict(X)
    print(np.average(np.asarray(pred) == y))
