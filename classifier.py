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
import matplotlib.pyplot as plt



def Iris_sanity_check():
    """
    Sanity check using Iris data set. Should give perfect training accuracy
    """
    X, y = load_iris(True)
    lambda_ = 1/len(X)
    epsilon_ = [0.1, 0.8, 5]
    for e in epsilon_:
        sanity_model = classifier(X,y,e,lambda_)
        sanity_model.train_lp()
        obj_val = sanity_model.objective_value()
        pred, _ = sanity_model.predict(X)
        acc = np.average(pred==y)
        print("Radius: %f, objective value: %f, training acc: %f, number of prototypes: %f" % (e, obj_val, acc*100, np.sum(sanity_model.alpha)))


def Breast_variation(fold=4):
    """
    Run the classifier on Breast cancer data set.

    Input:
        fold: how many subsets should the data set be split into
    :return:
        A plot of test error and cover error
        radius_list: list of radius that were used to classifier
        test_errors: list of corresponding test errors
        cover_errors: list of corresponding cover errors
        num_prototypes: list of number of prototypes
    """
    X_train, y_train = load_breast_cancer(True)
    lambda_ = fold/(len(X_train)*(fold-1))

    radius_list = []
    test_errors = []
    cover_errors = []
    num_prototypes = []

    # Get the 2% and 40% interpoint distance
    sample = classifier(X_train, y_train, np.random.randn(), lambda_, True)
    inter_point = []
    for i in range(sample.dist.shape[0]):
        for j in range(i+1):
            inter_point.append(sample.dist[i][j])
    inter_point = np.asarray(inter_point)
    for percent in range(2,41,2):
        radius = np.percentile(inter_point, percent)
        try:
            score, prots, obj_val, cover_err = cross_val(X_train,y_train, radius, lambda_, fold)
        except:
            continue
        radius_list.append(radius)
        test_errors.append(1-score)
        cover_errors.append(cover_err)
        num_prototypes.append(prots)
        print('%d percent of the inter-point distance done! Accuracy: %f%%' % (percent,score*100))

    plt.plot(num_prototypes, test_errors, 'bs-', num_prototypes, cover_errors, 'r^-')
    plt.legend(['test error', 'cover error'])
    plt.xlabel('Average number of prototypes')
    plt.ylabel('error')
    plt.show()

    return radius_list, test_errors, cover_errors, num_prototypes


def Train_digits():
    """
    Train the classifier on Digits data set.
    :return:
        A plot of test error and cover error
        radius_list: list of radius(epsilon) considered
        obj_val: list of corresponding objective value
        num_prototypes: list of number of prototypes
    """
    X_train, y_train = load_digits(return_X_y=True)

    candid_radius = np.linspace(0.1, 1, 5)
    candid_radius = np.append(candid_radius, np.linspace(2.5, 17.5, 5))
    candid_radius = np.append(candid_radius, np.linspace(20, 70, 6))

    lambda_ = 1/len(X_train)
    radius_list = []
    obj_val = []
    num_prototypes = []

    for radius in candid_radius:
        digit_model = classifier(X_train, y_train, radius, lambda_)
        try:
            digit_model.train_lp()
        except:
            continue
        obj = digit_model.objective_value()
        obj_val.append(obj)
        num_prototypes.append(np.sum(digit_model.alpha))
        radius_list.append(radius)
        print('Training success with Epsilon: %f' % (radius))

    plt.plot(radius_list, obj_val, 'rs-')
    plt.legend(['ILP objective value'])
    plt.xlabel('Epsilon')
    plt.show()

    return radius_list, obj_val, num_prototypes


class classifier():
    """Contains functions for prototype selection"""

    def __init__(self, X, y, epsilon_, lambda_, l2=True):
        """
        Store data points as unique indexes, and initialize
        the required member variables eg. epsilon, lambda,
        interpoint distances, points in neighborhood
        :param X: training data
        :param y: training class
        :param epsilon_: radius, constant
        :param lambda_: weight, constant
        :param l2: set True to use L2 distance, False to use L1 distance
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
        if l2:
            self.compute_dist = self.compute_dist_l2
        else:
            self.compute_dist = self.compute_dist_l1
        # self.dist = self.compute_dist(self.X_train, self.X_train)
        self.dist = self.compute_dist(self.X_train, self.X_train)

        # Create points in neighborhood
        self.nbr_mask = self.dist<self.epsilon_


    """Implement modules which will be useful in the train_lp() function
    for example
    1) operations such as intersection, union etc of sets of datapoints
    2) check for feasibility for the randomized algorithm approach
    3) compute the objective value with current prototype set
    4) fill in the train_lp() module which will make 
    use of the above modules and member variables defined by you
    5) any other module that you deem fit and useful for the purpose."""

    def _feasibility(self, train_sets, opt, A_l, S_i, nbr_list, Cl):
        """
        In class private function to determine if random rounded solution is feasible
        :param train_sets:
        :param opt:
        :param A_l:
        :param S_i:
        :param nbr_list:
        :param Cl:
        :return:
        """
        for true_idx, ref_idx in enumerate(train_sets):
            if ((1-S_i[true_idx])>np.sum(nbr_list*A_l)):
                print('Violate constraint 1')
                return False

        if np.sum(S_i)+np.sum(Cl*A_l) > 2*np.log(len(train_sets))*opt:
            print('Violate objective value')
            return False
        else:
            return True


    def train_lp(self, verbose = False):
        """
        Implement the linear programming formulation and solve using cvxpy for prototype selection
        Input:
            verbose:
        """
        self.inner_dist = {}

        def _union_size(self, train_sets, idx):
            count = 0
            idx_list = [tmp for tmp, x in enumerate(self.nbr_mask[idx]==True) if x]
            for j in idx_list:
                if not (j in train_sets):
                    count += 1
            return count, idx_list

        #Separate into L prize-collecting set cover problems
        train_sets = {}
        cls_count = np.unique(self.y_train)
        # cls_num = len(cls_count)
        for cls in cls_count:
            # keys in train_sets are indeices of the training data of the same class
            train_sets[cls] = [i for i, x in enumerate(self.y_train==cls) if x]
            m = len(train_sets[cls])
            self.inner_dist[cls] = self.compute_dist(self.X_train[train_sets[cls]], self.X_train[train_sets[cls]])

            # Initialize variables/constants
            alpha_ = cvx.Variable(m)
            xi_ = cvx.Variable(m)
            Cl = np.zeros(alpha_.shape)

            # Set up obj and constraints
            cons = [0 <= alpha_, alpha_ <= 1, 0 <= xi_]

            for true_idx, ref_idx in enumerate(train_sets[cls]):
                tmp, _ = _union_size(self, train_sets[cls], ref_idx)
                Cl[true_idx] = self.lambda_ + tmp
                nbr_list = self.inner_dist[cls][true_idx] < self.epsilon_
                cons += [1-xi_[true_idx] <= nbr_list*alpha_]

            obj = cvx.Minimize(Cl * alpha_ + sum(xi_))

            # Solve for class "cls"
            prob_cls = cvx.Problem(obj, cons)
            prob_cls.solve()

            alpha_.value[alpha_.value<0] = 0
            alpha_.value[alpha_.value>1] = 1
            xi_.value[xi_.value<0] = 0
            xi_.value[xi_.value>1] = 1
            # Bernoulli rounding
            A_l = np.zeros(alpha_.shape)
            S_i = np.zeros(xi_.shape)

            flag = False
            count = 0
            while not flag:
                for t in range(int(2 * np.log(m))):
                    A_tilt = np.random.binomial(1, alpha_.value)
                    S_tilt = np.random.binomial(1, xi_.value)
                    A_l = np.maximum(A_l, A_tilt)
                    S_i = np.maximum(S_i, S_tilt)
                flag = self._feasibility(train_sets[cls], prob_cls.value, A_l, S_i, nbr_list, Cl)
                count += 1
                if count > 100: break

            for ref_idx, true_idx in enumerate(train_sets[cls]):
                self.alpha[true_idx] = A_l[ref_idx]
                self.xi[true_idx] = S_i[ref_idx]
            del alpha_, xi_, A_l, S_i


    def objective_value(self):
        """
        Implement a function to compute the objective value of the integer optimization
        problem after the training phase

		may not be equal to the optimal value of LP

        """
        self.obj_val = np.sum(self.xi) + self.lambda_*np.sum(self.alpha)
        return self.obj_val


    def compute_dist_l2(self, X, instances):
        """
        Compute the L2 distance between each point
        Use broadcast to achieve computational efficient
        :return dists: an array of shape (instances.shape[0], X.shape[0])
        """
        mat_mul = np.matmul(instances, X.transpose())
        X_norm = ((np.linalg.norm(instances, axis=1))**2)[np.newaxis].transpose()
        Xtr_norm = ((np.linalg.norm(X.transpose(), axis=0))**2)[np.newaxis]

        dists = -2*mat_mul+X_norm
        dists += Xtr_norm
        dists[dists<0] = 0
        dists = np.sqrt(dists)

        return dists


    def compute_dist_l1(self, X, instances):
        """
        Compute the L1 distance between each point
        Use 1loop method to save computation
        :return dists: an array of shape (instances.shape[0], X.shape[0])
        """
        num_test = instances.shape[0]
        num_train = X.shape[0]
        dists = np.zeros((num_test, num_train), dtype=np.float64)

        for i in range(instances.shape[0]):
            dists[i] = np.sum(abs(X-instances[i]), axis=1)
        return dists


    def predict(self, instances, test_error=True):
        """
        Predicts the label for an array of instances using the framework learnt
        Input:
            instances: 2D array testing data of shape (N,D)
            test_error: a scalar
        Return:
            pred: prediction result as an array (D,) for test error
            miscls_count: scalar counts the number of uncovered data
        """
        distance = self.compute_dist(self.X_train, instances)
        pred = np.empty(len(instances), dtype='int')
        miscls_count = 0

        for i in range(len(instances)):
            j = np.argmin(distance[i])

            # Test error prediction
            tmp = self.y_train[j]
            pred[i] = tmp

            # Uncover count
            if self.epsilon_<=distance[i][j]:
                miscls_count += 1
        return pred, miscls_count


def cross_val(data, target, epsilon_, lambda_, k, verbose=False):
    """
    Implement a function which will perform k fold cross validation
    for the given epsilon and lambda and returns the average test error and number of prototypes
    Output:
         average test error
         average number of prototypes
         average objective value
         average cover error
    """
    kf = KFold(n_splits=k, random_state = 40)
    score = 0
    prots = 0
    obj_val = 0
    uncover_error = 0
    fold = 0
    for train_index, test_index in kf.split(data):
        fold += 1
        ps = classifier(data[train_index], target[train_index], epsilon_, lambda_, True)
        ps.train_lp(verbose)
        obj_val += ps.objective_value()
        pred, miscls = ps.predict(data[test_index])
        tmp = sklearn.metrics.accuracy_score(target[test_index], pred)
        score += tmp
        miscls /= len(test_index)
        uncover_error += miscls
        '''implement code to count the total number of prototypes learnt and store it in prots'''
        prots = np.sum(ps.alpha)

    score /= k
    prots /= k
    obj_val /= k
    uncover_error /= k

    return score, prots, obj_val, uncover_error


if __name__ == '__main__':
    """
    For sanity check, please run Iris_sanity_check()
    
    For k-fold validation on Breast cancer data set, please run Breast_variation(fold=k)
    
    For training on Digits data set and get the objective value, please run Train_digits()
    
    To switch to L1 distance, initialize in classifier(X, y, epsilon, lambda, L2=False). Otherwise the default is L2 distance
    """
    pass
