# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
# from sklearn.model_selection import ShuffleSplit


# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


# %%
#from sklearn.datasets import load_digits
#data2 = np.genfromtxt()
data2 = np.genfromtxt('sat.trn')


# %%
X, y = data2[:, :-1], data2[:, -1]
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


# %%
def plot_learning_curve(estimator, title, X, y, 
                        train_sizes=np.linspace(.1, 1.0, 6)):

    plt.figure()
    _, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].set_title('Data2:' + title)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ =         learning_curve(estimator, X, y, cv=5,
                       train_sizes=train_sizes,
                       shuffle=True,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_time")
    axes[1].set_title(f"Data2: Scalability of {title}")

    plt.savefig(f"data2_{title}.png", bbox_inches='tight')
    # plt.show()


# %%
def plot_validation_curve(estimator, title, X, y, param_name, param_range):

    train_scores, valid_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=5)

    train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    # valid_scores_std = np.std(valid_scores, axis=1)

    plt.figure()
    plt.title('Data2: ' + title)
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.plot(param_range, train_scores_mean, label='Training score')
    plt.plot(param_range, valid_scores_mean, label='Cross-validation score')
    plt.legend()

    plt.savefig(f"data2_{title}.png", bbox_inches='tight')
    # plt.show()


# %%
plot_validation_curve(KNeighborsClassifier(weights='uniform'), 'KNN_k_validation_curve', X, y, 'n_neighbors', np.arange(1, 105, 5))


# %%
plot_validation_curve(KNeighborsClassifier(weights='uniform', n_neighbors=12), 'KNN_p_validation_curve', X, y, param_name='p', param_range=[1, 2, 3, 4, 5])


# %%
knn_clf = KNeighborsClassifier(n_neighbors=12, weights='uniform', p=1)
plot_learning_curve(knn_clf, 'KNN_learning_curve', X, y)


# %%
plot_validation_curve(DecisionTreeClassifier(), 'DT_max_depth_validation_curve', X, y, param_name='max_depth', param_range=np.arange(1, 32))


# %%
plot_validation_curve(DecisionTreeClassifier(), 'DT_min_sample_split_validation_curve', X, y, param_name='min_samples_split', param_range=np.arange(2, 500, 2))


# %%
plot_learning_curve(DecisionTreeClassifier(max_depth=20, min_samples_split=5), 'DT_learning_curve', X, y)


# %%
plot_validation_curve(MLPClassifier(solver='adam', max_iter=500, early_stopping=True), 'ANN_hls_validation_curve', X, y, param_name='hidden_layer_sizes', param_range=[(i,) for i in range(5, 205, 5)])


# %%
plot_validation_curve(MLPClassifier(solver='adam', hidden_layer_sizes=(200), max_iter=500), 'ANN_activation_validation_curve', X, y, param_name='activation', param_range=['identity', 'logistic', 'tanh', 'relu'])


# %%
plot_validation_curve(MLPClassifier(solver='adam', hidden_layer_sizes=(100, 100), max_iter=500), 'ANN_activation_validation_curve', X, y, param_name='activation', param_range=['identity', 'logistic', 'tanh', 'relu'])


# %%
ann_clf = MLPClassifier(solver='adam', hidden_layer_sizes=(200, 200), activation='logistic', max_iter=500)
plot_learning_curve(ann_clf, 'ANN_learning_curve', X, y)


# %%
plot_validation_curve(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1)), 'Boost_validation_curve_1', X, y, param_name='n_estimators', param_range=range(1, 50, 3))


# %%
plot_validation_curve(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5)), 'Boost_validation_curve_5', X, y, param_name='n_estimators', param_range=range(1, 50, 3))


# %%
plot_validation_curve(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10)), 'Boost_validation_curve_10', X, y, param_name='n_estimators', param_range=range(1, 50, 3))


# %%
plot_validation_curve(AdaBoostClassifier(base_estimator=DecisionTreeClassifier()), 'Boost_validation_curve_max', X, y, param_name='n_estimators', param_range=range(1, 50, 3))


# %%
plot_learning_curve(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), n_estimators=32), 'Boost_DT_learning_curve', X, y)


# %%
plot_validation_curve(SVC(), 'SVM_kernel_validation_curve', X, y, param_name='kernel', param_range=['linear', 'rbf', 'poly', 'sigmoid'])


# %%
plot_validation_curve(SVC(kernel='poly'), 'SVM_degree_validation_curve', X, y, param_name='degree', param_range=[1, 2, 3, 4, 5])


# %%
clf =SVC(kernel='poly', degree=3)
plot_learning_curve(clf, 'SVM_learning_curve', X, y)


# %%



