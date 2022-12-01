
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from nltk.corpus import stopwords

from sklearn.model_selection import KFold
# import mse
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
# import linear regression
from sklearn.linear_model import LinearRegression
# set matplotlib style
# import dummy regressor
from sklearn.dummy import DummyRegressor
plt.style.use('seaborn-whitegrid')


corpus = pd.read_csv("data/coffee_review.csv")
# corpus text to str
corpus['text'] = corpus['text'].astype(str)

# remove stop words
stop_words = stopwords.words('english')
# add and to stop words
stop_words.append('and')

corpus['text'] = corpus['text'].apply(lambda x: ' '.join(
    [word for word in x.split() if word not in (stop_words)]))


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    corpus['text'], corpus['rating'], test_size=0.2, random_state=42)
# to numpy array
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


def get_ridge_model(c=1.0, bigrams=False):
    """Return a Ridge model pipeline with or without bigrams.
    Parameters
    ----------
    alpha : float
        The regularization strength of the Ridge regressor.
    bigrams : bool
        Whether to include bigrams in the model.
    """
    if bigrams:
        ngram_range = (2, 2)
    else:
        ngram_range = (1, 1)
    return Pipeline([
        ('vect', TfidfVectorizer(ngram_range=ngram_range)),
        ('clf', Ridge(alpha=1/(2*c), fit_intercept=True))
    ])


def get_linear_regresssion(bigrams=False, vectorizer="TfIDF"):
    """Return a Linear Regression model pipeline with or without bigrams.
    Parameters
    ----------
    bigrams : bool
        Whether to include bigrams in the model.
    vectorizer : str
        The vectorizer to use.
    """

    if bigrams:
        ngram_range = (2, 2)
    else:
        ngram_range = (1, 1)
    if vectorizer == "TfIDF":
        return Pipeline([
            ('vect', TfidfVectorizer(ngram_range=ngram_range)),
            ('clf', LinearRegression())
        ])
    elif vectorizer == "Count":
        return Pipeline([
            ('vect', CountVectorizer(ngram_range=ngram_range)),
            ('clf', LinearRegression(fit_intercept=True))
        ])


def get_knn_model(n_neighbors=5, bigrams=False):
    """Return a KNN model pipeline with or without bigrams.
    Parameters
    ----------
    n_neighbors : int
        The number of neighbors to use by default for kneighbors queries.
    bigrams : bool
        Whether to include bigrams in the model.
    """
    if bigrams:
        ngram_range = (2, 2)
    else:
        ngram_range = (1, 1)
    return Pipeline([
        ('vect', TfidfVectorizer(ngram_range=ngram_range)),
        ('clf', KNeighborsRegressor(n_neighbors=n_neighbors))
    ])


def get_error_plot(models, values, x_train, y_train, title='', x_label=''):
    """Return a plot of the error for each model.
    Parameters
    ----------
    models : list
        A list of models.
    values : list
        A list of values to test.
    x_train : array
        The training data.
    y_train : array
        The training labels.
    title : str
        The title of the plot.
    x_label : str
        The label of the x-axis.
    """
    # do k-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mean_errors = []
    std_errors = []
    best_model = None
    best_score = 100
    best_value = None
    for i, model in enumerate(models):
        errors = []
        for train_index, test_index in kf.split(x_train):
            X_train, X_val = x_train[train_index], x_train[test_index]
            Y_train, Y_val = y_train[train_index], y_train[test_index]
            model.fit(X_train, Y_train)
            error = mean_squared_error(Y_val, model.predict(X_val))
            errors.append(error)
        mean_errors.append(np.mean(errors))
        std_errors.append(np.std(errors))
        if np.mean(errors) < best_score:
            best_model = model
            best_score = np.mean(errors)
            best_value = values[i]
    fig, ax = plt.subplots(1, 1)
    # error plot
    ax.errorbar(
        values,
        mean_errors,
        yerr=std_errors,
        label="validation",
        capsize=5,
        capthick=2,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("mean squared error")
    ax.set_title(title)
    # set position of legend
    ax.legend(loc='upper right')
    print("Best value: ", best_value)
    return fig, best_model


c_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]
k_values = [1, 11, 21, 51, 101, 201]


knn_models_ug = [get_knn_model(k) for k in k_values]
ridge_models_ug = [get_ridge_model(c) for c in c_values]
knn_models_bg = [get_knn_model(k, bigrams=True) for k in k_values]
ridge_models_bg = [get_ridge_model(c, bigrams=True) for c in c_values]


# plot error vs hyperparameter value
erroplot_knn_ug = get_error_plot(
    knn_models_ug, k_values, X_train, y_train, title='KNN unigram', x_label='Kf')
# save plot tightly
erroplot_knn_ug[0].savefig('plots/knn_ug.png', dpi=300, bbox_inches='tight')
best_knn_ug = erroplot_knn_ug[1]


# plot error vs hyperparameter value
erroplot_ridge_ug = get_error_plot(
    ridge_models_ug, c_values, X_train, y_train, title='Ridge unigram', x_label='C')
# save plot tightly
erroplot_ridge_ug[0].savefig(
    'plots/ridge_ug.png', dpi=300, bbox_inches='tight')
best_ridge_ug = erroplot_ridge_ug[1]


# plot error vs hyperparameter value
erroplot_knn_bg = get_error_plot(
    knn_models_bg, k_values, X_train, y_train, title='KNN bigram', x_label='Kf')
# save plot tightly
erroplot_knn_bg[0].savefig('plots/knn_bg.png', dpi=300, bbox_inches='tight')
best_knn_bg = erroplot_knn_bg[1]


# ridge bigram
erroplot_ridge_bg = get_error_plot(
    ridge_models_bg, c_values, X_train, y_train, title='Ridge bigram', x_label='C')
# save plot tightly
erroplot_ridge_bg[0].savefig(
    'plots/ridge_bg.png', dpi=300, bbox_inches='tight')
best_ridge_bg = erroplot_ridge_bg[1]


# get linear regression models
lr_ug = get_linear_regresssion()
lr_bg = get_linear_regresssion(bigrams=True)
# for coount vectorizer
lr_ug_count = get_linear_regresssion(vectorizer="Count")
lr_bg_count = get_linear_regresssion(bigrams=True, vectorizer="Count")


# fit each model on the training data
lr_ug.fit(X_train, y_train)
lr_bg.fit(X_train, y_train)
lr_ug_count.fit(X_train, y_train)
lr_bg_count.fit(X_train, y_train)
# same for ridge and knn
best_ridge_ug.fit(X_train, y_train)
best_ridge_bg.fit(X_train, y_train)
best_knn_ug.fit(X_train, y_train)
best_knn_bg.fit(X_train, y_train)


# dummy regressor
dummy = DummyRegressor(strategy='mean')


models = [
    ('Dummy', dummy),
    ('Linear Regression unigram BoW', lr_ug_count),
    ('Linear Regression bigram BoW', lr_bg_count),
    ('Linear Regression unigram TfIdF', lr_ug),
    ('Linear Regression bigram TfIdF', lr_bg),
    ('Ridge unigram TfIdF', best_ridge_ug),
    ('Ridge bigram TfIdF', best_ridge_bg),
    ('KNN unigram', best_knn_ug),
    ('KNN bigram', best_knn_bg),

]


# score each model and put results in a dataframe
results = []
for name, model in models:
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    results.append([name, mse, mae])

results = pd.DataFrame(results, columns=['Model', 'MSE', 'MAE'])


results

# save results to csv
results.to_csv('results/results.csv', index=False)


feature_names = best_ridge_bg.named_steps['vect'].get_feature_names()
coefs_with_fns = sorted(
    zip(best_ridge_bg.named_steps['clf'].coef_, feature_names))
top = zip(coefs_with_fns[:10], coefs_with_fns[:-(10 + 1):-1])
df = pd.DataFrame()
for (coef_1, fn_1), (coef_2, fn_2) in top:
    df = df.append({"coef_1": coef_1, "fn_1": fn_1,
                   "coef_2": coef_2, "fn_2": fn_2}, ignore_index=True)


# make a barplot with the most important features color positive and negative in blue and red in the same plot
# !pip install seaborn

plt.rc('font', size=30)
# change plot size
plt.figure(figsize=(18, 15))
# set sns style
# set matplotlib style
plt.style.use("seaborn")
sns.set_style("whitegrid")
# sns.set(style="whitegrid")
values_neg = list(df["fn_1"].values)
values_neg.reverse()
coef_neg = list(df["coef_1"].values)
coef_neg.reverse()
sns.set_color_codes("pastel")
x = list(df["fn_2"].values) + values_neg
y = list(df["coef_2"].values) + coef_neg
# colors = ["red" if y_ < 0 else "blue" for y_ in y]
# set pastel  solor manurally pastel red and blue for negative and positive
colors = ["#e74c3c" if y_ < 0 else "#3498db" for y_ in y]
# vertical barplot
sns.barplot(x=y, y=x, palette=colors)
# rotate the labels
# add labels
plt.xlabel("Coefficient", fontsize=45)
plt.ylabel("Feature", fontsize=45)
# bigger text for axis labels
plt.xticks(fontsize=45)
plt.yticks(fontsize=45)

# show the seaborn plot
plt.show()

# save plot tightly
plt.savefig('plots/ridge_bg_coef.png', dpi=300, bbox_inches='tight')
