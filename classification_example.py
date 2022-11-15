import os
import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph.
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

#print(os.getcwd()) # check working directory
#os.chdir("C:/Users/Joyce Cheah/Documents") # change working directory

df = pd.read_csv("rows.csv")
print(df.shape)
print(df.head())
df['Consumer complaint narrative'].dropna()

# Create a new dataframe with two columns
df1 = df[['Product', 'Consumer complaint narrative']].copy()
df1 = df1.dropna()
print(df1)

# Renaming second column for a simpler name
df1.columns = ['Product', 'Consumer_complaint']
df1.Product.unique()

# Because the computation is time consuming (in terms of CPU), the data was sampled
df2 = df1.sample(10000, random_state=1).copy()

# Renaming categories
df2.replace({'Product': 
             {'Credit reporting, credit repair services, or other personal consumer reports': 
              'Credit reporting, repair, or other', 
              'Credit reporting': 'Credit reporting, repair, or other',
             'Credit card': 'Credit card or prepaid card',
             'Prepaid card': 'Credit card or prepaid card',
             'Payday loan': 'Payday loan, title loan, or personal loan',
             'Money transfer': 'Money transfer, virtual currency, or money service',
             'Virtual currency': 'Money transfer, virtual currency, or money service'}}, 
            inplace= True)
df2.Product.unique()
pd.DataFrame(df2.Product.unique())

# Create a new column 'category_id' with encoded categories 
df2['category_id'] = df2['Product'].factorize()[0]
category_id_df = df2[['Product', 'category_id']].drop_duplicates()

category_id_df

# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Product']].values)

fig = plt.figure(figsize=(10,3))
colors = ['grey','grey','grey','grey','grey','grey','grey','grey','grey',
    'grey','darkblue','darkblue','darkblue']
df2.groupby('Product').Consumer_complaint.count().sort_values().plot.barh(
    ylim=0, color=colors, title= 'NUMBER OF COMPLAINTS IN EACH PRODUCT CATEGORY')
plt.xlabel('Number of ocurrences', fontsize = 10)


#transform the texts into vectors using Term Frequency-Inverse Document Frequency (TFIDF) 
#and evaluate how important a particular word is in the collection of words

#min_df: remove the words which has occurred in less than ‘min_df’ number of files.
#Sublinear_tf: if True, then scale the frequency in logarithmic scale.
#Stop_words: it removes stop words which are predefined in ‘english’.

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#can be changed to CountVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2), #this only considers unigram and bigram
                        stop_words='english')

# We transform each complaint into a vector
features = tfidf.fit_transform(df2.Consumer_complaint).toarray()
labels = df2.category_id
print("Each of the %d complaints is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))

# Finding the three most correlated terms with each of the product categories

N = 3
for Product, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id) #returns a tuple of (Random Variates, Probability Distribution)
    indices = np.argsort(features_chi2[0]) #sort the random variates into index from 1 to 27507, from largest Chi2 to smallest Chi2
    feature_names = np.array(tfidf.get_feature_names())[indices] #sort the features by the index
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1] #features with one word only
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2] #features with two words
    print("n==> %s:" %(Product))
    print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:]))) #smallest Chi2 means highest correlation
    print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))
    

#Split into traning and testing data

X = df2['Consumer_complaint'] # Collection of documents
y = df2['Product'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)


#keep models in a list for looping

models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0, max_iter=400) #this did not run as not enough memory
]

# 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models))) # set up a dataframe with index 0 to 19

# to find out which model is the best performing
entries = []
for model in models:
    model_name = model.__class__.__name__
        # cross_val_score(model, X, y, scoring=, cv=)
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV) # accuracies is an array of 5 scores
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy)) #fold_idx is the fold number index
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy']) #turn into a dataframe
print(cv_df)

mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
print(acc)

sns.boxplot(x='model_name', y='accuracy', 
            data=cv_df, 
            color='lightblue', 
            showmeans=True)
plt.title("MEAN ACCURACY (cv = 5)", size=14)

#training model using LinearSVC, our best model, to evaluate performance on test data
X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, 
                                                                               labels, 
                                                               df2.index, test_size=0.25, 
                                                               random_state=1)
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#checking out the predicted labels
with np.printoptions(threshold=np.inf):
    print(y_pred)

# Classification report
print('CLASSIFICATIION METRICS')
print(metrics.classification_report(y_test, y_pred, 
                                    target_names= df2['Product'].unique())) # target_names show the product names instead of product index   

#confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=category_id_df.Product.values, 
            yticklabels=category_id_df.Product.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC", size=16)
plt.autoscale() 

#checking out the incorrectly predicted complaints
pd.set_option('display.max_colwidth', None)
from IPython.display import display
for predicted in category_id_df.category_id:
  for actual in category_id_df.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 10:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Product', 'Consumer complaint narrative']])
      print('')
      







#quick predictions on unseen data
X = df2['Consumer_complaint'] # Collection of documents
y = df2['Product'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer

tfidf = TfidfVectorizer(min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english')

fitted_vectorizer = tfidf.fit(X_train)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)
model = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)


#prediction example
complaint = """Respected Sir/ Madam, I am exploring the possibilities for financing my daughter 's 
XXXX education with private loan from bank. I am in the XXXX on XXXX visa. 
My daughter is on XXXX dependent visa. As a result, she is considered as international student. 
I am waiting in the Green Card ( Permanent Residency ) line for last several years.
I checked with Discover, XXXX XXXX websites. While they allow international students to apply for loan, they need cosigners who are either US citizens or Permanent Residents. I feel that this is unfair.
I had been given mortgage and car loans in the past which I closed successfully. I have good financial history. 
I think I should be allowed to remain cosigner on the student loan. I would be much obliged if you could look into it. Thanking you in advance. Best Regards"""
print(model.predict(fitted_vectorizer.transform([complaint])))




from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

X = df2['Consumer_complaint'] # Collection of documents
y = df2['Product'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)

kbest = 100
Vectorizer = CountVectorizer

pipeline2 = Pipeline([
    ('vectorizer', Vectorizer(tokenizer    = 'word_tokenize',
                              stop_words   = 'english', 
                              ngram_range  = (1, 2))),
    ('selector', SelectKBest()),
    ('classifier', MultinomialNB()), #here is where you would specify an alternative classifier
])

search_space = [{'selector__k'              : range(1000, 3001, 100), # this starts at 50 and ends at 700 with steps of 50
                 #'selector__score_func'     : [mutual_info_classif, chi2],
                 'vectorizer'               : [CountVectorizer()], # , TfidfVectorizer()
                 #'vectorizer__max_features' : [700, 2000]
                }]

scoring = {'Accuracy': make_scorer(accuracy_score)}

clf2 = GridSearchCV(estimator          = pipeline2, 
                    param_grid         = search_space, 
                    scoring            = scoring,
                    cv                 = 5, 
                    refit              = 'Accuracy',
                    return_train_score = True,
                    verbose            = 1)
clf2 = clf2.fit(X_train, y_train)

means = clf2.cv_results_['mean_test_Accuracy']
stds  = clf2.cv_results_['std_test_Accuracy']

for mean, std, params in zip(means, stds, clf2.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print(clf2.best_estimator_)

print(clf2.best_score_)

# get our results
results = clf2.cv_results_

plt.figure(figsize=(16, 16))
plt.title("GridSearchCV evaluating parameters using the Accuracy scorer.",
          fontsize=16)

plt.xlabel("k")
plt.ylabel("Accuracy")

ax = plt.gca()

# adjust these according to your accuracy results and range values.
ax.set_xlim(1000, 3000)
ax.set_ylim(0.60, 1)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_selector__k'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['b']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f with k=%s" % (best_score, X_axis[best_index]),
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()






#now that we have chosen the best K, we will search for ther hyperparameters
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
Vectorizer = CountVectorizer

pipeline3 = Pipeline([
    ('vectorizer', Vectorizer(tokenizer    = 'word_tokenize',
                              stop_words   = 'english',
                              lowercase    = True)),
    ('selector', SelectKBest(k=2800)),
    ('classifier', MultinomialNB()), #here is where you would specify an alternative classifier
])

search_space2 = [{'selector__score_func'     : [chi2],
                 'vectorizer'               : [CountVectorizer() , TfidfVectorizer()], 
                 'vectorizer__ngram_range'  : [(1,1),(1,2)],
                 'vectorizer__min_df'       : [3,5]
                }]

scoring2 = {'Accuracy': make_scorer(accuracy_score)}

clf3 = GridSearchCV(estimator          = pipeline3, 
                    param_grid         = search_space2, 
                    scoring            = scoring2,
                    cv                 = 5, 
                    refit              = 'Accuracy',
                    return_train_score = True,
                    verbose            = 1)
clf3 = clf3.fit(X_train, y_train)

means = clf3.cv_results_['mean_test_Accuracy']
stds  = clf3.cv_results_['std_test_Accuracy']

for mean, std, params in zip(means, stds, clf3.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
print(clf3.best_estimator_)

print(clf3.best_score_)
