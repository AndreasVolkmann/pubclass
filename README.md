# pubclass
A model to predict the LOV of a publication.

This is a simple model, trained on dictionary / wikipedia definitions of medical terms (Endo, Derma, Gyno).

It uses the CountVectorizer to extract the features from the unstructured text.
Then, LogisticRegression and MultinominalNaiveBayes is used to fit and predict.

The sample unobserved input is taken from the abstracts of publications from PubMed.

