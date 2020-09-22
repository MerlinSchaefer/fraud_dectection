# What I learned doing this project

The first thing I learned was how to handle (highly) skewed data. I was familiar with slightly skewed data and checking whether the data follows certain distributions, but I never worked with a dataset, which was as skewed as this one.
I learned how far reaching this is, from metrics over splitting the data to the actual model selection, highly skewed data can lead to strong deviation from the "default" way of handling data.

I also learned to not only check for NAs, which I always did but also to check for possible duplicates, something I missed during my first exploration of the data.

After starting to implement my own solutions for cross validating the data, I realized that the scikit-learn function already had a way of implementing what I was looking for, this taught me to really read the documentation down to the last detail if necessary.

I also realized how quickly GridSearchCV and even RandomSearchCV can require large amounts of computing power. This taught me to be more careful when selecting the Hyperparameters to tune.

Overall I learned new things but also used and practiced many existing skills.

Unfortunately I couldn't build a Streamlit app or something interactive as the data was already the product of PCA and the features can't really be given through user input.