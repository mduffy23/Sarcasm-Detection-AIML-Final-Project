# Sarcasm Detection

This project leverages a dataset from https://www.kaggle.com/datasets/danofer/sarcasm to train sarcasm detection models. The dataset contains sarcastic and non-sarcastic Reddit posts, which give a solid representation of how people use sarcasm in textual conversation. For modelling, Term Frequency-Inverse Document Frequency (TF-IDF) was trained as a baseline model, and a fine-tuned RoBERTa model was trained as the final model. The Jupyter notebook associated with this project contains commentary and code for the model fitting and testing. Please view that to gain a full understanding of this project. Below are some key notes and resources.

- TF-IDF evaluates how important a word is to a document in a collection, or corpus, by multiplying the term frequency (TF)—how often a word appears in a document—by the inverse document frequency (IDF), which measures how rare a word is across all documents. Certain *give away* phrases in sarcastic statements should be caught by this method, which makes it a viable solution, but TF-IDF will not be able to pick up on word order, context, and sentiment contrast, which are all important for sarcasm detection. Sarcasm can often be subtle, and a model needs to understand these concepts to be viable. Once the TF-IDF vectors are applied, a logistic regression is used for modelling.

- RoBERTa is a large contextual language transformer model that solves a lot of the shortcomings of TF-IDF. RoBERTa has the same architecture as BERT, but was trained on ten times more data, which makes it generally better than BERT. RoBERTa has a few other training improvements like dynamic masking (versus static masking), larger batches, and more training iterations, which also lead to it performing better than the original BERT model. Because of all these training methods, RoBERTa is a great model to fine-tune for sarcasm detection, as it can pick up on subtle context changes based on sentence structure, punctuation, and contradictions, because it is trained on such a large corpus of text. 

# Methods

- For TF-IDF, the comments were lemmatized (transforms words that mean the same thing but are in different parts of speech to be the same for the tokenizer) to help with understanding similar words. RoBERTa, on the other hand, does not need the words lemmatized since it uses a subword tokenizer and contextual embeddings, allowing RoBERTa to understand relationships between word forms based on shared subwords, without explicit lemmatization. Similar words like run, runs, ran, and running will have similar vectors when they appear in similar contexts. Since RoBERTa was trained on raw text, it should actually perform better on the raw statements; it understands things like tense and tone. 

- Both models were given a secondary variable that measures if the statement has common patterns of sarcasm. This is not a hack to say all statements that start with "Oh yeah," are sarcastic, but rather a baseline understanding that these particular words, phrases, and punctuation are often used when typing out sarcasm, trying to further guide the models into understanding what makes a statement sarcastic. Conventions such as elongated spelling ("woooow"), exaggerated words ("literally"), and ironic punctuation ("!!") are set as *tells* for sarcasm in statements.

- RoBERTa is a HuggingFace transformer model, but to utilize the lexical features, a torch model was defined to leverage the pretrained RoBERTa model and then add the lexical features to the model. The model was then hyperparameter-tuned with the optuna library, which offered the best learning rate, dropout, and weight decay. The regular transformer training arguments and trainer were used to fine-tune the model to predict sarcasm!

- This model was saved for usage in a small web application on Hugging Face and deployed on a Hugging Face space. FastAPI was used to generate a simple endpoint to hit the model from an HTML page.

# Considerations
- This project shows that machines can understand sarcasm. From a philosophical perspective, this can be used to show how close AI can come to modelling human behavior. The big question is whether the machine is beginning to find these statements funny. A large part of understanding sarcasm is picking up on the humorous nature of the statement, how different is *getting a joke* from understanding the underlying linguistic aspects that make it interesting? Humour is one of those human phenomena that seems too creative and unpredictable for a machine to understand, but it is difficult to argue with fairly accurate results. While I do not think machines at this point *get* humour in a way that they can make a perfect quip at the perfect time or deliver a punch line just right, but it does seem to know something substantial about sarcasm. Curious where that may go. 

# External Resources
- Deployed Application: https://huggingface.co/spaces/mduffy-23/sarcasm-detection

- Model Repo: https://huggingface.co/mduffy-23/RoBertaWithLexicalFeatures-Sarcasm

- Slides deck for project: https://docs.google.com/presentation/d/1IfV45Ue9LjGM3RReL6iFR1dRjt7vOaRMjJT1EO4iQWE/edit?usp=sharing
