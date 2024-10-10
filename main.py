import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
import math
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def tokenize(text, remove_stop_words = False):
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Replace punctuation with spaces, except for apostrophes
    text = re.sub(r'[^\w\s\']', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Handle contractions (simplified approach)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'m", " am", text)

    # Split on whitespace
    tokens = text.split()

    # Remove stop words and apply stemming
    if remove_stop_words:
        tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and re.search(r'\w', token)]
    else:
        tokens = [stemmer.stem(token) for token in tokens if re.search(r'\w', token)]
    return tokens

def unknown_word_handle(tokens):
    UNK_TOKEN = "<UNK>"
    UNK_THRESHOLD = 1
    token_freq = defaultdict(int)
    for token in tokens:
        token_freq[token] += 1

    # Step 2: Replace low-frequency tokens with the UNK_TOKEN
    tokens = [token if token_freq[token] > UNK_THRESHOLD else UNK_TOKEN for token in tokens]
    return tokens

def apply_smoothing_unigram(unigram_count, total_tokens, vocab_size, smoothing, k=1):
    if not smoothing:
        return unigram_count / total_tokens
    elif smoothing == "laplace":
        return (unigram_count + 1) / (total_tokens + vocab_size)
    elif smoothing == "add-k":
        return (unigram_count + k) / (total_tokens + k * vocab_size)

def apply_smoothing_bigram(unigram_count, bigram_count, vocab_size, smoothing, k=1):
    if not smoothing:
        return bigram_count / unigram_count if unigram_count != 0 else 0
    elif smoothing == "laplace":
        return (bigram_count + 1) / (unigram_count + vocab_size)
    elif smoothing == "add-k":
        return (bigram_count + k) / (unigram_count + k * vocab_size)

def unigram_and_bigram_model(tokens, smoothing='laplace', k=1):
    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    tokens = ['<s>'] + tokens + ['</s>']
    total_tokens = len(tokens)
    vocab_size = len(set(tokens))
    tokens = unknown_word_handle(tokens)
    # Count unigrams and bigrams in one pass
    for i in range(len(tokens) - 1):
        unigram_counts[tokens[i]] += 1
        bigram = (tokens[i], tokens[i + 1])
        bigram_counts[bigram] += 1

    # Add the last token for unigram count
    unigram_counts[tokens[-1]] += 1

    unigram_model = {}
    bigram_model = {}

    for token in unigram_counts:
        unigram_prob = apply_smoothing_unigram(unigram_counts[token], total_tokens, vocab_size, smoothing, k)
        unigram_model[token] = unigram_prob

    for bigram in bigram_counts:
        bigram_prob = apply_smoothing_bigram(unigram_counts[bigram[0]], bigram_counts[bigram], vocab_size, smoothing, k)
        bigram_model[bigram] = bigram_prob

    return unigram_model, bigram_model

def calculate_perplexity(tokens, model, ngram_type='bigram'):
    N = len(tokens)  # Total number of tokens
    log_sum = 0

    for i in range(1, N):  # Start from index 1 because for bigrams, we need a previous token
        if ngram_type == 'bigram':
            prev_token = tokens[i - 1]
            token = tokens[i]
            bigram = (prev_token, token)
            if bigram in model:
                prob = model[bigram]
            else:
                if (prev_token, '<UNK>') in model:
                    prob = model[(prev_token, '<UNK>')]
                else:
                    prob = model[('<UNK>', '<UNK>')]
        elif ngram_type == 'unigram':
            token = tokens[i]

            # Get unigram probability
            if token in model:
                prob = model[token]
            else:
                prob = model['<UNK>']
        log_sum += -math.log(prob)

    # Perplexity formula
    perplexity = math.exp(log_sum / N)
    return perplexity


if __name__ == '__main__':

    with open("A1_DATASET/train.txt", "r") as corpus:
        reviews = corpus.readlines()

    all_tokens = []
    for review in reviews:
        tokens = tokenize(review, remove_stop_words=False)
        all_tokens.extend(tokens)

    with open("A1_DATASET/val.txt", "r") as validation_file:
        validation_reviews = validation_file.readlines()

    validation_tokens = []
    for review in validation_reviews:
        tokens = tokenize(review, remove_stop_words=False)
        validation_tokens.extend(tokens)

    unsmoothed_unigram, unsmoothed_bigram = unigram_and_bigram_model(all_tokens, False)
    # Build unigram and bigram models from the tokens
    laplace_unigram_model, laplace_bigram_model = unigram_and_bigram_model(all_tokens, 'laplace', 1)

    addk_unigram_model_1, addk_bigram_model_1 = unigram_and_bigram_model(all_tokens, 'add-k', 0.01)
    addk_unigram_model_2, addk_bigram_model_2 = unigram_and_bigram_model(all_tokens, 'add-k', 0.1)

    # Compute and print perplexity for unsmoothed models (unigram and bigram)
    print("\n--- Unsmoothened Model Perplexities ---")

    # Training set perplexity
    unsmoothed_unigram_perplexity_train = calculate_perplexity(all_tokens, unsmoothed_unigram, ngram_type='unigram')
    print(f"Unigram Perplexity on Train set: {unsmoothed_unigram_perplexity_train:.6f}")

    unsmoothed_bigram_perplexity_train = calculate_perplexity(all_tokens, unsmoothed_bigram, ngram_type='bigram')
    print(f"Bigram Perplexity on Train set: {unsmoothed_bigram_perplexity_train:.6f}")

    # Validation set perplexity
    unsmoothed_unigram_perplexity_val = calculate_perplexity(validation_tokens, unsmoothed_unigram, ngram_type='unigram')
    print(f"Unigram Perplexity on Validation set: {unsmoothed_unigram_perplexity_val:.6f}")

    unsmoothed_bigram_perplexity_val = calculate_perplexity(validation_tokens, unsmoothed_bigram, ngram_type='bigram')
    print(f"Bigram Perplexity on Validation set: {unsmoothed_bigram_perplexity_val:.6f}")

    # Compute and print perplexity for Laplace-smoothed models (unigram and bigram)
    print("\n--- Laplace-Smoothened Model Perplexities ---")

    # Training set perplexity
    laplace_unigram_perplexity_train = calculate_perplexity(all_tokens, laplace_unigram_model, ngram_type='unigram')
    print(f"Unigram Perplexity on Train set: {laplace_unigram_perplexity_train:.6f}")

    laplace_bigram_perplexity_train = calculate_perplexity(all_tokens, laplace_bigram_model, ngram_type='bigram')
    print(f"Bigram Perplexity on Train set: {laplace_bigram_perplexity_train:.6f}")

    # Validation set perplexity
    laplace_unigram_perplexity_val = calculate_perplexity(validation_tokens, laplace_unigram_model, ngram_type='unigram')
    print(f"Unigram Perplexity on Validation set: {laplace_unigram_perplexity_val:.6f}")

    laplace_bigram_perplexity_val = calculate_perplexity(validation_tokens, laplace_bigram_model, ngram_type='bigram')
    print(f"Bigram Perplexity on Validation set: {laplace_bigram_perplexity_val:.6f}")

    # Compute and print perplexity for Add-K smoothed models (unigram and bigram)
    print("\n--- Add-K Smoothened Model Perplexities (k=0.01) ---")

    # Training set perplexity
    addk_unigram_perplexity_train = calculate_perplexity(all_tokens, addk_unigram_model_1, ngram_type='unigram')
    print(f"Unigram Perplexity on Train set: {addk_unigram_perplexity_train:.6f}")

    addk_bigram_perplexity_train = calculate_perplexity(all_tokens, addk_bigram_model_1, ngram_type='bigram')
    print(f"Bigram Perplexity on Train set: {addk_bigram_perplexity_train:.6f}")

    # Validation set perplexity
    addk_unigram_perplexity_val = calculate_perplexity(validation_tokens, addk_unigram_model_1, ngram_type='unigram')
    print(f"Unigram Perplexity on Validation set: {addk_unigram_perplexity_val:.6f}")

    addk_bigram_perplexity_val = calculate_perplexity(validation_tokens, addk_bigram_model_1, ngram_type='bigram')
    print(f"Bigram Perplexity on Validation set: {addk_bigram_perplexity_val:.6f}")
    print("\n--- Add-K Smoothened Model Perplexities (k=0.1) ---")
    # Training set perplexity
    addk_unigram_perplexity_train = calculate_perplexity(all_tokens, addk_unigram_model_2, ngram_type='unigram')
    print(f"Unigram Perplexity on Train set: {addk_unigram_perplexity_train:.6f}")

    addk_bigram_perplexity_train = calculate_perplexity(all_tokens, addk_bigram_model_2, ngram_type='bigram')
    print(f"Bigram Perplexity on Train set: {addk_bigram_perplexity_train:.6f}")

    # Validation set perplexity
    addk_unigram_perplexity_val = calculate_perplexity(validation_tokens, addk_unigram_model_2, ngram_type='unigram')
    print(f"Unigram Perplexity on Validation set: {addk_unigram_perplexity_val:.6f}")

    addk_bigram_perplexity_val = calculate_perplexity(validation_tokens, addk_bigram_model_2, ngram_type='bigram')
    print(f"Bigram Perplexity on Validation set: {addk_bigram_perplexity_val:.6f}")
