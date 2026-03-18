# ================================================
#         Naive Bayes Classifier for Beginners
# ================================================
# We will classify emails as SPAM or HAM (not spam)
# using simple math and Python — no libraries needed!


# -----------------------------------------------
# STEP 1: Our Training Data
# (text, label) pairs the model will learn from
# -----------------------------------------------

dataset = [
    ("win free money now",        "spam"),
    ("click here to claim prize", "spam"),
    ("you won a lottery",         "spam"),
    ("buy cheap medicine online", "spam"),
    ("free offer limited time",   "spam"),

    ("hey how are you",           "ham"),
    ("lets meet for lunch",       "ham"),
    ("please send me the file",   "ham"),
    ("see you tomorrow morning",  "ham"),
    ("happy birthday to you",     "ham"),
]


# -----------------------------------------------
# STEP 2: Count words in each class
# -----------------------------------------------

spam_words = {}   # how many times each word appears in spam
ham_words  = {}   # how many times each word appears in ham

spam_count = 0    # total number of spam messages
ham_count  = 0    # total number of ham messages

total_messages = len(dataset)

for (text, label) in dataset:
    words = text.lower().split()   # split sentence into words

    if label == "spam":
        spam_count += 1
        for word in words:
            spam_words[word] = spam_words.get(word, 0) + 1

    else:
        ham_count += 1
        for word in words:
            ham_words[word] = ham_words.get(word, 0) + 1

# All unique words across the entire dataset
vocab = set(list(spam_words.keys()) + list(ham_words.keys()))

print("=== Training Complete ===")
print(f"Spam messages : {spam_count}")
print(f"Ham messages  : {ham_count}")
print(f"Unique words  : {len(vocab)}")


# -----------------------------------------------
# STEP 3: Define the probability functions
# -----------------------------------------------

# P(spam) = how often spam appears in training data
def p_spam():
    return spam_count / total_messages

def p_ham():
    return ham_count / total_messages

# P(word | spam) = how likely a word appears in spam
# We add 1 to avoid zero probabilities (Laplace smoothing)
def p_word_given_spam(word):
    count = spam_words.get(word, 0) + 1          # add 1 (smoothing)
    total = sum(spam_words.values()) + len(vocab) # add vocab size
    return count / total

def p_word_given_ham(word):
    count = ham_words.get(word, 0) + 1
    total = sum(ham_words.values()) + len(vocab)
    return count / total


# -----------------------------------------------
# STEP 4: Classify a new message
# -----------------------------------------------

def classify(message):
    words = message.lower().split()

    # Start with the probability of each class
    # We use multiplication: P(spam) × P(w1|spam) × P(w2|spam) × ...
    prob_spam = p_spam()
    prob_ham  = p_ham()

    for word in words:
        prob_spam *= p_word_given_spam(word)
        prob_ham  *= p_word_given_ham(word)

    # Whichever probability is higher wins!
    if prob_spam > prob_ham:
        result = "SPAM"
    else:
        result = "HAM"

    return result, prob_spam, prob_ham


# -----------------------------------------------
# STEP 5: Test it on new messages!
# -----------------------------------------------

print("\n=== Classifying New Messages ===\n")

test_messages = [
    "win free prize now",
    "lets have lunch tomorrow",
    "click here free offer",
    "good morning how are you",
    "buy cheap lottery ticket",
]

for msg in test_messages:
    label, prob_spam, prob_ham = classify(msg)
    print(f"Message : \"{msg}\"")
    print(f"Result  : {label}")
    print(f"P(spam) = {prob_spam:.8f}  |  P(ham) = {prob_ham:.8f}")
    print()
