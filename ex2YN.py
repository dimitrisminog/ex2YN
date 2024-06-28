import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import numpy as np
import os
import sys
import tensorflow as tf
import random
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse

sys.stdout.reconfigure(encoding='utf-8')

# Συνάρτηση custom_error
def custom_error(y_true, y_pred):
    error = tf.where(y_pred <= y_true, y_true - y_pred, y_pred - y_true)
    return error

# Καθορισμός της διαδρομής του script
script_path = os.path.abspath(__file__)
print("Script path:", script_path)

# Φόρτωση αρχείου CSV
print("Loading data...")
dataframe = pd.read_csv(r"C:\\python\\iphi2802.csv", header=0, sep='\t', encoding='utf-8', engine='python')

# Φιλτράρισμα για τις γραμμές με region_main_id=1683
dataframe = dataframe[dataframe['region_main_id'] == 1683]

# Αφαίρεση σημείων στίξης
dataframe['text'] = dataframe['text'].str.replace(r'[.,\[\]\(\)\-]', '', regex=True)

# Συγκέντρωση όλων των λέξεων από τις επιγραφές
all_words = ' '.join(dataframe['text']).split()
print(f" number of words : {len(all_words)}")

# Print words before vectorization
print("\nWords before vectorization:")
print(all_words)

# Δημιουργία Counter για τις συχνότητες των λέξεων
word_cnt = Counter(all_words)

# Λίστα με όλες τις μοναδικές λέξεις
all_words_list = list(word_cnt.keys())
print(f"Total number of unique words: {len(all_words_list)}")

# Vectorizer με vocabulary όλες τις λέξεις
vectorizer = TfidfVectorizer(vocabulary=all_words_list)

# Υπολογισμός tf-idf τιμών για όλα τα κείμενα
tfidf_matrix = vectorizer.fit_transform(dataframe['text'])

# Επιλογή όλων των λέξεων
selected_indices = [vectorizer.vocabulary_[word] for word in all_words_list]
tfidf_values = tfidf_matrix[:, selected_indices]

#  κανονικοποίηση
dense_tfidf_values = tfidf_values.toarray()
scaler = MinMaxScaler(feature_range=(0, 1))
nrmlzd_tfidf_values = scaler.fit_transform(dense_tfidf_values)

# Εμφάνιση κανονικοποιημένων tf-idf τιμών
print("Normalized TF-IDF values:")
print(nrmlzd_tfidf_values)

# Λίστα των επιγραφών
texts = dataframe['text'].tolist()

# Συνολικός αριθμός λέξεων στο λεξικό
print(f"Total number of words in the lexicon: {len(all_words_list)}")

# Διασφάλιση ότι οι λέξεις που θα εισάγονται είναι εντός του λεξικού
def check_words(indices, max_index):
    return [min(max(idx, 1), max_index) for idx in indices]

# Μέγιστος δείκτης στο λεξικό
max_lexicon_index = len(all_words_list)

# Δημιουργία αρχικού πληθυσμού
def create_first_population(pop_size, max_index):
    population = []
    for _ in range(pop_size):
        word1_idx = random.randint(1, max_index)
        word2_idx = random.randint(1, max_index)
        individual = (word1_idx, word2_idx)
        population.append(individual)
    return population

# Μέγεθος πληθυσμού
population_size = 100
initial_population = create_first_population(population_size, max_lexicon_index)
print(f"Initial population: {initial_population}")

# Υπολογισμός ομοιότητας συνημιτόνου
def find_similarity(normalized_tfidf_matrix, target_vector):
    target_vector = scaler.transform(target_vector.toarray())  # Normalize the target vector
    similarities = cosine_similarity(normalized_tfidf_matrix, target_vector)
    return similarities

# Επιλογή όλων των λέξεων
selected_indices = [vectorizer.vocabulary_[word] for word in all_words_list]
tfidf_values = tfidf_matrix[:, selected_indices]

# Υπολογισμός ομοιότητας για την φθαρμένη επιγραφή
target_vector = vectorizer.transform(["αλεξανδρε ουδις"])
print("Διάνυσμα του 'αλεξανδρε ουδις':")
print(target_vector.toarray())  # Convert sparse matrix to dense for easier viewing

# Use the normalized TF-IDF matrix for similarity calculations
similarities = find_similarity(nrmlzd_tfidf_values, target_vector)
print("Πίνακας Ομοιοτήτων:")
print(similarities)

top_n = 10
top_n_indices = np.argsort(similarities.flatten())[-top_n:]
top_n_similarities = similarities.flatten()[top_n_indices]

print("Top 10 πιο κοντινές επιγραφές και οι ομοιότητές τους:")
for idx, similarity in zip(top_n_indices, top_n_similarities):
    print(f"Επιγραφή: {dataframe.iloc[idx]['text']}, Ομοιότητα: {similarity}")


# Συνάρτηση καταλληλότητας
def fitness_function(individual, tfidf_matrix, target_vector, top_n=10):
    word1, word2 = individual
    # Συμπλήρωση επιγραφής
    filled_inscription = f"{all_words_list[word1-1]} αλεξανδρε ουδις {all_words_list[word2-1]}"
    filled_vector = vectorizer.transform([filled_inscription])
    similarities = find_similarity(tfidf_matrix, filled_vector)
    top_similarities = np.sort(similarities.flatten())[-top_n:]
    fitness = np.mean(top_similarities)
    return fitness

# Υπολογισμός καταλληλότητας για τον αρχικό πληθυσμό
fitness_values = [fitness_function(ind, tfidf_matrix, target_vector, top_n=10) for ind in initial_population]
print(f"Fitness values: {fitness_values[:10]}")

# Επιλογή με ρουλέτα
def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]
    selected_idx = np.random.choice(len(population), p=selection_probs)
    return population[selected_idx]

# Διασταύρωση μονού σημείου
def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(0, 1)
    if crossover_point == 0:
        return (parent1[0], parent2[1])
    else:
        return (parent2[0], parent1[1])

# Μετάλλαξη με ελιτισμό
def mutate(individual, mutation_rate, max_index):
    if random.random() < mutation_rate:
        mutation_idx = random.randint(0, 1)
        new_value = random.randint(1, max_index)
        if mutation_idx == 0:
            individual = (new_value, individual[1])
        else:
            individual = (individual[0], new_value)
    return individual

# Εφαρμογή γενετικών τελεστών
def genetic_alg(tfidf_matrix, target_vector, pop_size, generations, mutation_rate, top_n):
    population = create_first_population(pop_size, max_lexicon_index)
    best_fitness_history = []  # Λίστα για την καταγραφή των καλύτερων fitness τιμών ανά γενιά
    best_fitness = 0  # Αρχική τιμή του καλύτερου fitness
    no_improvement_count = 0  # Μετρητής για τον έλεγχο της βελτίωσης
    generations_without_change = 0  # Μετρητής για τον έλεγχο των γενεών χωρίς αλλαγή
    
    for gen in range(generations):
        fitness_values = [fitness_function(ind, tfidf_matrix, target_vector, top_n) for ind in population]
        best_current_fitness = max(fitness_values)
        best_fitness_history.append(best_current_fitness)
        
        # Έλεγχος των κριτηρίων τερματισμού
        if gen > 0:
            improvement = (best_fitness_history[-2] - best_current_fitness) / best_fitness_history[-2]
            if improvement < 0.01:
                print(f"Termination criteria met at generation {gen+1}")
                break
            if best_current_fitness <= best_fitness:
                no_improvement_count += 1
                if no_improvement_count >= 10:  # Τερματισμός αν δεν υπάρξει βελτίωση για 10 γενεές
                    print(f"No improvement for {no_improvement_count} generations. Terminating...")
                    break
            else:
                no_improvement_count = 0  # Επαναφορά του μετρητή σε περίπτωση βελτίωσης
        
        best_fitness = best_current_fitness
        
        new_population = []
        for _ in range(pop_size):
            parent1 = roulette_wheel_selection(population, fitness_values)
            parent2 = roulette_wheel_selection(population, fitness_values)
            offspring = single_point_crossover(parent1, parent2)
            offspring = mutate(offspring, mutation_rate, max_lexicon_index)
            new_population.append(offspring)
        
        population = new_population
        
        generations_without_change += 1
        if generations_without_change >= 10:
            current_best_fitness = max(fitness_values)
            if current_best_fitness == best_fitness:
                print(f"No improvement in best fitness for 10 generations. Terminating...")
                break
        
        print(f"Generation {gen+1}, best fitness: {best_current_fitness}")
    
    return population

# Εκτέλεση ΓΑ με τα νέα κριτήρια τερματισμού
best_population = genetic_alg(tfidf_matrix, target_vector, pop_size=20, generations=1000, mutation_rate=0.1, top_n=10)
