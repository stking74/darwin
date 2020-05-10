import numpy as np
import matplotlib.pyplot as plt

def evolve(input, score_function, mode='max', perturbation_modifier=1, noise_modifier=1, iterations=30, n_offspring=50):
    itercount = 0
    score_history = []
    perturbation_factor = 0.01 * perturbation_modifier
    noise_factor = 0.01 * noise_modifier
    noise_factor = int(noise_factor * len(input))
    while itercount < iterations:
        offspring = [np.array(input) for i in range(n_offspring)]
        for i, o in enumerate(offspring):
            random_indices = [np.random.randint(0, len(o)) for i in range(noise_factor)]
            for j in random_indices:
                v = o[j]
                o[j] = np.random.normal(v, perturbation_factor)
            offspring[i] = o
        scores = [score_function(o) for o in offspring]
        if mode == 'max': high_score = max(scores)
        elif mode == 'min': high_score = min(scores)
        elif mode == 'approx':
            scores = [abs(s) for s in scores]
            high_score = min(scores)
        winner = offspring[scores.index(high_score)]
        input = winner
        score_history.append(high_score)
        itercount += 1
    return winner, score_history

def _gaussian_(x, a, x0, sigma):
    '''
    In probability theory, the normal (or Gaussian or Gauss or Laplace–Gauss)
    distribution is a very common continuous probability distribution. Normal
    distributions are important in statistics and are often used in the
    natural and social sciences to represent real-valued random variables
    whose distributions are not known. A random variable with a Gaussian
    distribution is said to be normally distributed and is called a normal deviate.
    '''
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def score(input):
    standard = np.linspace(-1,1,len(input))
    standard = np.array([_gaussian_(x, 1, 0, 0.2) for x in standard])
    diff = sum(standard - input)
    return diff

input = np.zeros((200))
optimized, scores = evolve(input, score, mode='approx', iterations=100000, n_offspring=2000)
print(scores)
