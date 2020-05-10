import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool

def evolve(arr, score_function, mode='max', perturbation_modifier=1, noise_modifier=1, iterations=30, n_offspring=50, cutoff=None):
    
    def score_offspring(o):
        random_indices = [np.random.randint(0, len(o)) for i in range(noise_factor)]
        for j in random_indices:
            o[j] = np.random.normal(o[j], perturbation_factor)
        return score_function(o)
    
    itercount = 0
    score_history = []
    winner_history = []
    perturbation_factor = 0.01 * perturbation_modifier
    noise_factor = 0.01 * noise_modifier
    noise_factor = int(noise_factor * len(arr))
    if noise_factor < 1: noise_factor = 1
    
    pool = Pool(4)
    
    while itercount < iterations:
        offspring = [np.array(arr) for i in range(n_offspring)]
        scores = pool.map(score_offspring, offspring)
        if mode == 'max': high_score = max(scores)
        elif mode == 'min': high_score = min(scores)
        elif mode == 'approx': 
            scores = [abs(s) for s in scores]
            high_score = min(scores)
        winner = offspring[scores.index(high_score)]
        arr = winner
        winner_history.append(winner)
        score_history.append(high_score)
        itercount += 1
        print('Iteration:', itercount+1)
        print('Score:', high_score)
        if cutoff is not None:
            if mode == 'max': 
                if high_score > cutoff: 
                    break
            elif mode == 'min' or mode == 'approx':
                if high_score < cutoff: 
                    break
    return winner, score_history

def rmse(measured, estimated):
    import numpy as np
    measured = np.array(measured)
    estimated = np.array(estimated)
    error = measured - estimated
    error = np.square(error)
    mse = np.mean(error)
    return np.sqrt(mse)

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

def score(arr):
    standard = np.linspace(-1,1,len(arr))
    standard = np.array([_gaussian_(x, 1, 0, 0.2) for x in standard])
    return rmse(standard, arr)

arr = np.zeros((2000))
optimized, scores = evolve(arr, score, mode='approx', iterations=1000000, n_offspring=10000, cutoff=1E-1, noise_modifier=10)
plt.figure()
plt.plot(optimized)
plt.figure()
plt.plot(scores)
