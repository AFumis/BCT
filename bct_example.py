import numpy as np
from bct_main_with_log import bct
from draw_tree import draw_tree as draw

def generate_variable_length_markov_chain(n,initial_ctxt = "000",seed=None):
    '''
    Generate a Markov chain of 5th order with the same parameter vector of the 
    5th order chain in Section 4.1 of Kontoyiannis et al (2022). 
        Bayesian context trees: modelling and exact inference for discrete time series
        I. Kontoyiannis et al, 2022

    Args:
    -----
    n: int 
        length of the chain.
    initial_ctxt: string
        necessary initial context to start the algorithm
    seed: : int or 1-d array_like, optional
        Seed for RandomState. Must be convertible to 32 bit unsigned integers.



    Returns:
    -----
    sample: string
        the resulting sample
    '''
    np.random.seed(seed)
    alphabet = ['0', '1', '2']

    # Transition matrices
    theta1 = np.array([[0.4, 0.4, 0.2]])
    theta2 = np.array([[0.2, 0.4, 0.4]])
    theta00 = np.array([[0.4, 0.2, 0.4]])
    theta01 = np.array([[0.3, 0.6, 0.1]])
    theta022 = np.array([[0.5, 0.3, 0.2]])
    theta0212 = np.array([[0.1, 0.3, 0.6]])
    theta0211 = np.array([[0.05, 0.25, 0.7]])
    theta0210 = np.array([[0.35, 0.55, 0.1]])
    theta0202 = np.array([[0.1, 0.2, 0.7]])
    theta0201 = np.array([[0.8, 0.05, 0.15]])
    theta02002 = np.array([[0.7, 0.2, 0.1]])
    theta02001 = np.array([[0.1, 0.1, 0.8]])
    theta02000 = np.array([[0.3, 0.45, 0.25]])

    # Generate values
    values = initial_ctxt
    for _ in range(n):
        if values[-1:] == "1":
            current_state = np.random.choice(np.arange(3), p=theta1[0])
        elif values[-1:] == "2":
            current_state = np.random.choice(np.arange(3), p=theta2[0])
        elif values[-2:] == "00":
            current_state = np.random.choice(np.arange(3), p=theta00[0])
        elif values[-2:] == "10":
            current_state = np.random.choice(np.arange(3), p=theta01[0])
        elif values[-3:] == "220":
            current_state = np.random.choice(np.arange(3), p=theta022[0])

        elif values[-4:] == "2120":
            current_state = np.random.choice(np.arange(3), p=theta0212[0])
        elif values[-4:] == "1120":
            current_state = np.random.choice(np.arange(3), p=theta0211[0])
        elif values[-4:] == "0120":
            current_state = np.random.choice(np.arange(3), p=theta0210[0])
        elif values[-4:] == "2020":
            current_state = np.random.choice(np.arange(3), p=theta0202[0])
        elif values[-4:] == "1020":
            current_state = np.random.choice(np.arange(3), p=theta0201[0])
        elif values[-5:] == "20020":
            current_state = np.random.choice(np.arange(3), p=theta02002[0])
        elif values[-5:] == "10020":
            current_state = np.random.choice(np.arange(3), p=theta02001[0])
        elif values[-5:] == "00020":
            current_state = np.random.choice(np.arange(3), p=theta02000[0])
        else:
            print("error, the following context wasn't predicted in the algorithm: ",values[-5:])
            break
        values += alphabet[current_state]
    sample = ''.join(values[len(initial_ctxt):])
    return sample



def example(seed):
    '''
    Generate two samples from a variable length markov chain, obtain their
    respective maximum a posteriori probability context tree and plot them.
    The examples are the same of Section 4.1 from Kontoyiannis (2022).
    Reference:
        Bayesian context trees: modelling and exact inference for discrete time series
        I. Kontoyiannis et al, 2022

    Args:
    -----
    seed: : int or 1-d array_like, optional
        Seed for RandomState. Must be convertible to 32 bit unsigned integers.

    Returns:
    -----
    None: Nonetype
    '''
    
    # Example 1:
        # 5th order variable-memory chain {Xn} on the alphabet A = {0, 1, 2} of 
        # m = 3 letters, with model given by the tree T shown in Figure 1 of 
        # Section 2.2; the associated parameter vector θ = {θs ; s ∈ T} is given 
        # in Section E of the supplementary material.
        # 
        # D = 10
        # Beta = 0.75
        # Sample size = 10_000
    
    # Generate the sample with length 10_000 * 2 
    # The sample length is multiplied by 2 to try to generate the sample from
    # the stationary distribution. Only the second half will count as sample 
    # with the last few values from the first half serving as initial context
    lines = generate_variable_length_markov_chain(10_000 * 2, seed=seed) 
    
    dmax = 10 # The initial context length considered
    
    # Use the BCT to estimate the maximum a posteriori probability context tree
    tt, pmpm = bct(s = "012", dmax = dmax, beta = 0.75, sample = lines[10_000:],
                   initial_ctxt = lines[10_000-dmax:10_000])
    
    # Plot the estimated tree
    draw(tt,title=f"N=10,000 and seed = {seed}")

    # Example 2:
        # 5th order variable-memory chain {Xn} on the alphabet A = {0, 1, 2} of 
        # m = 3 letters, with model given by the tree T shown in Figure 1 of 
        # Section 2.2; the associated parameter vector θ = {θs ; s ∈ T} is given 
        # in Section E of the supplementary material.
        # 
        # D = 10
        # Beta = 0.75
        # Sample size = 1_000
    
    # Generate the sample with length 1_000 * 2 
    lines = generate_variable_length_markov_chain(1_000 * 2, seed=seed) 
    
    dmax = 10 # The initial context length considered
    
    # Use the BCT to estimate the maximum a posteriori probability context tree
    tt, pmpm = bct(s = "012", dmax = dmax, beta = 0.75, sample = lines[1_000:],
                   initial_ctxt = lines[1_000-dmax:1_000])
    
    # Plot the estimated tree
    draw(tt,title=f"N=1,000 and seed = {seed}")

def show_examples():
    '''
    Call example four times with four different seeds

    Args:
    -----

    Returns:
    -----
    None: Nonetype
    '''
    example(1)
    example(2)
    example(3)
    example(4)


