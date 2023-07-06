import numpy as np

from tree_class import Tree

def bct(s, dmax, beta, sample, initial_ctxt):
    '''
    Implements the Bayesian Context Trees (BCT) algorithm to obtain the maximum a
    posteriori probability context tree in a variable length Markov chain
    (if beta >= 0.5). All calculations are done in log scale for numerical precision.
    Reference:
        Bayesian context trees: modelling and exact inference for discrete time series
        I. Kontoyiannis et al, 2022

    Args:
    -----
    s: iterable
        Set of possible states of process.
    dmax: int
        Maximum possible depth of context tree.
    beta: float
        Penalization constant for tree depth. Must be between 0 and 1 (not inclusive)
        and recommended to be between 0.5 and 1. A common choice is 1-2**(-len(s)+1).
    sample: string
        Sample from chain.
    initial_ctxt: string
        Sample from chain of size bigger or equal to dmax.

    Returns:
    -----
    tau: Tree object
        The maximum a posteriori probability context tree.
    pm_root:
        The joint marginal probability of tau and the sample.
    '''

    # constructs maximal tree while counting transitions from all potential
    # contexts (all nodes in the tree) to each state
    tau, counter = construct_tau_max(s, dmax, sample, initial_ctxt)

    # calculates the estimated probability for all nodes in maximal tree
    log_pe_dict = {u: calc_log_pe(s, counter, u) for u in tau.nodes}

    # calculates the maximal probability for all nodes in maximal tree
    # using recursive algorithm from leaves to root
    log_pm_dict = construct_log_pm_dict(s, dmax, beta, tau, log_pe_dict)

    # starting from the root, recursively along descendants until reaching all leaves:
    #   if beta*Pe(u) > (1-beta)*prod_{b in s} Pm(bu):
    #       prunes tree in u (removes descendants, turning u into a leaf)
    #   else:
    #       moves downward in tree along descendants

    queue = [""] # queue to add descendants to
    while queue: # while not empty
        node = queue[0]
        queue.remove(node)

        if node in tau.edges: # node is not a leaf
            sum_child_log_pm = 0.0
            for child in tau.edges[node]:
                sum_child_log_pm += log_pm_dict[child]

            if np.log(beta) + log_pe_dict[node] > np.log(1-beta) + sum_child_log_pm:
                tau.prune_at_node(node) # prunes
            else:
                queue += tau.edges[node] # adds descendants to queue

    pm_root = np.exp(log_pm_dict[""])
    return tau, pm_root

def construct_tau_max(s, dmax, sample, initial_ctxt):
    '''
    Constructs the maximal tree of depth dmax, composed of all sequences
    of states for which the sequence followed by another state (representing
    a transition from the sequence) was observed in the sample, as well as
    all siblings of the node in the tree structure, even if not observed.

    Args:
    -----
    s: iterable
        Set of possible states of process.
    dmax: int
        Maximum possible depth of context tree.
    sample: string
        Sample from chain.
    initial_ctxt: string
        Sample from chain of size bigger or equal to dmax.

    Returns:
    -----
    tree: Tree object
        The maximal tree of depth dmax.
    counter: dict
        Amount of times a given sequence of states has appeared. Contains all
        sequences of the form node + state, where node is a tree node and
        state is a state in s, even if the count is 0.
    '''

    joint_sample = initial_ctxt + sample
    tree = Tree(s)
    counter = {"": len(sample)}

    for i in range(len(initial_ctxt), len(joint_sample)):
        current_string = joint_sample[i-dmax : i+1] # last dmax states + transition
        for j in range(0, dmax+1): # iterates over possible depths
            present = current_string[-j-1 : ] # potential context + transition
            past = present[:-1] # potential context
            if j >= 1 and past not in tree.nodes:
                tree.add_node(past) # adds siblings to tree as well
            if present not in counter:
                counter[present] = 1
            else:
                counter[present] += 1

    # adds unseen potential context + transition pairs to dict with count 0
    for node in tree.nodes:
        node_transitions = [node+b for b in s]
        for transition in node_transitions:
            if transition not in counter:
                counter[transition] = 0

    return tree, counter

def calc_log_pe(s, counter, u):
    '''
    Calculates the estimated probability of a node, as defined in Lemma 2.2
    of Kontoyiannis et al, 2022, page 8.

    Args:
    -----
    s: iterable
        Set of possible states of process.
    counter: dict
        Amount of times a given sequence of states has appeared.
    u: string
        Node in tree.

    Returns:
    -----
    log_pe: float
        Logarithm of the estimated probability of node u.
    '''

    # all transition counts N(ub) for node u
    try:
        transition_counts = [counter[u+b] for b in s]
    except KeyError as err:
        print("Transition not in counter")
        raise err

    # Pe(u)=1 if N(ub)=0 for all b in S (Kontoyiannis et al 2022, pg 9)
    if transition_counts == [0]*len(s):
        return 0.0 #log(1)

    # log_numerator
    log_numerator = 0.0
    for count in transition_counts:
        for n in range(1, count+1):
            log_numerator += np.log(n - 0.5)

    # log_denominator
    m = len(s)
    nu = np.sum(transition_counts)
    log_denominator = 0.0
    for n in range(1, nu+1):
        log_denominator += np.log(n + m/2 - 1)

    return log_numerator - log_denominator

def construct_log_pm_dict(s, dmax, beta, tree, log_pe_dict):
    '''
    Calculates the estimated probability of a node, as defined in Algorithm
    3.2 of Kontoyiannis et al, 2022, page 10.

    Args:
    -----
    s: iterable
        Set of possible states of process.
    dmax: int
        Maximum possible depth of context tree.
    beta: float
        Penalization constant for tree depth.
    tree: Tree object
        The maximal tree of depth dmax.
    log_pe_dict: dict
        Logarithm of the estimated probability of each node in tree.

    Returns:
    -----
    log_pm_dict: dict
        Logarithm of the maximal probability of each node in tree.
    '''

    log_pm_dict = dict()

    for size in range(dmax, 0-1, -1): # starts from dmax and goes down
                                      # in depth until root
        cur_nodes = [node for node in tree.nodes if len(node)==size]

        for node in cur_nodes:
            if size == dmax: # dmax sized leaf
                log_pm_dict[node] = log_pe_dict[node]

            elif node not in tree.edges: # leaf with size < dmax
                log_pm_dict[node] = np.log(beta)

            else: # not a leaf
                try:
                    sum_child_log_pm = 0.0
                    for child in tree.edges[node]:
                        sum_child_log_pm += log_pm_dict[child]
                except KeyError as err:
                    print("Child not in pm_dict")
                    raise err

                log_pm_dict[node] = max(np.log(beta) + log_pe_dict[node],
                                        np.log(1-beta) + sum_child_log_pm)

    return log_pm_dict




