import json
import warnings
import itertools
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import numpy as np
from sbm.sbm import stochastic_block_model
from node2vec.node2vec import node2vec
import matplotlib
matplotlib.use('Agg')
from matplotlib.offsetbox import AnchoredText

import matplotlib.pyplot as plt

def score_bhamidi(memberships, predicted_memberships):
    """
    Scores the predicted clustering labels vs true labels using Bhamidi's scoring method.

    Parameters
    ----------
    memberships: actual true labels
    predicted_memberships: clustering labels

    Output
    ------
    a percentage of correct vertice pairings
    """
    score = 0
    vertex_count = len(memberships)
    for i in range(vertex_count):
        for j in range(vertex_count):
            actual_match = memberships[i]==memberships[j]
            predicted_match = predicted_memberships[i]==predicted_memberships[j] 
            if actual_match == predicted_match:
                score += 1
    #convert to percent of total vertex pairs
    score = score / (vertex_count*vertex_count)
    return score


def score_purity(memberships, predicted_memberships):
    """
    Scores the predicted clustering labels vs true labels using purity.

    Parameters
    ----------
    memberships: actual true labels
    predicted_memberships: clustering labels

    Output
    ------
    a percentage of correct labels
    """
    num_nodes = len(memberships)
    
    #identify unique labels
    true_labels = set(memberships)
    predicted_labels = set(predicted_memberships)
    
    #make a set for each possible label
    true_label_sets = {}
    predicted_label_sets = {}
    for label in true_labels:
        true_label_sets[label] = set()
        predicted_label_sets[label] = set()
    
    #go through each vertex and assign it to a set based on label
    for i in range(num_nodes):
        true_label = memberships[i]
        predicted_label = predicted_memberships[i]
        true_label_sets[true_label].add(i)
        predicted_label_sets[predicted_label].add(i)
        
    #now can perfrom purity algorithm
    score = 0
    for true_label, true_label_set in true_label_sets.items():
        max_intersection = 0
        for predicted_label, predicted_label_set in predicted_label_sets.items():
            intersection = len(set.intersection(predicted_label_set,true_label_set))
            if max_intersection < intersection:
                max_intersection = intersection
        score += max_intersection

    #divide score by total vertex count
    score = score / num_nodes
    return score

def score_agreement(y, y_hat):
    '''
    calculates agreement score for the labeling of a community

    input variables
    x - true labels of verticies
    y - predicted labels of verticies
    output variable
    score - agreement score
    '''
    max_score = 0.0
    lookups = itertools.permutations(set(y))
    for lookup in lookups:
        lookup = np.array(lookup)
        relabeling = np.array(lookup[y])
        score = float(np.sum(y == relabeling) / len(y))
        # record if the largest so far
        if score > max_score:
            max_score = score
    return max_score

def score_auc(x,y):
    '''
    calculates the area under the curve of the x,y pairs using the Trapezoidal rule
    x - vector of x values
    y - vector of y values
    '''
    return np.trapz(y,x=x)

def get_empirical_threshold(epsilons=[], scores=[], PROP_SUCCESS_CUT_OFF=0.75, SCORE_CUT_OFF=0.75):
    '''
    calculates the epmirical threshold using a list of lists of scores
    scores - vector of vectors full of scores
    PROP_SUCCESS_CUT_OFF - proportion of iterations that are successful above this will be considered 'successful' epsilon values
    SCORE_CUT_OFF - scores greater than this are considered 'successful'
    '''
    assert len(epsilons)==len(scores), "must have an epsilon for every score and vice versa -> must be same length!"
    # store all scores to find argmax_epsilon later on
    prop_successes = []
    for score_list in scores:
        # record the proportion that are successful
        successes = [1 if score > SCORE_CUT_OFF else 0 for score in score_list]
        prop_success = sum(successes) / len(successes)
        prop_successes.append(prop_success)
    return max([ep for ep, p_s in zip(epsilons, prop_successes) if p_s > PROP_SUCCESS_CUT_OFF], default=None)

def make_block_probs(in_class_prob=0.5, out_class_prob=0.5):
    return np.array([[in_class_prob, out_class_prob],
                     [out_class_prob, in_class_prob]])

def multiple_sbm_iterate(start = 30,
                        stop = 91,
                        step = 2,
                        walk_length = 50,
                        num_walks = 25,
                        num_nodes = 400,
                        n_classes = 2,
                        in_class_prob = 0.8,
                        iterations = 10,
                        p = 1.0,
                        q = 1.0,
                        samples = 10):
    # will be y-axis on plot
    bhamidi_scores_plot = []
    bhamidi_medians = []
    purity_scores_plot = []
    purity_medians = []
    agreement_scores_plot = []
    agreement_medians = []
    out_class_probs = []
    # will be x-axis on plot
    epsilons = []
    critical_point = None
    recoverable_point = None
    
    first_iter = True
    for i in range(samples):
        tmp_bhamidi_scores_plot,\
        _0,\
        tmp_purity_scores_plot,\
        _1,\
        tmp_agreement_scores_plot,\
        _2,\
        tmp_out_class_probs,\
        tmp_epsilons,\
        tmp_critical_point,\
        tmp_recoverable_point = iterate_out_of_class_probs(start = start,
                                        stop = stop,
                                        step = step,
                                        walk_length = walk_length,
                                        num_walks = num_walks,
                                        num_nodes = num_nodes,
                                        n_classes = n_classes,
                                        in_class_prob = in_class_prob,
                                        iterations = iterations,
                                        p = p,
                                        q = q)
        if first_iter:
            bhamidi_scores_plot = tmp_bhamidi_scores_plot
            purity_scores_plot = tmp_purity_scores_plot
            agreement_scores_plot = tmp_agreement_scores_plot
            out_class_probs = tmp_out_class_probs
            epsilons = tmp_epsilons
            first_iter = False
        else:
            bhamidi_scores_plot = [bhamidi_scores_plot[j]+tmp_bhamidi_scores_plot[j] for j in range(len(bhamidi_scores_plot))]
            purity_scores_plot = [purity_scores_plot[j]+tmp_purity_scores_plot[j] for j in range(len(purity_scores_plot))]
            agreement_scores_plot = [agreement_scores_plot[j]+tmp_agreement_scores_plot[j] for j in range(len(agreement_scores_plot))]
        if tmp_critical_point is not None:
            critical_point = tmp_critical_point
        if tmp_recoverable_point is not None:
            recoverable_point = tmp_recoverable_point
            
    bhamidi_medians = [np.median(bhamidi_scores_plot[j]) for j in range(len(bhamidi_scores_plot))]
    purity_medians = [np.median(purity_scores_plot[j]) for j in range(len(purity_scores_plot))]
    agreement_medians = [np.median(agreement_scores_plot[j]) for j in range(len(agreement_scores_plot))]
    return bhamidi_scores_plot, bhamidi_medians, purity_scores_plot, purity_medians, agreement_scores_plot, agreement_medians, out_class_probs, epsilons, critical_point, recoverable_point
    

def eval_multiple_walks(sbm, w_length=50, n_classes=2, num_walks=25, p=1, q=1, iterations=5):
    '''
    Return the bhamidi, purity, and agreement scores after sampling node2vec walks for the specified number of iterations

    Parameters
    ------------
    sbm : stochastic block matrix from which the graph object should be defined
    w_length : length of node2vec walk
    n_classes : number of classes; also number of clusters because we will use kmeans to evaluate
    num_walks : number of node2vec walks to generate per node in graph object
    p : Return parameter; lower values result in more "local" walks
    q : In-out parameter; lower values result in more Depth-First Search behaving walks

    Output
    ------
    bhamidi_scores: a list of floats
    purity_scores: a list of floats
    agreement_scores: a list of floats
    '''
    bhamidi_scores = []
    purity_scores = []
    agreement_scores = []
    for i in range(iterations):
        node_embeds = node2vec(G=None,
                                Adj_M=sbm.A,
                                labels=sbm.memberships,
                                n_classes=n_classes,
                                evaluate=True,
                                p=p,
                                q=q,
                                walk_length=w_length,
                                num_walks=num_walks,
                                window_size=10,
                                embedding_size=128,
                                num_iter=4,
                                min_count=0,
                                sg=1,
                                workers=8,
                                )
        bhamidi_scores.append(node_embeds.bhamidi_score)
        purity_scores.append(node_embeds.purity_score)
        agreement_scores.append(node_embeds.agreement_score)
    # all are of type : list
    return bhamidi_scores, purity_scores, agreement_scores

def iterate_out_of_class_probs(start = 30,
                               stop = 91,
                               step = 2,
                               walk_length = 50,
                               num_walks = 25,
                               num_nodes = 400,
                               n_classes = 2,
                               in_class_prob = 0.8,
                               iterations = 10,
                               p = 1.0,
                               q = 1.0):
    # will be y-axis on plot
    bhamidi_scores_plot = []
    bhamidi_medians = []
    purity_scores_plot = []
    purity_medians = []
    agreement_scores_plot = []
    agreement_medians = []
    out_class_probs = []
    # will be x-axis on plot
    epsilons = []
    critical_point = None
    recoverable_point = None

    for i in range(start, stop, step):
        # change i into a probability
        i *= 0.001
        # i will become out_class_prob
        # in_class_prob is static
        block_probs = make_block_probs(in_class_prob=in_class_prob, out_class_prob=i)
        sbm = stochastic_block_model(size=num_nodes,
                                     block_probabilities=block_probs,
                                     num_classes=n_classes)
        bhamidi_scores, purity_scores, agreement_scores = eval_multiple_walks(sbm,
                                                            w_length=walk_length,
                                                            n_classes=n_classes,
                                                            num_walks=num_walks,
                                                            p=p,
                                                            q=q,
                                                            iterations=iterations)
        # record for plotting purposes
        bhamidi_scores_plot.append(bhamidi_scores)
        bhamidi_medians.append(np.median(bhamidi_scores))
        purity_scores_plot.append(purity_scores)
        purity_medians.append(np.median(purity_scores))
        agreement_scores_plot.append(agreement_scores)
        agreement_medians.append(np.median(agreement_scores))
        out_class_probs.append(i)
        epsilons.append(sbm.epsilon)
        # first point at which sbm is not recoverable
        if not sbm.detectable and not isinstance(critical_point,float):
            critical_point = float(sbm.epsilon)
            print(sbm.critical_p)
        # first point at which sbm is not recoverable
        if not sbm.is_recoverable and not isinstance(recoverable_point,float):
            recoverable_point = float(sbm.epsilon)
            print(sbm.critical_p)
    return bhamidi_scores_plot, bhamidi_medians, purity_scores_plot, purity_medians, agreement_scores_plot, agreement_medians, out_class_probs, epsilons, critical_point, recoverable_point

def save_current_status(file_name = 'current_status_resample_walks',
                        out_class_probs=[],
                        bhamidi_scores_plot=[],
                        purity_scores_plot=[],
                        agreement_scores_plot=[],
                        bhamidi_medians=[],
                        purity_medians=[],
                        agreement_medians=[],
                        walk_length = 'N/a',
                        num_walks = 'N/a',
                        num_nodes = 'N/a',
                        n_classes = 'N/a',
                        in_class_prob = 'N/a',
                        iterations = 'N/a',
                        p = 'N/a',
                        q = 'N/a',
                        epsilons='N/a',
                        critical_point='N/a',
                        recoverable_point='N/a'):
    # saving current status
    current_status = {
                    'iterations' : iterations,
                    'walk_length' : walk_length,
                    'num_walks' : num_walks,
                    'num_nodes' : num_nodes,
                    'n_classes' : n_classes,
                    'in_class_prob' : in_class_prob,
                    'p' : p,
                    'q' : q,
                    'bhamidi_scores_plot' : bhamidi_scores_plot,
                    'bhamidi_medians' : bhamidi_medians,
                    'purity_scores_plot' : purity_scores_plot,
                    'purity_medians' : purity_medians,
                    'agreement_scores_plot' : agreement_scores_plot,
                    'agreement_medians' : agreement_medians,
                    'out_class_probs' : out_class_probs,
                    'epsilons' : epsilons,
                    'critical_point' : critical_point,
                    'recoverable_point' : recoverable_point
                    }
    # save to file (as json, obviously)
    with open(file_name+'.json', 'w') as fp:
        json.dump(current_status, fp)
    return current_status

def plot_save_scores(epsilons=[],
                     bhamidi_scores_plot=None,
                     purity_scores_plot=None,
                     agreement_scores_plot=None,
                     bhamidi_medians=None,
                     purity_medians=None,
                     agreement_medians=None,
                     file_name='current_status_resample_walks',
                     walk_length = 'N/a',
                     num_walks = 'N/a',
                     num_nodes = 'N/a',
                     n_classes = 'N/a',
                     in_class_prob = 'N/a',
                     iterations = 'N/a',
                     p = 'N/a',
                     q = 'N/a',
                     critical_point='N/a',
                     recoverable_point='N/a',
                     out_class_probs=[],
                     PROP_SUCCESS_CUT_OFF=0.75,
                     SCORE_CUT_OFF=0.75,
                     display=False):
    plt.style.use('fivethirtyeight')
    # first plot : plot scores
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Explore Phase Change: Walk Length = {0}'.format(walk_length),color='black',fontsize=18)
    ax.set_ylabel('Scores',color='black',fontsize=14)
    ax.set_xlabel('Epsilon (c_out/c_in)',color='black',fontsize=14)
    ax.tick_params(axis='both',color='black')
    # set ticks to be the color black
    plt.setp(ax.get_xticklabels(), color='black')
    plt.setp(ax.get_yticklabels(), color='black')

    prop_successes = []
    # plot scores as scatter plot first
    for i in range(len(epsilons)):
        x = epsilons[i]

        if isinstance(bhamidi_scores_plot,list):
            # bhamidi scores plot
            y = bhamidi_scores_plot[i]
            if i == 0:
                ax.scatter([x]*len(y), y, alpha=0.4, marker='.', c='c', label='raw bhamidi scores')
            else:
                ax.scatter([x]*len(y), y, alpha=0.4, marker='.', c='c')
        
        if isinstance(purity_scores_plot,list):
            # purity scores plot
            y = purity_scores_plot[i]
            if i == 0:
                ax.scatter([x]*len(y), y, alpha=0.4, marker='.', c='y', label='raw purity scores')
            else:
                ax.scatter([x]*len(y), y, alpha=0.4, marker='.', c='y')

        if isinstance(agreement_scores_plot,list):
            # agreement scores plot
            y = agreement_scores_plot[i]
            if i == 0:
                ax.scatter([x]*len(y), y, alpha=0.4, marker='.', c='r', label='raw agreement scores')
            else:
                ax.scatter([x]*len(y), y, alpha=0.4, marker='.', c='r')
            prop_success = [1 if val > SCORE_CUT_OFF else 0 for val in agreement_scores_plot[i]]
            prop_success = sum(prop_success) / len(prop_success)
            prop_successes.append(prop_success)

    ax.xaxis.set_ticks(np.arange(0.0, 1.1, 0.1)) # up to, but not including 1.1
    ax.yaxis.set_ticks(np.arange(0.0, 1.1, 0.1)) # up to, but not including 1.1
    
    if isinstance(bhamidi_scores_plot,list):
        median_bhamidi, = ax.plot(epsilons, bhamidi_medians, '-', color='c', label='median bhamidi scores')
    if isinstance(purity_scores_plot,list):
        median_purity, = ax.plot(epsilons, purity_medians, '-', color='y', label='median purity scores')
    if isinstance(agreement_scores_plot,list):
        median_agreement, = ax.plot(epsilons, agreement_medians, '-', color='r', label='median agreement scores')
        plot_prop_success, = ax.plot(epsilons, prop_successes, '-', color='g', label='Proportion of Successful Iterations')
    # get empirical_threshold
    last_prop_success = get_empirical_threshold(epsilons=epsilons,
                                                scores=agreement_scores_plot,
                                                PROP_SUCCESS_CUT_OFF=PROP_SUCCESS_CUT_OFF,
                                                SCORE_CUT_OFF=SCORE_CUT_OFF)
    # the undetectable point
    if (not isinstance(critical_point,str)) and (critical_point is not None):
        ax.axvline(critical_point, c='b', label='Undetectable Threshold')
    # the undetectable point
    if (not isinstance(recoverable_point,str)) and (recoverable_point is not None):
        ax.axvline(recoverable_point, c='c', label='Unrecoverable Threshold')
    # last successful epsilon value
    if last_prop_success is not None:
        ax.axvline(last_prop_success, c='m', label='Empirical Threshold')

    # create the legend
    legd = ax.legend(loc=3,fancybox=True,fontsize=12,scatterpoints=3)
    for text in legd.get_texts():
        text.set_color('black')
    anchored_text = AnchoredText('''---------PARAMS---------
walk length : {0}
num of classes : {3}
in-class prob. : {4}
iterations : {5}
p : {6}
q : {7}'''.format(walk_length, num_walks, num_nodes, n_classes, round(in_class_prob,2), iterations, round(p,2), round(q,2)),loc=7, bbox_to_anchor=(1, 0.5))
    ax.add_artist(anchored_text)
    if display:
        plt.show()
    else:
        if ('.png' not in file_name) and ('.jp' not in file_name) and ('.pdf' not in file_name):
            file_name = file_name+'.png'
        plt.savefig(file_name)
    plt.close()

def plot_vs_parameter(file_name='walk_len_plot', param_values={}, y_values={}, display=False):
    '''
    plot parameter vs list of y's; ex: empirical threshold vs. walk_length
    param_values - dict with single entry; key is name of param
    y_values - dict of lists of y values to plot (all between 0 and 1), key is name of Y value
    '''
    plt.style.use('fivethirtyeight')
    # first plot : plot scores
    fig, ax = plt.subplots(figsize=(12, 8))
    y_title = ', '.join(['{'+str(i)+'}' for i in range(len(y_values.keys()))])
    x_title = list(param_values)[0]
    title = y_title + ' vs. ' + x_title
    ax.set_title(title.format(*list(y_values)),color='black',fontsize=18)
    ax.set_ylabel(y_title.format(*list(y_values)),color='black',fontsize=14)
    ax.set_xlabel(x_title,color='black',fontsize=14)
    ax.tick_params(axis='both',color='black')
    # set ticks to be the color black
    plt.setp(ax.get_xticklabels(), color='black')
    plt.setp(ax.get_yticklabels(), color='black')

    ax.xaxis.set_ticks(np.arange(0, max(param_values[list(param_values)[0]])+5, 5)) # up to, but not including
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1)) # up to, but not including
    # plot all y values
    x = param_values[list(param_values)[0]]
    colors = ['g','r','m','b','c','y']
    plot_dict = {}
    for i,y_key in enumerate(list(y_values.keys())):
        plot_dict[y_key], = ax.plot(x, y_values[y_key], '-', color=colors[i], label=y_key)

    # create the legend
    legd = ax.legend(loc='best',fancybox=True,fontsize=12,scatterpoints=3)
    for text in legd.get_texts():
        text.set_color('black')

    if display:
        plt.show()
    else:
        if ('.png' not in file_name) and ('.jp' not in file_name) and ('.pdf' not in file_name):
            file_name = file_name+'.png'
        plt.savefig(file_name)
    plt.close()