import json
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import numpy as np
from sbm.sbm import stochastic_block_model
from node2vec.node2vec import node2vec
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
    score = score/(vertex_count*vertex_count)
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
    score = score/num_nodes
    return score

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
    # will be x-axis on plot
    out_class_probs = []
    
    first_iter = True
    for i in range(samples):
        tmp_bhamidi_scores_plot,\
        _0,\
        tmp_purity_scores_plot,\
        _1,\
        tmp_out_class_probs = iterate_out_of_class_probs(start = start,
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
            out_class_probs = tmp_out_class_probs
            first_iter = False
        else:
            bhamidi_scores_plot = [bhamidi_scores_plot[j]+tmp_bhamidi_scores_plot[j] for j in range(len(bhamidi_scores_plot))]
            purity_scores_plot = [purity_scores_plot[j]+tmp_purity_scores_plot[j] for j in range(len(purity_scores_plot))]
            
    bhamidi_medians = [np.median(bhamidi_scores_plot[j]) for j in range(len(bhamidi_scores_plot))]
    purity_medians = [np.median(purity_scores_plot[j]) for j in range(len(purity_scores_plot))]
    
    return bhamidi_scores_plot, bhamidi_medians, purity_scores_plot, purity_medians, out_class_probs 
    

def eval_multiple_walks(sbm, w_length=50, n_classes=2, num_walks=25, p=1, q=1, iterations=5):
    '''
    Return the bhamidi and purity scores after sampling node2vec walks for the specified number of iterations

    Parameters
    ------------
    sbm : stochastic block matrix from which the graph object should be defined
    w_length : length of node2vec walk
    n_classes : number of classes; also number of clusters because we will use kmeans to evaluate
    num_walks : number of node2vec walks to generate per node in graph object
    p : Return parameter; lower values result in more "local" walks
    q : In-out parameter; lower values result in more Depth-First Search behaving walks
    iterations : number of times the node2vec walks should be regenerated, understanding that the node embeddings must
                    be recalculated every time the walks are regenerated
    '''
    print('At eval_multiple_walks(...)')
    bhamidi_scores = []
    purity_scores = []
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
    # both are of type : list
    return bhamidi_scores, purity_scores

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
    print('At iterate_out_of_class_probs(...)')
    # will be y-axis on plot
    bhamidi_scores_plot = []
    bhamidi_medians = []
    purity_scores_plot = []
    purity_medians = []
    # will be x-axis on plot
    out_class_probs = []

    iteration_counter = 1
    for i in range(start, stop, step):
        # keep track of where the program is:
        if iteration_counter%5==0:
            print('Currently at iteration : {0}'.format(iteration_counter))

        iteration_counter += 1
        # change i into a probability
        i *= 0.01
        # i will become out_class_prob
        # in_class_prob is static
        block_probs = make_block_probs(in_class_prob=in_class_prob, out_class_prob=i)
        sbm = stochastic_block_model(size=num_nodes,
                                     block_probabilities=block_probs,
                                     num_classes=n_classes)
        bhamidi_scores, purity_scores = eval_multiple_walks(sbm,
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
        out_class_probs.append(i)
    return bhamidi_scores_plot, bhamidi_medians, purity_scores_plot, purity_medians, out_class_probs

def save_current_status(file_name = 'current_status_resample_walks',
                        out_class_probs=[],
                        bhamidi_scores_plot=[],
                        purity_scores_plot=[],
                        bhamidi_medians=[],
                        purity_medians=[],
                        walk_length = 'N/a',
                        num_walks = 'N/a',
                        num_nodes = 'N/a',
                        n_classes = 'N/a',
                        in_class_prob = 'N/a',
                        iterations = 'N/a',
                        p = 'N/a',
                        q = 'N/a'):
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
                    'out_class_probs' : out_class_probs
                    }
    # save to file (as json, obviously)
    with open(file_name+'.json', 'w') as fp:
        json.dump(current_status, fp)
    return current_status

def plot_save_scores(out_class_probs=[],
                     bhamidi_scores_plot=[],
                     purity_scores_plot=[],
                     bhamidi_medians=[],
                     purity_medians=[],
                     file_name='current_status_resample_walks',
                     walk_length = 'N/a',
                     num_walks = 'N/a',
                     num_nodes = 'N/a',
                     n_classes = 'N/a',
                     in_class_prob = 'N/a',
                     iterations = 'N/a',
                     p = 'N/a',
                     q = 'N/a'):
    plt.style.use('ggplot')
    # first plot : plot scores
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Explore Scores Jump : Resampling Walks',color='black')
    ax.set_ylabel('Bhamidi/Purity Scores',color='black')
    ax.set_xlabel('Out-Class Probability',color='black')
    ax.tick_params(axis='both',color='black')
    # set ticks to be the color black
    plt.setp(ax.get_xticklabels(), color='black')
    plt.setp(ax.get_yticklabels(), color='black')


    # plot scores as scatter plot first
    for i in range(len(out_class_probs)):
        x = out_class_probs[i]

        # bhamidi scores plot
        y = bhamidi_scores_plot[i]
        if i == 0:
            ax.scatter([x]*len(y), y, alpha=0.55, marker='.', c='g', label='raw bhamidi scores')
        else:
            ax.scatter([x]*len(y), y, alpha=0.55, marker='.', c='g')
           
        # purity scores plot
        y = purity_scores_plot[i]
        if i == 0:
            ax.scatter([x]*len(y), y, alpha=0.55, marker='.', c='b', label='raw purity scores')
        else:
            ax.scatter([x]*len(y), y, alpha=0.55, marker='.', c='b')
    
    
    median_bhamidi, = ax.plot(out_class_probs, bhamidi_medians, '-', color='g', label='median bhamidi scores')
    median_purity, = ax.plot(out_class_probs, purity_medians, '-', color='b', label='median purity scores')
    # create the legend
    legd = ax.legend(loc=3,fancybox=True)
    for text in legd.get_texts():
        text.set_color('black')
    anchored_text = AnchoredText('''---------PARAMS---------
walk length : {0}
num of walks : {1}
num of nodes : {2}
num of classes : {3}
in-class prob. : {4}
iterations : {5}
p : {6}
q : {7}'''.format(walk_length, num_walks, num_nodes, n_classes, round(in_class_prob,2), iterations, round(p,2), round(q,2)),loc=4)
    ax.add_artist(anchored_text)
    plt.savefig(file_name+'.png')
    plt.show()
