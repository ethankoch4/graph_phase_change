def main():
    ROUND_TO = 2    
    import sys
    import os

    from utilities import iterate_out_of_class_probs, save_current_status, plot_save_scores, multiple_sbm_iterate, score_auc
    import warnings
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    import numpy as np

    parameters = [round(float(arg), ROUND_TO) for arg in sys.argv[1:3]] + [int(arg) for arg in sys.argv[3:-1]]
    print('Parameters are: p={0}, q={1}, walk_length={2}, num_walk={3}, embedding_size={4}, num_iter={5}'.format(*parameters))
    assert len(parameters)==6, 'Parameters to script must be: p, q, walk_length, num_walk, embedding_size, num_iter, R'
    R = int(sys.argv[-1])
    print('R equals: {0}'.format(R))

    # to be interated over
    out_class_probs = [round(i*0.01, ROUND_TO) for i in range(1,80)]

    # dont do the same thing twice
    file_name = 'p{0}_q{1}_wl{2}_nw{3}_es{4}_ni{5}_R{6}.json'.format(p, q, walk_length, num_walk, embedding_size, num_iter, R).format(in_class_prob,walk_length)
    if os.path.isfile(data_dir + file_name):
        print('SIMULATION HAS BEEN PERFORMED. SKIPPING: p{0}_q{1}_wl{2}_nw{3}_es{4}_ni{5}_s{6}'.format(*parameters))
        raise ValueError('Simulation has been run before. Change file_name if wanting to run again.')

    data_to_save = []
    for r in range(len(R)):
        for out_class_prob in out_class_probs:
            tmp_status = {}
            ## PARAMETERS
            p = parameters[0]
            tmp_status['p'] = p
            q = parameters[1]
            tmp_status['q'] = q
            walk_length = parameters[2]
            tmp_status['walk_length'] = walk_length
            num_walk = parameters[3]
            tmp_status['num_walk'] = num_walk
            embedding_size = parameters[4]
            tmp_status['embedding_size'] = embedding_size
            num_iter = parameters[5]
            tmp_status['num_iter'] = num_iter
            num_nodes = 400
            tmp_status['num_nodes'] = num_nodes
            n_classes = 2
            tmp_status['n_classes'] = n_classes
            in_class_prob = 0.8
            tmp_status['in_class_prob'] = in_class_prob
            out_class_prob = out_class_prob
            tmp_status['out_class_prob'] = out_class_prob
            iterations = 1
            tmp_status['iterations'] = iterations
            samples = 1
            tmp_status['samples'] = samples

            # for saving purposes; where we store data
            data_dir = 'data/'
            if os.path.isdir(data_dir):
                pass
            elif os.path.isdir('../' + data_dir):
                data_dir = '../' + data_dir
            elif os.path.isdir('../' + '../' + data_dir):
                data_dir = '../' + '../' + data_dir
            else:
                data_dir = '../' + data_dir

            block_probs = make_block_probs(in_class_prob=in_class_prob, out_class_prob=out_class_prob)

            sbm = stochastic_block_model(size=num_nodes,
                                         block_probabilities=block_probs,
                                         num_classes=n_classes
                                         )

            node_embeds = node2vec(G=None,
                                   Adj_M=sbm.A,
                                   labels=sbm.memberships,
                                   n_classes=n_classes,
                                   evaluate=True,
                                   p=p,
                                   q=q,
                                   walk_length=walk_length,
                                   num_walks=num_walks,
                                   window_size=5,
                                   embedding_size=embedding_size,
                                   num_iter=num_iter,
                                   min_count=0,
                                   sg=1,
                                   workers=8
                                   )
            true_labels = node_embeds.labels
            predicted_labels = node_embeds.predicted_labels
            # save true and predicted labels

    # save labels
    current_status = save_current_status(file_name = file_name,
                                         data = data_to_save
                                         )


print('Beginning to generate phase change plots.')
main()
print('Script completed generating phase change plots..')
