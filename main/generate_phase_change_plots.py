def main():
    import sys
    import os

    from utilities import iterate_out_of_class_probs, save_current_status, plot_save_scores, multiple_sbm_iterate, score_auc
    import warnings
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    import numpy as np

    in_class_probs = [0.8]
    walk_lengths = [int(arg) for arg in sys.argv[1:]]
    areas_under_curve = {}
    for walk_length in walk_lengths:
        for in_class_prob in in_class_probs:
            # change out_class_prob
            ## PARAMETERS
            start = 8
            stop = 800 # up to, but not including
            step = 8
            walk_length = walk_length
            num_walks = 75
            num_nodes = 400
            n_classes = 2
            in_class_prob = in_class_prob
            # out_class_prob = 0.6
            iterations = 1
            samples = 20
            p = 1.0
            q = 1.0
            # for saving purposes
            file_name = 'q_{0}_walk_len_{1}'.format(in_class_prob,walk_length)
            if os.path.isfile('../plots/'+file_name+'.png') or os.path.isfile('../data/'+file_name+'.json'):
                print('SIMULATION HAS BEEN PERFORMED. SKIPPING.')
                raise ValueError('Simulation has been run before. Change file_name if wanting to run again.')

            # bhamidi_scores_plot,\
            # bhamidi_medians,\
            # purity_scores_plot,\
            # purity_medians,\
            # out_class_probs = iterate_out_of_class_probs(start = start,
            #                                             stop = stop,
            #                                             step = step,
            #                                             walk_length = walk_length,
            #                                             num_walks = num_walks,
            #                                             num_nodes = num_nodes,
            #                                             n_classes = n_classes,
            #                                             in_class_prob = in_class_prob,
            #                                             iterations = iterations,
            #                                             p = p,
            #                                             q = q)
            bhamidi_scores_plot, \
            bhamidi_medians, \
            purity_scores_plot, \
            purity_medians, \
            agreement_scores_plot, \
            agreement_medians, \
            out_class_probs, \
            epsilons, \
            critical_point, \
            recoverable_point = multiple_sbm_iterate(start = start,
                                            stop = stop,
                                            step = step,
                                            walk_length = walk_length,
                                            num_walks = num_walks,
                                            num_nodes = num_nodes,
                                            n_classes = n_classes,
                                            in_class_prob = in_class_prob,
                                            iterations = iterations,
                                            p = p,
                                            q = q,
                                            samples = samples)
            # because we don't currently want to plot bhamidi/purity scores
            bhamidi_scores_plot = None
            bhamidi_medians = None
            purity_scores_plot = None
            purity_medians = None
            current_status = save_current_status(file_name = '../data/'+file_name,
                                                    walk_length = walk_length,
                                                    num_walks = num_walks,
                                                    num_nodes = num_nodes,
                                                    n_classes = n_classes,
                                                    in_class_prob = in_class_prob,
                                                    iterations = iterations,
                                                    p = p,
                                                    q = q,
                                                    bhamidi_scores_plot = bhamidi_scores_plot,
                                                    bhamidi_medians = bhamidi_medians,
                                                    purity_scores_plot = purity_scores_plot,
                                                    purity_medians = purity_medians,
                                                    agreement_scores_plot = agreement_scores_plot,
                                                    agreement_medians = agreement_medians,
                                                    out_class_probs = out_class_probs,
                                                    epsilons = epsilons,
                                                    critical_point = critical_point,
                                                    recoverable_point = recoverable_point)
            areas_under_curve[walk_length] = score_auc(np.arange(start,stop,step),agreement_medians)
            plot_save_scores(file_name='../plots/'+file_name, **current_status)

    print(areas_under_curve)

print('Beginning to generate phase change plots.')
main()
print('Script completed generating phase change plots..')
