def main():
	from utilities import iterate_out_of_class_probs, save_current_status, plot_save_scores, multiple_sbm_iterate
	import warnings
	import time
	warnings.filterwarnings('ignore')
	warnings.simplefilter('ignore')
	import numpy as np

	in_class_probs = [0.8, 0.2, 0.6, 0.4]

	for in_class_prob in in_class_probs:
		start_time = time.clock()
		# change out_class_prob
		## PARAMETERS
		start = 1
		stop = 101 # up to, but not including
		step = 1
		walk_length = 75
		num_walks = 25
		num_nodes = 400
		n_classes = 2
		# in_class_prob = 0.8
		# out_class_prob = 0.6
		iterations = 1
		samples = 2
		p = 1.0
		q = 1.0
		# for saving purposes
		file_name = '2_sbms_evaluate_phase_change_{0}'.format(in_class_prob)

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
		out_class_probs  = multiple_sbm_iterate(start = start,
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

		current_status = save_current_status(file_name = file_name,
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
		                                        out_class_probs = out_class_probs)

		plot_save_scores(file_name=file_name, **current_status)
		print('\nTOTAL TIME FOR THE DATA/PLOT AT THESE SETTINGS TO BE GENERATED:{0}\n'.format(time.clock()-start_time))

if __name__ == "__main__":
	print('Beginning to generate phase change plots.')
	main()
	print('Script completed generating phase change plots..')