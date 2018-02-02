def main():
	from utilities import iterate_out_of_class_probs, save_current_status, plot_save_scores
	import warnings
	warnings.filterwarnings('ignore')
	warnings.simplefilter('ignore')
	import numpy as np

	# change out_class_prob
	## PARAMETERS
	start = 1
	stop = 100
	step = 1
	walk_length = 50
	num_walks = 25
	num_nodes = 400
	n_classes = 2
	in_class_prob = 0.8
	# out_class_prob = 0.6
	iterations = 10
	p = 1.0
	q = 1.0
	# for saving purposes
	file_name = 'evaluate_phase_change_{0}'.format(in_class_prob)

	bhamidi_scores_plot,\
	bhamidi_medians,\
	purity_scores_plot,\
	purity_medians,\
	out_class_probs = iterate_out_of_class_probs(start = start,
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

if __name__ == "__main__":
	print('Beginning to generate phase change plots.')
	main()
	print('Script completed generating phase change plots..')