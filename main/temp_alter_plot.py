import json
from utilities import plot_save_scores

def main():
	in_class_prob = 0.8
	file_name = 'evaluate_phase_change_{0}'.format(in_class_prob)
	with open(file_name+'.json', 'r') as fp:
		current_status = json.load(fp)
	# utilities.save_current_status(file_name = file_name,
	#                                         walk_length = walk_length,
	#                                         num_walks = num_walks,
	#                                         num_nodes = num_nodes,
	#                                         n_classes = n_classes,
	#                                         in_class_prob = in_class_prob,
	#                                         iterations = iterations,
	#                                         p = p,
	#                                         q = q,
	#                                         bhamidi_scores_plot = bhamidi_scores_plot,
	#                                         bhamidi_medians = bhamidi_medians,
	#                                         purity_scores_plot = purity_scores_plot,
	#                                         purity_medians = purity_medians,
	#                                         out_class_probs = out_class_probs)
	file_name += '_ex'
	plot_save_scores(file_name=file_name, **current_status)

if __name__ == "__main__":
	print('Beginning to re-graph data.')
	main()
	print('Script completed re-graphing.')