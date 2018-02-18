import math
import numpy as np
import random

class stochastic_block_model(object):
    def __init__(self, size, block_probabilities, num_classes):
        self.A, self.memberships = self.createA(size, block_probabilities, num_classes)
        self.critical_p, \
        self.avg_degree_in, \
        self.avg_degree_out, \
        self.detectable = self.undetectable_point(memberships=self.memberships, block_probabilities=block_probabilities)
        self.is_recoverable = self.calculate_threshold(memberships=self.memberships, block_probabilities=block_probabilities)
        self.epsilon = self.avg_degree_out / self.avg_degree_in
        
    def sample_stochastic_block_model(self, memberships, block_probabilities, undirected=True):
        """
        Samples from a stochastic block model.

        Parameters
        ----------
        memberships: a vector of length n denoting the community memberships

        block_probabilities: a K x K symmetric matrix whose i,jth entry is the probability of an edge from community i to community j

        Output
        ------
        A numpy matrix
        """
        node_count = memberships.shape[0]
        adj_matrix = np.zeros([node_count, node_count])
        for i in range(node_count):
            for j in range(node_count):
                memberships_pair = (memberships[i],memberships[j])
                if random.random() < block_probabilities[memberships_pair[0],memberships_pair[1]]:
                    adj_matrix[i,j] = 1
                else:
                    adj_matrix[i,j] = 0
        if undirected:
            adj_matrix = np.maximum(adj_matrix, adj_matrix.transpose())
        return adj_matrix


    def createA(self, size, block_probabilities, num_classes):
        nums = np.zeros(size, dtype=np.int8)
        offset = 0
        for i in range(num_classes):
            if i != 0:
                offset = offset + size // num_classes
            nums[offset:offset + size // num_classes] = i
        np.random.shuffle(nums)
        memberships = nums
        return self.sample_stochastic_block_model(memberships,block_probabilities), memberships
       
    def create_node_labels(self, num_nodes, num_classes, matrix_form=True):
        """
        Creates a numpy matrix of class labels for a given number of nodes. Class labels are evenly distributed and shuffled.

        Parameters
        ----------
        num_nodes: the number of nodes to label

        num_classes: the number of different labels a node can be classified as
        
        Output
        ------
        A [num_nodes X 1] numpy matrix where V[i] = l means node i is of class l and l ranges from 0 to (num_classes-1).
        """
        memberships = np.zeros(num_nodes, dtype=np.int8)
        offset = 0
        for i in range(num_classes):
            if i != 0:
                offset = offset + num_nodes // num_classes
            memberships[offset:offset + num_nodes // num_classes] = i
        np.random.shuffle(memberships)
        return memberships
            
    def vector_labels_to_matrix(self, labels):
        """
        Converts a numpy vector of labels to an equivalent boolean matrix
        
        Parameters
        ----------
        labels: A [num_nodes X 1] numpy matrix where V[i] = l means node i is of class l and l ranges from 0 to (num_classes-1).
        
        Output
        ------
        A [num_nodes X num_classes] numpy matrix will be returned where M[i,j] = 1 means node i is of class j 
        """
        num_nodes =  len(labels)
        num_classes = np.amax(labels) + 1
        matrix_labels = np.zeros([num_nodes, num_classes])
        for i in range(num_nodes):
            label = labels[i]
            matrix_labels[i,label] = 1
        return matrix_labels

    def undetectable_point(self, memberships=[], block_probabilities=[]):
        """
        Calculates the critical point at which the algorithm should not be able to differentiate between groups according to:
        https://arxiv.org/pdf/1109.3041.pdf

        Parameters
        ----------
        memberships: a vector of length n denoting the community memberships
        block_probabilities: a K x K symmetric matrix whose i,jth entry is the probability of an edge from community i to community j

        Output
        ------
        critical_point: type float between 0.0 and 1.0
        avg_degree_in: mean degree of within-group connections
        avg_degree_out: mean degree of out-of-group connections
        detectable: boolean of whether the sbm's communities should be recoverable
        """
        group_totals = [0 for _ in range(len(set(memberships)))]
        for mem in memberships:
            group_totals[mem]+=1
        assert len(group_totals) == len(block_probabilities),"Number of groups must be equal to number of groups in block_probabilities"
        
        num_nodes = sum(group_totals)
        num_groups = len(group_totals)
        avg_degree = 0
        avg_degree_in = 0
        avg_degree_out = 0
        
        for a,row_group in enumerate(block_probabilities):
            for b,column_value in enumerate(row_group):
                # from paper
                avg_degree_ab = num_nodes*column_value
                proportion_in_a = group_totals[a]/num_nodes
                proportion_in_b = group_totals[b]/num_nodes
                if a==b:
                    avg_degree_in += avg_degree_ab*proportion_in_a*proportion_in_b
                else:
                    avg_degree_out += avg_degree_ab*proportion_in_a*proportion_in_b

        assert avg_degree_in > avg_degree_out,"avg_degree_in must be greater than avg_degree_out -- see paper"
        avg_degree = avg_degree_in + avg_degree_out
        # FROM PAPER: p_c = [c + (q-1)*math.sqrt(c)]/(q*c), when c_in > c_out
        critical_p = (avg_degree + (num_groups - 1)*math.sqrt(avg_degree))/(num_groups*avg_degree)
        detectable = round(avg_degree_in-avg_degree_out,8) > round(num_groups*math.sqrt(avg_degree),8)
        print('DETECTABLE:',detectable)
        return critical_p, avg_degree_in, avg_degree_out, detectable

    def calculate_threshold(self,block_probabilities=[],memberships=[]):
        """
        Calculates whether the current sbm's communities are recoverable according to:
        https://arxiv.org/pdf/1405.3267.pdf

        Parameters
        ----------
        memberships: a vector of length n denoting the community memberships
        block_probabilities: a K x K symmetric matrix whose i,jth entry is the probability of an edge from community i to community j

        Output
        ------
        is_recoverable: boolean of whether the sbm's communities should be recoverable
        """
        assert block_probabilities[0,1]==block_probabilities[1,0],"block_probabilities must be symmetric!"
        assert block_probabilities[0,0]==block_probabilities[1,1],"block_probabilities must be symmetric!"
        num_nodes = len(memberships)
        alpha = num_nodes*block_probabilities[0,0]/math.log(num_nodes)
        beta = num_nodes*block_probabilities[1,0]/math.log(num_nodes)
        assert alpha > beta,"required that alpha is greater than beta (a.k.a. graph is assortative)"
        is_recoverable = ((alpha + beta)/2 - math.sqrt(alpha*beta)) > 1
        print('RECOVERABLE:',is_recoverable)
        return is_recoverable