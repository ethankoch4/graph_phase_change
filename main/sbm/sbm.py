import time
import numpy as np
import random

class stochastic_block_model(object):
    def __init__(self, size, block_probabilities, num_classes):
        print('At createA(...)')
        start_time = time.clock()
        self.A, self.memberships = self.createA(size, block_probabilities, num_classes)
        print("Time elapsed while running 'createA' function: {0}".format(time.clock()-start_time))
        
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
            adj_matrix = np.maximum( adj_matrix, adj_matrix.transpose() )
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