def main():
    # set to true to make script submit jobs instead of running it here
    RUNNING_ON_CLUSTER = True
    # for uniformity and file_name saving
    ROUND_TO = 2

    # imports
    import random
    import itertools

    # define values for parameters
    p_s = [i*0.1 for i in range(1,21)]# + [i*0.5 for i in range(5,11)]
    q_s = [i*0.1 for i in range(1,21)]# + [i*0.5 for i in range(5,11)]
    print('Values of p,q are:\n {0}'.format(p_s))
    walk_lengths = [i for i in range(1,5)]# + [i*5 for i in range(1,11)]
    print('Values of walk_length are:\n {0}'.format(walk_lengths))
    num_walks = [1, 5]# + [i*10 for i in range(1,21)]
    print('Values of num_walk are:\n {0}'.format(num_walks))
    embedding_sizes = [10]# + [i*25 for i in range(1,9)]
    print('Values of embedding_size are:\n {0}'.format(embedding_sizes))
    num_iters = [2, 4, 8]
    print('Values of num_iter are:\n {0}'.format(num_iters))

    # used later
    total_combinations = len(p_s)*len(q_s)*len(walk_lengths)*len(num_walks)*len(embedding_sizes)*len(num_iters)
    print('Total number of combinations of parameters:\n {0}'.format(total_combinations))

    # get random seeds
    random.seed(42)
    random_seeds = [random.randint(1, 100*total_combinations) for _ in range(total_combinations)]
    print('Values of random seeds are:\n {0}'.format(random_seeds))

    # generate all combinations of the parameters
    parameters = [p_s, q_s, walk_lengths, num_walks, embedding_sizes, num_iters]
    combinations = list(itertools.product(*parameters))
    # print('All combinations are:')
    # for c in sorted(combinations):
    #     print(c)
    print('Following 3 numbers should be equal:')
    print(len(random_seeds))
    print(len(combinations))
    print(len(set(combinations)))

    # repetitions at each setting
    R = 100

    # submit jobs in running on cluster
    MEM = 5000
    for combination in combinations:
        # order: p, q, walk_length, num_walks, embedding_size, num_iters
        p = round(combination[0], ROUND_TO)
        q = round(combination[1], ROUND_TO)
        walk_length = combination[2]
        num_walk = combination[3]
        embedding_size = combination[4]
        num_iter = combination[5]
        LOG = 'p{0}_q{1}_wl{2}_nw{3}_es{4}_ni{5}_R{6}'.format(p, q, walk_length, num_walk, embedding_size, num_iter, R)

        COMMAND = 'python3 generate_predictions_sampling.py {0} {1} {2} {3} {4} {5} {6}'.format(p, q, walk_length, num_walk, embedding_size, num_iter, R)
        


print('BEGINNING TO RUN WITH ALL COMBINATIONS.')
print()
main()
print('COMPLETED RUNNING WITH ALL COMBINATIONS.')