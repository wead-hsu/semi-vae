This project is to replicate the results reported in '2015-kingma-Semi-supervised-learning-with-deep-generative-models'.
The author has pushed his code in 'https://github.com/dpkingma/nips14-ssl'.

During programming, I found that the hyper-parameters should be carefully chosen, otherwise the learning will not converge:
    
 1. batch_size (1000 -> 250): when batch_size is 1000, not converge at all
 2. num_hidden_layer (2 -> 1): learnable but slow
 3. learning_rate (1e-3 -> 3e-4): learning becomes faster
 4. prior_cost (5e-3 -> 5e-4): become stable and finally replicate the accuary of '96.x%'
