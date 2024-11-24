import numpy as np
import tensorflow as tf
import sys

def generate_one_hot_encoding(
    N: int,
    N_classes: int,
    random_seed: int,
    data=[]
):
    if data == []:
        np.random.seed(random_seed)
        data = np.random.randint(N_classes, size=N)
        
    data = np.ndarray.flatten(data)
    X_label = np.zeros((data.size, data.max()+1))
    X_label[np.arange(data.size),data] = 1
    
    return X_label


def generate_data(param):
    # Input
    # param: parameters dictionary
    # Outputs
    # X_, Y_, [X_label, Y_label]
    # param
    if param['N_dim'] == None:
        param['N_dim'] = 2
        
    sys.path.append('../')
    
    if param['dataset'] == 'Learning_gaussian':
        from data.Random_samples import generate_gaussian
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        X_ = generate_gaussian(size=(param['N_samples_Q'], param['N_dim']), m=0.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Mixture_of_gaussians':
        from data.Random_samples import generate_gaussian, generate_four_gaussians
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        X_ = generate_four_gaussians(size=(param['N_samples_Q'], param['N_dim']), dist=4.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
    
    elif param['dataset'] == 'Learning_student_t':
        from data.Random_samples import generate_student_t, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['nu']
        X_ = generate_student_t(size=(param['N_samples_Q'], param['N_dim']), m=0.0, nu=param['nu'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Learning_translated_student_t':
        from data.Random_samples import generate_student_t, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['nu']
        X_ = generate_student_t(size=(param['N_samples_Q'], param['N_dim']), m=0.0, nu=param['nu'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Learning_translated_directsum_student_t':
        from data.Random_samples import generate_directsum_student_t, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['nu']
        X_ = generate_directsum_student_t(size=(param['N_samples_Q'], param['N_dim']), m=0.0, nu=param['nu'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Learning_from_student_t':
        from data.Random_samples import generate_student_t, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['nu']
        Y_ = generate_student_t(size=(param['N_samples_P'], param['N_dim']), m=0.0, nu=param['nu'], random_seed=param['random_seed']+100) # initial
        X_ = generate_gaussian(size=(param['N_samples_Q'], param['N_dim']), m=0.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        
    elif param['dataset'] == 'Learning_from_laplacian_to_student_t':
        from data.Random_samples import generate_student_t, generate_laplacian
        param['expname'] = param['expname'] +'_%.2f' % (param['nu'])
        Y_ = generate_laplacian(size=(param['N_samples_P'], param['N_dim']), random_seed=param['random_seed']+100) # initial
        X_ = generate_student_t(size=(param['N_samples_Q'], param['N_dim']), m=0.0, nu=param['nu'], random_seed=param['random_seed']) # target
        
    elif param['dataset'] == 'Learning_from_cauchy_to_student_t':
        from data.Random_samples import generate_student_t, generate_cauchy
        param['expname'] = param['expname'] +'_%.2f' % (param['nu'])
        Y_ = generate_cauchy(size=(param['N_samples_P'], param['N_dim']), random_seed=param['random_seed']+100) # initial
        X_ = generate_student_t(size=(param['N_samples_Q'], param['N_dim']), m=0.0, nu=param['nu'], random_seed=param['random_seed']) # target
        
    elif param['dataset'] == 'Learning_from_student_t_to_student_t':
        from data.Random_samples import generate_student_t, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f_%.2f' % (param['nus'][0], param['nus'][1])
        Y_ = generate_student_t(size=(param['N_samples_P'], param['N_dim']), m=0.0, nu=param['nus'][0], random_seed=param['random_seed']+100) # initial
        X_ = generate_student_t(size=(param['N_samples_Q'], param['N_dim']), m=0.0, nu=param['nus'][1], random_seed=param['random_seed']) # target

    elif param['dataset'] == 'Stretched_exponential':
        from data.Random_samples import generate_stretched_exponential, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['beta']
        X_ = generate_stretched_exponential(size=(param['N_samples_Q'], param['N_dim']), beta=param['beta'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Learning_from_Stretched_exponential':
        from data.Random_samples import generate_stretched_exponential, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['beta']
        Y_ = generate_stretched_exponential(size=(param['N_samples_P'], param['N_dim']), beta=param['beta'], random_seed=param['random_seed']+100) # initial
        X_ = generate_gaussian(size=(param['N_samples_Q'], param['N_dim']), m=10.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        
    elif param['dataset'] == 'Keystrokes':
        from data.Random_samples import generate_gaussian
        param['N_dim'] = 1
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=param['sigma_P'], random_seed=param['random_seed']) # initial
        
        filename = "data/inter_stroke_time.txt"
        X_ = np.reshape(np.loadtxt(filename), (-1,1))
        param['N_samples_Q'] = X_.size
        
    elif param['dataset'] == 'Heavytail_submanifold':
        from data.Random_samples import generate_gaussian, generate_cauchy, embed_data
        
        df = param['N_dim'] - param['N_submnfld_dim']
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=param['sigma_P'], random_seed=param['random_seed']) # initial
        x = generate_cauchy(size=(param['N_samples_Q'], param['N_submnfld_dim']), random_seed=param['random_seed']+100)
        X_ = np.abs(x)
        np.random.seed(param['random_seed']+100)
        y = np.random.uniform(low=0.5, high=2.0, size=(1, param['N_submnfld_dim']))
        X_ = embed_data(np.sign(x)*(X_ ** y), di=0, df=df, offset=0.0)
        
    elif param['dataset'] == 'Lorenz63':
        from data.Random_samples import generate_gaussian
        
        param['N_dim'] = 3
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=param['sigma_P'], random_seed=param['random_seed']) # initial
        filename = f"data/lorenzdataset_{param['N_samples_Q']}.npy"
        X_ = np.load(filename)

    if param['mb_size_P'] > param['N_samples_P']:
        param['mb_size_P'] = param['N_samples_P']
    if param['mb_size_Q'] > param['N_samples_Q']:
        param['mb_size_Q'] = param['N_samples_Q']
    
    if 'X_label' not in locals():
        X_label = None
    if 'Y_label' not in locals():
        Y_label = None
        
    if param['unseen'] == False:
        return param, X_, Y_, X_label, Y_label
    else:
        if 'Y_unseen_label' not in locals():
            Y_unseen_label = None
        param['Y_unseen_label'] = Y_unseen_label
        return param, X_, Y_, X_label, Y_label, Y_unseen, Y_unseen_label
