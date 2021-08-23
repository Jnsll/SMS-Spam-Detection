import random





def extract_sub_dataset_from_new_number_of_rows(initial_dataset, new_number_of_rows):
    '''
        :type: CSR (Compressed Sparse Row format) matrix (from scipy)
        
        
        TO DO: description of function
    '''
    nb_rows, nb_features = initial_dataset.get_shape()
    print(nb_rows, nb_features)
    selected_lines = random.sample(range(0, nb_rows-1), new_number_of_rows)
    print(selected_lines)
    print(initial_dataset.tocsr()[selected_lines,:].get_shape())  #.todense()
    return initial_dataset.tocsr()[selected_lines,:]