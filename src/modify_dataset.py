import random



def extract_dataset_and_raw_data_from_number_of_rows(initial_dataset, raw_data, new_number_of_rows):
    # Get info from initial dataset
    nb_rows, nb_features = initial_dataset.get_shape()

    selected_rows = get_randow_integers(nb_rows-1, new_number_of_rows)
    # Extraction of dataset and raw_data
    new_dataset = extract_sub_dataset_from_new_number_of_rows(initial_dataset, selected_rows)
    new_raw_data = extract_corresponding_raw_data_from_new_number_of_rows(raw_data, selected_rows)

    return new_dataset, new_raw_data




def get_randow_integers(max_value, number_of_int_to_get):
    return random.sample(range(0, max_value), number_of_int_to_get) # 0 can be picked !

def extract_sub_dataset_from_new_number_of_rows(initial_dataset, selected_rows):
    '''
        :type: CSR (Compressed Sparse Row format) matrix (from scipy)
        
        
        TO DO: description of function
    '''
    print(initial_dataset.tocsr()[selected_rows,:].get_shape())  #.todense()
    return initial_dataset.tocsr()[selected_rows,:]


def extract_corresponding_raw_data_from_new_number_of_rows(raw_data, selected_rows):
    return raw_data.iloc[selected_rows]

