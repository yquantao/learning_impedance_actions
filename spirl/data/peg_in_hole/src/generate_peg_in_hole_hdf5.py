"""Script for generating the datasets for peg-in-hole tasks."""
import h5py
import numpy as np
import pandas as pd
import os
import pickle

def main():
    dataset_df = pd.read_csv('peg_in_hole_dataset.csv', header=None)
    #print(action_df)
    #action_df.to_hdf('action.hdf5', key='action', mode='w')

    with h5py.File('peg_in_hole_dataset.hdf5', 'w') as hdf:
        # create state group
        #group1 = hdf.create_group('observations')
        hdf.create_dataset('observations', data=dataset_df.iloc[:,0:19])

        # create action group
        #group2 = hdf.create_group('actions')
        hdf.create_dataset('actions', data=dataset_df.iloc[:,19:27])

        # create reward group
        hdf.create_dataset('rewards', data=dataset_df.iloc[:,27])

        #group3 = hdf.create_group('terminals')
        hdf.create_dataset('terminals', data=dataset_df.iloc[:,28])


if __name__ == '__main__':
    main()