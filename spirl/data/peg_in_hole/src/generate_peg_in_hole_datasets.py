"""Script for generating the datasets for peg-in-hole tasks."""
import h5py
import numpy as np
import pandas as pd
import os
import pickle

def main():
    state_df = pd.read_csv('states.csv', header=None)
    action_df = pd.read_csv('actions_terminals_infos.csv', header=None)
    #print(action_df)
    #action_df.to_hdf('action.hdf5', key='action', mode='w')

    with h5py.File('peg_in_hole_dataset.hdf5', 'w') as hdf:
        # create state group
        #group1 = hdf.create_group('states')
        hdf.create_dataset('states', data=state_df)

        # create action group
        #group2 = hdf.create_group('actions')
        hdf.create_dataset('actions', data=action_df.iloc[:,0:9])

        #group3 = hdf.create_group('terminals')
        hdf.create_dataset('terminals', data=action_df.iloc[:,9])


if __name__ == '__main__':
    main()