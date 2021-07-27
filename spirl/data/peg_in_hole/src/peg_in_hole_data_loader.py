
from spirl.components.data_loader import Dataset


class PegInHoleSequenceDataset(Dataset):

    def __getitem__(self, index):
        data = super().__getitem__(index)
        for key in data.keys():
            if key.endswith('states') and data[key].shape[-1] == 40:
                # remove quatenion dimensions
                data[key] = data[key][:, :20]
            elif key.endswith('states') and data[key].shape[-1] == 43:
                data[key] = data[key][:, :23]
            if key.endswith('actions') and data[key].shape[-1] == 4:
                # remove rotation dimension
                data[key] = data[key][:, [0, 1, 3]]
        return data