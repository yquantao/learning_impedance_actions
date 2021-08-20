from spirl.utils.general_utils import AttrDict
#from spirl.data.kitchen.src.kitchen_data_loader import D4RLSequenceSplitDataset
from spirl.data.peg_in_hole.src.peg_in_hole_data_loader import PegInHoleSequenceSplitDataset

data_spec = AttrDict(
    dataset_class=PegInHoleSequenceSplitDataset,
    n_actions=8,
    state_dim=19,
    env_name="peg-in-hole-v0",
    res=128,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 280
