"""
exp_config.py

This module implements the class ExpConfig which is derived from class AttrDict.

An instance of ExpConfig holds all the information needed for a single experiment.
In particular: the dataset, the training and active learning schedule, the
sampling policy and the current random seed.
"""

from .attrdict import AttrDict


class ExpConfig(AttrDict):
    def __init__(
        self, name, ds, model, train_schedule, al_schedule, sample_policy, seed
    ):
        self.name = name
        self.ds = ds
        self.model = model
        self.train_schedule = train_schedule
        self.al_schedule = al_schedule
        self.sample_policy = sample_policy
        self.seed = seed

    # returns a sting with important information about the ExpConfig instance
    def important_stats(self):
        s = "Model: " + str(type(self.model).__name__) + "\n"
        s += "  Layers: " + str(self.model.net_config.layer_sizes) + "\n"
        s += "  Learn Rate: " + str(self.model.net_config.learning_rate) + "\n"
        s += "  Batch size: " + str(self.train_schedule.batch_size) + "\n"
        s += "  Num epochs: " + str(self.train_schedule.num_epochs) + "\n"
        s += "  Active Learning: " + str(self.al_schedule) + "\n"
        s += "  Sample Policy: " + str(type(self.sample_policy).__name__) + "\n"
        try:
            s += (
                "  Sample batch_size: "
                + str(self.sample_policy.batch_size)
                + "\n  Number MC samples: "
                + str(self.sample_policy.n_mc_samples)
                + "\n"
            )
        except:
            s += "  No batch size and MC samples for this policy.\n"
        s += "  Seed: " + str(self.seed) + "\n"
        return s
