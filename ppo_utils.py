from collections import namedtuple


experience = namedtuple('experience_sample',['obs','action','reward','next_obs','done'])

class TrajectoryDataset():
    def __init__(self):
        self.storage_container = []
        self.index = 0 
    
    def store(self,sample):
        self.storage_container.append(sample)
    
    def __len___(self):
        return len(self.storage_container)

    def __getitem__(self,i):
        return self.storage_container[i]
    
    def clear(self):
        self.storage_container = []
    







