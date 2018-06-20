import torch
import torch.utils.data
import os
import numpy as np

def train_val_dataset(dataset_root_path, which, prop):
    nratings = {'ml-latest-small': 100004,
                'ml-1m': 1000208,
                'ml-20m' : 20000263}['ml-'+which]
    idx = np.arange(nratings)
    np.random.shuffle(idx)
    train_idx = idx[:int(np.ceil(prop*nratings))]
    val_idx = idx[int(np.ceil(prop*nratings)):]
    print(len(train_idx), len(val_idx), train_idx[-1], val_idx[0])

    train_set = MovieLensDataset(dataset_root_path, which, train_idx)
    val_set = MovieLensDataset(dataset_root_path, which, val_idx)
    return train_set, val_set, train_set.nusers, train_set.nmovies
        

class MovieLensDataset(torch.utils.data.Dataset):

    """Dataset object for loading the movie lens datasets"""
    
    def __init__(self, dataset_root_path, which, indices=None):

        if which not in ['latest-small', '1m', '20m']:
            return "You must select one of the MovieLens datasets\
                    in latest-small, 1m or 20m"

        dataset_filename = os.path.join(dataset_root_path, "ml-"+which, "ratings.csv")
        ############ Load the dataset
        # userId, movieId, rating, timestamp
        
        self.ratings = np.loadtxt(dataset_filename, delimiter=',', skiprows=1)
        self.nusers = len(np.unique(self.ratings[:,0]))
        self.nmovies = len(np.unique(self.ratings[:,1]))
        
        
        ############ Remap ids to continuous ids
        # The ids are not necessarily contiguous...
        # e.g. movieId goes from 1 -> 163949
        #      but len(np.unique(ratings[:,1])) = 9066
        
        self.map_uid = {}
        for kid, uid in enumerate(np.unique(self.ratings[:,0])):
            self.map_uid[str(uid)] = kid
        self.map_mid = {}
        for kid, mid in enumerate(np.unique(self.ratings[:,1])):
            self.map_mid[str(mid)] = kid
    
        for i in range(self.ratings.shape[0]):
            self.ratings[i,0] = self.map_uid[str(self.ratings[i,0])]
            self.ratings[i,1] = self.map_mid[str(self.ratings[i,1])]

        if indices is not None:
            self.ratings = self.ratings[indices]

    def __len__(self):
        return self.ratings.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray([self.ratings[i,0],self.ratings[i,1]], dtype=int)), \
                torch.from_numpy(np.asarray(self.ratings[i, 2], dtype=np.float32))
    #return (torch.from_numpy(np.asarray([self.ratings[i,0],self.ratings[i,1], dtype=int)), \
        #           torch.from_numpy(np.asarray(self.ratings[i,1], dtype=int))), \
        #           torch.from_numpy(np.asarray(self.ratings[i, 2], dtype=np.float32))
