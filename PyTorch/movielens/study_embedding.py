import sys
import model as rs_model

import torch
import matplotlib.pyplot as plt
import tsne
import pickle

import argparse

def tsne_embedding(embedding):
    print("Running t-SNE : {} -> 2".format(embedding.shape[1]))
    ar = embedding.detach().numpy()
    return tsne.tsne(ar, 2, ar.shape[1], 20)
    
def display_embeddings(user_embedding, movie_embedding):
    """
    embedding is a pyTorch tensor of size  n_samples x embed_size
    """
    # We first normalize the embeddings
    user_ar = user_embedding.detach().numpy()
    print("User embedding range: [{},{}]".format(user_ar.min(), user_ar.max()))
    movie_ar = movie_embedding.detach().numpy()
    print("Movie embedding range: [{},{}]".format(movie_ar.min(), movie_ar.max()))

    plt.figure()
    plt.subplot(121)
    plt.imshow(user_ar)
    plt.gca().set_aspect('auto')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(movie_ar)
    plt.gca().set_aspect('auto')
    plt.colorbar()

    #plt.show()
    
    

parser = argparse.ArgumentParser(description='Visualization of the embeddings')
parser.add_argument('model_filename', type=str, action='store', help='The pytorch saved model')
parser.add_argument('--user_tsne', type=str, required=False, action='store', help='Some already computed user tsne if any')
parser.add_argument('--movie_tsne', type=str, required=False, action='store', help='Some already computed movie tsne if any')

args = parser.parse_args()

print(args)

    
model_filename = sys.argv[1]

state = torch.load(model_filename)
model = rs_model.Model(state['nusers'], state['nmovies'],
                       state['embed_size'], state['ratings_range'])
model.load_state_dict(state['state_dict'])

user_embeddings = model.embed_user.weight
movie_embeddings = model.embed_movie.weight

display_embeddings(user_embeddings, movie_embeddings)
if args.user_tsne:
    f = open(args.user_tsne, 'rb')
    user_tsne = pickle.load(f)
    f.close()
else:
    user_tsne = tsne_embedding(user_embeddings)
    f = open('user_tsne.pkl','wb')
    pickle.dump(user_tsne, f)
    f.close()
    print("t-SNE saved to user_tsne.pkl")
    
    


#movie_tsne = tsne_embedding(movie_embeddings)

plt.figure()
plt.subplot(211)
plt.scatter(user_tsne[:,0], user_tsne[:,1])
#plt.subplot(212)
#plt.scatter(movie_tsne[:,0], movie_tsne[:,1])


plt.show()

