import sys
import model as rs_model

import torch
import matplotlib.pyplot as plt
from sklearn import manifold
import tsne

if len(sys.argv) != 2:
    print("Usage : {} model.pt".format(sys.argv[0]))
    sys.exit(-1)


def tsne_embeddings(user_embedding, movie_embedding):
    user_ar = user_embedding.detach().numpy()
    print("Computing user embedding t-SNE")
    #user_tsne = manifold(n_components=2,perplexity=40).fit_transform(user_ar)
    user_tsne = tsne.tsne(user_ar, 2, user_ar.shape[1], 40)
    
    movie_ar = movie_embedding.detach().numpy()
    print("Computing movie embedding t-SNE")
    #movie_tsne = manifold.TSNE(n_components=2, perplexity=40).fit_transform(movie_ar)
    user_tsne = tsne.tsne(movie_ar, 2, movie_ar.shape[1], 40)
    
    plt.figure()
    plt.subplot(211)
    plt.scatter(user_tsne[:,0], user_tsne[:,1])
    plt.subplot(212)
    plt.scatter(movie_tsne[:,0], movie_tsne[:,1])
    #plt.show()
    

    
def display_embeddings(user_embedding, movie_embedding):
    """
    embedding is a pyTorch tensor of size  n_samples x embed_size
    """
    # We first normalize the embeddings
    user_ar = user_embedding.detach().numpy()
    print("User embedding range: [{},{}]".format(user_ar.min(), user_ar.max()))
    movie_ar = movie_embedding.detach().numpy()
    print("Movie embedding range: [{},{}]".format(movie_ar.min(), movie_ar.max()))

    user_tsne = manifold.TSNE(n_components=2,perplexity=40).fit_transform(user_ar)
    movie_tsne = manifold.TSNE(n_components=2, perplexity=40).fit_transform(movie_ar)
    
    
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
    
    

model_filename = sys.argv[1]

state = torch.load(model_filename)
model = rs_model.Model(state['nusers'], state['nmovies'],
                       state['embed_size'], state['ratings_range'])
model.load_state_dict(state['state_dict'])

user_embeddings = model.embed_user.weight
movie_embeddings = model.embed_movie.weight

display_embeddings(user_embeddings, movie_embeddings)
tsne_embeddings(user_embeddings, movie_embeddings)

plt.show()

