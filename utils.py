#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:43:23 2023

@author: vbr
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def embeddings_by_layer(embeddings,tokens):
    layer,rows,cols = 1,3,4
    fig, ax_full = plt.subplots(rows, cols)
    fig.set_figheight(rows*4)
    fig.set_figwidth(cols*4+3)
    plt.rcParams.update({'font.size': 10})
    emb_n = 0
    for r in range(rows):
        for c in range(cols):
           print()
           print(tokens)
           twodim = TSNE(n_components=2, learning_rate='auto',init='random',perplexity=embeddings[emb_n][0].shape[0]-1).fit_transform(embeddings[emb_n][0])
           print(twodim)
           ax = ax_full[r,c]
           ax.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
           for word, (x,y) in zip(tokens, twodim):
               ax.text(x+0.05, y+0.05, word)
           emb_n+=1
    plt.savefig('embs_12.png')

def attention_by_layer(embeddings,tokens,layer):
    if tokens[0][:3] == 'pat':
        rows=3
    else:
        rows=2
    attention,layer,rows,cols = embeddings,layer,rows,4
    fig, ax_full = plt.subplots(rows, cols)
    fig.set_figheight(rows*4)
    fig.set_figwidth(cols*4+3)
    plt.rcParams.update({'font.size': 10})
    j = 0
    for r in range(rows):
        for c in range(cols):
            ax = ax_full[r,c]
            sattention = attention[layer][0][j].numpy()
            sattention = np.flip(sattention, 0)
            plt.rcParams.update({'font.size': 10})
            im = ax.pcolormesh(sattention, cmap='gnuplot')
            # Show all ticks and label them with the respective list entries
            ax.set_title("Head " + str(j))
            ax.set_yticks(np.arange(len(tokens)))
            if c == 0:
                ax.set_yticklabels(reversed(tokens))
                ax.set_ylabel("Queries")
            else:
                ax.set_yticks([])

            ax.set_xticks(np.arange(len(tokens)))
            if r == rows-1:
                ax.set_xticklabels(tokens)
                ax.set_xlabel("Keys")
            
                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
            else:
                ax.set_xticks([])

            
            # Loop over data dimensions and create text annotations.
            j = j + 1

    fig.suptitle("Layer" + str(layer) + " Multi-head Self-attentions")
    cbar = fig.colorbar(im, ax=ax_full, location='right', shrink=0.5)
    cbar.ax.set_ylabel("Selt-attention", rotation=-90, va="bottom")
    plt.savefig('attention.png')