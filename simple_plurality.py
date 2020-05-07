import numpy as np
from sklearn.metrics import euclidean_distances
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from collections import Counter
import matplotlib.pyplot as plt
import scipy 


'''simple plurality tactics strategy for three Candidates'''


class Population:
    def __init__(self, size):
        self.size = size
        self.Agents = np.random.randn(self.size,2)
        self.Candidates = np.random.randn(3,2)
        self.pmatrix = euclidean_distances(self.Agents, self.Candidates)
        self.vote_orders = [row.argsort() for row in self.pmatrix]  
        self.honest = [np.argmin(p) for p in self.pmatrix]
        self.perm_matrices = [np.eye(3)[list(v)] for v in self.vote_orders]
        self.inv_matrices = [np.linalg.inv(m) for m in self.perm_matrices]
        self.training_matrix = [np.eye(3) for i in range(self.size)] #these will 
        self.ordered_pmatrix = np.array([self.pmatrix[i].dot(self.inv_matrices[i]) for i in range(size)])
        self.votes=np.zeros(3)
        for i in range(self.size): self.votes[np.argmin(self.pmatrix[i])]+=1
        self.W2 = self.pmatrix.sum(axis=0)
        for i in range(self.size):
            m = np.array([self.pmatrix[i], self.votes, self.W2])
            self.training_matrix[i] = m.dot(self.inv_matrices[i])
    
    def plot(self, model):
        plt.scatter(self.Candidates[0,0],self.Candidates[0,1], s=300, c='b')
        plt.scatter(self.Candidates[1,0],self.Candidates[1,1], s=300, c='g')
        plt.scatter(self.Candidates[2,0],self.Candidates[2,1], s=300, c='r')
        svotes = SVote(self.training_matrix,self.perm_matrices, model)
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='b') for i in range(self.size) if svotes[i]==0]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='g') for i in range(self.size) if svotes[i]==1]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='r') for i in range(self.size) if svotes[i]==2]
        win = Counter(svotes).most_common()[0][0]
        plt.scatter(self.Candidates[win,0],self.Candidates[win,1], s=300, marker="*")
        plt.show()


model2 = Sequential()
model2.add(Dense(20, activation='sigmoid', input_dim=9))
model2.add(Dropout(0.15))
model2.add(Dense(30, activation='sigmoid'))
model2.add(Dropout(0.15))
model2.add(Dense(10, activation='sigmoid'))
model2.add(Dropout(0.15))
model2.add(Dense(3, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam')



def SVote(mm,perms, mod):
    train = np.array([m.reshape(9,) for m in mm])
    output = mod.predict(train)
    output = [output[i].dot(perms[i]) for i in range(len(mm))]
    votes = [np.argmax(o) for o in output]
    return(votes)

def OptVote(Svotes, pmatrix, SampleParameter=8):
    VC = Counter(Svotes)
    tally = [VC[0], VC[1], VC[2]]
    pivots = cpN(SampleParameter, tally)
    pivots = pivots+pivots.transpose()
    DIFF = [np.tile(p, (3,1))-np.tile(p, (3,1)).transpose() for p in pmatrix]
    PR = [np.diag(pivots.dot(-d)) for d in DIFF]
    optvote = [np.argmax(pr) for pr in PR]
    one_hot = np.zeros(shape = pmatrix.shape)
    for i in range(len(pmatrix)): one_hot[i,optvote[i]]+=1
    return(one_hot, optvote)


''' The following is a very basic pivot probability function, using different methods than in Large Poisson Games. Instead, you give the function a set of three votes, say 134,189,150 and a sample size S, which I take to be somewhat low, like 8 or 11, and then sample 8 from the population of votes and compute probability that any two are tied for the lead.  It's not monte carlo, it's explicity n choose k computations.  Returns an unsymmetrized matrix'''



def cpN(S,tv):  
    ssc = scipy.special.comb  
    pivot_matrix = np.zeros([3,3])
    NT = tv[0]+tv[1]+tv[2]
    combs = [(0,1,2), (1,2,0), (2,0,1)]
    tie_values = [k for k in range(int(S/2)+1) if 3*k>S]
    for (a,b,c) in combs:
        for k in tie_values:
            value = ssc(tv[a], k)*ssc(tv[b], k)*ssc(tv[c], S-2*k)/ssc(NT, S)
            pivot_matrix[a,b]+=value
    return(pivot_matrix/pivot_matrix.sum())

    

def mixer(a,b,q):
    aa = a.copy()
    for i in range(len(a)):
        if np.random.rand()<q : aa[i]=b[i]
    return(aa)




def train_model(tr, mod, psize):
    for k in range(tr):
        pop = Population(psize)
        svote = SVote(pop.training_matrix,pop.perm_matrices, mod)
        mix = mixer(svote, pop.honest, .25)
        opt = OptVote(mix, pop.pmatrix)[0]
        train = np.array([m.reshape(9,) for m in pop.training_matrix])
        target = np.array([opt[i].dot(pop.inv_matrices[i]) for i in range(pop.size)])
        mod.fit(train, target, epochs = 10)


TRAIN_ROUNDS = 10000000
POP_SIZE=150

train_model(TRAIN_ROUNDS, model2, POP_SIZE)

test_pop = Population(200)
test_pop.plot(model2)



