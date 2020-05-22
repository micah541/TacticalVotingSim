import numpy as np
from sklearn.metrics import euclidean_distances
import random
from collections import Counter
import matplotlib.pyplot as plt


'''simple monte carlo tactics strategy for three Candidates'''


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
    
    def plot1(self):
        plt.scatter(self.Candidates[0,0],self.Candidates[0,1], s=300, c='b')
        plt.scatter(self.Candidates[1,0],self.Candidates[1,1], s=300, c='g')
        plt.scatter(self.Candidates[2,0],self.Candidates[2,1], s=300, c='r')
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='b') for i in range(self.size) if self.vote_orders[i][0]==0]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='g') for i in range(self.size) if self.vote_orders[i][0]==1]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='r') for i in range(self.size) if self.vote_orders[i][0]==2]
        plt.show()
    
    def plot2(self, votelist, winner, changevotes=[], pause=True):
        plt.scatter(self.Candidates[0,0],self.Candidates[0,1], s=300, c='b')
        plt.scatter(self.Candidates[1,0],self.Candidates[1,1], s=300, c='g')
        plt.scatter(self.Candidates[2,0],self.Candidates[2,1], s=300, c='r')
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='b') for i in range(self.size) if votelist[i]==0 and i not in changevotes]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='g') for i in range(self.size) if votelist[i]==1 and i not in changevotes]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='r') for i in range(self.size) if votelist[i]==2 and i not in changevotes]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='y') for i in range(self.size) if votelist[i]==3 and i not in changevotes]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='k') for i in range(self.size) if votelist[i]==4 and i not in changevotes]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='c') for i in range(self.size) if votelist[i]==5 and i not in changevotes]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='b', marker='^') for i in range(self.size) if votelist[i]==0 and i in changevotes]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='g', marker='^') for i in range(self.size) if votelist[i]==1 and i in changevotes]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='r',marker='^') for i in range(self.size) if votelist[i]==2 and i in changevotes]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='y',marker='^') for i in range(self.size) if votelist[i]==3 and i in changevotes]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='k',marker='^') for i in range(self.size) if votelist[i]==4 and i in changevotes]
        [plt.scatter(self.Agents[i,0], self.Agents[i,1], s=10, c='c',marker='^') for i in range(self.size) if votelist[i]==5 and i in changevotes]
        if(winner==0): plt.scatter(self.Candidates[0,0],self.Candidates[0,1], s=400, marker='*')
        if(winner==1):plt.scatter(self.Candidates[1,0],self.Candidates[1,1], s=400, marker='*')
        if(winner==2):plt.scatter(self.Candidates[2,0],self.Candidates[2,1], s=400, marker='*')
        if(pause) : plt.show()
        if(not pause) : 
            plt.show(block=False)
            plt.pause(2)
            plt.close()


def is_tied(vote_list):
    firsts = [v[0] for v in vote_list]
    fcounts = Counter(firsts)
    try: 
        if fcounts.most_common(3)[2][1]==fcounts.most_common(3)[1][1]: return(True)
    except: pass ## if there's only 2 with first counts, 2nd and 3rd aren't tied
    try: top2 = fcounts.most_common(2)[0][0], fcounts.most_common(2)[1][0]
    except: return(False)
    fpv = [list(vl).index(top2[0])<list(vl).index(top2[1]) for vl in vote_list]
    if Counter(fpv)[True]==Counter(fpv)[False]: return(True)
    else: return(False)
    

VP = [[0,1,2], [1,2,0], [2,0,1], [1,0,2], [0,2,1], [2,1,0]]

def winner(vote_list):
    firsts = [v[0] for v in vote_list]
    fcounts = Counter(firsts)
    try: 
        if fcounts.most_common(3)[2][1]==fcounts.most_common(3)[1][1]: return(None)
    except: pass ## if there's only 2 with first counts, 2nd and 3rd aren't tied
    try: top2 = fcounts.most_common(2)[0][0], fcounts.most_common(2)[1][0]
    except: return(fcounts.most_common(2)[0][0])
    fpv = [list(vl).index(top2[0])<list(vl).index(top2[1]) for vl in vote_list]
    if Counter(fpv)[True]>Counter(fpv)[False]: return(top2[0])
    if Counter(fpv)[True]<Counter(fpv)[False]: return(top2[1])
    else: return(None)
    


def cpMC(S,pop):  
    pm = pop.pmatrix
    vo = pop.vote_orders
    N = len(pm)
    samples = [random.sample(range(N), S) for i in range(10000)]
    vote_samples = [[vo[s] for s in ss] for ss in samples] 
    tally = np.zeros(shape = (6,3))
    for i in range(6):
        winners = [winner(t+[VP[i]]) for t in vote_samples]
        tally[i] = [Counter(winners)[i] for i in range(3)]
    tally = tally/tally.sum(axis =1 ).reshape(6,1)
    smatrix = tally.dot(pm.transpose()).transpose()
    best_option = [np.argmin(sm) for sm in smatrix]
    return(best_option)



def mixer(a,b,q):  #replace a with b with prob q
    aa = a.copy()
    for i in range(len(a)):
        if np.random.rand()<q : aa[i]=b[i]
    return(aa)


S_FRACTION = .5

p1 = Population(180)
ov = p1.vote_orders.copy()
vo = p1.vote_orders
p1.plot1()
votewinner = winner(vo)
p1.plot2([VP.index(list(v)) for v in vo],votewinner)
mixed_votes = vo.copy()
streak = 0
for i in range(200):
    old_vo = mixed_votes.copy()
    best = cpMC(S,p1)
    vo = [VP[b] for b in best]
    mixed_votes = mixer(old_vo, vo, S_FRACTION )
    p1.vote_orders = mixed_votes
    votewinner = winner(mixed_votes)
    changes = [k for k in range(len(best)) if list(old_vo[k])!=list(mixed_votes[k])]
    p1.plot2([VP.index(list(v)) for v in mixed_votes], votewinner, changes, False)
    if (changes==[]): streak+=1
    else : streak = 0
    print(streak, len(changes))


    


