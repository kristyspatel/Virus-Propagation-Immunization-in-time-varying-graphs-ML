__author__ = 'Kristy'

from igraph import *
from os.path import join
from os import listdir
import scipy.linalg as sc
from pylab import *
import numpy as np
import random
import __builtin__ as base
import time

#Get the highest eigen value of the graph
def highest_eigen_value(g):
    length = len(g.vs)
    eg_val = sc.eigh(g.get_adjacency().data,eigvals_only=True,eigvals=(length-1,length-1))
    return eg_val[0]

def shield_score(g):
    length = len(g.vs)
    A = np.matrix(g.get_adjacency().data)
    eg = sc.eigh(A,eigvals_only=False,eigvals=(length-1,length-1))
    lamda = eg[0][0]
    u = eg[1]
    S=[]
    v=[0,0,0,0,0,0,0,0,0,0]
    score = [0,0,0,0,0,0,0,0,0,0]
    for j in range(length):
        v[j] = (2*lamda - A[j,j])*u[j][0]*u[j][0]
    for i in range(3):
        B = A[:,S]
        b = np.dot(B,u[S])
        for j in range(length):
            if j in S:
                score[j] = -1
            else:
                score[j] = v[j] - 2 *b[j]*u[j]
        l = np.argmax(score)
        S.append(l)
    return S


#calculate effective strength = (beta/delta)*highest eigen value
def effective_strength(high_eig,beta,delta):
    return round((beta/delta)*high_eig , 3)

# To see the effect of changing beta on the effective strength of virus, keeping delta fixed.
def plot_beta(high_eig,delta,filename,save):

    """
    Plot the 10 beta values from 0 to 1.0, step by 0.1 to see the effective strength at each beta value
    """
    betas = np.arange(0.,1.,0.1)
    strengths = [effective_strength(high_eig,b,delta) for b in betas]
    min_beta = delta/high_eig

    #plot the results
    figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(betas,strengths, "-o")
    axhline(y=min_beta, ls='-', c='r',
            label='Minimum Transimission Probability: %.2f'%(min_beta),
            lw=2)
    plt.grid(True)
    plt.legend(loc='best')
    plt.title("Virus Propogation")
    plt.xlabel('Beta')
    plt.ylabel('Effective Strength')
    if save:
        savefig(join('graph', filename+"_beta"+str(time.time())+".png"),bbox_inches='tight')
    else:
        plt.show()
    print 'Minimum transmission probability: %.3f' % (min_beta)


# To see the effect of changing delta on the effective strength of virus, keeping beta fixed.
def plot_delta(high_eig,beta,filename,save):
    deltas = np.arange(0.,1.,0.1)
    strengths = [effective_strength(high_eig,beta,d) for d in deltas]
    max_delta = beta*high_eig

    #plot the results
    figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(deltas,strengths,'o-')
    axhline(y=max_delta, ls='-', c='r',
            label='Maximum Healing Probability: %.3f'%(max_delta),
            lw=2)
    plt.grid(True)
    plt.legend(loc='best')
    plt.title("Virus Propogation")
    plt.xlabel('Delta')
    plt.ylabel('Effective Strength')
    if save:
        savefig(join('graph', filename+"_delta"+str(time.time())+".png"),bbox_inches='tight')
    else:
        plt.show()
    print 'Maximum healing probability: %.3f' % (max_delta)

#for given node in the graph, find the infected neighbors
def get_infected_neighbors(g,n,beta):
    infected =  [k for k in g.neighbors(g.vs[n]) if np.random.uniform(0,1) <= beta]
    return infected;


"""
 simulate for t=100 and return the number of infected nodes at each time step in a list.
 select c random nodes from the graph to infect initially. At each step infect other neighboring nodes by randomly choosing a number for each node from the uniform
 probability distribution. if the randomly chosen number for the node is <= beta then the node is infected.
 At the same time, also cure the nodes infected in previous step. Generate random probability number from uniform distribution for each infected node in previous step and
 if the probability is <= delta then the node is cured.
 At current time step, total infected nodes = infected_in_previous_step + newly_infected in current_time_step - Cured in current time step.
"""
def simulate(g ,beta,delta,t,c):
    samples = random.sample(range(g.vs[0].index,g.vs[len(g.vs)-1].index),c)
    infected = samples
    num_infected = [len(samples)]
    for i in xrange(1,t):
        infected_at_t = base.sum([get_infected_neighbors(g,n,beta) for n in infected],[])
        cured =  [k for k in infected if np.random.uniform(0,1) <= delta]
        infected = list(set(infected)-set(cured))
        infected = list(set(infected + infected_at_t))
        num_infected.append(len(infected))
    return num_infected

#To run the simulate step for 'runs' times and get average number of infected nodes at each time step.
def run_simulation(g,beta,delta,t,c,runs):
    simulations=[]
    for i in xrange(0,runs):
        simulations = simulations + [simulate(g,beta,delta,t,c)]
    frac_affected = []
    for i in xrange(0,t):
        frac_affected.append(mean([simulations[j][i] for j in range(runs)])/len(g.vs))
    return frac_affected

#Randomly immunize the k nodes. Select k nodes from graph vertices randomly and immunize them. The immunized nodes are deleted form the graph.
def random_immunize(g,k):
    g1 = g.copy()
    nodes = random.sample(range(len(g1.vs)),k)
    g1.delete_vertices(nodes)
    return g1

# select k nodes from the graph with highest degree and immunize them. The immunized nodes are deleted form the graph.
def high_degree_immunize(g,k):
    g1 = g.copy()
    degree = {v: g1.degree(v) for v in g1.vs}
    sorted_deg = sorted(degree.keys(),key = lambda s : degree[s],reverse = True)
    g1.delete_vertices(sorted_deg[:k])
    return g1

"""
 select the node with highest degree and immunize it. Remove it from graph. Now from the graph again remove the node with highest degree and immunize it.
 Repeat this process until there are k nodes that are immunized.
"""
def high_degree_immunize_iteratively(g,k):
    g1 = g.copy()
    for i in range(k):
        g1 = high_degree_immunize(g1,1)
    return g1

"""
    Get the highest eigen value of the graph and corresponding eigen vector.
    Sort the absolute values of the elements of eigen vector.
    Remove the nodes corresponding to the maximum values of the eigen vectors from the graph and immunize them.
"""
def largest_eig_vec_immunize(g,k):
    g1 = g.copy()
    largest_eig = sc.eigh(g1.get_adjacency().data,eigvals=(len(g1.vs)-1,len(g1.vs)-1))
    eig_vec = {i: largest_eig[1][i] for i in range(len(largest_eig[1]))}
    sorted_deg = sorted(eig_vec.keys(),key = lambda s : abs(eig_vec[s][0]),reverse = True)
    max_eig_vertices =  [g1.vs[i] for i in sorted_deg][:k]
    g1.delete_vertices(max_eig_vertices)
    return g1



def plot_simulation(f_infected,t,filename,save):

    """
    Plot the simulation results. Number of infected nodes at each time step on y axis and time ticks on x axis.
    """
    t = range(0,t)
    figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(t,f_infected, "-o")
    plt.grid(True)
    plt.title("Virus Propogation")
    plt.xlabel('Time')
    plt.ylabel('Fraction of Infected nodes')
    if save:
        savefig(join('graph', filename+"_virus_simulation"+str(time.time())+".png"),bbox_inches='tight')
    else:
        plt.show()


"""
 Study the variations of changing k and the effective strength after immunization using k vaccines
 First immunize the graph based on the policy chosen and valu of k. Then calculate effective strength of virus on after immuniztion and plot the results for different k values
"""
def plot_k_variations(g,k_list,beta,delta,policy,filename,save):
    g1 = g.copy()
    eff_strength = []
    for k in k_list:
        if policy == "A":
            g1 = random_immunize(g1,k)
        if policy == "B":
            g1 = high_degree_immunize(g1,k)
        if policy == "C":
            g1 = high_degree_immunize_iteratively(g1,k)
        if policy == "D":
            g1 = largest_eig_vec_immunize(g1,k)
        h_eigen = highest_eigen_value(g1)
        eff_strength.append(effective_strength(h_eigen,beta,delta))
        g1 = g.copy()
    figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(k_list,eff_strength, "-o")
    plt.grid(True)
    plt.title("Immunization chart and Effective Strength")
    plt.xlabel('Number of vaccines')
    plt.ylabel('Effective Strength')
    if save:
        savefig(join('graph', filename+"_immunization_simulation"+str(time.time())+".png"),bbox_inches='tight')
    else:
        plt.show()


"""
Get system matrix
Calculate Si for each graph as follows:
Si = (1-delta) * I + beta * Adjacency matrix of graph i
Multiply all the Si to get system matrix
System matrix = S0 * S1 * ... * Sn
"""
def get_system_matrix(graphs,beta,delta):
    #S1 = (1-delta)*np.identity(len(graphs[0].vs)) + beta*graph.get_adjacency().data
    S=[]
    for graph in graphs:
        Si = (1-delta)*np.identity(len(graph.vs)) + beta*np.array(graph.get_adjacency().data)
        S.append(Si)
    system_matrix = S[0]
    for i in range(1,len(S)):
        system_matrix = system_matrix.dot(S[i])
    return system_matrix


def plot_beta_alternating(graphs,beta,delta,filename,save):

    """
    Plot the 10 beta values from 0 to 1.0, step by 0.1 to see the effective strength at each beta value
    Get the system matrix for each beta value and calculate the effective strength i.e. highest eigen value of the system matrix
    """
    betas = np.arange(0.,1.1,0.1)
    strengths = []
    for b in betas:
        S = get_system_matrix(graphs,b,delta)
        num_rows = np.shape(S)[0]
        highest_eigen_value = sc.eigh(S,eigvals_only=True,eigvals=(num_rows-1,num_rows-1))[0]
        strengths.append(highest_eigen_value)
    figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(betas,strengths, "-o")
    plt.grid(True)
    plt.title("Virus Propagation")
    plt.xlabel('Beta')
    plt.ylabel('Effective Strength')
    if save:
        savefig(join('graph', filename+"_alternating_beta"+str(time.time())+".png"),bbox_inches='tight')
    else:
        plt.show()


def plot_delta_alternating(graphs,beta,delta,filename,save):
    """
    Plot the 10 delta values from 0 to 1.0, step by 0.1 to see the effective strength at each beta value
    Get the system matrix for each beta value and calculate the effective strength i.e. highest eigen value of the system matrix
    """
    deltas = np.arange(0.,1.1,0.1)
    strengths = []
    for d in deltas:
        S = get_system_matrix(graphs,beta,d)
        num_rows = np.shape(S)[0]
        highest_eigen_value = sc.eigh(S,eigvals_only=True,eigvals=(num_rows-1,num_rows-1))[0]
        strengths.append(highest_eigen_value)
    figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(deltas,strengths, "-o")
    plt.grid(True)
    plt.title("Virus Propagation")
    plt.xlabel('delta')
    plt.ylabel('Effective Strength')
    if save:
        savefig(join('graph', filename+"_alternating_beta"+str(time.time())+".png"),bbox_inches='tight')
    else:
        plt.show()

"""
    Take graph[t % len(graphs)] at time t and infect c nodes initially in the chosen graph.
     Carry out the infecting process as explained before but on alternate graphs at each time step.
"""
def simulate_alternating(graphs ,beta,delta,t,c):
    samples = random.sample(range(len(graphs[0].vs)),c)
    total_graphs = len(graphs)
    infected = samples
    num_infected = [len(samples)]
    for i in xrange(1,t):
        g = graphs[i%total_graphs]
        infected_at_t = base.sum([get_infected_neighbors(g,n,beta) for n in infected],[])
        cured =  [k for k in infected if np.random.uniform(0,1) <= delta]
        infected = list(set(infected)-set(cured))
        infected = list(set(infected + infected_at_t))
        num_infected.append(len(infected))
    return num_infected

"""
    Plot the simulation results for alternate graphs.
"""
def run_simulation_alternating(graphs,beta,delta,t,c,runs):
    simulations=[]
    for i in xrange(0,runs):
        simulations = simulations + [simulate_alternating(graphs,beta,delta,t,c)]
    frac_affected = []
    for i in xrange(0,t):
        frac_affected.append(mean([simulations[j][i] for j in range(runs)])/len(graphs[0].vs))
    return frac_affected

"""
Randomly choose k nodes and remove those k nodes from each graph and immunize them
"""
def random_immunize_alternating(graphs,k):
    g1 = [g.copy() for g in graphs]
    nodes = random.sample(range(len(g1[0].vs)),k)
    for g in g1:
        g.delete_vertices(nodes)
    return g1

"""
 Choose k nodes with highest average degree across all networks. Remove those k nodes from all graphs and immunize them.
"""
def high_degree_immunize_alternating(graphs,k):
    g1 = [g.copy() for g in graphs]
    N = len(g1[0].vs)
    degree = {i: mean([g1[0].degree(i),g1[1].degree(i)]) for i in range(N)}
    sorted_deg = sorted(degree.keys(),key = lambda s : degree[s],reverse = True)
    for g in g1:
        g.delete_vertices(sorted_deg[:k])
    return g1

"""
 Choose a nodes with highest average degree across all networks. Remove that nodes from all graphs and immunize them.
  Repeat this process until k nodes are immunized.
"""
def high_degree_immunize_iteratively_alter(graphs,k):
    g1 = [g.copy() for g in graphs]
    for i in range(k):
        g1 = high_degree_immunize_alternating(g1,1)
    return g1

"""
Select the largest eigen value of the system matrix and the corresponding eigen vector.
Sort the eigen vector and immunize k nodes corresponding to the highest eigen vector values from all the graphs. Remove the immunized nodes
"""
def largest_eig_vec_immunize_alter(system_matrix,graphs,k):
    g1 = [g.copy() for g in graphs]
    num_rows = np.shape(system_matrix)[0]
    largest_eig = sc.eigh(system_matrix, eigvals=(num_rows-1, num_rows-1))
    eig_vec = {i: largest_eig[1][i] for i in range(len(largest_eig[1]))}
    sorted_deg = sorted(eig_vec.keys(),key = lambda s : abs(eig_vec[s][0]),reverse = True)
    max_eig_vertices =  sorted_deg[:k]
    for g in g1:
        g.delete_vertices(max_eig_vertices)
    return g1

"""
Plot the variations of k. Immunize the graphs depending on the policy chosen and study the effective strengths for each k value after immunization
"""
def plot_k_variations_alternating(graphs,k_list,beta,delta,policy,system_matrix,filename,save):
    g1 = [g.copy() for g in graphs]
    eff_strength = []
    for k in k_list:
        if policy == "A":
            g1 = random_immunize_alternating(g1,k)
        if policy == "B":
            g1 = high_degree_immunize_alternating(g1,k)
        if policy == "C":
            g1 = high_degree_immunize_iteratively_alter(g1,k)
        if policy == "D":
            g1 = largest_eig_vec_immunize_alter(system_matrix,g1,k)
        sys_matrix = get_system_matrix(g1,beta,delta)
        num_rows = np.shape(sys_matrix)[0]
        highest_eigen_value = sc.eigh(sys_matrix,eigvals_only=True,eigvals=(num_rows-1,num_rows-1))[0]
        eff_strength.append(highest_eigen_value)
        g1 = [g.copy() for g in graphs]
    figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(k_list,eff_strength, "-o")
    plt.grid(True)
    plt.title("Immunization chart and Effective Strength")
    plt.xlabel('Number of vaccines')
    plt.ylabel('Effective Strength')
    if save:
        savefig(join('graph', filename+"_immunization_simulation"+str(time.time())+".png"),bbox_inches='tight')
    else:
        plt.show()



def get_graph(file):
    g = Graph()
    with open(file) as fi:
            v,e = fi.next().split()
            e_list = [(int(line.split()[0]) , int(line.split()[1])) for line in list(fi)]
            g.add_vertices(int(v))
            g.add_edges(e_list)
    return g



if __name__ == '__main__':
    g= Graph
    g.add_vertices(10)
    g.add_edges([(0, 1), (0, 3), (1, 4), (2, 3), (2, 6), (5, 6), (5, 7), (6, 7),(7, 9), (8, 9)])
    S = shield_score(g)
    print S
"""if __name__ == '__main__':
    net_type = sys.argv[2]
    out_file = sys.argv[3]
    print net_type
    beta1 = 0.20
    beta2 = 0.01
    delta1 = 0.70
    delta2 = 0.60
    k1 = 200
    if net_type == 'static':
        g = get_graph(sys.argv[1])
        highest_e_val = highest_eigen_value(g)
        eff_strength1 = effective_strength(highest_e_val,beta1,delta1)
        print "The effective strength of the virus for Beta = %.2f and Delta = %.2f is %.3f " %(beta1,delta1,eff_strength1)
        if eff_strength1 > 1:
            print "The infection will spread across the network."
        else:
            print "The virus will die quickly."

        eff_strength2 = effective_strength(highest_e_val,beta2,delta2)
        print "The effective strength of the virus for Beta = %.2f and Delta = %.2f is %.3f " %(beta2,delta2,eff_strength2)
        if eff_strength2 > 1:
            print "The infection will spread across the network."
        else:
            print "The virus will die quickly."

        plot_beta(highest_e_val,delta1,out_file,True)
        plot_delta(highest_e_val,beta1,out_file,True)
        plot_beta(highest_e_val,delta1,out_file,True)
        plot_delta(highest_e_val,beta2,out_file,True)

        frac_infected = run_simulation(g,beta1,delta1,100,len(g.vs)/10,10)
        plot_simulation(frac_infected,100,out_file,True)
        frac_infected = run_simulation(g,beta2,delta2,100,len(g.vs)/10,10)
        plot_simulation(frac_infected,100,out_file,True)

    if net_type == 'alernating':
        files = {f:join(sys.argv[1],f) for f in listdir(sys.argv[1])}
        graphs = [get_graph(f) for g,f in files.iteritems()]
        system_matrix = get_system_matrix(graphs,beta1,delta1)
        num_rows = np.shape(system_matrix)[0]
        highest_eigen_value = sc.eigh(system_matrix,eigvals_only=True,eigvals=(num_rows-1,num_rows-1))[0]
        #betas =
"""
