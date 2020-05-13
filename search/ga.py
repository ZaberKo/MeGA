import numpy as np
from .model_eval_ddp import evaluate_arch
from utils import path_idx2name

class Individual(object):
    def __init__(self,gene,fitness):
        self.gene=gene
        self.fitness=fitness

    def __eq__(self,other:Individual):
        for g1,g2 in zip(self.gene,other.gene):
            if g1!=g2:
                return False
        return True


class MGA(object):
    '''
        Microbial GA:
        only modified the bad gene
    '''
    def __init__(self, dna_size,dna_bound, pop_size, mutation_rate=None,crossover_rate=None):
        self.dna_size = dna_size
        self.dna_bound=dna_bound
        self.pop_size = pop_size
        self.pop = self._init_pop()
        # best fitness individual
        self.bsf_p=max(self.pop,key=lambda p: p.fitness)

        self.mutation_rate=mutation_rate
        self.crossover_rate=crossover_rate
        return

    def _init_pop(self):
        self.pop=[]
        genes= np.random.randint(0, self.dna_bound, size=(self.pop_size, self.dna_size))
        for gene in genes:
            fitness=self.get_fitness(gene)
            p=Individual(gene,fitness)
            self.pop.append(p)

    def binary_tournament_select_parants(self):
        indices=np.random.choice(self.pop_size,4)
        p1_i,p2_i,p3_i,p4_i=indices
        p1,p2,p3,p4=self.pop[indices]
        pa=p1 if p1.fitness>p2.fitness else p2
        pa_i=p1_i if p1.fitness>p2.fitness else p2_i
        pb=p3 if p3.fitness>p4.fitness else p4
        pb_i=p3_i if p3.fitness>p4.fitness else p4_i
        return pa,pa_i,pb,pb_i



    def get_fitness(self, gene):
        path=path_idx2name([gene])[0]
        return evaluate_arch(path)


    def crossover2(self, p1_gene, p2_gene):
        mask=np.random.rand(self.dna_size)<self.crossover_rate
        child_gene=np.copy(p1_gene)
        child_gene[mask]=p2_gene[mask]
        return child_gene

    def crossover(self, child_gene, p_winner_gene):
        if np.random.rand()<self.crossover_rate:
            i1,i2=np.sort(np.random.randint(self.dna_size,size=2))
            
            mask=np.zeros(self.dna_size)
            mask[i1:i2+1]=1
            mask=mask.astype(np.bool)

            child_gene[mask]=p_winner_gene[mask]
        return child_gene

    def mutate(self, child_gene):
        if np.random.rand()<self.mutation_rate:
            choice=np.random.randint(self.dna_size)
            child_gene[choice]=np.random.randint(self.dna_bound)

        return child_gene

    def evolve(self,n):
        for i in range(n):
            # Step 1. Mating selection
            p1,p1_i,p2,p2_i=self.binary_tournament_select_parants()
            if p1.fitness<p2.fitness:
                (p_losser,p_losser_i)=(p1,p1_i)
            else: 
                (p_winner,p_winner_i)=(p2,p2_i)

            child_gene=p_losser.gene.copy()
            # Step 2. Variation operator1: Crossover
            child_gene=self.crossover(child_gene,p_winner.child_gene)
            # Step 3. Variation operator2: Mutation
            child_gene=self.mutate(child_gene)
            # Step 4. Evaluate the child u
            fitness=self.get_fitness(child_gene)
            child=Individual(child_gene,fitness)
            # Step 5. Selection: replace the losser parent
            self.pop[p_losser_i]=Individual(child_gene,fitness)
            # Step 6. Update the best-so-far solution
            if child.fitness>self.bsf_p.fitness:
                self.bsf=child



