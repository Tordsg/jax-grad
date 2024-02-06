import math
import random
import jax
import jax.numpy as jnp
class Plant:
    def __init__(self) -> None:
        self.U = .0
        self.D = .0
        self.Y = .0

class Bathtub(Plant):
    G = 9.8
    def __init__(self, A: float, H: float, C: float, D: float) -> None:
        super().__init__()
        self.H0 = H
        self.A = A
        self.H = H
        self.C = C
        self.D = D
        self.V = math.sqrt(2 * self.G * self.H)
        self.Q = self.C * self.V
        self.B = H*A
        
    def reset(self) -> None:
        self.B = self.H0*self.A
        self.H = self.H0
        
    def update(self, U: float) -> float:
        self.U = U
        D = random.uniform(-self.D, self.D)
        dB = U + D - self.Q
        
        
        self.B += dB
        self.H += dB / self.A
        #print('U: ', U, ' H: ', self.H, 'db: ', dB, ' B: ', self.B)
        return self.H
    
class Cournot(Plant):
    def __init__(self, pMax, cM, noise) -> None:
        super().__init__()
        self.Q1 = 0
        self.Q2 = 0
        self.Q = 0
        self.pMax = pMax
        self.cM = cM
        self.noise = noise
    
    def p(self, Q: float) -> float:
        return self.pMax - Q
    
    def reset(self) -> None:
        self.Q1 = 0
        self.Q2 = 0
        self.Q = 0
        self.U = 0
    
    def add(self, Y: float, X: float) -> float:
        Y += X
        if Y < 0:
            return 0
        elif Y > 1:
            return 1
        return Y
        
    
    def update(self, U: float) -> float:
        self.U = U
        self.Q1 = self.add(self.Q1, U)
        self.Q2 = self.add(self.Q2, random.uniform(-self.noise, self.noise))
        self.Q = self.Q1 + self.Q2
        p = self.p(self.Q)
        return self.Q1*(p-self.cM)
    
class ChickenPopulation:
    def __init__(self, initPopulation: int, foxes, noise, reproductiveRate) -> None:
        self.foxes = foxes
        self.noise = noise
        self.initPopulation = initPopulation
        self.population = initPopulation
        self.reproductiveRate = reproductiveRate
        
        
    def reset(self) -> None:
        self.population = self.initPopulation
        self.dPopulation = 0
        self.food = 0
    
    def update(self, U) -> int:
        #Fox population is static
        U = jnp.maximum(0, U)
        noise = random.uniform(self.noise[0], self.noise[1])
        reproduction_factor = self.population / (1 + jnp.exp(-U))
        offspring = jnp.floor(reproduction_factor*self.reproductiveRate*(1+noise))
        jax.debug.print("population: {pop}, food: {food}, offspring: {offspring}, reproductive rate: {reproductiveProbability}", pop=self.population, food=U, offspring=offspring, reproductiveProbability=self.reproductiveRate*(1+noise))
        self.dPopulation = jnp.floor(offspring - jnp.maximum(0, jnp.tanh(self.population*self.foxes/100)))
        self.population += self.dPopulation
        if(self.population < 0):
            self.population = 0
        return self.population