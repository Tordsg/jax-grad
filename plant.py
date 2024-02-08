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
    def __init__(self, A: float, H: float, C: float, noise) -> None:
        super().__init__()
        self.noise = noise
        self.H0 = H
        self.A = A
        self.H = H
        self.C = C
        self.V = math.sqrt(2 * self.G * self.H)
        self.Q = self.C * self.V
        self.B = H*A
        
    def reset(self) -> None:
        self.B = self.H0*self.A
        self.H = self.H0
        
    def update(self, U: float, noise) -> float:
        self.U = U
        D = noise
        dB = U + D - self.Q
        self.B += dB
        self.H += dB / self.A
        #print('U: ', U, ' H: ', self.H, 'db: ', dB, ' B: ', self.B)
        return self.H
    
class Cournot(Plant):
    def __init__(self, pMax, cM, noise) -> None:
        super().__init__()
        self.Q1 = 0.0
        self.Q2 = 0.0
        self.Q = 0.0
        self.pMax = pMax
        self.cM = cM
        self.noise = noise
    
    def p(self, Q: float) -> float:
        return self.pMax - Q
    
    def reset(self) -> None:
        self.Q1 = 0.0
        self.Q2 = 0.0
        self.Q = 0.0
        self.U = 0.0
    
    def add(self, F: float, X: float) -> float:
        Y = F + X
        if Y < 0:
            return 0
        elif Y > 1:
            return 1
        return Y
        
    
    # def update(self, U: float, noise) -> float:
    #     self.U = U
    #     self.Q1 = self.add(self.Q1, U)
    #     self.Q2 = self.add(self.Q2, noise)
    #     #jax.debug.print("U: {U}, Q1: {Q1}, Q2: {Q2}", U=U, Q1=self.Q1, Q2=self.Q2)
    #     self.Q = self.Q1 + self.Q2
    #     p = self.p(self.Q)
    #     return self.Q1*(p-self.cM)
    def update(self, U: float, noise: float) -> None:
        self.U = U
        self.D = noise

        # Update q1 based on the controller output (U)
        self.Q1 = jax.lax.clamp(0.0, self.Q1 + U, 1.0)

        # Update q2 based on the disturbance (D)
        self.Q2 = jax.lax.clamp(0.0, self.Q2 + self.D, 1.0)

        # Calculate total production q = q1 + q2
        q = self.Q1 + self.Q2

        # Calculate price p(q) = pMax - q
        price = self.pMax - q

        # Calculate profit for producer 1
        P1 = self.Q1 * (price - self.cM)
        return P1
    
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
    # def update(self, U, noise) -> float:
    #     # Plant dynamics with noise
    #     if self.population < 1:
    #         return 0
    #     population_change = U/self.population*self.reproductiveRate*(1+noise)*self.population
    #     self.population += population_change - self.foxes*noise

    #     self.population = jnp.maximum(0, self.population)

    #     return self.population
    def update(self, U, noise) -> float:
        if self.population < 1:
            return 0
        else:
            offspring = jnp.tanh(U/self.population)*self.reproductiveRate*(1+noise)*self.population
            killed = jnp.maximum(0, jnp.tanh(10*self.foxes/self.population)*self.population*(1+noise))
            #jax.debug.print("population: {pop}, food: {food}, offspring: {offspring}, reproductive rate: {reproductiveProbability}, killed: {killed}", pop=self.population, food=U, offspring=offspring, reproductiveProbability=self.reproductiveRate*(1+noise), killed=killed)
            self.dPopulation = offspring - killed
            self.population += self.dPopulation
        return self.population