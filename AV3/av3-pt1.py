import numpy as np
import matplotlib . pyplot as plt
from typing import Union, Literal

problem = 1 # Variavel para definir qual o problema a ser analisado
alghoritm: Union[str, Literal['HillClimbing', 'LRS', 'GRS', 'SimulatedAnnealing']] = 'HillClimbing' # Variavel para setar o algoritmo
isToPlotTheGraph = True

# Definição de variaveis
MaximoIteracao = 1000
MaximoCanditados = 10
Epsilon = 0.1
Sigma = 0.1
Alpha = 0.9
Temperatura = 10

if(problem == 1):
    Low = -100
    High = 100
elif(problem == 2):
    Low = -2
    High = 4
elif(problem == 3):
    Low = -8
    High = 8
elif(problem == 4):
    Low = -5.12
    High = 5.12
elif(problem == 5):
    Low = -2
    High = 2
elif(problem == 6):
    Low = -1
    High = 3
elif(problem == 7):
    Low = 0
    High = np.pi
elif(problem == 8):
    Low = -200
    High = 20

def fn(x1 , x2):
    if(problem == 1):
        return ( x1 **2+ x2 **2)
    elif(problem == 2):
        return np.exp(-x1**2 - x2**2) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 1.7)**2))
    elif(problem == 3):
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) + -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)
    elif(problem == 4):
        return x1**2 - 10 * np.cos(2 * np.pi * x1) + 10 + x2**2 - 10 * np.cos(2 * np.pi * x2) + 10
    elif(problem == 5):
        return (x1 - 1)**2 + 100 * (x2 - x1**2)**2
    elif(problem == 6):
        return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1
    elif(problem == 7):
        return -np.sin(x1) * np.sin((x1**2) / np.pi)**2.10 + -np.sin(x2) * np.sin((2 * x2**2) / np.pi)**2.10
    else:
        return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + x2 + 47))) + -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

#Função para plotar o gráfico
def plotGraph(x1Cand, x2Cand):
    if(isToPlotTheGraph):
        x1 = np . linspace( Low ,High , MaximoIteracao)
        X1 , X2 = np.meshgrid(x1 , x1 )
        Y = fn(X1 , X2 )
        f_cand = fn( x1Cand , x2Cand )
        fig = plt . figure ()
        ax = fig . add_subplot (projection = '3d')
        ax.plot_surface ( X1 ,X2 , Y, rstride =10 , cstride =10 , alpha =0.6 , cmap ='jet')
        ax.scatter ( x1Cand , x2Cand , f_cand , marker ='x',s =90 , linewidth =3 , color ='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('f(x1 ,x2)')
        plt.tight_layout()
        plt.show ()

# Algoritmo Hill Climbing
def HillClimbing(low, high, epsilon, maxIt, maxCad):
    x1Best = np.random.uniform(low, high)
    x2Best = np.random.uniform(low, high)
    fBest = fn(x1Best, x2Best)
    
    i = 0
    while i < maxIt and epsilon > 1e-5:  # Adiciona um critério de parada baseado em epsilon
        j = 0
        improvement = False
        
        while j < maxCad:
            j += 1
            candidate1 = np.random.uniform(x1Best - epsilon, x1Best + epsilon)  # Gera um candidato vizinho
            candidate2 = np.random.uniform(x2Best - epsilon, x2Best + epsilon)  # Gera um candidato vizinho
            
            F = fn(candidate1, candidate2)
            
            if F > fBest:
                x1Best, x2Best = candidate1, candidate2
                fBest = F
                improvement = True
                break
        
        if not improvement:
            epsilon *= 0.9  # Reduz o tamanho do passo se não houver melhoria
        i += 1

    return [x1Best, x2Best], fBest

# Algoritmo Local Random Search (LRS)
def LocalRandomSearch(low, high, maxIt, sigma):
    xBest = np.random.uniform(low, high, size=2)  # Ponto inicial aleatório dentro do intervalo
    fBest = fn(*xBest)
    
    i = 0
    while i < maxIt:
        n = np.random.normal(0, sigma, size=2)  # Perturbação
        
        xCand = xBest + n  # Candidato gerado
        
        # Verificação de limites (restrição em caixa)
        xCand = np.clip(xCand, low, high)
        
        fCand = fn(*xCand)
        
        if fCand > fBest:
            xBest = xCand
            fBest = fCand
        
        i += 1
    
    return xBest, fBest

# Algoritmo Global Random Search (GRS)
def GlobalRandomSearch(low, high, maxIt):
    xBest = np.random.uniform(low, high, size=2)  # Ponto inicial aleatório dentro do intervalo
    fBest = fn(*xBest)
    
    i = 0
    while i < maxIt:
        xCand = np.random.uniform(low, high, size=2)  # Candidato gerado
        
        fcand = fn(*xCand)
        
        if fcand > fBest:
            xBest = xCand
            fBest = fcand
        
        i += 1
    
    return xBest, fBest

# Algoritmo Simulated Annealing
def SimulatedAnnealing(low, high, maxIt, temp ,sigma, alpha):
    xBest = np.random.uniform(low, high, size=2)  # Ponto inicial aleatório dentro do intervalo
    fBest = fn(*xBest)
    
    i = 0
    while i < maxIt:
        n = np.random.normal(0, sigma, size=2)  # Perturbação aleatória
        
        xCand = xBest + n  # Candidato gerado
        
        # Verificação de limites (restrição em caixa)
        xCand = np.clip(xCand, low, high)
        
        fCand = fn(*xCand)
        
        if fCand < fBest:
            xBest = xCand
            fBest = fCand
        elif np.random.uniform(0, 1) < np.exp((fBest - fCand) / T):
            xBest = xCand
            fBest = fCand
        
        i += 1
        T = temp * alpha  # Reduz a temperatura
        
    return xBest, fBest

# Execução dos algoritmos
def RunAlgorithm():
    if(alghoritm == 'HillClimbing'):
        xBest, fBest = HillClimbing(low=Low, high=High, epsilon=Epsilon, maxIt=MaximoIteracao, maxCad=10)
    elif(alghoritm == 'LRS'):
        xBest, fBest = LocalRandomSearch(low=Low, high=High, maxIt=MaximoIteracao, sigma=Sigma)
    elif(alghoritm == 'GRS'):
        xBest, fBest = GlobalRandomSearch(low=Low, high=High, maxIt=MaximoIteracao)
    else:
        xBest, fBest = SimulatedAnnealing(low=Low, high=High, maxIt=MaximoIteracao, temp=Temperatura, sigma=Sigma, alpha=Alpha)
    print(f"Melhor solução encontrada: \n x1, x2 = {xBest} \n valor da função = {fBest}")
    plotGraph(x1Cand=xBest[0], x2Cand=xBest[1])

RunAlgorithm()
