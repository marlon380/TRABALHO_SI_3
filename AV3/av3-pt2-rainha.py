import random
import time

#Definição de variáveis
quantidadeIndividuos=100
maxGeracoes=1000

def createPopulation():
    return [[random.randint(0, 7) for _ in range(8)] for _ in range(quantidadeIndividuos)]

def calculateFitness(individual):
    conflicts = 0
    for i in range(len(individual)):
        for j in range(i + 1, len(individual)):
            if individual[i] == individual[j] or abs(individual[i] - individual[j]) == abs(i - j):
                conflicts += 1
    return 28 - conflicts

def calculateProbabilities(fitnesses):
    totalFitness = sum(fitnesses)
    return [fitness / totalFitness for fitness in fitnesses]

def generateMask(nd):
    return [random.randint(0, 1) for _ in range(nd)]

def crossoverSinglePoint(parent1, parent2):
    cutPoint = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cutPoint] + parent2[cutPoint:]
    child2 = parent2[:cutPoint] + parent1[cutPoint:]
    return child1, child2

def crossoverTwoPoints(parent1, parent2):
    cutPoint1 = random.randint(1, len(parent1) - 2)
    cutPoint2 = random.randint(cutPoint1 + 1, len(parent1) - 1)
    child1 = parent1[:cutPoint1] + parent2[cutPoint1:cutPoint2] + parent1[cutPoint2:]
    child2 = parent2[:cutPoint1] + parent1[cutPoint1:cutPoint2] + parent2[cutPoint2:]
    return child1, child2

def crossover(parents):
    pc = random.uniform(0.85, 0.95)
    if random.random() <= pc:
        return crossoverSinglePoint(parents[0], parents[1]) if random.choice([True, False]) else crossoverTwoPoints(parents[0], parents[1])
    else:
        return parents[0], parents[1]

def mutation(individual):
    if random.random() <= 0.01:
        gene = random.randint(0, 7)
        newValue = random.randint(0, 7)
        individual[gene] = newValue

def geneticAlgorithm():
    population = createPopulation()
    t = 0
    foundSolutions = set()
    startTime = time.time()
    while len(foundSolutions) < 92:
        fitnesses = [calculateFitness(individual) for individual in population]
        probabilities = calculateProbabilities(fitnesses)
        newPopulation = []
        for _ in range(quantidadeIndividuos // 2):
            parents = []
            for _ in range(2):
                r = random.random()
                cumulativeProb = 0
                i = 0
                while cumulativeProb < r:
                    cumulativeProb += probabilities[i]
                    i += 1
                parents.append(population[i - 1])

            child1, child2 = crossover(parents)
            mutation(child1)
            mutation(child2)
            newPopulation.extend([child1, child2])

        population = newPopulation
        t += 1

        for individual in population:
            foundSolutions.add(tuple(individual))

        if t == maxGeracoes:
            break

    totalTime = time.time() - startTime
    print('solutions:', foundSolutions)
    return len(foundSolutions), totalTime

numSolutions, elapsedTime = geneticAlgorithm()

print(f"Número de soluções diferentes encontradas: {numSolutions}")
print(f"Tempo computacional: {elapsedTime} segundos")
