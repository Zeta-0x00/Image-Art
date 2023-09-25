#!/usr/lcal/bin/env python
# -*- coding: utf-8 -*-

#region Imports
import cv2
import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from math import ceil
from typing import Callable, Final
import random
import time
#endregion

# This code implements a genetic algorithm to solve the following problem:
#
#   * Given a grid of dimensions GRID_DIMENSION x GRID_DIMENSION, place
#     NUM_CHROMOSOMES queens on the grid such that no two queens can attack
#     each other (i.e., they cannot share the same row, column, or diagonal).
#
# This code is a modified version of the code found at

#region args
parser = argparse.ArgumentParser(description='Image processing script')
parser.add_argument('-i', '--input', required=True, help='Input image file path')
parser.add_argument('-o', '--output', required=True, help='Output image file name')
parser.add_argument('-f', '--fitness', type=int, default=1, choices=[1,2], help='Fitness function to use')
parser.add_argument('-t', '--time', action='store_true', help='Print the start time and end time with timestap')
parser.add_argument('-d', '--dimension', type=int, default=6, help='Grid dimension')
parser.add_argument('-n', '--num', type=int, default=90, help='Number of chromosomes')
parser.add_argument('-m', '--mutation', type=float, default=0.5, help='Mutation probability')
parser.add_argument('-c', '--crossover', type=float, default=0.3, help='Crossover chance')
parser.add_argument('-v', '--verbose', action='store_true', help='Print the parameters')
args: argparse.Namespace = parser.parse_args()
#endregion


#region Parámetros
GRID_DIMENSION: Final[int] = args.dimension
NUM_CHROMOSOMES: Final[int] = args.num
MUTATION_PROBABILITY: Final[float] = args.mutation
CROSSOVER_CHANCE: Final[float] = args.crossover
#endregion


#region fitness
# fitness calculating function
""" The code above does the following:
1. Calculates the mean values of each color channel of the source image within the boundary defined by the chromosome.
2. Calculates the distance between each color channel of the chromosome and the mean values of the source image.
3. Returns the inverse of the distance as a fitness score. """
def fitness1(chromosome: list[int], coord: tuple, source_img: Image)  -> float: #fitness v1

    # calculate mean values on each channel of rgb scheme
    means = source_img[coord[1]:coord[3], coord[0]:coord[2]].mean(axis=(0,1))

    # calculate 3d distance squared between chromosome and mean values
    score:float = 0.0
    for i in range(len(chromosome)):
        score += (chromosome[i] / means[i]) ** (chromosome[i] / means[i])
    
    # return fitness score
    return 1 / (1 + score)
def fitness2(chromosome: list[int], coord: tuple, source_img: Image)  -> float: #fitness v2

    # calculate mean values on each channel of rgb scheme
    means = source_img[coord[1]:coord[3], coord[0]:coord[2]].mean(axis=(0,1))

    # calculate 3d distance squared between chromosome and mean values
    score:float = 0.0
    for i in range(len(chromosome)):
        score += (chromosome[i] - means[i]) * (chromosome[i] - means[i])
    
    # return fitness score
    return 1 / (1 + score)

# fitness function selector
fitness: Callable[..., float] = fitness2 if args.fitness == 2 else fitness1
#endregion

#region selection
def selection(population: list, probability: list) -> list:
    """ The code above does the following:
    1. Calculate the sum of all probabilities
    2. Calculate the probability to be selected in next generation
    3. Calculate cumulative probability distribution
    4. Select the chromosomes according to cumulative probability distribution """
    # calculate the sum of all probabilities
    total = 0.0
    for i in range(NUM_CHROMOSOMES):
        total += probability[i]

    # calculate the probability to be selected in next generation
    for i in range(NUM_CHROMOSOMES):
        probability[i] /= total # normalize the probability

    # generate new array for population constructed from selected chromosomes
    new_population: list[None] = [None] * NUM_CHROMOSOMES

    # calculate cumulative probability distribution
    cumulative_probability: list[None] = [None] * NUM_CHROMOSOMES
    cumulative_probability[0] = probability[0]
    for i in range(1,NUM_CHROMOSOMES):
        cumulative_probability[i] = probability[i] + cumulative_probability[i-1]

    # select the chromosomes according to cumulative probability distribution
    for i in range(NUM_CHROMOSOMES):
        r: float = random.uniform(0,1)
        for j in range(NUM_CHROMOSOMES):
            if j == 0:
                if 0 <= r and r < cumulative_probability[j]:
                    new_population[i] = population[j].copy()
            else:
                if cumulative_probability[j-1] <= r and r < cumulative_probability[j]:
                    new_population[i] = population[j].copy()

    return new_population
#endregion

#region crossover
def crossover(population: list) -> list:
    """ The code above does the following:
    1. We choose parents for crossover (randomly, according to CROSSOVER_CHANCE)
    2. We choose a random cut point
    3. We apply one-cut crossover technique to chosen parents """
    parents:list = [] # list of parents
    for i in range(NUM_CHROMOSOMES):
        r: float = random.uniform(0,1) # generate random number between 0 and 1
        if r < CROSSOVER_CHANCE:
            parents.append(i) # add the chromosome to the list of parents
    for i in range(len(parents)):
        cut = random.randint(1,3) # generate random cut point
        for k in range(cut, len(population[parents[i]])):
            population[parents[i]][k] = population[parents[(i + 1)%(i + 1)]][k] # apply one-cut crossover technique to chosen parents

    return population
#endregion

#region mutation
def mutation(population: list) -> list:
    """ The code above does the following:
    1. The population is a list of integers, where each integer is a move in the game.
    2. We go through every chromosome in the population.
    3. We go through every move in the chromosome.
    4. We generate a random number r between 0 and 1.
    5. If r is less than MUTATION_PROBABILITY, we change the move to a random number between 0 and 256.
    6. We return the new population. """
    for chromosome in population:
        for move in range(len(chromosome)):
            r: float = random.uniform(0, 1) # generate random number between 0 and 1
            if r < MUTATION_PROBABILITY:
                chromosome[move] = random.randint(0,256) # change the move to a random number between 0 and 2568

    return population
#endregion

#region gets
def getChromosome(coord: tuple, source_img: Image)  -> list[list[int]]:
    """ The code above does the following:
    1. Generates a population of chromosomes (RGB color values)
    2. Calculates the fitness score for each chromosome
    3. Selects the best chromosome
    4. Crossover
    5. Mutation
    6. Repeat 2-5 until the best chromosome is good enough """
    # generate population of chromosomes
    population: list[list[int]] = [[random.randint(0,256), random.randint(0,256), random.randint(0,256)].copy() for i in range(NUM_CHROMOSOMES)] # generate random chromosomes
    score: list[None] = [None] * NUM_CHROMOSOMES # list of fitness scores for each chromosome
    best_score : float = 0.0 # best score of the best chromosome
    while(best_score < 0.1):
        for i in range(NUM_CHROMOSOMES):
            score[i] = fitness(chromosome=population[i], coord=coord, source_img=source_img) # calculate fitness score for each chromosome
            if score[i] > best_score:
                best_score = score[i] # update the best score
                best_chromosome: list[int] = population[i] # update the best chromosome
        
        population = selection(population=population, probability=score) # select the best chromosomes
        population = crossover(population=population) # run the crossover procedure
        population = mutation(population=population) # run the mutation procedure

    return best_chromosome
#endregion

#region main
def main() -> None:
    """ The code above does the following:
    1. Read the image and convert it to RGB
    2. Make the image divisible by squares of pixels with GRID_DIMENSION
    3. Create an empty image of the same size
    4. For each square, find the best chromosome and draw a circle with the appropriate color on the image """
    if args.verbose:
        print(f'GRID_DIMENSION: {GRID_DIMENSION}')
        print(f'NUM_CHROMOSOMES: {NUM_CHROMOSOMES}')
        print(f'MUTATION_PROBABILITY: {MUTATION_PROBABILITY}')
        print(f'CROSSOVER_CHANCE: {CROSSOVER_CHANCE}')
        print(f'fitness: {fitness.__name__}')
        print(f'input: {args.input}')
        print(f'output: {args.output}')
        print(f'time: {args.time}')
    start_time: float = time.time() # start timing
    if args.time:
        print(f'Start time: {time.ctime(start_time)}')
    image = cv2.imread(f'./img/{args.input}') # read the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert the image to RGB

    # add some pixels by reflecting the image to make it divisible by squares of pixels with GRID_DIMENSION
    height, width, _ = image.shape # get the height and width of the image
    height_offset: int = ceil(height / GRID_DIMENSION) * GRID_DIMENSION - height # calculate the number of pixels to add to the height
    width_offset: int = ceil(width / GRID_DIMENSION) * GRID_DIMENSION - width # calculate the number of pixels to add to the width
    image = cv2.copyMakeBorder(src=image, top=0, bottom=height_offset, left=0, right=width_offset, borderType=cv2.BORDER_REFLECT) # add the pixels
    height, width, _ = image.shape # get the height and width of the image

    im: Image = Image.new(mode='RGB', size=(width, height), color='black') # create an empty image of the same size
    draw: ImageDraw = ImageDraw.Draw(im=im) # create a drawing object

    for y in range(0, height, GRID_DIMENSION): 
        for x in range(0, width, GRID_DIMENSION):
            # find the best chromosome for square with coordinates (x,y) and (x + GRID_DIMENSION, y + GRID_DIMENSION)  
            solution: list[list[int]] = getChromosome(coord=(x, y, x + GRID_DIMENSION, y + GRID_DIMENSION), source_img=image)
            # draw that the most appropriate chromosome on the image with form of circle
            draw.ellipse((x, y, x + GRID_DIMENSION, y + GRID_DIMENSION), fill=(solution[0], solution[1], solution[2]))

    # end timing and print it to the console
    if args.time:
        print(f'End time: {time.ctime(time.time())}')
    
    print(f'Elapsed time: {time.time() - start_time} seconds')

    # show the generated image
    plt.imshow(im)
    plt.show()
    # save the image
    im.save(f'./img/{args.output}')   
#endregion


if __name__ == '__main__':
    main()