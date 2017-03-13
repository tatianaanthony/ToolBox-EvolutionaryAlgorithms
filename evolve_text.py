"""
Evolutionary algorithm, attempts to evolve a given message string.

Uses the DEAP (Distributed Evolutionary Algorithms in Python) framework,
http://deap.readthedocs.org

Usage:
    python evolve_text.py [goal_message]

Full instructions are at:
https://sites.google.com/site/sd15spring/home/project-toolbox/evolutionary-algorithms
"""

import random
import string

import numpy    # Used for statistics
from deap import algorithms
from deap import base
from deap import tools


# -----------------------------------------------------------------------------
#  Global variables
# -----------------------------------------------------------------------------

# Allowable characters include all uppercase letters and space
# You can change these, just be consistent (e.g. in mutate operator)
VALID_CHARS = string.ascii_uppercase + " "
num_levenshtein = 0
memo = {}

# Control whether all Messages are printed as they are evaluated
VERBOSE = True


# -----------------------------------------------------------------------------
# Message object to use in evolutionary algorithm
# -----------------------------------------------------------------------------

class FitnessMinimizeSingle(base.Fitness):
    """
    Class representing the fitness of a given individual, with a single
    objective that we want to minimize (weight = -1)
    """
    weights = (-1.0, )


class Message(list):
    """
    Representation of an individual Message within the population to be evolved

    We represent the Message as a list of characters (mutable) so it can
    be more easily manipulated by the genetic operators.
    """
    def __init__(self, starting_string=None, min_length=4, max_length=30):
        """
        Create a new Message individual.

        If starting_string is given, initialize the Message with the
        provided string message. Otherwise, initialize to a random string
        message with length between min_length and max_length.
        """
        # Want to minimize a single objective: distance from the goal message
        self.fitness = FitnessMinimizeSingle()

        # Populate Message using starting_string, if given
        if starting_string:
            self.extend(list(starting_string))

        # Otherwise, select an initial length between min and max
        # and populate Message with that many random characters
        else:
            initial_length = random.randint(min_length, max_length)
            for i in range(initial_length):
                self.append(random.choice(VALID_CHARS))

    def __repr__(self):
        """Return a string representation of the Message"""
        # Note: __repr__ (if it exists) is called by __str__. It should provide
        #       the most unambiguous representation of the object possible, and
        #       ideally eval(repr(obj)) == obj
        # See also: http://stackoverflow.com/questions/1436703
        template = '{cls}({val!r})'
        return template.format(cls=self.__class__.__name__,     # "Message"
                               val=self.get_text())

    def get_text(self):
        """Return Message as string (rather than actual list of characters)"""
        return "".join(self)


# -----------------------------------------------------------------------------
# Genetic operators
# -----------------------------------------------------------------------------

def levenshtein_distance(stringS,stringT):
    """
    >>> levenshtein_distance("kitten", len("kitten"), "sitting", len("sitting"))
    3
    """
    # print("evaluating levenshtein of", stringS, "and", stringT, "Iteration", num_levenshtein)
    global num_levenshtein
    num_levenshtein +=1
    cost = 0;
    # Checks for empty strings
    if len(stringS) ==0:
        return len(stringT)
    if len(stringT) == 0:
        return len(stringS)
## If it isn't empty:
    if stringS[-1] == stringT[-1]:
        cost =0
    else:
        cost = 1
    key1 = (stringS[:-1], stringT)
    if not key1 in memo:
        memo[key1] = levenshtein_distance(*key1)
    key2 = (stringS, stringT[:-1])
    if not key2 in memo:
        memo[key2] = levenshtein_distance(*key2)
    key3 = (stringS[:-1], stringT[:-1])
    if not key3 in memo:
        memo[key3] = levenshtein_distance(*key3)
    return min(memo[key1] + 1,
               memo[key2]+1,
               memo[key3]+cost)
#  Implement levenshtein_distance function (see Day 9 in-class exercises)
# HINT: Now would be a great time to implement memoization if you haven't

def evaluate_text(message, goal_text, verbose=VERBOSE):
    """
    Given a Message and a goal_text string, return the Levenshtein distance
    between the Message and the goal_text as a length 1 tuple.
    If verbose is True, print each Message as it is evaluated.
    """
    distance = levenshtein_distance(goal_text,message.get_text())
    if verbose:
        print("{msg!s}\t[Distance: {dst!s}]".format(msg=message, dst=distance))
    return (distance, )     # Length 1 tuple, required by DEAP


def mutate_text(message, prob_ins=0.05, prob_del=0.05, prob_sub=0.05):
    """
    Given a Message and independent probabilities for each mutation type,
    return a length 1 tuple containing the mutated Message.

    Possible mutations are:
        Insertion:      Insert a random (legal) character somewhere into
                        the Message
        Deletion:       Delete one of the characters from the Message
        Substitution:   Replace one character of the Message with a random
                        (legal) character
    """

    if random.random() < prob_ins:
        pos_ins = random.randint(0,len(message))
        print(pos_ins)
        # part1 = message[:pos_ins]
        # ins_message = []
        # ins_message.extend(part1)
        # ins_message.extend(random.choice(VALID_CHARS))
        # ins_message.extend(message[pos_ins:])
        message.insert(pos_ins,random.choice(VALID_CHARS))
        # print(message)
    if random.random()<prob_del:
        pos_del = random.randint(0,len(message)-1)
        message.pop(pos_del)
        # print(pos_del, message)
    if random.random()<prob_sub:
        pos_sub = random.randint(0,len(message)-1)
        message[pos_sub] = random.choice(VALID_CHARS)
        # print(pos_sub,message)

    return (message, )   # Length 1 tuple, required by DEAP

def crossover_two(String1,String2):
    """
    Takes 2 strings, makes a crossover of the two of them.
    Note:  I did what the pseudocode says, but there's an error that 'str' object has no attribute 'fitness.'
    """
    max_cross_len = max(len(String1),len(String2))
    start_pos = random.randint(0,max_cross_len-1)
    end_pos = random.randint(start_pos,max_cross_len-1)
    string1_start = "".join(String1[:start_pos])
    string1_cross = "".join(String1[start_pos:end_pos])
    string1_end = "".join(String1[end_pos:])
    string2_start = "".join(String2[:start_pos])
    string2_cross = "".join(String2[start_pos:end_pos])
    string2_end = "".join(String2[end_pos:])
    new_string1 = "".join((string1_start,string2_cross,string1_end))
    print(new_string1)
    new_string2 = "".join((string2_start,string1_cross,string2_end))
    return(new_string1,new_string2)

# -----------------------------------------------------------------------------
# DEAP Toolbox and Algorithm setup
# -----------------------------------------------------------------------------

def get_toolbox(text):
    """Return DEAP Toolbox configured to evolve given 'text' string"""

    # The DEAP Toolbox allows you to register aliases for functions,
    # which can then be called as "toolbox.function"
    toolbox = base.Toolbox()

    # Creating population to be evolved
    toolbox.register("individual", Message)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("evaluate", evaluate_text, goal_text=text)
    toolbox.register("mate", tools.cxTwoPoints)
    toolbox.register("mutate", mutate_text)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # NOTE: You can also pass function arguments as you define aliases, e.g.
    #   toolbox.register("individual", Message, max_length=200)
    #   toolbox.register("mutate", mutate_text, prob_sub=0.18)

    return toolbox


def evolve_string(text):
    """Use evolutionary algorithm (EA) to evolve 'text' string"""

    # Set random number generator initial seed so that results are repeatable.
    # See: https://docs.python.org/2/library/random.html#random.seed
    #      and http://xkcd.com/221
    random.seed(4)

    # Get configured toolbox and create a population of random Messages
    toolbox = get_toolbox(text)
    pop = toolbox.population(n=500)

    # Collect statistics as the EA runs
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Run simple EA
    # (See: http://deap.gel.ulaval.ca/doc/dev/api/algo.html for details)
    pop, log = algorithms.eaSimple(pop,
                                   toolbox,
                                   cxpb=0.5,    # Prob. of crossover (mating)
                                   mutpb=0.2,   # Probability of mutation
                                   ngen=500,    # Num. of generations to run
                                   stats=stats)

    return pop, log


# -----------------------------------------------------------------------------
# Run if called from the command line
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Get goal message from command line (optional)
    # import sys
    # if len(sys.argv) == 1:
    #     # Default goal of the evolutionary algorithm if not specified.
    #     # Pretty much the opposite of http://xkcd.com/534
    #     goal = "SKYNET IS NOW ONLINE"
    # else:
    #     goal = " ".join(sys.argv[1:])
    # print(goal)
    goal = "ALL IS NOT LOST"
    # Verify that specified goal contains only known valid characters
    # (otherwise we'll never be able to evolve that string)
    for char in goal:
        if char not in VALID_CHARS:
            msg = "Given text {goal!r} contains illegal character {char!r}.\n"
            msg += "Valid set: {val!r}\n"
            raise ValueError(msg.format(goal=goal, char=char, val=VALID_CHARS))
    
    # Run evolutionary algorithm
    pop, log = evolve_string(goal)
    # mutate_text(["A","B","C","D","E"],prob_ins=1,prob_del=1,prob_sub=1)