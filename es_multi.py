import gym
import copy
import torch
from tqdm import tqdm, trange
import torch.nn as nn
import numpy as np
from functools import partial

import wandb
wandb.init(project="Warehouse_Evolution")

torch.set_grad_enabled(False) # important af
device = "cuda" if torch.cuda.is_available() else "cpu"

def _init_weights(m):
    if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)

def createmodel(input_dim, output_dim):
    model =  nn.Sequential(
        nn.Linear(input_dim[0], 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, output_dim)
    )
    model.apply(_init_weights)
    model.to(device)
    return model

def mutate(agent):
    child_agent = copy.deepcopy(agent)
    mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    for param in child_agent.parameters():
        param += mutation_power * torch.randn(param.shape, device=device, dtype=param.dtype)
    return child_agent

def _evaluate_agent(agent, envname):
    # return a pyint
    env = gym.make(envname)
    observations = env.reset()
    total_reward = 0
    for _ in trange(300, desc="Env Stepping", leave=False):
        inp = torch.tensor(observations).to(device)
        output_probabilities = agent(inp).cpu().numpy()
        cdf = np.cumsum(output_probabilities, axis=1)
        rndselect = np.random.uniform(size=(cdf.shape[0],1))
        actions = np.sum(rndselect < cdf, axis=1)
        new_observations, rewards, done, _ = env.step(actions)
        total_reward += rewards.sum()
        observations = new_observations
        if(done):
            break

    return total_reward

def _evaluate_agent_mega(agent, envname, nrepeats):
    # return a pyint
    env = gym.make(envname)
    observations = env.reset(nrepeats)
    total_reward = 0
    for _ in trange(300, desc="Env Stepping", leave=False):
        inp = torch.tensor(observations).to(device)
        output_probabilities = agent(inp).cpu().numpy()
        cdf = np.cumsum(output_probabilities, axis=1)
        rndselect = np.random.uniform(size=(cdf.shape[0],1))
        actions = np.sum(rndselect < cdf, axis=1)
        new_observations, rewards, done, _ = env.step(actions)
        total_reward += rewards.sum()
        observations = new_observations
        if(done):
            break

    return total_reward

def evaluate(nrepeats, envname, agent):
    """
    Evaluate an agent (ONE MODEL) in envname nrepeats number of times
    """
    if "mega" in envname:
        agent_score = _evaluate_agent_mega(agent, envname, nrepeats) / nrepeats
        return agent_score
    else:
        agent_score = [_evaluate_agent(agent, envname) for _ in trange(nrepeats, desc="Repeat Evals", leave=False)]
        return sum(agent_score) / len(agent_score)

def simplegenetic(psi, phi, F, F2, N, T, C, G):
    SAVE_EVERY = 5
    """
    SAVE_EVERY: Save agent every time 
    psi: agent mutation function
    phi: single agent initialization function
    F: fitness function
    F: fitness function for elites (loaded with a different number of repeats)
    N: population size
    T: number of selected individuals (beam size)
    C: number of candidate elites per generation
    G: generations
    """
    print(f"Population Size: {N}")
    print(f"Top T: {T}")
    print(f"Candidate Elites: {C}")
    print(f"Generations: {G}")
    topT = [phi() for _ in range(T + 1)]
    population = [None for _ in range(N)]
    pscores = [None for _ in range(N)]
    elite = None
    for g in range(G):
        print(f"Generation {g + 1}/{G}")
        if g % SAVE_EVERY == 0:
            torch.save(topT[0].state_dict(), f"./saved_weights/{wandb.run.name}_gen_{g}.pt")
        for idx, k in enumerate(tqdm(np.random.randint(T + 1, size=(N,)), desc="Create next population")):
            population[idx] = psi(topT[k])
            pscores[idx] = F(population[idx])
        
        topTidx = np.argsort(-np.array(pscores))[:T]
        print("Top T Scores:")
        topTscores = [pscores[idx] for idx in topTidx]
        print(topTscores)
        print("")

        topT = [population[idx] for idx in topTidx]

        print("Calculating Elite...")
        if g == 0:
            elites = [population[idx] for idx in topTidx[:C]]
        else:
            elites = [population[idx] for idx in topTidx[:(C - 1)]] + [elite]
        elitescores = [F2(elitecandidate) for elitecandidate in elites]
        elite = elites[np.array(elitescores).argmax()]
        print(f"Elite Score: {max(elitescores)}")
        wandb.log({"TopT Avg Score": sum(topTscores) / len(topTscores), "Elite Score": max(elitescores)})

        topT.append(elite)

    return elite

def main():
    GENERATIONS = 100
    BEAM_SIZE = 10
    AGENTS = 20
    REPEATS = 3
    CANDIDATE_ELITES = 4
    ELITEREPEATS = 7
    # ENVNAME = "warehouse_simple_multi"
    # ENVNAME = "warehouse_simple_multi_gpu"
    ENVNAME = "warehouse_simple_mega_gpu"

    _hold = gym.make(ENVNAME)
    _createmodel = partial(createmodel, _hold.observation_space.shape, _hold.action_space.n)
    del _hold
    _evaluate = partial(evaluate, REPEATS, ENVNAME)
    _evaluateelite = partial(evaluate, ELITEREPEATS, ENVNAME)
    
    simplegenetic(mutate, _createmodel, _evaluate, _evaluateelite, AGENTS, BEAM_SIZE, CANDIDATE_ELITES, GENERATIONS)

if __name__ == "__main__":
    main()
