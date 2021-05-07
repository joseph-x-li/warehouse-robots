import gym
import copy
import torch
import torch.nn as nn
torch.set_grad_enabled(False) # important af

def init_weights(m):
    import pdb; pdb.set_trace()
    if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)

def createmodel(input_dim, output_dim):
    model =  nn.Sequential(
        nn.Linear(input_dim[0], 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, output_dim)
    )
    model.apply(init_weights)


def mutate(agent):
    child_agent = copy.deepcopy(agent)
    mutation_power = 0.02 #hyper-parameter, set from https:/    /arxiv.org/pdf/1712.06567.pdf
    for param in child_agent.parameters():
        param += mutation_power * np.random.randn(*(param.shape))
    return child_agent

def create_agents(nagents, input_dim, output_dim):
    return [createmodel(input_dim, output_dim) for _ in range(nagents)]

def _evaluate_agent(agent, envname):
    # return a pyint
    env = gym.make(envname)
    observation = env.reset()
    rewards = []
    for _ in range(300):
        inp = torch.tensor(observation)
        output_probabilities = agent(inp).detach().numpy()[0]
        action = np.random.choice(range(game_actions), 1, p=output_probabilities).item()
        new_observation, reward, done, _ = env.step(action)
        rewards.append(reward)
        observation = new_observation
        if(done):
            break

    return sum(rewards)

def evaluate(agents, nrepeats, envname):
    scores = []
    for agent in agents:
        if "gpu" in envname:
            pass # do fancy ass shit here
        else:
            agent_score = [_evaluate_agent(agent, envname) for _ in range(nrepeats)]
            scores.append(sum(agent_score) / len(agent_score))
    return scores

def main():
    AGENTS = 200
    REPEATS = 16 # number of times to repeat evaluation of an agent
    GENERATIONS = 100
    BEAM_SIZE = 20
    envname = "warehouse_simple_multi"
    _hold = gym.make(envname)
    agents = create_agents(AGENTS, env.observation_space.shape, env.action_space.n)
    del env
    for _ in range(GENERATIONS):
        rewards = evaluate(agents, REPEATS)
        topk_args = np.argsort(rewards)[::-1][:BEAM_SIZE]
        children = 
        agents = children


if __name__ == "__main__":
    main()
