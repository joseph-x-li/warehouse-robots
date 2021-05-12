import gym
import torch
import torch.nn as nn
import subprocess
import numpy as np

torch.set_grad_enabled(False) # important af
device = "cuda" if torch.cuda.is_available() else "cpu"

def createmodel(input_dim, output_dim):
    model =  nn.Sequential(
        nn.Linear(input_dim[0], 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, output_dim)
    )
    model.to(device)
    return model

def record(envname, run_name, frames, region):
    """
    envname: bruh
    run_name: name of run from which to load state_dict
    frames: Number of frames(steps) to record (300 is good)
    region: (r1, r2, c1, c2) Area to record [r1, r2); [c1, c2)
    """
    r1, r2, c1, c2 = region
    env = gym.make(envname)
    observations = env.reset()
    agent = createmodel(env.observation_space.shape, env.action_space.n)
    agent.load_state_dict(torch.load(f"saved_weights/{run_name}.pt"))

    frame_accum = []

    for _ in trange(frames, desc="Env Stepping..."):
        _, _, field = env.state.copy_gpu_data()
        frame_accum.append(field[r1:r2, c1:c2])
        inp = torch.tensor(observations).to(device)
        output_probabilities = agent(inp).cpu().numpy()
        cdf = np.cumsum(output_probabilities, axis=1)
        rndselect = np.random.uniform(size=(cdf.shape[0],1))
        actions = np.sum(rndselect < cdf, axis=1)
        observations, _, _, _ = env.step(actions)
    
    frame_accum = np.array(frame_accum)
    print(frame_accum.shape)
    np.save("_hold.npy", frame_accum)

    subprocess.run(["write_mp4.py", "--runname", run_name])  

if __name__ == "__main__":
    record(
        "warehouse_simple_multi_gpu", 
        "faithful-surf-5_gen_0", 
        300,
        (0, 500, 0, 500)
    )