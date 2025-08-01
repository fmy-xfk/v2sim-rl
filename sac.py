from copy import deepcopy
import itertools
from collections import deque
from typing import Iterable
import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
import core
from env import *
from utils import EpochLogger, time2str

OneBuffer = tuple[
    ObsRes,     # obs
    np.ndarray, # act
    float,      # rew
    ObsRes,     # next_obs
    bool,       # done
]

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, size):
        self._q:deque[OneBuffer] = deque(maxlen=size)

    def store(self, obs:ObsRes, act:np.ndarray, rew:float, next_obs:ObsRes, done:bool):
        self._q.append((obs, act, rew, next_obs, done))
    
    @property
    def size(self):
        return len(self._q)

    def __obs_cat(self, obs:Iterable[ObsRes]):
        nets = []
        prices = []
        for o in obs:
            nets.append(o["net"])
            prices.append(o["price"])
        return {
            "net": torch.as_tensor(np.stack(nets, axis=0), dtype=torch.float32),
            "price": torch.as_tensor(np.stack(prices, axis=0), dtype=torch.float32),
        }
    
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        sampled = [self._q[i] for i in idxs]
        obs, act, rew, obs2, done = zip(*sampled)
        batch = dict(
            obs=self.__obs_cat(obs),
            obs2=self.__obs_cat(obs2),
            act=torch.as_tensor(np.stack(act, axis=0, dtype=np.float32), dtype=torch.float32),
            rew=torch.as_tensor(np.stack(rew, axis=0, dtype=np.float32), dtype=torch.float32),
            done=torch.as_tensor(np.stack(done, axis=0, dtype=np.float32), dtype=torch.float32),
        )
        return batch

def create_env(obj):
    #return gym.make("Pendulum-v1")
    mcase = str(Path(__file__).parent / "cases" / G_CASE)
    return V2SimEnv(mcase, G_ET, G_TS, G_RLS, res_path=obj)

def test_baseline(out_dir, num_test_episodes:int, max_ep_len:int):
    test_env = create_env("./test_temp")
    assert isinstance(test_env.action_space, gym.spaces.Box)
    action = np.ones(test_env.action_space.shape, dtype=np.float32)
    results = []
    for j in range(num_test_episodes):
        o, _ = test_env.reset()
        d, ep_ret, ep_len = False, 0, 0
        this_st = time.time()
        last_print_time = -1
        while not(d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            o, r, d, _, _ = test_env.step(action)
            cur_time = time.time()
            if ep_len > 0 and cur_time - last_print_time > 1:
                dur = cur_time - this_st
                rem = dur / ep_len * (max_ep_len - ep_len)
                print(f"Baseline, TestEp {j}: {ep_len}/{max_ep_len}   ", end = "\r")
                last_print_time = cur_time
            ep_ret += r
            ep_len += 1
        results.append((ep_ret, ep_len))
    test_env.close()
    print("\nBaseline test results:")
    for i, (ret, length) in enumerate(results):
        print(f"Episode {i+1}: Return = {ret:.2f}, Length = {length}")
    print(f"Average Return = {np.mean([r for r, _ in results]):.2f}")
    print(f"Max Return = {np.max([r for r, _ in results]):.2f}")
    print(f"Min Return = {np.min([r for r, _ in results]):.2f}")

    with open(out_dir + "/baseline_results.csv", "w") as f:
        f.write("epret,eplen\n")
        for ret, length in results:
            f.write(f"{ret:.2f},{length}\n")


def sac(actor_critic:str, ac_kwargs=dict(),
        seed=0, 
        steps_per_epoch=1000,   # Number of steps of interaction (state-action pairs) for the agent and the environment in each epoch
        epochs=100,             # Number of epochs to run
        replay_size=int(1e6),   # Size of the replay buffer
        gamma=0.99,             # Discount factor
        polyak=0.995,           # Interpolation factor in polyak averaging for target networks
        lr=1e-3,                # Learning rate for both actor and critic
        alpha=0.2,              # Entropy regularization coefficient
        adaptive_alpha=True,    # Whether to use adaptive entropy coefficient
        batch_size=50,          # Number of episodes to optimize at the same time
        start_steps=1000,       # Number of steps for uniform-random action selection, before running real policy.
        update_after=500,       # Number of env interactions to collect before starting to do gradient descent updates.
        update_every=50,        # Number of env interactions that should elapse between gradient descent updates.
        num_test_episodes=1,    # Number of episodes to test the deterministic policy at the end of each epoch.
        max_ep_len=1000,        # Maximum length of trajectory / episode / rollout
        logger_kwargs=dict()):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = create_env("./train_temp")
    obs_dim = env.observation_space.shape

    assert isinstance(env.action_space, gym.spaces.Box)
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    if actor_critic == "MLP":
        ac = core.ActorCritic(env.observation_space, env.action_space, **ac_kwargs, 
            actor=core.ActorMLP, q_function=core.QFunctionMLP)
    elif actor_critic == "LSTM":
        ac = core.ActorCritic(env.observation_space, env.action_space, **ac_kwargs,
            actor=core.ActorLSTM, q_function=core.QFunctionLSTM)
    else:
        raise ValueError(f"Unknown actor-critic type: {actor_critic}")
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q, q1, q2

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data) -> tuple[torch.Tensor, torch.Tensor]:
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi:torch.Tensor = (alpha * logp_pi - q_pi).mean()

        return loss_pi, logp_pi

    if adaptive_alpha:
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        target_entropy = -act_dim
        # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
        log_alpha = torch.zeros(1, requires_grad=True)
        alpha = log_alpha.exp().item()
        alpha_optimizer = Adam([log_alpha], lr=lr)
    
    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q1, q2 = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(
            LossQ=loss_q.item(),
            Q1Vals=q1.detach().numpy(),
            Q2Vals=q2.detach().numpy(),
        )

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, logp_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(
            LossPi=loss_pi.item(), 
            LogPi=logp_pi.detach().numpy()
        )

        # Update alpha
        if adaptive_alpha:
            alpha_optimizer.zero_grad()
            alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean()
            alpha_loss.backward()
            alpha_optimizer.step()
            alpha = log_alpha.exp().item()

            # Record things
            logger.store(LossAlpha=alpha_loss.item(), Alpha=alpha)
        
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(o, deterministic)

    def test_agent(num_test_episodes:int, max_ep_len:int):
        test_env = create_env("./test_temp")
        print("\n")
        for j in range(num_test_episodes):
            o, _ = test_env.reset()
            d, ep_ret, ep_len = False, 0, 0
            this_st = time.time()
            last_print_time = -1
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                action = ac.act(o, True)
                o, r, d, _, _ = test_env.step(action)
                cur_time = time.time()
                if ep_len > 0 and cur_time - last_print_time > 1:
                    dur = cur_time - this_st
                    rem = dur / ep_len * (max_ep_len - ep_len)
                    print(f"Epoch {test_count+1}, TestEp {j}: {ep_len}/{max_ep_len}, Used: {time2str(dur)}, EST: {time2str(rem)}         ", end = "\r")
                    last_print_time = cur_time
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        test_env.close()
    
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, _ = env.reset()
    ep_ret, ep_len = 0, 0
    
    test_count = 0
    st_time = time.time()
    last_print_time = -1
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        cur_time = time.time()
        if cur_time - last_print_time > 1 and t > 0:
            dur = cur_time - st_time
            rem = (total_steps - t) * (dur / t)
            print(f"Epoch: {test_count+1}, Step: {t}, Time: {env.reset_count}-{env.ctime}, Used: {time2str(dur)}, EST: {time2str(rem)}   ", end="\r")
            last_print_time = cur_time
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _,  _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _ = env.reset()
            ep_ret, ep_len = 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model: ATTN: NV2SimEnv cannot be saved!
            # if (epoch % save_freq == 0) or (epoch == epochs):
            #    logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent(num_test_episodes, max_ep_len)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Alpha', alpha)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            test_count += 1

if __name__ == '__main__':
    from feasytools import ArgChecker
    args = ArgChecker()

    do_baseline = args.pop_bool("baseline")

    # Case
    G_CASE = args.pop_str("d", "drl_12nodes")

    # End time
    G_ET = args.pop_int("e", 115200 + 4 * 3600)  # Default is 4 hours, can be adjusted

    # Traffic step
    G_TS = args.pop_int("ts", 15)

    # Reinforcement learning step: How many traffic steps per RL step
    G_RLS = args.pop_int("rls", 40)

    # Road capacity
    G_RC = args.pop_float("rc", 100)

    # Hidden size
    hidden_size0 = args.pop_int("hid", 256)

    # Hidden layer
    hidden_layer = args.pop_int("l", 2)

    # Seed
    seed = args.pop_int("s", 0)
    
    # Epochs
    epochs = args.pop_int("epochs", 50)

    # Model type
    model = args.pop_str("m", "MLP")

    assert model in ("MLP", "LSTM"), "Model type must be either 'MLP' or 'LSTM'."


    from utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(f"{G_CASE}_{model}_ep{epochs}", seed)
    
    torch.set_num_threads(torch.get_num_threads())
    
    if do_baseline:
        test_baseline(logger_kwargs["output_dir"], num_test_episodes=1, max_ep_len=1000)
    else:
        sac(
            actor_critic=model,
            ac_kwargs=dict(hidden_sizes=[hidden_size0] * hidden_layer), 
            gamma=args.pop_float("gamma", 0.99),
            seed=seed,
            epochs=epochs,
            logger_kwargs=logger_kwargs
        )
