'''
Environment for the 12-node V2Sim case
'''
from itertools import chain
from multiprocessing.connection import PipeConnection
from pathlib import Path
import shutil
import time, random
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
import libsumo, os
import v2sim
import multiprocessing as mp

ObsRes = Dict[str, np.ndarray]

class _drlenv:
    def __create_inst(self):
        pp = Path(self._res_path) / Path(self._case_path).name
        shutil.rmtree(str(pp), ignore_errors=True)
        self._inst = v2sim.V2SimInstance(
            cfgdir = self._case_path,
            outdir = self._res_path, 
            traffic_step = self._traffic_step,
            start_time = 0, # Useless, will be covered by case initial state
            end_time = self._end_time,
            log = "",
            silent = True,
            seed = 0, # random.randint(0, 1000000),
            initial_state = self._case_state
        )

        self._evlog:dict[str,tuple[float,int]] = {}
        self._trip_speed:list[float] = []

        def arrive_listener(simT:int, veh:v2sim.EV, _0):
            nonlocal self
            if veh.ID not in self._evlog: return
            dist = veh.odometer - self._evlog[veh.ID][0]
            t = simT - self._evlog[veh.ID][1]
            if t > 0 and dist > 0:
                self._trip_speed.append(dist / t)
        
        self._inst.trips_logger.add_arrive_listener(arrive_listener)
        self._inst.trips_logger.add_arrive_fcs_listener(arrive_listener)

        def depart_listener(simT:int, veh:v2sim.EV, _0, _1 = None, _2 = None):
            nonlocal self
            self._evlog[veh.ID] = (0, simT)
        
        self._inst.trips_logger.add_depart_listener(depart_listener)
        self._inst.trips_logger.add_depart_cs_listener(depart_listener)

        if self._SoC_change:
            # Randomize the initial state of SoC
            for ev in self._inst.vehicles.values():
                ev._elec *= random.uniform(0.95, 1.05)
                ev._elec = min(ev._elec, ev._bcap)
                ev._elec = max(ev._elec, 0.0)
        self._cs_cnt = len(self._inst.fcs)
        self._inst.start()
        self._start_time = self._inst.ctime
        assert self._start_time < self._end_time, f"Start time {self._start_time} must be less than end time {self._end_time}."
        self._inst.step()
        self._enames = [e for e in self._inst.edge_names if not e.startswith("CS")]
        self._t = self._inst.ctime
        if self._inst.pdn:
            self._vmax = 0
            self._vmin = 1e6
            for b in self._inst.pdn.Grid.Buses:
                if b.V is not None:
                    self._vmax = max(self._vmax, b.V)
                    self._vmin = min(self._vmin, b.V)
            self._buses = list(self._inst.pdn.Grid.Buses)
        else:
            self._buses = []
        self._state = []
        self.__add_state()
    
    def __init__(self, 
            case_path:str,
            end_time:int,
            traffic_step:int = 15,
            rl_step:int = 4,
            res_path:str = "",
            road_cap:float = 500.0,
            SoC_change:bool = False
        ):
        self._SoC_change = SoC_change
        self.road_cap = road_cap
        self._case_path = case_path
        self._case_state = os.path.join(self._case_path, "saved_state")
        self._res_path = res_path
        self._end_time = end_time
        self._traffic_step = traffic_step
        self._rl_step = rl_step
        Path(self._res_path).mkdir(parents=True, exist_ok=True)
        self.__create_inst()
        

    def __cp_gini(self) -> float:
        '''Gini coefficient of the charge prices in FCSs'''
        t = self._inst.ctime
        prices = np.array(sorted(c.pbuy(t) for c in self._inst.fcs))
        if len(prices) == 0:
            return 0.0
        n = prices.shape[0]
        index = np.arange(1, n + 1)
        ret = (2 * np.sum(index * prices) / np.sum(prices) - (n + 1)) / n
        if ret < 0 or ret > 1:
            raise ValueError(f"Invalid Gini coefficient {ret} at time {t}. Prices: {prices}")
        return ret
    
    def __nv_stddev(self) -> float:
        '''Standard deviation of the number of vehicles in FCSs'''
        veh_counts = np.fromiter((c.veh_count() for c in self._inst.fcs), dtype=np.float32)
        if len(veh_counts) == 0:
            return 0.0
        return (np.max(veh_counts) - np.min(veh_counts)).item()
    
    def __pc_stddev(self) -> float:
        '''Standard deviation of the power consumption in FCSs'''
        pc_MW = np.fromiter((c.Pc_MW for c in self._inst.fcs), dtype=np.float32)
        if len(pc_MW) == 0:
            return 0.0
        return (np.max(pc_MW) - np.min(pc_MW)).item()

    def _bus_overlim(self):
        s = 0.0
        for b in self._buses:
            assert b.V is not None
            if b.V > self._vmax:
                s += b.V - self._vmax
            elif b.V < self._vmin:
                s += self._vmin - b.V
        return s
    
    @property
    def road_vcnt(self):
        return (libsumo.edge.getLastStepVehicleNumber(e) for e in self._enames)
    
    @property
    def road_density(self):
        return (libsumo.edge.getLastStepVehicleNumber(e)/self.road_cap for e in self._enames)

    @property
    def road_avg_soc(self):
        for e in self._enames:
            cnt = 0
            sum_soc = 0
            for vid in libsumo.edge.getLastStepVehicleIDs(e):
                v = self._inst.vehicles[vid]
                sum_soc += v.SOC
                cnt += 1
            if cnt == 0: 
                yield 0
            else:
                yield sum_soc / cnt
    
    @property
    def fcs_vcnt(self):
        return (c.veh_count() for c in self._inst.fcs)
    
    @property
    def fcs_usage(self):
        return (c.veh_count()/c.slots for c in self._inst.fcs)
    
    @property
    def fcs_vcnt_wait(self):
        return (c.wait_count() for c in self._inst.fcs)
    
    @property
    def fcs_wait_rate(self):
        return (c.wait_count()/c.slots for c in self._inst.fcs)
    
    @property
    def fcs_avg_soc(self):
        for c in self._inst.fcs:
            cnt = 0
            sum_soc = 0
            for vid in c.vehicles():
                v = self._inst.vehicles[vid]
                sum_soc += v.SOC
                cnt += 1
            if cnt == 0: 
                yield 0
            else: 
                yield sum_soc / cnt

    @property
    def fcs_price(self):
        return (c.pbuy(self._inst.ctime) for c in self._inst.fcs)
    
    @property
    def fcs_load(self):
        return (c.Pc_MW for c in self._inst.fcs)
    
    def _get_obs(self):
        while len(self._state) < self._rl_step:
            self.__add_state()
        ret = {
            "net": np.stack(self._state), 
            "price": np.fromiter(chain([self.ctime01],self.fcs_price), dtype=np.float32)
        }
        return ret
    
    def _get_info(self):
        return {}
    
    def reset(self):
        self.close()
        self.__create_inst()
        self._t = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def close(self):
        if self._inst.is_working:
            self._inst.stop()
            time.sleep(0.2) # Wait for the instance to stop. Never remove this line!
    
    @property
    def ctime01(self) -> float:
        return (self._inst.ctime - self._start_time) / (self._end_time - self._start_time)
    
    def __add_state(self):
        state = np.fromiter(chain(
            [self.ctime01],
            self.road_density,
            self.road_avg_soc,
            self.fcs_usage,
            self.fcs_avg_soc,
            self.fcs_load,
        ), dtype=np.float32)
        self._state.append(state)
    
    def step(self, action:np.ndarray):
        assert action.shape == (self._cs_cnt, )
        for i, v in enumerate(action):
            assert v >= 0.0 and v <= 5.0, f"Invalid action value {v} at index {i}. Must be in [0.0, 5.0]. {action}"
            self._inst.fcs[i].pbuy.setOverride(v)
        time_to = self._t + self._traffic_step * self._rl_step

        self._state.clear()
        while self._t < time_to:
            try:
                self._t = self._inst.step()
            except Exception as e:
                print("\nV2SimEnv Error @", self._t)
                raise e
            self.__add_state()
        
        if self._t >= self._inst.etime:
            terminated = True
        else:
            terminated = False
        truncated = False

        #avg_ts = sum(self._trip_speed) / max(1, len(self._trip_speed))

        reward = - self.__nv_stddev() - self.__pc_stddev() * 10
        #print(self._bus_overlim(), self.__cp_gini(), self.__nv_stddev(), (avg_ts - 12) * 20, reward)

        self._trip_speed.clear()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def _mainloop(self, q:PipeConnection):
        while True:
            try:
                op, data = q.recv()
            except EOFError:
                break
            if op == 's':
                q.send(self.step(data))
            elif op == 'r':
                q.send(self.reset())
            elif op == 'o':
                q.send(self._get_obs())
            elif op == 'ct':
                q.send(self._inst.ctime)
            elif op == 'cf':
                q.send(self._cs_cnt)
            elif op == 'ce':
                q.send(len(self._enames))
            elif op == 'q':
                self.close()
                q.send('__done__')
                break
            else:
                raise ValueError(f"Unknown operation {op}")
    
    @staticmethod
    def __exit(q:PipeConnection):
        import traceback
        traceback.print_exc()
        try:
            q.send('__error__')
        except Exception as e:
            pass

    @staticmethod
    def _worker(q:PipeConnection, *args, **kwargs):
        try:
            env = _drlenv(*args, **kwargs)
        except Exception as e:
            print("\n********************** V2SimEnv Initialization Error **********************")
            _drlenv.__exit(q)
            print("***************************************************************************\n")
            return
        try:
            env._mainloop(q)
        except Exception as e:
            print("\n************************ V2SimEnv Main Loop Error ************************")
            _drlenv.__exit(q)
            print("**************************************************************************\n")

class V2SimEnv(gym.Env):
    def __fetch(self, op:str, par) -> Any:
        self._par.send((op,par))
        self._par.poll()
        ret = self._par.recv()
        if isinstance(ret, str) and ret == '__error__':
            raise RuntimeError("V2SimEnv encountered an error in the worker process. See the details above.")
        return ret
    
    def __init__(self, 
            case_path:str,
            end_time:int,
            traffic_step:int = 15,
            rl_step:int = 40,            
            res_path:str = "",
            road_cap:float = 100.0,
            SoC_change:bool = True,
            seed:int = 0,
        ):
        super().__init__()
        self._par, chd = mp.Pipe()
        self._p = mp.Process(target=_drlenv._worker, args=(
            chd, case_path, end_time, traffic_step, rl_step, res_path, road_cap, SoC_change),
        daemon=True)
        self._p.start()

        self.__rst_cnt = 0

        self.__cs_cnt = self.__fetch('cf', None)
        assert isinstance(self.__cs_cnt, int) and self.__cs_cnt > 0

        self.__e_cnt = self.__fetch('ce', None)
        assert isinstance(self.__e_cnt, int) and self.__e_cnt > 0

        # Observation space: N dims of number of vehicles on the road, and M dims of number of vehicles in the FCS
        net_state_dim = self.__e_cnt * 2 + self.__cs_cnt * 3
        self.observation_space = gym.spaces.Dict({
            "net": gym.spaces.Sequence(gym.spaces.Box(-1.0, 10.0, (1 + net_state_dim,)), stack = True, seed = seed),
            "price": gym.spaces.Box(0.0, 5.0, (1 + self.__cs_cnt,), seed = seed),
        })
        
        # Action space: Price of all the FCS
        self.action_space = gym.spaces.Box(0.0, 5.0, (self.__cs_cnt,), seed = seed)
    
    @property
    def reset_count(self):
        return self.__rst_cnt
    
    @property
    def cs_count(self):
        return self.__cs_cnt
    
    @property
    def edge_count(self):
        return self.__e_cnt

    @property
    def ctime(self) -> int:
        return self.__fetch('ct', None)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsRes, dict]:
        super().reset(seed=seed)
        self._par.send(('r', None))
        self.__rst_cnt += 1
        self._par.poll()
        return self._par.recv()
    
    def step(self, action:np.ndarray) -> Tuple[ObsRes, float, bool, bool, dict]:
        return self.__fetch('s', action)
    
    def observe(self) -> ObsRes:
        return self.__fetch('o', None)
    
    def close(self):
        self.__fetch('q', None)
    
    def __del__(self):
        try:
            self.close()
        except:
            pass
        self._p.join()
    
gym.register(id="v2sim/v2simenv-v0", entry_point=V2SimEnv) # type:ignore

def removeprefix(s:str, prefix:str) -> str:
    if s.startswith(prefix):
        return s[len(prefix):]
    return s

def removesuffix(s:str, suffix:str) -> str:
    if s.endswith(suffix):
        return s[:-len(suffix)]
    return s

if __name__ == "__main__":
    from feasytools import ArgChecker
    args = ArgChecker()
    mcase = args.pop_str("d", "drl_12nodes")
    verbose = args.pop_bool("verbose")
    et = args.pop_int("t", 115200 + 4 * 3600)  # 4 hours
    price_str = args.pop_str("p", "1.0")
    output = args.pop_str("o", "")
    suffix = args.pop_int("s", 0)
    res_path = f"./env_temp/{mcase}_{suffix}"
    env = V2SimEnv(str(Path("./cases") / mcase), et, res_path=res_path)
    #obs, _ = env.reset()
    #print(obs)
    terminated = False
    try:
        p = float(price_str)
        assert p >= 0.0 and p <= 5.0, f"Invalid price {p}. Must be in [0.0, 5.0]."
        assert env.action_space.shape is not None
        price = np.ones(env.action_space.shape, dtype=np.float32) * p
    except ValueError:
        price_arr = removesuffix(removeprefix(price_str.strip(),"["),"]").split(",")
        try:
            price = np.array([float(x.strip()) for x in price_arr], dtype=np.float32)
        except:
            raise ValueError(f"Invalid price string {price_str}. Must be a float or a list of floats in [0.0, 5.0].")
    ep_ret = 0.0
    while not terminated:
        obs, reward, terminated, _, _ = env.step(price)
        if verbose: print(env.ctime, reward, obs["net"].shape, obs["price"].shape)
        ep_ret += reward
    price_list = list(price)
    print(f"Price: {price_list}, Episode return: {ep_ret:.2f}")
    if output != "":
        # 等待文件解锁
        while True:
            try:
                with open(output, "a") as f:
                    price_str = ';'.join(map(str, price_list))
                    f.write(f"{price_str},{ep_ret}\n")
                break
            except PermissionError:
                time.sleep(0.1)
    env.close()
    shutil.rmtree(res_path, ignore_errors=True)