'''
Environment for the 12-node V2Sim case
'''
from itertools import chain
from multiprocessing.connection import PipeConnection
from pathlib import Path
import time, random
from typing import Optional
import gymnasium as gym
import numpy as np
import libsumo, os
import v2sim
import multiprocessing as mp

class _drlenv:
    def __create_inst(self):
        self._inst = v2sim.V2SimInstance(
            cfgdir = self._case_path,
            outdir = self._res_path, 
            traffic_step = self._traffic_step,
            start_time = 0, # Useless, will be covered by case initial state
            end_time = self._end_time,
            log = "",
            silent = True,
            seed = random.randint(0, 1000000),
            initial_state = self._case_state
        )

        for ev in self._inst.vehicles.values():
            ev._w = 1e6

        if self._SoC_change:
            # Randomize the initial state of SoC
            for ev in self._inst.vehicles.values():
                ev._elec *= random.uniform(0.8, 1.2)
                ev._elec = min(ev._elec, ev._bcap)
                ev._elec = max(ev._elec, 0.0)
        self._cs_cnt = len(self._inst.fcs)
        self._inst.start()
        self._inst.step()
        self._enames = [e for e in self._inst.edge_names if not e.startswith("CS")]
        self._t = 0
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
    
    def __init__(self, 
            case_path:str,
            end_time:int,
            traffic_step:int = 15,
            rl_step:int = 4,
            res_path:str = "",
            road_cap:float = 500.0,
            SoC_change:bool = True
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
        
    def _get_bus_overlim(self):
        s = 0.0
        for b in self._buses:
            assert b.V is not None
            if b.V > self._vmax:
                s += b.V - self._vmax
            elif b.V < self._vmin:
                s += self._vmin - b.V
        return s
    
    def _get_road_vcnt(self):
        return (libsumo.edge.getLastStepVehicleNumber(e) for e in self._enames)
    
    def _get_road_density(self):
        return (libsumo.edge.getLastStepVehicleNumber(e)/self.road_cap for e in self._enames)

    def _get_road_avg_soc(self):
        ret = []
        for e in self._enames:
            cnt = 0
            sum_soc = 0
            for vid in libsumo.edge.getLastStepVehicleIDs(e):
                v = self._inst.vehicles[vid]
                sum_soc += v.SOC
                cnt += 1
            if cnt == 0: ret.append(0)
            else: ret.append(sum_soc / cnt)
        return ret
    
    def _get_fcs_vcnt(self):
        return [c.veh_count() for c in self._inst.fcs]
    
    def _get_fcs_usage(self):
        return [c.veh_count()/c.slots-1 for c in self._inst.fcs]
    
    def _get_fcs_vcnt_wait(self):
        return [c.wait_count() for c in self._inst.fcs]
    
    def _get_fcs_wait_rate(self):
        return [c.wait_count()/c.slots for c in self._inst.fcs]
    
    def _get_fcs_avg_soc(self):
        ret = []
        for c in self._inst.fcs:
            cnt = 0
            sum_soc = 0
            for vid in c.vehicles():
                v = self._inst.vehicles[vid]
                sum_soc += v.SOC
                cnt += 1
            if cnt == 0: ret.append(0)
            else: ret.append(sum_soc / cnt)
        return ret
    
    def _get_obs(self):
        return np.array(
            list(chain(
                self._get_road_density(),
                self._get_road_avg_soc(),
                self._get_fcs_usage(),
                self._get_fcs_avg_soc(),
            )),
            dtype = np.float32
        )
    
    def _get_info(self):
        return {}
    
    def reset(self):
        if self._inst.is_working:
            self._inst.stop()
            time.sleep(0.5) # Wait for the instance to stop. Never remove this line!
        self.__create_inst()
        self._t = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action:np.ndarray):
        assert action.shape == (self._cs_cnt, )
        for i, v in enumerate(action):
            self._inst.fcs[i].pbuy.setOverride(v)
        try:
            self._t = self._inst.step_until(
                self._t + self._traffic_step * self._rl_step
            )
        except Exception as e:
            print("\nV2SimEnv Error @", self._t)
            raise e
        if self._t >= self._inst.etime:
            terminated = True
            #self._inst.stop()
        else:
            terminated = False
        truncated = False

        # The more vehicle in FCSs and on roads, the more penalty
        reward = - (
            self._get_bus_overlim() * 1e5 +
            sum(self._get_fcs_wait_rate()) * 100.0 +
            sum(self._get_road_vcnt()) / 100.0
        )

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
                break
            else:
                raise ValueError(f"Unknown operation {op}")
    
    @staticmethod
    def _worker(q:PipeConnection, *args, **kwargs):
        env = _drlenv(*args, **kwargs)
        env._mainloop(q)

class V2SimEnv(gym.Env):
    def __init__(self, 
            case_path:str,
            end_time:int,
            traffic_step:int = 15,
            rl_step:int = 4,            
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

        self._par.send(('cf', None))
        self._par.poll()
        self.__cs_cnt = self._par.recv()
        assert isinstance(self.__cs_cnt, int) and self.__cs_cnt > 0

        self._par.send(('ce', None))
        self._par.poll()
        self.__e_cnt = self._par.recv()
        assert isinstance(self.__e_cnt, int) and self.__e_cnt > 0

        # Observation space: N dims of number of vehicles on the road, and M dims of number of vehicles in the FCS
        self.observation_space = gym.spaces.Box(-1.0, 5.0, (self.__e_cnt * 2 + self.__cs_cnt * 2,), seed = seed)
        # Action space: Price of the 12 FCS
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
        self._par.send(('ct', None))
        self._par.poll()
        return self._par.recv()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._par.send(('r', None))
        self.__rst_cnt += 1
        self._par.poll()
        return self._par.recv()
    
    def step(self, action:np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._par.send(('s', action))
        self._par.poll()
        return self._par.recv()
    
    def observe(self) -> np.ndarray:
        self._par.send(('o', None))
        self._par.poll()
        return self._par.recv()
    
    def close(self):
        self._par.send(('q', None))
    
    def __del__(self):
        try:
            self.close()
        except:
            pass
        self._p.join()
    
gym.register(id="v2sim/v2simenv-v0", entry_point=V2SimEnv) # type:ignore

if __name__ == "__main__":
    import feasytools as ft
    args = ft.ArgChecker()
    mcase = args.pop_str("d", "drl_12nodes")
    et = args.pop_int("t", 129600)
    env = V2SimEnv(str(Path("./cases") / mcase), et)
    #obs, _ = env.reset()
    #print(obs)
    terminated = False
    price = np.array([1.0]*env.cs_count)
    while not terminated:
        obs, reward, terminated, _, _ = env.step(price)
        print(env.ctime, reward)
    env.close()