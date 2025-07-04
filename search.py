import os
from typing import List
from multiprocessing import Pool

def run_cmd(cmd):
    os.system(cmd)

def search(case:str, n_dim:int = 2, prices:List[float] = [1,2], output:str = "./data/search_results.csv", n_core:int = 1):
    """
    Search for optimal prices in a grid of dimensions.
    
    Args:
        n_dim (int): Number of dimensions for the price grid.
        output (str): Path to the output CSV file.
    """
    with open(output, "w") as f:
        f.write("price,return\n")

    cmds = []

    # Iterate over a grid of prices from 1 to 5 for both dimensions
    cnt = 0
    n = len(prices)
    a = [0] * (n_dim + 1)
    while a[-1] == 0:
        this_prices = [prices[i] for i in a[:-1]]
        cmds.append(f'python env.py -d {case} -p "{this_prices}" -o {output} -s {cnt}')
        cnt += 1
        # Increment the last dimension
        p = 0
        while p <= n_dim and a[p] == n - 1:
            a[p] = 0
            p += 1
        a[p] += 1
    
    if n_core > 1:
        with Pool(n_core) as pool:
            pool.map(run_cmd, cmds)
    else:
        for cmd in cmds:
            run_cmd(cmd)
    
if __name__ == "__main__":
    from feasytools import ArgChecker
    parser = ArgChecker()
    case = parser.pop_str("case", default="drl_2cs")
    n_dim = parser.pop_int("n_dim", default=2)
    prices = eval(parser.pop_str("prices", default="[1, 2, 3, 4, 5]"))
    output = parser.pop_str("output", default="./data/search_results.csv")
    n_core = parser.pop_int("n_core", default=0)

    if n_core == 0:
        n_core = os.cpu_count() or 1
        print(f"Using {n_core} cores for parallel processing.")
    search(case=case, n_dim=n_dim, prices=prices, output=output, n_core=n_core)