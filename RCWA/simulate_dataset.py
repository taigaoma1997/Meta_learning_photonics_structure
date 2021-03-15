import numpy as np
import pandas as pd
import matlab.engine
import tqdm

if __name__ == '__main__':
    eng = matlab.engine.start_matlab()

    # period, height, diameter, acc, stepcase, structure, material, plot_fig

    acc = 10
    stepcase = 10
    structure = 'rod'
    material = 'Si'

    for i in tqdm.tqdm(range(10000)):
        period = np.random.randint(100, 300)
        height = np.random.randint(50, 200)
        diameter = np.random.randint(50, 0.75 * period)
        ret = eng.run_rcwa_simulation(period, height, diameter, acc, stepcase, structure, material, False)