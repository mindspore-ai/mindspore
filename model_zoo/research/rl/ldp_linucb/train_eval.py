# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
train/eval.
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import MovieLensEnv
from src.linucb import LinUCB


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='ua.base',
                        help='data file for movielens')
    parser.add_argument('--rank_k', type=int, default=20,
                        help='rank for data matrix')
    parser.add_argument('--num_actions', type=int, default=20,
                        help='movie number for choices')
    parser.add_argument('--epsilon', type=float, default=8e5,
                        help='epsilon for differentially private')
    parser.add_argument('--delta', type=float, default=1e-1,
                        help='delta for differentially private')
    parser.add_argument('--alpha', type=float, default=1e-1,
                        help='failure probability')
    parser.add_argument('--iter_num', type=float, default=1e6,
                        help='iteration number for training')

    args_opt = parser.parse_args()
    return args_opt


if __name__ == '__main__':
    # build environment
    args = parse_args()
    env = MovieLensEnv(args.data_file, args.num_actions, args.rank_k)

    # Linear UCB
    lin_ucb = LinUCB(
        args.rank_k,
        epsilon=args.epsilon,
        delta=args.delta,
        alpha=args.alpha,
        T=args.iter_num)

    print('start')
    start_time = time.time()
    cumulative_regrets = []
    for i in range(int(args.iter_num)):
        x = env.observation()
        rewards = env.current_rewards()
        lin_ucb.update_status(i + 1)
        xaxat, xay, max_a = lin_ucb(x, rewards)
        cumulative_regrets.append(float(lin_ucb.regret))
        lin_ucb.server_update(xaxat, xay)
        diff = np.abs(lin_ucb.theta.asnumpy() - env.ground_truth).sum()
        print(
            f'--> Step: {i}, diff: {diff:.3f},'
            f'current_regret: {lin_ucb.current_regret:.3f},'
            f'cumulative regret: {lin_ucb.regret:.3f}')
    end_time = time.time()
    print(f'Regret: {lin_ucb.regret}, cost time: {end_time-start_time:.3f}s')
    print(f'theta: {lin_ucb.theta.asnumpy()}')
    print(f'   gt: {env.ground_truth}')

    np.save(f'e_{args.epsilon:.1e}.npy', cumulative_regrets)
    plt.plot(
        range(len(cumulative_regrets)),
        cumulative_regrets,
        label=f'epsilon={args.epsilon:.1e}')
    plt.legend()
    plt.savefig(f'regret_{args.epsilon:.1e}.png')
