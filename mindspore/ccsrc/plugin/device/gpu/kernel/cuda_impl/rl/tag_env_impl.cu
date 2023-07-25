/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/gpu/kernel/cuda_impl/rl/tag_env_impl.cuh"
#include <assert.h>
#include <algorithm>

__global__ void InitKernel(const int env_num, const int agent_num, const GameSetting *setting,
                           AgentState *agent_state) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < env_num * agent_num; i += gridDim.x * blockDim.x) {
    curand_init(setting->seed, i, 0, &agent_state->rand_state[i]);
  }
}

cudaError_t InitEnv(const int env_num, const int agent_num, const GameSetting *setting, AgentState *state,
                    cudaStream_t stream) {
  InitKernel<<<(env_num * agent_num + 255) / 256, 256, 0, stream>>>(env_num, agent_num, setting, state);
  return GetCudaStatus();
}

__device__ __forceinline__ void ConstructFullyObservation(const int &x, const int &y, const int &eid, const int &aid,
                                                          const int &agent_num, const int &width, const int &length,
                                                          const int &max_timestep, const bool &is_prey,
                                                          const int &time_step, float *observation) {
  const int obs_size_per_agent = agent_num * kFeatureNum + 1;
  const size_t base = eid * agent_num * obs_size_per_agent + aid * kFeatureNum;
  for (int id = 0; id < agent_num; id++) {
    size_t offset = base + id * obs_size_per_agent;
    observation[offset] = static_cast<float>(x) / width;
    observation[offset + 1] = static_cast<float>(y) / length;
    observation[offset + 2] = is_prey;
    observation[offset + 3] = aid == id;
  }
  observation[eid * agent_num * obs_size_per_agent + aid * obs_size_per_agent + agent_num * kFeatureNum] =
    static_cast<float>(time_step) / max_timestep;
}

__device__ __forceinline__ void ConstructPartialObservation(const int &x, const int &y, const int &eid, const int &aid,
                                                            const int &agent_num, const int &width, const int &length,
                                                            const int max_timestep, const bool &is_prey,
                                                            const int &time_step, const int adv_x, const int &adv_y,
                                                            float *observation) {
  const int obs_size_per_agent = 6;
  const size_t base = eid * agent_num * obs_size_per_agent + aid * obs_size_per_agent;
  auto observation_base = observation + base;
  observation_base[0] = static_cast<float>(x) / width;
  observation_base[1] = static_cast<float>(y) / length;
  observation_base[2] = static_cast<float>(adv_x) / width;
  observation_base[3] = static_cast<float>(adv_y) / length;
  observation_base[4] = is_prey;
  observation_base[5] = static_cast<float>(time_step) / max_timestep;
}

__global__ void ResetKernel(const int env_num, const int agent_num, const GameSetting *setting, AgentState *agent_state,
                            float *state) {
  // Reset the agent state
  int eid = blockIdx.x;
  for (int aid = threadIdx.x; aid < agent_num; aid += blockDim.x) {
    int gaid = eid * agent_num + aid;
    // Static reset.
    bool is_prey = (aid >= setting->predator_num);
    int x = is_prey ? 0 : 0.5 * setting->map_width;
    int y = is_prey ? 0 : 0.5 * setting->map_length;

    agent_state->loc_x[gaid] = x;
    agent_state->loc_y[gaid] = y;
    agent_state->still_in_game[gaid] = true;
    agent_state->time_step[eid] = 0;
    __syncthreads();

    if (setting->partially_observation) {
      int prey_index = (eid + 1) * agent_num - 1;
      int prey_x = agent_state->loc_x[prey_index];
      int prey_y = agent_state->loc_y[prey_index];

      extern __shared__ int distance[];
      // Manhattan distance.
      distance[aid] = std::abs(prey_x - x) + std::abs(prey_y - y);
      __syncthreads();

      int adv_x = prey_x;
      int adv_y = prey_y;
      if (is_prey) {
        int nearest_index = 0;
        int min_distance = setting->map_width + setting->map_length;
        for (int i = 0; i < agent_num - 1; i++) {
          if (min_distance > distance[i]) {
            nearest_index = i;
            min_distance = distance[i];
          }
        }
        adv_x = agent_state->loc_x[nearest_index];
        adv_y = agent_state->loc_x[nearest_index];
      }

      ConstructPartialObservation(x, y, eid, aid, agent_num, setting->map_width, setting->map_length,
                                  setting->max_timestep, is_prey, agent_state->time_step[eid], adv_x, adv_y, state);
    } else {
      ConstructFullyObservation(x, y, eid, aid, agent_num, setting->map_width, setting->map_length,
                                setting->max_timestep, is_prey, agent_state->time_step[eid], state);
    }
  }
}

cudaError_t ResetEnv(const int env_num, const int agent_num, const GameSetting *setting, AgentState *agent_state,
                     float *state, cudaStream_t stream) {
  size_t shm_size = agent_num * sizeof(int);
  ResetKernel<<<env_num, 256, shm_size, stream>>>(env_num, agent_num, setting, agent_state, state);
  return GetCudaStatus();
}

__global__ void StepBindBlockKernel(const int env_num, const int agent_num, const GameSetting *setting,
                                    AgentState *agent_state, const int *action, float *state, float *reward,
                                    bool *done) {
  __shared__ int team_reward;
  extern __shared__ int agent_loc[];
  int *loc_x = agent_loc;
  int *loc_y = agent_loc + agent_num;

  int eid = blockIdx.x;
  for (int aid = threadIdx.x; aid < agent_num; aid += blockDim.x) {
    int gaid = eid * agent_num + aid;
    float agent_reward = 0.0;

    // Parse discrete action.
    int action_offset = action[gaid] * 2;
    assert(action_offset <= 8);
    int action_x = setting->index_to_action[action_offset];
    int action_y = setting->index_to_action[action_offset + 1];

    // Update agent location.
    int x = agent_state->loc_x[gaid] + action_x;
    int y = agent_state->loc_y[gaid] + action_y;

    int map_width = setting->map_width;
    int map_length = setting->map_length;
    if (x < 0 || y < 0 || x > map_width || y > map_length) {
      x = min(max(0, x), setting->map_width);
      y = min(max(0, y), setting->map_length);
      agent_reward -= setting->wall_hit_penalty;
    }

    loc_x[aid] = x;
    loc_y[aid] = y;
    agent_state->loc_x[gaid] = x;
    agent_state->loc_y[gaid] = y;

    // Update time step
    if (aid == 0) {
      team_reward = 0;
      agent_state->time_step[eid]++;
    }
    __syncthreads();

    // Calculate team reward.
    bool is_prey = aid >= setting->predator_num;
    if (is_prey && agent_state->still_in_game[gaid]) {
      for (int tid = 0; tid < setting->predator_num; tid++) {
        // Every prey only caught by one predator.
        if (x == loc_x[tid] && y == loc_y[tid]) {
          agent_state->still_in_game[gaid] = false;
          team_reward = 1;
          break;
        }
      }
    }
    __syncthreads();

    // Auto reset done environment.
    bool is_done = (agent_state->time_step[eid] >= setting->max_timestep) || (team_reward == 1);
    if (is_done) {
      x = is_prey ? 0 : 0.5 * setting->map_width;
      y = is_prey ? 0 : 0.5 * setting->map_length;
      agent_state->loc_x[gaid] = x;
      agent_state->loc_y[gaid] = y;
      agent_state->still_in_game[gaid] = true;
    }

    // Construct observation.
    if (setting->partially_observation) {
      int prey_index = (eid + 1) * agent_num - 1;
      int prey_x = agent_state->loc_x[prey_index];
      int prey_y = agent_state->loc_y[prey_index];

      // Reuse shared memory.
      int *distance = loc_x;
      // Manhattan distance.
      distance[aid] = std::abs(prey_x - x) + std::abs(prey_y - y);
      __syncthreads();

      int adv_x = prey_x;
      int adv_y = prey_y;
      if (is_prey) {
        int nearest_index = 0;
        int min_distance = setting->map_width + setting->map_length;
        for (int i = 0; i < agent_num - 1; i++) {
          if (min_distance > distance[i]) {
            nearest_index = i;
            min_distance = distance[i];
          }
        }
        adv_x = agent_state->loc_x[nearest_index];
        adv_y = agent_state->loc_x[nearest_index];
      }

      ConstructPartialObservation(x, y, eid, aid, agent_num, setting->map_width, setting->map_length,
                                  setting->max_timestep, is_prey, agent_state->time_step[eid], adv_x, adv_y, state);
    } else {
      ConstructFullyObservation(x, y, eid, aid, agent_num, map_width, map_length, setting->max_timestep, is_prey,
                                agent_state->time_step[eid], state);
    }

    // Construct reward.
    if (team_reward > 0) {
      agent_reward += is_prey ? -setting->caught_penalty * team_reward : setting->catch_reward * team_reward;
    } else {
      agent_reward += is_prey ? setting->step_cost : -setting->step_cost;
    }
    reward[gaid] = agent_reward * agent_state->still_in_game[gaid];

    // Construct environment done flag.
    if (aid == 0) {
      done[eid] = is_done;
      agent_state->time_step[eid] = 0;
    }
  }
}

cudaError_t StepBindBlock(const int env_num, const int agent_num, const GameSetting *setting, AgentState *agent_state,
                          const int *action, float *state, float *reward, bool *done, cudaStream_t stream) {
  size_t shm_size = env_num * agent_num * sizeof(float) * 2;
  StepBindBlockKernel<<<env_num, 256, shm_size, stream>>>(env_num, agent_num, setting, agent_state, action, state,
                                                          reward, done);
  return GetCudaStatus();
}

__global__ void UpdateAgentLoc(const int env_num, const int agent_num, const GameSetting *setting,
                               AgentState *agent_state, const int *action, float *state, float *reward) {
  int total_agent = env_num * agent_num;
  for (size_t gaid = blockIdx.x * blockDim.x + threadIdx.x; gaid < total_agent; gaid += gridDim.x * blockDim.x) {
    const int eid = gaid / agent_num;
    const int aid = gaid % agent_num;
    // Parse discrete action.
    int action_offset = action[gaid] * 2;
    assert(action_offset <= 8);
    int action_x = setting->index_to_action[action_offset];
    int action_y = setting->index_to_action[action_offset + 1];

    // Update agent location
    int x = agent_state->loc_x[gaid] + action_x;
    int y = agent_state->loc_y[gaid] + action_y;

    int map_width = setting->map_width;
    int map_length = setting->map_length;
    reward[gaid] = 0.0;
    if (x < 0 || y < 0 || x > map_width || y > map_length) {
      x = min(max(0, x), setting->map_width);
      y = min(max(0, y), setting->map_length);
      reward[gaid] -= setting->wall_hit_penalty;
    }
    agent_state->loc_x[gaid] = x;
    agent_state->loc_y[gaid] = y;

    if (aid == 0) {
      agent_state->time_step[eid]++;
    }
  }
}

__global__ void CalcTeamReward(const int env_num, const int agent_num, const GameSetting *setting,
                               AgentState *agent_state, float *team_reward, int *distance) {
  size_t total_agent = env_num * agent_num;
  for (size_t gaid = blockIdx.x * blockDim.x + threadIdx.x; gaid < total_agent; gaid += gridDim.x * blockDim.x) {
    const int eid = gaid / agent_num;
    const int aid = gaid % agent_num;

    const int prey_id = eid * agent_num + setting->predator_num;
    int prey_x = agent_state->loc_x[prey_id];
    int prey_y = agent_state->loc_y[prey_id];
    distance[eid * agent_num + aid] =
      std::abs(prey_x - agent_state->loc_x[gaid]) + std::abs(prey_y - agent_state->loc_y[gaid]);

    if (gaid != prey_id && agent_state->still_in_game[prey_id] && agent_state->loc_x[gaid] == prey_x &&
        agent_state->loc_y[gaid] == prey_y) {
      agent_state->still_in_game[prey_id] = false;
      team_reward[eid] = 1;
    }
  }
}

__global__ void ConstructStepOutput(const int env_num, const int agent_num, const GameSetting *setting,
                                    AgentState *agent_state, float *state, float *reward, bool *done,
                                    const float *team_reward, const int *distance) {
  int total_agent_num = env_num * agent_num;
  for (size_t gaid = blockIdx.x * blockDim.x + threadIdx.x; gaid < total_agent_num; gaid += gridDim.x * blockDim.x) {
    int eid = gaid / agent_num;
    int aid = gaid % agent_num;
    bool is_prey = aid >= setting->predator_num;

    int x = agent_state->loc_x[gaid];
    int y = agent_state->loc_y[gaid];
    int map_width = setting->map_width;
    int map_length = setting->map_length;

    // Auto reset done environment.
    bool is_done = (agent_state->time_step[eid] >= setting->max_timestep) || (team_reward[eid] == 1);
    if (is_done) {
      x = is_prey ? 0 : 0.5 * setting->map_width;
      y = is_prey ? 0 : 0.5 * setting->map_length;
      agent_state->loc_x[gaid] = x;
      agent_state->loc_y[gaid] = y;
      agent_state->still_in_game[gaid] = true;
    }

    // Construct observation.
    if (setting->partially_observation) {
      int prey_index = (eid + 1) * agent_num - 1;
      int prey_x = agent_state->loc_x[prey_index];
      int prey_y = agent_state->loc_y[prey_index];

      int adv_x = prey_x;
      int adv_y = prey_y;
      if (is_prey) {
        int nearest_index = 0;
        int min_distance = setting->map_width + setting->map_length;
        for (int i = eid * agent_num; i < (eid + 1) * agent_num - 1; i++) {
          if (min_distance > distance[i]) {
            nearest_index = i;
            min_distance = distance[i];
          }
        }
        adv_x = agent_state->loc_x[nearest_index];
        adv_y = agent_state->loc_y[nearest_index];
      }

      ConstructPartialObservation(x, y, eid, aid, agent_num, setting->map_width, setting->map_length,
                                  setting->max_timestep, is_prey, agent_state->time_step[eid], adv_x, adv_y, state);
    } else {
      ConstructFullyObservation(x, y, eid, aid, agent_num, map_width, map_length, setting->max_timestep, is_prey,
                                agent_state->time_step[eid], state);
    }

    // Construct agent reward.
    if (team_reward[eid] > 0) {
      reward[gaid] += is_prey ? -team_reward[eid] * setting->caught_penalty : team_reward[eid] * setting->catch_reward;
    } else {
      reward[gaid] += is_prey ? setting->step_cost : -setting->step_cost;
    }

    // Construct environment done flag.
    if (aid == 0) {
      done[eid] = is_done;
      agent_state->time_step[eid] = 0;
    }
  }
}

cudaError_t StepCrossBlock(const int env_num, const int agent_num, const GameSetting *setting, AgentState *agent_state,
                           const int *action, float *state, float *reward, bool *done, float *team_reward,
                           int *distance, cudaStream_t stream) {
  // Update agent location, construct observation, done.
  int block_dim = 256;
  int grid_dim = (env_num * agent_num + block_dim - 1) / block_dim;
  UpdateAgentLoc<<<grid_dim, block_dim, 0, stream>>>(env_num, agent_num, setting, agent_state, action, state, reward);

  // Calculate team reward.
  cudaMemsetAsync(team_reward, 0, sizeof(float) * env_num, stream);
  CalcTeamReward<<<grid_dim, block_dim, 0, stream>>>(env_num, agent_num, setting, agent_state, team_reward, distance);

  // Construct step output.
  ConstructStepOutput<<<grid_dim, block_dim, 0, stream>>>(env_num, agent_num, setting, agent_state, state, reward, done,
                                                          team_reward, distance);
  return GetCudaStatus();
}

__global__ void AgentStateCopyKernel(const int env_num, const int agent_num, AgentState *dst, AgentState *src) {
  int total_agent_num = env_num * agent_num;
  for (size_t gaid = blockIdx.x * blockDim.x + threadIdx.x; gaid < total_agent_num; gaid += gridDim.x * blockDim.x) {
    int eid = gaid / total_agent_num;
    int aid = gaid % total_agent_num;
    dst->loc_x[gaid] = src->loc_x[gaid];
    dst->loc_y[gaid] = src->loc_y[gaid];
    dst->rand_state[gaid] = src->rand_state[gaid];
    dst->still_in_game[gaid] = src->still_in_game[gaid];

    if (aid == 0) {
      dst->time_step[eid] = src->time_step[eid];
    }
  }
}

cudaError_t AgentStateCopy(const int env_num, const int agent_num, AgentState *dst, AgentState *src,
                           cudaStream_t stream) {
  int block_dim = 256;
  int grid_dim = (env_num * agent_num + block_dim - 1) / block_dim;
  AgentStateCopyKernel<<<grid_dim, block_dim, 0, stream>>>(env_num, agent_num, dst, src);
  return GetCudaStatus();
}
