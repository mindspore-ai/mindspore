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

#include "backend/kernel_compiler/gpu/cuda_impl/rl/tag_env_impl.cuh"
#include <assert.h>
#include <algorithm>

__global__ void InitKernel(const int env_num, const int agent_num, const GameSetting *setting,
                           AgentState *agent_state) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < env_num * agent_num; i += gridDim.x * blockDim.x) {
    curand_init(setting->seed, i, 0, &agent_state->rand_state[i]);
  }
}

void InitEnv(const int env_num, const int agent_num, const GameSetting *setting, AgentState *state,
             cudaStream_t stream) {
  InitKernel<<<(env_num * agent_num + 255) / 256, 256, 0, stream>>>(env_num, agent_num, setting, state);
}

__global__ void ResetKernel(const int env_num, const int agent_num, const GameSetting *setting, AgentState *agent_state,
                            float *state) {
  // Reset the agent state
  for (size_t gaid = blockIdx.x * blockDim.x + threadIdx.x; gaid < env_num * agent_num;
       gaid += gridDim.x * blockDim.x) {
    const int eid = gaid / agent_num;
    const int aid = gaid % agent_num;

    // Static reset.
    bool is_prey = (aid >= setting->predator_num);
    int x = is_prey ? 0 : 0.5 * setting->map_width;
    int y = is_prey ? 0 : 0.5 * setting->map_length;

    agent_state->loc_x[gaid] = x;
    agent_state->loc_y[gaid] = y;
    agent_state->still_in_game[gaid] = true;
    agent_state->time_step[eid] = 0;
    agent_state->prey_left[eid] = setting->prey_num;

    const int state_size_per_agent = agent_num * kFeatureNum + 1;
    const size_t base = eid * agent_num * state_size_per_agent + aid * kFeatureNum;
    for (int id = 0; id < agent_num; id++) {
      size_t offset = base + id * state_size_per_agent;
      state[offset] = static_cast<float>(x) / setting->map_width;
      state[offset + 1] = static_cast<float>(y) / setting->map_length;
      state[offset + 2] = aid >= setting->predator_num;
      state[offset + 3] = aid == id;
    }
    state[eid * agent_num * state_size_per_agent + aid * state_size_per_agent + agent_num * kFeatureNum] = 0;
  }
}

void ResetEnv(const int env_num, const int agent_num, const GameSetting *setting, AgentState *agent_state, float *state,
              cudaStream_t stream) {
  ResetKernel<<<(env_num * agent_num + 255) / 256, 256, 0, stream>>>(env_num, agent_num, setting, agent_state, state);
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
          agent_state->prey_left[eid]--;
          atomicAdd(&team_reward, 1);
          break;
        }
      }
    }
    __syncthreads();

    // Auto reset done environment.
    bool is_done = (agent_state->time_step[eid] >= setting->max_timestep) || (agent_state->prey_left[eid] == 0);
    if (is_done) {
      x = is_prey ? 0 : 0.5 * setting->map_width;
      y = is_prey ? 0 : 0.5 * setting->map_length;
      agent_state->loc_x[gaid] = x;
      agent_state->loc_y[gaid] = y;
      agent_state->still_in_game[gaid] = true;
    }

    // Construct observation.
    const int state_size_per_agent = agent_num * kFeatureNum + 1;
    const size_t base = eid * agent_num * state_size_per_agent + aid * kFeatureNum;
    for (int id = 0; id < agent_num; id++) {
      size_t offset = base + id * state_size_per_agent;
      state[offset] = static_cast<float>(x) / map_width;
      state[offset + 1] = static_cast<float>(y) / map_length;
      state[offset + 2] = is_prey;
      state[offset + 3] = (aid == id);
    }
    state[eid * agent_num * state_size_per_agent + aid * state_size_per_agent + agent_num * kFeatureNum] =
      static_cast<float>(agent_state->time_step[eid]) / setting->max_timestep;

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
      agent_state->prey_left[eid] = setting->prey_num;
    }
  }
}

void StepBindBlock(const int env_num, const int agent_num, const GameSetting *setting, AgentState *agent_state,
                   const int *action, float *state, float *reward, bool *done, cudaStream_t stream) {
  size_t shm_size = env_num * agent_num * sizeof(float) * 2;
  StepBindBlockKernel<<<env_num, 256, shm_size, stream>>>(env_num, agent_num, setting, agent_state, action, state,
                                                          reward, done);
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
                               AgentState *agent_state, float *team_reward) {
  const int prey_num_per_env = setting->prey_num;
  const int total_prey_num = env_num * prey_num_per_env;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_prey_num; i += gridDim.x * blockDim.x) {
    const int eid = i / prey_num_per_env;
    const int rid = eid * agent_num + i % prey_num_per_env + setting->predator_num;

    if (agent_state->still_in_game[rid]) {
      int x = agent_state->loc_x[rid];
      int y = agent_state->loc_y[rid];
      for (int j = 0; j < setting->predator_num; j++) {
        int tid = eid * agent_num + j;
        if (x == agent_state->loc_x[tid] && y == agent_state->loc_y[tid]) {
          agent_state->still_in_game[rid] = false;
          agent_state->prey_left[eid]--;
          atomicAdd(&team_reward[eid], 1);
        }
      }
    }
  }
}

__global__ void ConstructStepOutput(const int env_num, const int agent_num, const GameSetting *setting,
                                    AgentState *agent_state, float *state, float *reward, bool *done,
                                    float *team_reward) {
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
    bool is_done = (agent_state->time_step[eid] >= setting->max_timestep) || (agent_state->prey_left[eid] == 0);
    if (is_done) {
      x = is_prey ? 0 : 0.5 * setting->map_width;
      y = is_prey ? 0 : 0.5 * setting->map_length;
      agent_state->loc_x[gaid] = x;
      agent_state->loc_y[gaid] = y;
      agent_state->still_in_game[gaid] = true;
    }

    // Construct observation.
    const int state_size_per_agent = agent_num * kFeatureNum + 1;
    const size_t base = eid * agent_num * state_size_per_agent + aid * kFeatureNum;
    for (int id = 0; id < agent_num; id++) {
      size_t offset = base + id * state_size_per_agent;
      state[offset] = static_cast<float>(x) / map_width;
      state[offset + 1] = static_cast<float>(y) / map_length;
      state[offset + 2] = is_prey;
      state[offset + 3] = (aid == id);
    }
    state[eid * agent_num * state_size_per_agent + aid * state_size_per_agent + agent_num * kFeatureNum] =
      static_cast<float>(agent_state->time_step[eid]) / setting->max_timestep;

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
      agent_state->prey_left[eid] = setting->prey_num;
    }
  }
}

void StepCrossBlock(const int env_num, const int agent_num, const GameSetting *setting, AgentState *agent_state,
                    const int *action, float *state, float *reward, bool *done, float *team_reward,
                    cudaStream_t stream) {
  // Update agent location, construct observation, done.
  int block_dim = 256;
  int grid_dim = (env_num * agent_num + block_dim - 1) / block_dim;
  UpdateAgentLoc<<<grid_dim, block_dim, 0, stream>>>(env_num, agent_num, setting, agent_state, action, state, reward);

  // Calculate team reward.
  cudaMemsetAsync(team_reward, 0, sizeof(float) * env_num, stream);
  CalcTeamReward<<<grid_dim, block_dim, 0, stream>>>(env_num, agent_num, setting, agent_state, team_reward);

  // Construct step output.
  ConstructStepOutput<<<grid_dim, block_dim, 0, stream>>>(env_num, agent_num, setting, agent_state, state, reward, done,
                                                          team_reward);
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
      dst->prey_left[eid] = src->prey_left[eid];
    }
  }
}

void AgentStateCopy(const int env_num, const int agent_num, AgentState *dst, AgentState *src, cudaStream_t stream) {
  int block_dim = 256;
  int grid_dim = (env_num * agent_num + block_dim - 1) / block_dim;
  AgentStateCopyKernel<<<grid_dim, block_dim, 0, stream>>>(env_num, agent_num, dst, src);
}
