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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_TAG_ENV_IMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_TAG_ENV_IMPL_H_

#include <curand_kernel.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

constexpr int kFeatureNum = 4;
constexpr int kPartiallyObsFeatureNum = 6;

struct GameSetting {
  int seed;
  int predator_num;
  int max_timestep;
  int map_length;
  int map_width;
  float wall_hit_penalty;
  float catch_reward;
  float caught_penalty;
  float step_cost;
  bool partially_observation;
  int index_to_action[10] = {0, 0, 1, 0, -1, 0, 0, 1, 0, -1};
};

// Structure of array (short for SOA) for parallel.
// member shape: [env_num, agent_num]
struct AgentState {
  int *loc_x;
  int *loc_y;
  curandState *rand_state;
  bool *still_in_game;
  int *time_step;
};

CUDA_LIB_EXPORT void InitEnv(const int env_num, const int agent_num, const GameSetting *setting, AgentState *state,
                             cudaStream_t stream);
CUDA_LIB_EXPORT void ResetEnv(const int env_num, const int agent_num, const GameSetting *setting,
                              AgentState *agent_state, float *state, cudaStream_t stream);
CUDA_LIB_EXPORT void StepBindBlock(const int env_num, const int agent_num, const GameSetting *setting,
                                   AgentState *agent_state, const int *action, float *state, float *reward, bool *done,
                                   cudaStream_t stream);
CUDA_LIB_EXPORT void StepCrossBlock(const int env_num, const int agent_num, const GameSetting *setting,
                                    AgentState *agent_state, const int *action, float *state, float *reward, bool *done,
                                    float *team_reward, int *distance, cudaStream_t stream);
CUDA_LIB_EXPORT void AgentStateCopy(const int env_num, const int agent_num, AgentState *dst, AgentState *src,
                                    cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_TAG_ENV_IMPL_H_
