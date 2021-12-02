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
#include "backend/kernel_compiler/gpu/rl/tag_environment.h"

#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <utility>
#include "runtime/device/gpu/cuda_driver.h"
#include "runtime/device/gpu/cuda_common.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kSeedAttr = "seed";
constexpr auto kPredatorNumAttr = "predator_num";
constexpr auto kPreyNumrAttr = "prey_num";
constexpr auto kMaxTimestepAttr = "max_timestep";
constexpr auto kMapLengthAttr = "map_length";
constexpr auto kMapWidthAttr = "map_width";
constexpr auto kWallHitPenaltyAttr = "wall_hit_penalty";
constexpr auto kCatchRewardAttr = "catch_reward";
constexpr auto kCaughtPenaltyAttr = "caught_penalty";
constexpr auto kStepCostAttr = "step_cost";
constexpr auto kEnvNumAttr = "environment_num";
}  // namespace

TagEnvironment::~TagEnvironment() {
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  allocator.FreeTensorMem(agent_state_device_);
  allocator.FreeTensorMem(game_setting_device_);
  FinalizeAgentState(agent_state_host_);
}

bool TagEnvironment::InitGameSetting(const CNodePtr &cnode, GameSetting *setting_host) {
  MS_EXCEPTION_IF_NULL(setting_host);

  setting_host->seed = AnfAlgo::GetNodeAttr<int64_t>(cnode, kSeedAttr);
  setting_host->predator_num = AnfAlgo::GetNodeAttr<int64_t>(cnode, kPredatorNumAttr);
  setting_host->prey_num = AnfAlgo::GetNodeAttr<int64_t>(cnode, kPreyNumrAttr);
  setting_host->max_timestep = AnfAlgo::GetNodeAttr<int64_t>(cnode, kMaxTimestepAttr);
  setting_host->map_length = AnfAlgo::GetNodeAttr<int64_t>(cnode, kMapLengthAttr);
  setting_host->map_width = AnfAlgo::GetNodeAttr<int64_t>(cnode, kMapWidthAttr);
  setting_host->wall_hit_penalty = AnfAlgo::GetNodeAttr<float>(cnode, kWallHitPenaltyAttr);
  setting_host->catch_reward = AnfAlgo::GetNodeAttr<float>(cnode, kCatchRewardAttr);
  setting_host->caught_penalty = AnfAlgo::GetNodeAttr<float>(cnode, kCaughtPenaltyAttr);
  setting_host->step_cost = AnfAlgo::GetNodeAttr<float>(cnode, kStepCostAttr);

  env_num_ = AnfAlgo::GetNodeAttr<int64_t>(cnode, kEnvNumAttr);
  agent_num_ = setting_host->predator_num + setting_host->prey_num;
  return true;
}

bool TagEnvironment::InitAgentState(int predator_num, int prey_num, AgentState *agent_state) {
  MS_EXCEPTION_IF_NULL(agent_state);

  int total_agents_num = env_num_ * agent_num_;
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  agent_state->loc_x = static_cast<int *>(allocator.AllocTensorMem(sizeof(int) * total_agents_num));
  agent_state->loc_y = static_cast<int *>(allocator.AllocTensorMem(sizeof(int) * total_agents_num));
  agent_state->still_in_game = static_cast<bool *>(allocator.AllocTensorMem(sizeof(bool) * total_agents_num));
  agent_state->rand_state =
    static_cast<curandState *>(allocator.AllocTensorMem(sizeof(curandState) * total_agents_num));
  agent_state->time_step = static_cast<int *>(allocator.AllocTensorMem(sizeof(int) * env_num_));
  agent_state->prey_left = static_cast<int *>(allocator.AllocTensorMem(sizeof(int) * env_num_));
  return true;
}

bool TagEnvironment::FinalizeAgentState(const AgentState &agent_setting) {
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  allocator.FreeTensorMem(agent_setting.prey_left);
  allocator.FreeTensorMem(agent_setting.time_step);
  allocator.FreeTensorMem(agent_setting.still_in_game);
  allocator.FreeTensorMem(agent_setting.rand_state);
  allocator.FreeTensorMem(agent_setting.loc_x);
  allocator.FreeTensorMem(agent_setting.loc_y);
  return true;
}

bool TagEnvironment::Init(const CNodePtr &cnode, void *stream_ptr) {
  InitGameSetting(cnode, &game_setting_host_);
  InitAgentState(game_setting_host_.predator_num, game_setting_host_.prey_num, &agent_state_host_);

  // Move the game setting to device.
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  game_setting_device_ = static_cast<GameSetting *>(allocator.AllocTensorMem(sizeof(GameSetting)));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(game_setting_device_, &game_setting_host_, sizeof(GameSetting), cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpy failed.");

  // Move the agent state to device.
  agent_state_device_ = static_cast<AgentState *>(allocator.AllocTensorMem(sizeof(AgentState)));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(agent_state_device_, &agent_state_host_, sizeof(AgentState), cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpy failed.");

  InitEnv(env_num_, agent_num_, game_setting_device_, agent_state_device_, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

bool TagEnvironment::Reset(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto state = reinterpret_cast<float *>(outputs[0]->addr);
  ResetEnv(env_num_, agent_num_, game_setting_device_, agent_state_device_, state,
           reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

bool TagEnvironment::Step(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto action = reinterpret_cast<int *>(inputs[0]->addr);
  auto state = reinterpret_cast<float *>(outputs[0]->addr);
  auto reward = reinterpret_cast<float *>(outputs[1]->addr);
  auto done = reinterpret_cast<bool *>(outputs[2]->addr);
  auto team_reward = reinterpret_cast<float *>(workspace[0]->addr);
  StepKernelProfiling(action, state, reward, done, team_reward, reinterpret_cast<cudaStream_t>(stream_ptr));
  if (optimal_kernel_ == kBindBlock) {
    StepBindBlock(env_num_, agent_num_, game_setting_device_, agent_state_device_, action, state, reward, done,
                  reinterpret_cast<cudaStream_t>(stream_ptr));
  } else {
    StepCrossBlock(env_num_, agent_num_, game_setting_device_, agent_state_device_, action, state, reward, done,
                   team_reward, reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  return true;
}

size_t TagEnvironment::ActionSizeInBytes() {
  // Action with shape (env_num, agent_num, movement)
  return env_num_ * agent_num_ * sizeof(int);
}

size_t TagEnvironment::StateSizeInBytes() {
  // State with shape (env_num, agent_num, agent_num * position)
  return env_num_ * agent_num_ * (agent_num_ * kFeatureNum + 1) * sizeof(float);
}

size_t TagEnvironment::RewardSizeInBytes() {
  // Reward with shape (env_num, agent_num, reward)
  return env_num_ * agent_num_ * sizeof(float);
}

size_t TagEnvironment::WorkspaceSizeInBytes() {
  // Team reward with shape (env_num,)
  return sizeof(float) * env_num_;
}

size_t TagEnvironment::DoneSizeInBytes() { return env_num_ * sizeof(bool); }

void TagEnvironment::StepKernelProfiling(const int *action, float *state, float *reward, bool *done, float *team_reward,
                                         cudaStream_t stream) {
  if (!enable_profiling_) {
    return;
  }

  size_t shared_mem_size = env_num_ * agent_num_ * sizeof(float) * 2;
  if (shared_mem_size >= device::gpu::CudaCommon::GetInstance().share_memory_size()) {
    optimal_kernel_ = kCrossBlock;
    enable_profiling_ = false;
    return;
  }

  MS_LOG(INFO) << "Start Tag environment profiling.";

  // Prepare agent state for profiling.
  AgentState agent_state;
  InitAgentState(game_setting_host_.predator_num, game_setting_host_.prey_num, &agent_state);
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  AgentState *agent_state_device = static_cast<AgentState *>(allocator.AllocTensorMem(sizeof(AgentState)));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(agent_state_device, &agent_state, sizeof(AgentState), cudaMemcpyHostToDevice, stream),
    "cudaMemcpy failed.");
  AgentStateCopy(env_num_, agent_num_, agent_state_device, agent_state_device_, stream);

  // Warmup
  StepBindBlock(env_num_, agent_num_, game_setting_device_, agent_state_device, action, state, reward, done, stream);
  StepCrossBlock(env_num_, agent_num_, game_setting_device_, agent_state_device, action, state, reward, done,
                 team_reward, stream);

  // Collect profiling info
  device::gpu::CudaDeviceStream start = nullptr;
  device::gpu::CudaDeviceStream end = nullptr;
  float bind_cost = 0;
  float cross_cost = 0;
  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::CreateEvent(&start), "Failed to create event.");
  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::CreateEvent(&end), "Failed to create event.");

  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::RecordEvent(start, stream), "Failed to record event to stream.");
  StepBindBlock(env_num_, agent_num_, game_setting_device_, agent_state_device, action, state, reward, done, stream);
  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::RecordEvent(end, stream), "Failed to record event to stream.");
  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::SyncEvent(start), "Failed to sync event.");
  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::SyncEvent(end), "Failed to sync event.");
  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::ElapsedTime(&bind_cost, start, end), "Record time failed.");

  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::RecordEvent(start, stream), "Failed to record event to stream.");
  StepCrossBlock(env_num_, agent_num_, game_setting_device_, agent_state_device, action, state, reward, done,
                 team_reward, stream);
  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::RecordEvent(end, stream), "Failed to record event to stream.");
  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::SyncEvent(start), "Failed to sync event.");
  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::SyncEvent(end), "Failed to sync event.");
  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::ElapsedTime(&cross_cost, start, end), "Record time failed.");

  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::DestroyEvent(start), "Failed to destroy event.");
  CHECK_OP_RET_WITH_EXCEPT(device::gpu::CudaDriver::DestroyEvent(end), "Failed to destroy event.");

  // Select optimal kernel
  optimal_kernel_ = bind_cost < cross_cost ? kBindBlock : kCrossBlock;

  // Free tmp agent state
  allocator.FreeTensorMem(agent_state_device);
  FinalizeAgentState(agent_state);

  MS_LOG(INFO) << "Tag environment profiling finish. Bind cost: " << bind_cost << ", cross cost: " << cross_cost;
  enable_profiling_ = false;
}
}  // namespace kernel
}  // namespace mindspore
