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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TAG_ENV_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TAG_ENV_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/rl/environment_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/rl/tag_env_impl.cuh"

namespace mindspore {
namespace kernel {
// Class for Tag environment.
// Tag is a multi-agent reinforcement learning environment.
// It is suppose that the predators learn cooperative stategy (for example surround) to catch the prey.
// The predators try to catch the prey. All of predators will get same reward when they catch the prey.
// Tag environment uses discrete action space(still, left, right, up, down), and result observations
// including agent location information. The tag environment supports multiple instances to speed sample collection.
// It also supports auto performance profiling and cuda-kernel selection.
class TagEnvironment : public Environment {
 public:
  TagEnvironment() = default;
  ~TagEnvironment();

  // Init environment. Parse environment setting, create device memory for environment setting and agent state etc.
  bool Init(const CNodePtr &cnode, void *stream_ptr) override;
  // Reset environment state include agent location and time step.
  bool Reset(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
             const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  // Execute time step.
  bool Step(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
            const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  size_t ActionSizeInBytes() override;
  size_t StateSizeInBytes() override;
  size_t RewardSizeInBytes() override;
  size_t DoneSizeInBytes() override;
  size_t WorkspaceSizeInBytes() override;

 private:
  // The GameSetting and AgentState are used in C-like compiling environment, use C style resource managerment.
  bool InitGameSetting(const CNodePtr &cnode, GameSetting *setting_host);
  bool InitAgentState(AgentState *agent_state);
  bool FinalizeAgentState(const AgentState &agent_state);

  int env_num_ = 0;
  int agent_num_ = 0;
  GameSetting game_setting_host_;
  GameSetting *game_setting_device_ = nullptr;
  AgentState agent_state_host_;
  AgentState *agent_state_device_ = nullptr;

  enum StepKernelType { kBindBlock = 0, kCrossBlock };
  void StepKernelProfiling(const int *action, float *state, float *reward, bool *done, float *team_reward,
                           int *distance, cudaStream_t stream);
  int enable_profiling_ = true;
  StepKernelType optimal_kernel_ = kBindBlock;
};

MS_REG_GPU_ENV(Tag, TagEnvironment)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TAG_ENV_KERNEL_H_
