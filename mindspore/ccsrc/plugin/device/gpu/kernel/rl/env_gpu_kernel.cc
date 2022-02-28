/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/rl/env_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kEnvTypeName = "name";
constexpr auto kHandleAttrName = "handle";
}  // namespace

bool EnvCreateKernelMod::Init(const CNodePtr &cnode) {
  const auto &name = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kEnvTypeName);
  std::tie(handle_, env_) = EnvironmentFactory::GetInstance().Create(name);
  MS_EXCEPTION_IF_NULL(env_);
  env_->Init(cnode, nullptr);
  InitSizeLists();
  return true;
}

void EnvCreateKernelMod::InitSizeLists() { output_size_list_.push_back(sizeof(handle_)); }

bool EnvCreateKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &outputs, void *stream) {
  auto handle = GetDeviceAddress<int64_t>(outputs, 0);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(handle, &handle_, sizeof(handle_), cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)),
    "cudaMemcpy failed.");
  return true;
}

bool EnvResetKernelMod::Init(const CNodePtr &cnode) {
  handle_ = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kHandleAttrName);
  kernel_node_ = cnode;
  env_ = EnvironmentFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(env_);
  InitSizeLists();
  return true;
}

void EnvResetKernelMod::InitSizeLists() {
  output_size_list_.push_back(env_->StateSizeInBytes());
  workspace_size_list_.push_back(env_->WorkspaceSizeInBytes());
}

bool EnvResetKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs, void *stream) {
  return env_->Reset(inputs, workspace, outputs, stream);
}

bool EnvStepKernelMod::Init(const CNodePtr &cnode) {
  handle_ = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kHandleAttrName);
  env_ = EnvironmentFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(env_);
  InitSizeLists();
  return true;
}

void EnvStepKernelMod::InitSizeLists() {
  input_size_list_.push_back(env_->ActionSizeInBytes());
  output_size_list_.push_back(env_->StateSizeInBytes());
  output_size_list_.push_back(env_->RewardSizeInBytes());
  output_size_list_.push_back(env_->DoneSizeInBytes());
  workspace_size_list_.push_back(env_->WorkspaceSizeInBytes());
}

bool EnvStepKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                              const std::vector<AddressPtr> &outputs, void *stream) {
  return env_->Step(inputs, workspace, outputs, stream);
}
}  // namespace kernel
}  // namespace mindspore
