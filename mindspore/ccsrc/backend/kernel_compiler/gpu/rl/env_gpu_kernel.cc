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
#include "backend/kernel_compiler/gpu/rl/env_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kEnvTypeName = "name";
constexpr auto kHandleAttrName = "handle";
}  // namespace

const std::vector<size_t> &EnvCreateKernel::GetInputSizeList() const { return input_size_list_; }
const std::vector<size_t> &EnvCreateKernel::GetOutputSizeList() const { return output_size_list_; }
const std::vector<size_t> &EnvCreateKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool EnvCreateKernel::Init(const CNodePtr &cnode) {
  const auto &name = AnfAlgo::GetNodeAttr<std::string>(cnode, kEnvTypeName);
  std::tie(handle_, env_) = EnvironmentFactory::GetInstance().Create(name);
  MS_EXCEPTION_IF_NULL(env_);
  env_->Init(cnode, nullptr);
  InitSizeLists();
  return true;
}

void EnvCreateKernel::InitSizeLists() { output_size_list_.push_back(sizeof(handle_)); }

bool EnvCreateKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                             const std::vector<AddressPtr> &outputs, void *stream) {
  auto handle = GetDeviceAddress<int64_t>(outputs, 0);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(handle, &handle_, sizeof(handle_), cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)),
    "cudaMemcpy failed.");
  return true;
}

const std::vector<size_t> &EnvResetKernel::GetInputSizeList() const { return input_size_list_; }
const std::vector<size_t> &EnvResetKernel::GetOutputSizeList() const { return output_size_list_; }
const std::vector<size_t> &EnvResetKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool EnvResetKernel::Init(const CNodePtr &cnode) {
  handle_ = AnfAlgo::GetNodeAttr<int64_t>(cnode, kHandleAttrName);
  env_ = EnvironmentFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(env_);
  InitSizeLists();
  return true;
}

void EnvResetKernel::InitSizeLists() {
  output_size_list_.push_back(env_->StateSizeInBytes());
  workspace_size_list_.push_back(env_->WorkspaceSizeInBytes());
}

bool EnvResetKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                            const std::vector<AddressPtr> &outputs, void *stream) {
  return env_->Reset(inputs, workspace, outputs, stream);
}

const std::vector<size_t> &EnvStepKernel::GetInputSizeList() const { return input_size_list_; }
const std::vector<size_t> &EnvStepKernel::GetOutputSizeList() const { return output_size_list_; }
const std::vector<size_t> &EnvStepKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool EnvStepKernel::Init(const CNodePtr &cnode) {
  handle_ = AnfAlgo::GetNodeAttr<int64_t>(cnode, kHandleAttrName);
  env_ = EnvironmentFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(env_);
  InitSizeLists();
  return true;
}

void EnvStepKernel::InitSizeLists() {
  input_size_list_.push_back(env_->ActionSizeInBytes());
  output_size_list_.push_back(env_->StateSizeInBytes());
  output_size_list_.push_back(env_->RewardSizeInBytes());
  output_size_list_.push_back(env_->DoneSizeInBytes());
  workspace_size_list_.push_back(env_->WorkspaceSizeInBytes());
}

bool EnvStepKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                           const std::vector<AddressPtr> &outputs, void *stream) {
  return env_->Step(inputs, workspace, outputs, stream);
}
}  // namespace kernel
}  // namespace mindspore
