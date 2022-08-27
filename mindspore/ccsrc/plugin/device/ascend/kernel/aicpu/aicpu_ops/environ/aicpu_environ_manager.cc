/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "environ/aicpu_environ_manager.h"
#include <string>

namespace aicpu {
constexpr auto kScalarTensorShapeDim = 1;
constexpr auto kScalarTensorShapeSize = 1;
constexpr auto kEnvValueTypeAttr = "value_type";

int64_t EnvironMgr::Create() {
  std::unique_lock<std::mutex> lock(mutex);
  if (env_handles_count_ >= INT64_MAX) {
    AICPU_LOGE(" The handles number:%d is out of range: ", env_handles_count_);
    return kAicpuKernelStateInvalid;
  }
  int64_t ret_handle = ++env_handles_count_;
  auto env = std::make_shared<Environ>(ret_handle);
  AICPU_CHECK_NULLPTR(env, kAicpuKernelStateInvalid, "env is null.");
  envs_[ret_handle] = env;

  return ret_handle;
}

EnvironPtr EnvironMgr::Get(int64_t handle) {
  std::unique_lock<std::mutex> lock(mutex);
  const auto &envIter = envs_.find(handle);
  if (envIter != envs_.end()) {
    auto &result = envIter->second;
    return result;
  }
  return nullptr;
}

void EnvironMgr::Clear() {
  std::unique_lock<std::mutex> lock(mutex);
  for (auto &env : envs_) {
    AICPU_CHECK_NULLPTR_VOID(env.second, "env is null.")
    env.second->Clear();
  }
  envs_.clear();
}

bool EnvironMgr::IsScalarTensor(const aicpuops::Tensor &tensor) const {
  aicpuops::TensorShape shape = tensor.tensor_shape();
  if (shape.dim_size() == 0) {
    AICPU_LOGD("The shape is empty.");
    return true;
  }

  if ((shape.dim_size() == kScalarTensorShapeDim) && (shape.dim(aicpu::kIndex0).size() == kScalarTensorShapeSize)) {
    AICPU_LOGD("The tensor is scalar.");
    return true;
  }
  return false;
}

bool EnvironMgr::CheckEnvInput(const aicpuops::NodeDef &node_def) const {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> nodedef_map = node_def.attrs();
  auto value_type_attr = nodedef_map[kEnvValueTypeAttr].i();
  if ((value_type_attr != kObjectTypeTensorType) && (value_type_attr != kObjectTypeEnvType)) {
    AICPU_LOGE("The value type is not supported: [%d]", value_type_attr);
    return false;
  }

  // Check the input handle.
  if (!IsScalarTensor(node_def.inputs(aicpu::kIndex0))) {
    AICPU_LOGE("The input handle checks invalid.");
    return false;
  }

  // Check the input key
  if (!IsScalarTensor(node_def.inputs(aicpu::kIndex1))) {
    AICPU_LOGE("The input key checks invalid.");
    return false;
  }

  // Check the input value
  if ((value_type_attr == kObjectTypeEnvType) && (!IsScalarTensor(node_def.inputs(aicpu::kIndex2)))) {
    AICPU_LOGE("The input value checks invalid.");
    return false;
  }

  return true;
}
}  // namespace aicpu
