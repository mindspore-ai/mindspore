/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/runtime/actor/control_flow/switch_actor.h"
#include "backend/ge_backend/runtime/actor/control_flow/entrance_actor.h"
#include "backend/ge_backend/runtime/actor/output_actor.h"
#include "utils/log_adapter.h"
#include "ir/tensor_new.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
constexpr size_t kMaxSwitchCondSize = 8;
constexpr size_t kSwitchDefaultOutputNum = 1;

SwitchActor::SwitchActor(const std::string &name, const AID &memory_manager_aid,
                         const std::vector<KernelWithIndex> &parameters, const AnfNodePtr &node)
    : ControlActor(name, KernelTransformType::kSwitchActor, memory_manager_aid, parameters, node) {
  output_data_by_output_index_.resize(kSwitchDefaultOutputNum);
}

void SwitchActor::FetchInput(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  // Call the base class interface to get input data and input partial.
  ControlActor::FetchInput(context);

  MS_LOG(INFO) << "Sync stream in the condition switch.";
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  size_t index = GetIndex(context);
  if (IsSkippedLaunch()) {
    // dry run switch index is always 0.
    index = input_partials_.size() - kSwitchCondPos - 1;
  }
  if (!output_partial_arrows_.empty()) {
    if (index + kSwitchCondPos >= input_partials_.size()) {
      MS_EXCEPTION(IndexError) << "Given index " << std::to_string(index)
                               << " out of range. Please make sure the value of index in ["
                               << std::to_string(1 - SizeToInt(input_partials_.size())) << ", "
                               << std::to_string(input_partials_.size() - 1) + "), and the type is int32.";
    }
    MS_EXCEPTION_IF_NULL(input_partials_[index + kSwitchCondPos]);
    auto func_graph = input_partials_[index + kSwitchCondPos]->func_graph_;
    MS_EXCEPTION_IF_NULL(func_graph);
    input_partials_[0] = input_partials_[index + kSwitchCondPos];
  }

  for (auto &output_data : output_data_by_output_index_[0]) {
    MS_EXCEPTION_IF_NULL(output_data);
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[index + kSwitchCondPos]);
    output_data->data_ = input_kernel_tensors_[index + kSwitchCondPos];
  }
}

size_t SwitchActor::GetIndex(const OpContext<KernelTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(input_kernel_tensors_[0]);
  TypeId type_id = input_kernel_tensors_[0]->dtype_id();
  size_t size = abstract::TypeIdSize(type_id);
  if (size > sizeof(int64_t)) {
    MS_LOG(ERROR) << "Index must be Int type.";
    return 0;
  }

  int64_t index = 0;
  char buf[kMaxSwitchCondSize] = {0};
  ShapeVector host_shape;
  auto tensor = tensor::from_buffer(type_id, host_shape, buf, size);
  if (!SyncCopy(input_kernel_tensors_[0].get(), tensor.get(), kDefaultStreamIndex)) {
    MS_LOG(ERROR) << GetAID().Name() << " get index from device address failed, type id:" << type_id;
    return 0;
  }

  if (type_id == TypeId::kNumberTypeInt32) {
    index = static_cast<int64_t>((static_cast<int32_t *>(static_cast<void *>(buf)))[0]);
    MS_LOG(DEBUG) << "Index:" << index << " for actor:" << GetAID();
  } else if (type_id == TypeId::kNumberTypeInt64) {
    index = (static_cast<int64_t *>(static_cast<void *>(buf)))[0];
    MS_LOG(DEBUG) << "Index:" << index << " for actor:" << GetAID();
  } else if (type_id == TypeId::kNumberTypeBool) {
    bool cond = (static_cast<bool *>(static_cast<void *>(buf)))[0];
    if (cond) {
      index = 1;
    }
    MS_LOG(DEBUG) << "Condition:" << cond << ", index:" << index << " for actor:" << GetAID();
  } else {
    MS_LOG(ERROR) << "Index must be Int type.";
    return 0;
  }

  // SwitchLayer node support negative index range [-size, -1].
  if (index < 0) {
    int64_t positive_index = index + SizeToLong(formal_parameters_.size() - 1);
    if (positive_index < 0) {
      MS_EXCEPTION(IndexError) << "Given index " << std::to_string(index)
                               << " out of range. Please make sure the value of index in ["
                               << std::to_string(1 - SizeToInt(input_partials_.size())) << ", "
                               << std::to_string(input_partials_.size() - 1) + "), and the type is int32.";
    }
    index = positive_index;
  }
  return LongToSize(index);
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
