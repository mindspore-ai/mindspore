/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/runtime/actors/control_flow/switch_actor.h"
#include "backend/ms_backend/runtime/actors/control_flow/entrance_actor.h"
#include "backend/ms_backend/runtime/actors/base/output_actor.h"
#include "runtime/hardware_abstract/utils.h"
#include "include/utils/pipeline/fallback.h"
#include "utils/log_adapter.h"
#include "include/utils/python_adapter.h"
#include "mindspore/core/include/ir/tensor_new.h"

namespace mindspore {
namespace runtime {
constexpr size_t kMaxSwitchCondSize = 8;
constexpr size_t kSwitchDefaultOutputNum = 1;

SwitchActor::SwitchActor(const std::string &name, const AID &memory_manager_aid,
                         const std::vector<KernelWithIndex> &parameters, const AnfNodePtr &node)
    : ControlActor(name, KernelTransformType::kSwitchActor, memory_manager_aid, parameters, node) {
  device_contexts_.resize(parameters.size());
  output_data_by_output_index_.resize(kSwitchDefaultOutputNum);
}

void SwitchActor::FetchInput(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  // Call the base class interface to get input data and input partial.
  ControlActor::FetchInput(context);

  MS_LOG(INFO) << "Sync stream in the switch actor:" << GetAID();
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
                               << std::to_string(input_partials_.size() - 1)
                               << "), and the type is int32 for switch actor:" << GetAID();
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

  auto device_tensor = input_kernel_tensors_[0]->device_address();
  TypeId type_id = input_kernel_tensors_[0]->dtype_id();
  size_t size = abstract::TypeIdSize(type_id);
  if (size > sizeof(int64_t)) {
    MS_LOG(ERROR) << "Index must be Int type.";
    return 0;
  }

  int64_t index = 0;
  char buf[kMaxSwitchCondSize] = {0};
  ShapeVector host_shape;
  if (input_kernel_tensors_[0]->user_data() != nullptr && input_kernel_tensors_[0]->need_sync_user_data() &&
      input_kernel_tensors_[0]->user_data()->has(kernel::PyExecuteOutputUserData::key)) {
    const auto &user_data_obj =
      input_kernel_tensors_[0]->user_data()->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
    MS_EXCEPTION_IF_NULL(user_data_obj);
    const auto &obj = user_data_obj->obj;
    py::gil_scoped_acquire gil_acquire;
    if (py::isinstance<py::bool_>(obj)) {
      MS_LOG(DEBUG) << "Index:" << py::cast<bool>(obj) << " for actor:" << GetAID();
      return index = static_cast<int64_t>(py::cast<bool>(obj) ? 1 : 0);
    }
  }

  device::DeviceContextKey host_key = {device_tensor->GetDeviceType(), device_tensor->device_id()};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  if (!host_context->device_res_manager_->SyncAllStreams(false) ||
      !host_context->device_res_manager_->Copy(buf, device_tensor->GetMutablePtr(), size, device::CopyType::kD2H,
                                               device_tensor->stream_id())) {
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
}  // namespace mindspore
