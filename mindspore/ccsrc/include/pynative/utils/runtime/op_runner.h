/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_OP_RUNNER_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_OP_RUNNER_H_

#include <vector>
#include <string>
#include "include/utils/pynative/op_runner_info.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/pynative/utils/runtime/op_compiler.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore::runtime {
class PYNATIVE_UTILS_EXPORT OpRunner {
 public:
  // Update Tensor or input node DeviceAddress before PyNative async running.
  static void UpdateDeviceAddress(const KernelGraphPtr &graph,
                                  const std::vector<tensor::TensorPtr> &tensors_without_value_mask,
                                  const device::DeviceContext *device_context, bool is_sync, size_t stream_id);

  static void RunSingleOpGraph(const session::BackendOpRunInfoPtr &op_run_info,
                               const OpCompilerInfoPtr &op_compiler_info,
                               const std::vector<tensor::TensorPtr> &input_tensors);

  static std::vector<tensor::TensorPtr> GetTensorWithoutValueMask(const session::BackendOpRunInfoPtr &op_run_info);
  static void LaunchKernelTask(const runtime::KernelTaskType &task_type, DeviceContext *device_context,
                               const tensor::TensorPtrList &input_tensors, const tensor::TensorPtrList &output_tensors,
                               size_t stream_id);
  static DeviceContext *GetDeviceContext(device::DeviceType device_type);
  static void ChildAfterFork();
};

class PYNATIVE_UTILS_EXPORT DynamicOpRunner {
 public:
  static void UpdateInputDeviceAddress(const OpCompilerInfoPtr &op_compiler_info,
                                       const std::vector<tensor::TensorPtr> &input_tensors, bool is_sync,
                                       size_t stream_id);
  static void RunSingleOpGraph(const session::BackendOpRunInfoPtr &op_run_info,
                               const OpCompilerInfoPtr &op_compiler_info,
                               const std::vector<tensor::TensorPtr> &input_tensors);
  static void CopyHostToDevice(const OpCompilerInfoPtr &op_compiler_info,
                               const std::vector<tensor::TensorPtr> &input_tensors);
};
}  // namespace mindspore::runtime
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_OP_RUNNER_H_
