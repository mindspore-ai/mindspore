/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DEBUG_STATISTIC_KERNEL_H_
#define MINDSPORE_CCSRC_DEBUG_STATISTIC_KERNEL_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "include/common/debug/common.h"
#include "ir/dtype/tensor_type.h"
#include "mindrt/include/async/async.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "runtime/hardware/device_context.h"
#include "utils/log_adapter.h"

namespace mindspore {

namespace datadump {
using device::DeviceAddressPtr;
using kernel::KernelTensor;
using mindspore::device::DeviceContext;
using TensorPtr = tensor::TensorPtr;

class StatisticKernel {
 public:
  StatisticKernel(const DeviceContext *device_context, string kernel_name, const std::set<TypeId> &dtype_id)
      : device_context_(device_context), kernel_name_(kernel_name), supported_dtype_(dtype_id) {
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context_->device_res_manager_);
    MS_LOG(DEBUG) << "Statistic kernel mod " << kernel_name_ << " construct.";
    kernel_mod_ = device_context_->GetKernelExecutor(false)->CreateKernelMod(kernel_name);
    MS_EXCEPTION_IF_NULL(kernel_mod_);
  }
  DeviceAddressPtr GenerateDeviceAddress(const uint32_t &stream_id, const size_t &mem_size, const TypeId &dtype_id,
                                         const ShapeVector &shape, const ValuePtr &value = nullptr);
  DeviceAddressPtr GetWorkSpaceDeviceAddress(const uint32_t stream_id, const vector<KernelTensor *> &inputs,
                                             const vector<KernelTensor *> &outputs);
  DeviceAddressPtr GetOutputDeviceAddress(const uint32_t stream_id, TypeId dtype_id);
  TensorPtr LaunchKernel(KernelTensor *input);
  TensorPtr SyncDeviceToHostTensor(DeviceAddressPtr device_addr);
  bool CheckDataType(const TypeId &dtype_id) { return supported_dtype_.find(dtype_id) != supported_dtype_.end(); }

 protected:
  const DeviceContext *device_context_{nullptr};
  string kernel_name_;
  kernel::KernelModPtr kernel_mod_;
  std::set<TypeId> supported_dtype_;
};

class DimStatisticKernel : public StatisticKernel {
 public:
  explicit DimStatisticKernel(const DeviceContext *device_context, string kernel_name, const std::set<TypeId> &dtype_id)
      : StatisticKernel(device_context, kernel_name, dtype_id) {}
  TensorPtr LaunchKernel(KernelTensor *input);
  TensorPtr Launch(vector<KernelTensor *> inputs, DeviceAddressPtr output_addr, uint32_t stream_id);
  DeviceAddressPtr GetAxisDeviceAddress(const uint32_t stream_id, size_t dim);
  DeviceAddressPtr GetKeepDimsDeviceAddress(const uint32_t stream_id);
  DeviceAddressPtr GetDtypeDeviceAddress(const uint32_t stream_id, const TypeId &);
};

class MeanStatisticKernel : public DimStatisticKernel {
 public:
  explicit MeanStatisticKernel(const DeviceContext *device_context, const std::set<TypeId> &dtype_id)
      : DimStatisticKernel(device_context, ops::kNameMeanExt, dtype_id) {}
};

class NormStatisticKernel : public DimStatisticKernel {
 public:
  explicit NormStatisticKernel(const DeviceContext *device_context, const std::set<TypeId> &dtype_id)
      : DimStatisticKernel(device_context, ops::kNameNorm, dtype_id) {}
  TensorPtr LaunchKernel(KernelTensor *input);
  DeviceAddressPtr GetScalar(const uint32_t stream_id, float scalar = 2.0);
};

TensorPtr CalL2Norm(const DeviceContext *device_context, KernelTensor *input);
TensorPtr CalMax(const DeviceContext *device_context, KernelTensor *input);
TensorPtr CalMin(const DeviceContext *device_context, KernelTensor *input);
TensorPtr CalMean(const DeviceContext *device_context, KernelTensor *input);
TensorPtr CalStatistic(const std::string &stat_name, const DeviceContext *device_context, KernelTensor *input);

}  // namespace datadump

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_STATISTIC_KERNEL_H_
