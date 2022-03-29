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

#include "plugin/device/cpu/kernel/bucketize_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kOutputNum = 1;
const size_t kInputNum = 1;
const size_t kParallelDataNumSameShape = 64 * 1024;
const size_t kParallelDataNumSameShapeMid = 35 * 1024;
}  // namespace

void BucketizeCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  boundaries_ = common::AnfAlgo::GetNodeAttr<std::vector<float>>(kernel_node, "boundaries");
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

bool BucketizeCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> & /* workspace */,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  if (dtype_ != kNumberTypeInt32 && dtype_ != kNumberTypeInt64 && dtype_ != kNumberTypeFloat32 &&
      dtype_ != kNumberTypeFloat64) {
    MS_LOG(EXCEPTION) << "Input data type must int32 or int64 or float32 or float64, but got data type." << dtype_;
    return false;
  }
  size_t input_sizes = input_shape_.size();
  size_t output_sizes = output_shape_.size();
  if (input_sizes != output_sizes) {
    MS_LOG(EXCEPTION) << "The tensor shape of input need be same with output.";
    return false;
  }
  // BucketizeCompute(inputs, outputs);
  switch (dtype_) {
    case kNumberTypeInt32:
      return BucketizeCompute<int32_t>(inputs, outputs);
    case kNumberTypeInt64:
      return BucketizeCompute<int64_t>(inputs, outputs);
    case kNumberTypeFloat32:
      return BucketizeCompute<float>(inputs, outputs);
    case kNumberTypeFloat64:
      return BucketizeCompute<double>(inputs, outputs);
    default:
      MS_LOG(ERROR) << "Unsupported data type.";
  }
  return true;
}

template <typename T>
bool BucketizeCpuKernelMod::BucketizeCompute(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &outputs) {
  auto input_data = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_data = reinterpret_cast<int32_t *>(outputs[0]->addr);
  size_t data_num_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());
  std::vector<float> boundaries_data = boundaries_;
  std::sort(boundaries_data.begin(), boundaries_data.end());
  if (data_num_ >= kParallelDataNumSameShape) {
    auto sharder_bucketize = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        auto first_bigger_it = std::upper_bound(boundaries_data.begin(), boundaries_data.end(), input_data[i]);
        output_data[i] = first_bigger_it - boundaries_data.begin();
      }
    };
    ParallelLaunchAutoSearch(sharder_bucketize, data_num_, this, &parallel_search_info_);
  } else {
    for (size_t i = 0; i < data_num_; i++) {
      auto first_bigger_it = std::upper_bound(boundaries_data.begin(), boundaries_data.end(), input_data[i]);
      output_data[i] = first_bigger_it - boundaries_data.begin();
    }
  }
  return true;
}

std::vector<KernelAttr> BucketizeCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Bucketize, BucketizeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
