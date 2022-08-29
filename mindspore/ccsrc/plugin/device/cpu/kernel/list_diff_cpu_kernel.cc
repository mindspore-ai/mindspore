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

#include "plugin/device/cpu/kernel/list_diff_cpu_kernel.h"
#include <unordered_set>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr uint32_t kListDiffInputNum = 2;
constexpr uint32_t kListDiffOutputNum = 2;

#define LIST_DIFF_COMPUTE_CASE(data_type, type)              \
  case (data_type): {                                        \
    if (idx_type_ == kNumberTypeInt32) {                     \
      result = LaunchKernel<type, int32_t>(inputs, outputs); \
    } else {                                                 \
      result = LaunchKernel<type, int64_t>(inputs, outputs); \
    }                                                        \
    break;                                                   \
  }
}  // namespace
void ListDiffCPUKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  cnode_ptr_ = kernel_node;
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  TypeId x_type = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex0);
  TypeId y_type = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex1);
  out_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, kIndex0);
  if (x_type != y_type || x_type != out_type_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input 'x', 'y' and output 'out' should be same type, but get x["
                      << TypeIdLabel(x_type) << "], y[" << TypeIdLabel(y_type) << "], out[" << TypeIdLabel(out_type_)
                      << "].";
  }
  auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex0);
  auto y_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex1);
  x_size_ = x_shape[0];
  y_size_ = y_shape[0];
  auto out_idx = common::AnfAlgo::GetNodeAttr<TypePtr>(kernel_node, kAttrOutIdx);
  MS_EXCEPTION_IF_NULL(out_idx);
  idx_type_ = out_idx->type_id();
  MS_EXCEPTION_IF_CHECK_FAIL((idx_type_ == kNumberTypeInt32 || idx_type_ == kNumberTypeInt64),
                             "attr 'out_idx' should be int32 or int64");
}

template <typename T, typename Tidx>
bool ListDiffCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto x_addr = static_cast<T *>(inputs[0]->addr);
  auto y_addr = static_cast<T *>(inputs[1]->addr);
  auto out_addr = static_cast<T *>(outputs[0]->addr);
  auto idx_addr = static_cast<Tidx *>(outputs[1]->addr);
  MS_EXCEPTION_IF_NULL(x_addr);
  MS_EXCEPTION_IF_NULL(y_addr);
  MS_EXCEPTION_IF_NULL(out_addr);
  MS_EXCEPTION_IF_NULL(idx_addr);

  std::unordered_set<T> y_set;
  y_set.reserve(y_size_);
  for (int64_t i = 0; i < y_size_; ++i) {
    (void)y_set.insert(y_addr[i]);
  }
  // Compute the size of the output.
  out_size_ = 0;
  for (int64_t i = 0; i < x_size_; ++i) {
    if (y_set.count(x_addr[i]) == 0) {
      ++out_size_;
    }
  }
  // calculate results
  for (Tidx i = 0, p = 0; i < static_cast<Tidx>(x_size_); ++i) {
    if (0 == y_set.count(x_addr[i])) {
      MS_EXCEPTION_IF_CHECK_FAIL(p < static_cast<Tidx>(out_size_),
                                 "Try to set output index failure for index out of out_size");
      out_addr[p] = x_addr[i];
      idx_addr[p] = i;
      p++;
    }
  }
  // update out
  if (!cnode_ptr_.expired()) {
    auto node_ = cnode_ptr_.lock();
    if (!node_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', cnode_ptr_(kernel_node) is expired. Error no: " << node_;
    }
    ShapeVector out_shape = {out_size_};
    ShapeVector idx_shape = {out_size_};
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(node_);
    std::vector<TypeId> dtypes(output_num);
    for (size_t i = 0; i < output_num; i++) {
      dtypes[i] = AnfAlgo::GetOutputDeviceDataType(node_, i);
    }
    common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, {out_shape, idx_shape}, node_.get());
  }
  return true;
}

bool ListDiffCPUKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                  const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kListDiffInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kListDiffOutputNum, kernel_name_);
  bool result = false;
  switch (out_type_) {
    LIST_DIFF_COMPUTE_CASE(kNumberTypeInt8, int8_t)
    LIST_DIFF_COMPUTE_CASE(kNumberTypeInt16, int16_t)
    LIST_DIFF_COMPUTE_CASE(kNumberTypeInt32, int32_t)
    LIST_DIFF_COMPUTE_CASE(kNumberTypeInt64, int64_t)
    LIST_DIFF_COMPUTE_CASE(kNumberTypeUInt8, uint8_t)
    LIST_DIFF_COMPUTE_CASE(kNumberTypeUInt16, uint16_t)
    LIST_DIFF_COMPUTE_CASE(kNumberTypeFloat16, float16)
    LIST_DIFF_COMPUTE_CASE(kNumberTypeFloat32, float)
    LIST_DIFF_COMPUTE_CASE(kNumberTypeFloat64, double)
    default: {
      MS_LOG(EXCEPTION)
        << "For [" << kernel_name_
        << "] input data type should be in [int8, int16, int32, int64, uint8, uint16, float16, float32, float64],"
        << " but get" << TypeIdLabel(out_type_) << ".";
    }
  }
  return result;
}
std::vector<KernelAttr> ListDiffCPUKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddOutputAttr(kNumberTypeUInt8)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddOutputAttr(kNumberTypeUInt16)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddOutputAttr(kNumberTypeInt8)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddOutputAttr(kNumberTypeInt16)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddOutputAttr(kNumberTypeUInt8)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddOutputAttr(kNumberTypeUInt16)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddOutputAttr(kNumberTypeInt8)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddOutputAttr(kNumberTypeInt16)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)};

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ListDiff, ListDiffCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
