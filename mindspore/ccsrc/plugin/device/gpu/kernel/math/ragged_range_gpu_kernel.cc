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

#include "plugin/device/gpu/kernel/math/ragged_range_gpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/ragged_range_impl.cuh"
#include "mindspore/core/ops/ragged_range.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
template <typename T, typename TSPLITS>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateRaggedRangeKernelPtr(const std::string &kernel_name,
                                                                          const uint32_t &device_id) {
  return std::make_unique<cukernel::RaggedRangeHelperGpuKernel<T, TSPLITS>>(kernel_name, device_id);
}
using RaggedRangePtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, RaggedRangePtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateRaggedRangeKernelPtr<int32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateRaggedRangeKernelPtr<int32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt64),
   &CreateRaggedRangeKernelPtr<int64_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &CreateRaggedRangeKernelPtr<int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32),
   &CreateRaggedRangeKernelPtr<float, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &CreateRaggedRangeKernelPtr<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64),
   &CreateRaggedRangeKernelPtr<double, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &CreateRaggedRangeKernelPtr<double, int64_t>},
};
}  // namespace

bool RaggedRangeGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool RaggedRangeGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::RaggedRange>(base_operator);
  std::string kernel_name = kernel_ptr->name();

  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name, device_id_));

  return true;
}

int RaggedRangeGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &others) {
  constexpr size_t startsIdx = 0;
  constexpr size_t limitsIdx = 1;
  constexpr size_t deltasIdx = 2;
  constexpr size_t nestedSplitsIdx = 0;
  constexpr size_t denseValuesIdx = 1;
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  for (const auto &output : outputs) {
    auto output_shape = output->GetShapeVector();
    if (!IsValidShape(output_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  auto starts_shape = inputs[startsIdx]->GetShapeVector();
  auto starts_type = inputs[startsIdx]->GetDtype();
  size_t starts_dim = starts_shape.size();
  auto limits_shape = inputs[limitsIdx]->GetShapeVector();
  auto limits_type = inputs[limitsIdx]->GetDtype();
  size_t limits_dim = limits_shape.size();
  auto deltas_shape = inputs[deltasIdx]->GetShapeVector();
  auto deltas_type = inputs[deltasIdx]->GetDtype();
  size_t deltas_dim = deltas_shape.size();
  if (starts_dim > 1) {
    MS_LOG(EXCEPTION) << "For RaggedRange, the dimension of RaggedRange input starts must be less than 2, but got "
                      << starts_dim << ".";
  }
  if (limits_dim > 1) {
    MS_LOG(EXCEPTION) << "For RaggedRange, the dimension of RaggedRange input limits must be less than 2, but got "
                      << limits_dim << ".";
  }
  if (deltas_dim > 1) {
    MS_LOG(EXCEPTION) << "For RaggedRange, the dimension of RaggedRange input deltas must be less than 2, but got "
                      << deltas_dim << ".";
  }
  if (starts_dim != limits_dim || starts_dim != deltas_dim || limits_dim != deltas_dim) {
    MS_LOG(EXCEPTION) << "For RaggedRange, starts, limits, and deltas must have the same shape"
                      << ", but got starts (" << starts_dim << ",)"
                      << ", limits (" << limits_dim << ",)"
                      << ", deltas (" << deltas_dim << ",).";
  }
  if (starts_type != limits_type || starts_type != deltas_type || limits_type != deltas_type) {
    MS_LOG(EXCEPTION) << "For RaggedRange, starts, limits, and deltas must have the same type, "
                      << "but got starts " << starts_type << ", limits " << limits_type << ", deltas " << deltas_type
                      << ".";
  }

  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> starts_shape_vector = inputs[startsIdx]->GetShapeVector();
  std::vector<int64_t> limits_shape_vector = inputs[limitsIdx]->GetShapeVector();
  std::vector<int64_t> deltas_shape_vector = inputs[deltasIdx]->GetShapeVector();
  std::vector<int64_t> rt_nested_splits_shape = outputs[nestedSplitsIdx]->GetShapeVector();
  std::vector<int64_t> rt_dense_values_shape = outputs[denseValuesIdx]->GetShapeVector();

  input_shapes.emplace_back(starts_shape_vector);
  input_shapes.emplace_back(limits_shape_vector);
  input_shapes.emplace_back(deltas_shape_vector);

  output_shapes.emplace_back(rt_nested_splits_shape);
  output_shapes.emplace_back(rt_dense_values_shape);

  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  input_size_list_ = helper_ptr_->GetInputSizeList();
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> RaggedRangeGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RaggedRangePtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, RaggedRange, RaggedRangeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
