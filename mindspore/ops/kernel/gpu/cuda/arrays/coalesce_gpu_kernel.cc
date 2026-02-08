/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/coalesce_gpu_kernel.h"
#include <utility>
#include <map>
namespace mindspore {
namespace kernel {
namespace {
template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateCoalesceKernelPtr(const std::string &kernel_name,
                                                                       const uint32_t &device_id) {
  return std::make_unique<cukernel::CoalesceHelperGpuKernel<T>>(kernel_name, device_id);
}
using CoalescePtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, CoalescePtrCreatorFunc>> kernel_attr = {{KernelAttr()
                                                                                   .AddInputAttr(kNumberTypeInt64)
                                                                                   .AddInputAttr(kNumberTypeFloat32)
                                                                                   .AddInputAttr(kNumberTypeInt64)
                                                                                   .AddOutputAttr(kNumberTypeInt64)
                                                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                                                   .AddOutputAttr(kNumberTypeInt64),
                                                                                 CreateCoalesceKernelPtr<float>},
                                                                                {KernelAttr()
                                                                                   .AddInputAttr(kNumberTypeInt64)
                                                                                   .AddInputAttr(kNumberTypeFloat16)
                                                                                   .AddInputAttr(kNumberTypeInt64)
                                                                                   .AddOutputAttr(kNumberTypeInt64)
                                                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                                                   .AddOutputAttr(kNumberTypeInt64),
                                                                                 CreateCoalesceKernelPtr<half>},
                                                                                {KernelAttr()
                                                                                   .AddInputAttr(kNumberTypeInt64)
                                                                                   .AddInputAttr(kNumberTypeFloat64)
                                                                                   .AddInputAttr(kNumberTypeInt64)
                                                                                   .AddOutputAttr(kNumberTypeInt64)
                                                                                   .AddOutputAttr(kNumberTypeFloat64)
                                                                                   .AddOutputAttr(kNumberTypeInt64),
                                                                                 CreateCoalesceKernelPtr<double>}};
}  // namespace

bool CoalesceGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &workspace,
                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  int ret = helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr);
  int FALSE_1 = 1;
  int FALSE_2 = 2;
  int FALSE_3 = 3;
  if (ret == FALSE_1) {
    MS_EXCEPTION(ValueError) << "For coalesce, indices cannot be less than 0";
  }
  if (ret == FALSE_2) {
    MS_EXCEPTION(ValueError) << "For coalesce, shape must be greater than 0";
  }
  if (ret == FALSE_3) {
    MS_EXCEPTION(ValueError) << "For coalesce, indices must be less than shape of the corresponding dimension";
  }
  return true;
}

bool CoalesceGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto [is_match, index] = MatchKernelAttr(GetKernelAttrFromTensors(inputs, outputs), GetOpSupport());
  if (!is_match) {
    return false;
  }
  helper_ptr_ = kernel_attr[index].second(kernel_name_, device_id_);
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  return true;
}

int CoalesceGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  std::vector<std::vector<int64_t>> input_shapes;
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
    input_shapes.emplace_back(input_shape);
  }
  auto input_indices_shape = inputs[kIndex0]->GetShapeVector();
  auto input_shape_shape = inputs[kIndex2]->GetShapeVector();
  // push back the max shape to output_size_list_
  auto output_max_indices_shape = input_indices_shape;
  std::vector<int64_t> output_max_values_shape = {input_indices_shape[1]};
  std::vector<std::vector<int64_t>> output_shapes;
  output_shapes.emplace_back(output_max_indices_shape);
  output_shapes.emplace_back(output_max_values_shape);
  output_shapes.emplace_back(input_shape_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

void CoalesceGpuKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "Coalesce cudaStreamSynchronized failed");
  auto dyn_out = helper_ptr_->GetOutputTensorInfo();
  size_t output_num = outputs.size();
  constexpr auto expect_out_num = 3;
  if (output_num != expect_out_num) {
    MS_LOG(EXCEPTION) << "Unexpected output num: " << output_num;
  }

  // update output0
  auto out0_shape = outputs[0]->GetShapeVector();
  constexpr auto kOut0Rank = 2;
  if (out0_shape.size() < kOut0Rank) {
    MS_LOG(EXCEPTION) << "Unexpected output0 shape size: " << out0_shape.size()
                      << ", shape: " << mindspore::ToString(out0_shape);
  }
  outputs[0]->set_size(LongToSize(std::accumulate(
    out0_shape.begin(), out0_shape.end(), UnitSizeInBytes(outputs[0]->dtype_id()), std::multiplies<int64_t>())));
  out0_shape[1] = dyn_out.shapes[0][0];
  outputs[0]->SetShapeVector(out0_shape);

  // update output1
  auto out1_shape = outputs[1]->GetShapeVector();
  if (out1_shape.empty()) {
    MS_LOG(EXCEPTION) << "Unexpected output1 shape size: " << out1_shape.size()
                      << ", shape: " << mindspore::ToString(out1_shape);
  }
  out1_shape[0] = dyn_out.shapes[0][0];
  outputs[1]->set_size(LongToSize(std::accumulate(
    out1_shape.begin(), out1_shape.end(), UnitSizeInBytes(outputs[1]->dtype_id()), std::multiplies<int64_t>())));
  outputs[1]->SetShapeVector(out1_shape);

  // cal output3 size
  auto out2_shape = outputs[2]->GetShapeVector();
  outputs[2]->set_size(LongToSize(std::accumulate(
    out2_shape.begin(), out2_shape.end(), UnitSizeInBytes(outputs[2]->dtype_id()), std::multiplies<int64_t>())));
}

std::vector<KernelAttr> CoalesceGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CoalescePtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Coalesce, CoalesceGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
