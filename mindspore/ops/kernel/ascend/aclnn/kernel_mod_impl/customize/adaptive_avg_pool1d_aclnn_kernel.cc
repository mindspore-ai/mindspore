/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/adaptive_avg_pool1d_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include <utility>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/ops/include/view/expand_dims_strides_calc.h"

namespace mindspore {
namespace kernel {
namespace adaptive_avg_pool1d {
std::vector<int64_t> GetOriStrides(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<int64_t> ret(shape.size(), 1);
  int64_t strides = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides *= shape[i];
    ret[i - 1] = strides;
  }
  return ret;
}

// Expand logical shape [N, C, L] -> [N, C, 1, L] and compute matching storage info.
static void ExpandTo2DView(KernelTensor *clone_tensor, const ShapeVector &orig_shape) {
  ShapeVector expand_shape = orig_shape;
  expand_shape.insert(expand_shape.end() - 1, 1);
  auto ts_list = ops::ExpandDimsStrideCalc(clone_tensor->GetShapeVector(), GetOriStrides(orig_shape),
                                           clone_tensor->tensor_storage_info(), -2);
  clone_tensor->SetShapeVector(expand_shape);
  clone_tensor->set_tensor_storage_info(ts_list[kIndex0]);
}

void AdaptivePool1DAscend::SetParaForPool2D(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  auto in_shape = inputs[kIndex0]->GetShapeVector();
  auto output_size = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  input_kernel_tensor_ = inputs[kIndex0]->CloneKernelTensor();
  ExpandTo2DView(input_kernel_tensor_.get(), in_shape);
  auto out_shape = outputs[kIndex0]->GetShapeVector();
  out_shape_ori = out_shape;
  output_kernel_tensors_.clear();
  output_kernel_tensors_.reserve(outputs.size());
  for (auto &output : outputs) {
    auto out_clone = output->CloneKernelTensor();
    ExpandTo2DView(out_clone.get(), out_shape);
    output_kernel_tensors_.push_back(std::move(out_clone));
  }
  output_size_for_2d_ = std::vector<int64_t>{1, output_size[0]};
}

void AdaptiveAvgPool1DAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  SetParaForPool2D(inputs, outputs);
  GetWorkspaceForResize(input_kernel_tensor_.get(), output_size_for_2d_, output_kernel_tensors_[kIndex0].get());
}

bool AdaptiveAvgPool1DAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  input_kernel_tensor_->set_device_ptr(inputs[kIndex0]->device_ptr());
  for (size_t i = 0; i < output_kernel_tensors_.size(); ++i) {
    output_kernel_tensors_[i]->set_device_ptr(outputs[i]->device_ptr());
  }
  RunOp(stream_ptr, workspace, input_kernel_tensor_.get(), output_size_for_2d_, output_kernel_tensors_[kIndex0].get());
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AdaptiveAvgPool1D, AdaptiveAvgPool1DAscend);
}  // namespace adaptive_avg_pool1d
}  // namespace kernel
}  // namespace mindspore
