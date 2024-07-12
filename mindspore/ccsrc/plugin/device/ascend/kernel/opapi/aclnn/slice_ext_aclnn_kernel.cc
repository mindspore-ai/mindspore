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
#include "plugin/device/ascend/kernel/opapi/aclnn/slice_ext_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

void SliceExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  dim_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  start_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  end_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  step_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex4]);

  shape_ = inputs[0]->GetShapeVector();
  dim_ = dim_ < 0 ? dim_ + shape_.size() : dim_;
  auto length_value = end_ - start_;
  start_ = start_ < 0 ? start_ + shape_[dim_] : start_;
  end_ = start_ + length_value;

  GetWorkspaceForResize(inputs[kIndex0], dim_, start_, end_, step_, outputs[kIndex0]);
}

bool SliceExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (start_ == end_) {
    auto output_shape = shape_;
    output_shape[dim_] = 0;
    outputs[kIndex0]->SetShapeVector(output_shape);
  } else {
    RunOp(stream_ptr, workspace, inputs[kIndex0], dim_, start_, end_, step_, outputs[kIndex0]);
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SliceExt, SliceExtAscend);
}  // namespace kernel
}  // namespace mindspore
