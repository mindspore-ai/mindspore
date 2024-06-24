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
  auto dim = transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  auto start = transform::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  auto end = transform::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  auto step = transform::ConvertKernelTensor<int64_t>(inputs[kIndex4]);

  shape_ = inputs[0]->GetShapeVector();
  dim = dim < 0 ? dim + shape_.size() : dim;
  auto length_value = end - start;
  start = start < 0 ? start + shape_[dim] : start;
  end = start + length_value;

  GetWorkspaceForResize(inputs[kIndex0], dim, start, end, step, outputs[kIndex0]);
}

bool SliceExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto dim = transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  auto start = transform::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  auto end = transform::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  auto step = transform::ConvertKernelTensor<int64_t>(inputs[kIndex4]);

  auto length_value = end - start;
  dim = dim < 0 ? dim + shape_.size() : dim;
  start = start < 0 ? start + shape_[dim] : start;
  end = start + length_value;

  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], dim, start, end, step, outputs[kIndex0]));
  if (start == end) {
    auto output_shape = shape_;
    output_shape[dim] = 0;
    outputs[kIndex0]->SetShapeVector(output_shape);
  } else {
    RunOp(stream_ptr, workspace);
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SliceExt, SliceExtAscend);
}  // namespace kernel
}  // namespace mindspore
