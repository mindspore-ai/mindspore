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

#include "plugin/device/ascend/kernel/opapi/aclnn/chunk_aclnn_kernel.h"
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
void ChunkAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  const auto &input_shape = inputs[kIndex0]->GetShape()->GetShapeVector();
  auto chunks = transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  dims_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  if (dims_ < 0) {
    dims_ += SizeToLong(input_shape.size());
  }
  int64_t dim_size = input_shape[dims_];
  split_size_ = (dim_size + chunks - 1) / chunks;
  if (split_size_ == 0 && dim_size == 0) {
    op_type_ = "aclnnSplitWithSize";
    split_sizes_ = std::vector<int64_t>(chunks, 0);
    GetWorkspaceForResize(inputs[kIndex0], split_sizes_, dims_, outputs);
  } else {
    op_type_ = "aclnnSplitTensor";
    GetWorkspaceForResize(inputs[kIndex0], split_size_, dims_, outputs);
  }
}

bool ChunkAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (op_type_ == "aclnnSplitWithSize") {
    RunOp(stream_ptr, workspace, inputs[kIndex0], split_sizes_, dims_, outputs);
  } else {
    RunOp(stream_ptr, workspace, inputs[kIndex0], split_size_, dims_, outputs);
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Chunk, ChunkAscend);
}  // namespace kernel
}  // namespace mindspore
