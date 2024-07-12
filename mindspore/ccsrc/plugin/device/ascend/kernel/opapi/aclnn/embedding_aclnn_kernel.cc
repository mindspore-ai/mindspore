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
#include "plugin/device/ascend/kernel/opapi/aclnn/embedding_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

void EmbeddingAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  GetWorkspaceForResize(inputs[kIndex1], inputs[kIndex0], outputs[kIndex0]);
  auto max_norm_opt = inputs[kIndex3]->GetValue<float>();
  if (max_norm_opt.has_value()) {
    auto max_norm = max_norm_opt.value();
    norm_type_ = static_cast<double>(transform::ConvertKernelTensor<float>(inputs[kIndex4]));
    max_norm_ = static_cast<double>(max_norm);
    SetWorkspaceForRenorm(inputs[1], inputs[0], max_norm_, norm_type_);
    do_renorm_ = true;
  }
}

bool EmbeddingAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (do_renorm_) {
    MS_EXCEPTION_IF_NULL(workspace.back());
    void *ws_addr = renorm_ws_size_ != 0 ? workspace.back()->device_ptr() : nullptr;
    transform::aclOpExecutor *executor;
    std::function<void()> release_func;
    std::tie(std::ignore, executor, release_func, std::ignore, std::ignore) = GEN_EXECUTOR_BOOST(
      embedding_renorm_name_, renorm_hash_id_, inputs[kIndex1], inputs[kIndex0], max_norm_, norm_type_);
    RUN_OP_API_ASYNC(embedding_renorm_name_, ws_addr, renorm_ws_size_, executor, stream_ptr, release_func);
  }

  RunOp(stream_ptr, workspace, inputs[kIndex1], inputs[kIndex0], outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Embedding, EmbeddingAscend);
}  // namespace kernel
}  // namespace mindspore
