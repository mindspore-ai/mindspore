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
#include "plugin/device/ascend/kernel/opapi/aclnn/binary_cross_entropy.h"
#include <vector>
#include <unordered_map>
#include "ir/tensor.h"
#include "transform/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace kernel {
void BinaryCrossEntropyAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                        const std::vector<KernelTensor *> &outputs) {
  auto reduction_imm = static_cast<Reduction>(transform::ConvertKernelTensor<int64_t>(inputs[kIndex3]));
  // transform reduction enum value to corresponding value
  std::unordered_map<Reduction, int64_t> reduction_map = {
    {Reduction::REDUCTION_SUM, 2}, {Reduction::MEAN, 1}, {Reduction::NONE, 0}};
  auto iter = reduction_map.find(reduction_imm);
  if (iter == reduction_map.end()) {
    MS_LOG(EXCEPTION) << "For BinaryCrossEntropy, the value of reduction is invalid.";
  }
  reduction_ = iter->second;

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], reduction_, outputs[kIndex0]);
}

bool BinaryCrossEntropyAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &workspace,
                                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], reduction_,
                                      outputs[kIndex0]));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(BinaryCrossEntropy, BinaryCrossEntropyAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
