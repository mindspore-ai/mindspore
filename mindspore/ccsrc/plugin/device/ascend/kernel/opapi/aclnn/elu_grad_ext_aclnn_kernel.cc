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
#include "plugin/device/ascend/kernel/opapi/aclnn/elu_grad_ext_aclnn_kernel.h"
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

void EluGradExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  alpha_ = transform::ConvertKernelTensor<ScalarPtr>(inputs[kIndex2]);
  MAKE_SCALAR(1.f, inputs[kIndex0]->dtype_id(), scale_);
  MAKE_SCALAR(1.f, inputs[kIndex0]->dtype_id(), input_scale_);
  GetWorkspaceForResize(inputs[kIndex0], alpha_, scale_, input_scale_, false, inputs[kIndex1], outputs[kIndex0]);
}

bool EluGradExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], alpha_, scale_, input_scale_, false,
                                      inputs[kIndex1], outputs[kIndex0]));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(EluGradExt, EluGradExtAscend);
}  // namespace kernel
}  // namespace mindspore
