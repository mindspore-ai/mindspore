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
#include "plugin/device/ascend/kernel/opapi/aclnn/quant_v2_aclnn_kernel.h"
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "transform/graph_ir/op_adapter_base.h"

namespace mindspore {
using mindspore::transform::AscendQuantRoundMode;
namespace kernel {

void QuantV2Ascend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto sqrt_mode = transform::ConvertKernelTensor<bool>(inputs[kIndex3]);
  auto rounding_mode = transform::ConvertKernelTensor<int64_t>(inputs[kIndex4]);
  std::string rounding_mode_str = AscendQuantRoundMode::ConvertEnumToString(rounding_mode);

  // Infer function has confirmed the actual dtype of output
  TypeId out_type = outputs[kIndex0]->dtype_id();

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], sqrt_mode, rounding_mode_str, out_type,
                        outputs[kIndex0]);
}

bool QuantV2Ascend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto sqrt_mode = transform::ConvertKernelTensor<bool>(inputs[kIndex3]);
  auto rounding_mode = transform::ConvertKernelTensor<int64_t>(inputs[kIndex4]);
  std::string rounding_mode_str = AscendQuantRoundMode::ConvertEnumToString(rounding_mode);

  // Infer function has confirmed the actual dtype of output
  TypeId out_type = outputs[kIndex0]->dtype_id();

  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], sqrt_mode, rounding_mode_str,
        out_type, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(QuantV2, QuantV2Ascend);
}  // namespace kernel
}  // namespace mindspore
