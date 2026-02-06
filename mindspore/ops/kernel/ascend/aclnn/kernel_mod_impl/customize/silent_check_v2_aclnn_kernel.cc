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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/silent_check_v2_aclnn_kernel.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "include/utils/utils.h"
#include "ir/tensor.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
namespace silent_check_v2 {
void SilentCheckV2Ascend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  c_min_steps_ = inputs[kIndex4]->GetValueWithCheck<int64_t>();
  c_thresh_l1_ = inputs[kIndex5]->GetValueWithCheck<pyfloat>();
  c_coeff_l1_ = inputs[kIndex6]->GetValueWithCheck<pyfloat>();
  c_thresh_l2_ = inputs[kIndex7]->GetValueWithCheck<pyfloat>();
  c_coeff_l2_ = inputs[kIndex8]->GetValueWithCheck<pyfloat>();
  npu_asd_detect_ = inputs[kIndex9]->GetValueWithCheck<int64_t>();
  ClearOpsWorkSpaceList();
  GetWorkspaceForResizeSilentCheck(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], c_min_steps_,
                                   c_thresh_l1_, c_coeff_l1_, c_thresh_l2_, c_coeff_l2_, npu_asd_detect_,
                                   outputs[kIndex3]);
  // copy input_grad to outputs[0]
  GetWorkspaceForResizeInputGradCopy(outputs[kIndex0], inputs[kIndex1]);
}

bool SilentCheckV2Ascend::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOpSilentCheck(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3],
                   c_min_steps_, c_thresh_l1_, c_coeff_l1_, c_thresh_l2_, c_coeff_l2_, npu_asd_detect_,
                   outputs[kIndex3]);
  // copy input_grad to outputs[0]
  RunOpInputGradCopy(stream_ptr, workspace, outputs[kIndex0], inputs[kIndex1]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SilentCheckV2, SilentCheckV2Ascend);
}  // namespace silent_check_v2
}  // namespace kernel
}  // namespace mindspore
