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
#include "plugin/device/ascend/kernel/opapi/aclnn/silent_check_v2_aclnn_kernel.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "include/common/utils/utils.h"
#include "ir/tensor.h"
#include "kernel/kernel.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace kernel {
namespace {
void DoMemcpy(const std::string &op_name, const std::vector<uint64_t> &copy_hash_ids,
              const std::vector<uint64_t> &copy_ws_sizes, size_t order, const KernelTensor *input,
              const KernelTensor *output, const std::vector<KernelTensor *> &workspace, void *stream_ptr) {
  auto copy_hash_id = copy_hash_ids[order];
  auto copy_ws_size = copy_ws_sizes[order];
  void *ws_addr = copy_ws_size != 0 ? (workspace.back() - kIndex3 + order)->device_ptr() : nullptr;
  transform::aclOpExecutor *executor;
  std::function<void()> release_func;
  std::tie(std::ignore, executor, release_func, std::ignore, std::ignore) =
    GEN_EXECUTOR_BOOST(op_name, copy_hash_id, output, input);
  RUN_OP_API_ASYNC(op_name, ws_addr, copy_ws_size, executor, stream_ptr, release_func);
}
}  // namespace
void SilentCheckV2Ascend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  c_min_steps_ = inputs[kIndex4]->GetValueWithCheck<int64_t>();
  c_thresh_l1_ = inputs[kIndex5]->GetValueWithCheck<pyfloat>();
  c_coeff_l1_ = inputs[kIndex6]->GetValueWithCheck<pyfloat>();
  c_thresh_l2_ = inputs[kIndex7]->GetValueWithCheck<pyfloat>();
  c_coeff_l2_ = inputs[kIndex8]->GetValueWithCheck<pyfloat>();
  npu_asd_detect_ = inputs[kIndex9]->GetValueWithCheck<int64_t>();
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], c_min_steps_, c_thresh_l1_,
                        c_coeff_l1_, c_thresh_l2_, c_coeff_l2_, npu_asd_detect_, outputs[kIndex3]);
  for (size_t i = 0; i < kIndex3; i++) {
    SetWorkspaceForInplaceCopy(outputs[i], inputs[i + kIndex1], i);
  }
}

bool SilentCheckV2Ascend::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2],
                                      inputs[kIndex3], c_min_steps_, c_thresh_l1_, c_coeff_l1_, c_thresh_l2_,
                                      c_coeff_l2_, npu_asd_detect_, outputs[kIndex3]));
  RunOp(stream_ptr, workspace);
  for (size_t i = 0; i < kIndex3; i++) {
    DoMemcpy(inplace_copy_str_, copy_hash_ids_, copy_ws_sizes_, i, inputs[i + kIndex1], outputs[i], workspace,
             stream_ptr);
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SilentCheckV2, SilentCheckV2Ascend);
}  // namespace kernel
}  // namespace mindspore
