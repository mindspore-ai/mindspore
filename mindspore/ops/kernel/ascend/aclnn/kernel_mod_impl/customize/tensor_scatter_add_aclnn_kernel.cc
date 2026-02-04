/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/tensor_scatter_add_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
namespace tensor_scatter_add {

void TensorScatterAddAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  GetWorkspaceForResizeInplaceCopy(outputs[kIndex0], inputs[kIndex0]);
  GetWorkspaceForResizeTfScatterAdd(outputs[kIndex0], inputs[kIndex1], inputs[kIndex2]);
}

bool TensorScatterAddAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOpInplaceCopy(stream_ptr, workspace, outputs[kIndex0], inputs[kIndex0]);
  RunOpTfScatterAdd(stream_ptr, workspace, outputs[kIndex0], inputs[kIndex1], inputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(TensorScatterAdd, TensorScatterAddAscend);
}  // namespace tensor_scatter_add
}  // namespace kernel
}  // namespace mindspore
