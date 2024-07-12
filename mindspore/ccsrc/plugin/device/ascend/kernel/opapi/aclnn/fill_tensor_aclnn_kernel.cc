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
#include "plugin/device/ascend/kernel/opapi/aclnn/fill_tensor_aclnn_kernel.h"
#include "transform/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {

void FillTensorAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  auto value_tensor = inputs[kIndex1];
  if (value_tensor->device_ptr() == nullptr) {
    MS_LOG(INFO) << "For " << primitive_->name() << ", Input [fill_value] is a host tensor, FillScalar will be used.";
    value_ = transform::ConvertKernelTensor<ScalarPtr>(inputs[kIndex1]);
    op_type_ = "aclnnInplaceFillScalar";
    GetWorkspaceForResize(outputs[kIndex0], value_);
    return;
  }
  GetWorkspaceForResize(outputs[kIndex0], inputs[kIndex1]);
}

bool FillTensorAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (op_type_ == "aclnnInplaceFillScalar") {
    ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, outputs[kIndex0], value_));
  } else {
    ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, outputs[kIndex0], inputs[kIndex1]));
  }
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(FillTensor, FillTensorAscend);
}  // namespace kernel
}  // namespace mindspore
