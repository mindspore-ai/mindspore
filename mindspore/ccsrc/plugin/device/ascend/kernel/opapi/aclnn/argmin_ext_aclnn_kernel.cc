/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/opapi/aclnn/argmin_ext_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
void ArgMinAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  dim_ = 0;
  dim_is_none_ = false;
  keepdim_ = false;
  input_real_shape_ = inputs[kIndex0]->GetShapeVector();
  output_real_shape_ = outputs[kIndex0]->GetShapeVector();
  auto dim_value_opt = inputs[kIndex1]->GetOptionalValueWithCheck<int64_t>();
  if (dim_value_opt.has_value()) {
    dim_ = dim_value_opt.value();
    keepdim_ = transform::ConvertKernelTensor<bool>(inputs[kIndex2]);
  } else {  // input dim is None set flatten size
    dim_is_none_ = true;
  }

  if (dim_is_none_) {
    input_kernel_tensor_ = inputs[kIndex0]->CloneKernelTensor();

    int input_flatten_size =
      std::accumulate(input_real_shape_.begin(), input_real_shape_.end(), 1, std::multiplies<int64_t>());
    auto input_flatten_shape = ShapeVector{input_flatten_size};
    input_kernel_tensor_->SetShapeVector(input_flatten_shape);

    size_t offset = 0;
    auto input_flatten_shape_ori = input_flatten_shape;
    auto input_flatten_shape_new = input_flatten_shape;
    std::vector<int64_t> strides_new = {1};
    std::vector<int64_t> strides_ori = {1};
    TensorStorageInfoPtr tensor_storage_info = std::make_shared<TensorStorageInfo>(
      input_flatten_shape_new, strides_new, offset, input_flatten_shape_ori, strides_ori, true);
    input_kernel_tensor_->set_tensor_storage_info(tensor_storage_info);
    GetWorkspaceForResize(input_kernel_tensor_.get(), dim_, keepdim_, outputs[kIndex0]);
  } else {
    GetWorkspaceForResize(inputs[kIndex0], dim_, keepdim_, outputs[kIndex0]);
  }
}

bool ArgMinAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (dim_is_none_) {
    input_kernel_tensor_->set_device_ptr(inputs[kIndex0]->device_ptr());
    ParseGenExecutor(
      GEN_EXECUTOR_BOOST(op_type_, hash_id_, input_kernel_tensor_.get(), dim_, keepdim_, outputs[kIndex0]));
    RunOp(stream_ptr, workspace);
  } else {
    ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], dim_, keepdim_, outputs[kIndex0]));
    RunOp(stream_ptr, workspace);
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(ArgMinExt, ArgMinAscend);
}  // namespace kernel
}  // namespace mindspore
