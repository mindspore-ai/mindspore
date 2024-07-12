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
#include "plugin/device/ascend/kernel/opapi/aclnn/convolution_grad_aclnn_kernel.h"
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

void ConvolutionGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  stride_ = transform::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex4]);
  padding_ = transform::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex5]);
  dilation_ = transform::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex6]);
  transposed_ = transform::ConvertKernelTensor<bool>(inputs[kIndex7]);
  output_padding_ = transform::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex8]);
  groups_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex9]);
  const auto &output_mask_vec = transform::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex10]);
  output_mask_.clear();
  std::transform(output_mask_vec.begin(), output_mask_vec.end(), std::back_inserter(output_mask_),
                 [](const int64_t &value) { return static_cast<uint8_t>(value); });

  const auto &dout_shape = inputs[kIndex0]->GetShape()->GetShapeVector();
  bias_size_ = {dout_shape[1]};

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], bias_size_, stride_, padding_, dilation_,
                        transposed_, output_padding_, groups_, output_mask_, OpApiUtil::GetCubeMathType(),
                        outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
}

bool ConvolutionGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], bias_size_, stride_, padding_,
        dilation_, transposed_, output_padding_, groups_, output_mask_, OpApiUtil::GetCubeMathType(), outputs[kIndex0],
        outputs[kIndex1], outputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(ConvolutionGrad, ConvolutionGradAscend);
}  // namespace kernel
}  // namespace mindspore
