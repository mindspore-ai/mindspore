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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/upsample_bilinear2d_aclnn_kernel.h"

#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <tuple>

#include "ir/tensor.h"
#include "mindapi/base/types.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace upsample_bilinear2d {
namespace {
std::tuple<std::vector<int64_t>, std::tuple<double, double>, bool> UpsampleBilinear2DGenerate(
  const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> output_size{output_shape.begin() + kIndex2, output_shape.end()};

  std::vector<pyfloat> scales{0., 0.};
  if (inputs[kIndex2]->GetType()->type_id() != kMetaTypeNone) {
    scales = inputs[kIndex2]->GetValueWithCheck<std::vector<pyfloat>>();
  }

  bool align_corners = inputs[kIndex3]->GetValueWithCheck<bool>();

  MS_ASSERT(scales.size() == kIndex2);
  double scales_h = scales[0];
  double scales_w = scales[1];

  return std::make_tuple(std::move(output_size), std::make_tuple(scales_h, scales_w), align_corners);
}
}  // namespace

void UpsampleBilinear2DAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  auto params = UpsampleBilinear2DGenerate(inputs, outputs);
  output_size_ = std::get<kIndex0>(params);
  std::tie(scales_h_, scales_w_) = std::get<kIndex1>(params);
  align_corners_ = std::get<kIndex2>(params);
  GetWorkspaceForResize(inputs[kIndex0], output_size_, align_corners_, scales_h_, scales_w_, outputs[kIndex0]);
}

bool UpsampleBilinear2DAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &workspace,
                                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], output_size_, align_corners_, scales_h_, scales_w_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(UpsampleBilinear2D, UpsampleBilinear2DAscend);
}  // namespace upsample_bilinear2d
}  // namespace kernel
}  // namespace mindspore
