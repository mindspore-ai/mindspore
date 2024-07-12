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

#include "plugin/device/ascend/kernel/opapi/aclnn/upsample_nearest3d_aclnn_kernel.h"

#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <tuple>
#include <functional>

#include "ir/tensor.h"
#include "mindapi/base/types.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace {
std::tuple<std::vector<int64_t>, std::tuple<double, double, double>> UpsampleNearest3DGenerate(
  const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> output_size{output_shape.begin() + kIndex2, output_shape.end()};

  std::vector<pyfloat> scales{0., 0., 0.};
  if (inputs[kIndex2]->GetType()->type_id() != kMetaTypeNone) {
    scales = inputs[kIndex2]->GetValueWithCheck<std::vector<pyfloat>>();
  }

  double scales_d = scales[0];
  double scales_h = scales[1];
  double scales_w = scales[2];

  return std::make_tuple(std::move(output_size), std::make_tuple(scales_d, scales_h, scales_w));
}
}  // namespace

void UpsampleNearest3DAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  auto params = UpsampleNearest3DGenerate(inputs, outputs);
  output_size_ = std::get<0>(params);
  std::tie(scales_d_, scales_h_, scales_w_) = std::get<1>(params);
  GetWorkspaceForResize(inputs[0], output_size_, scales_d_, scales_h_, scales_w_, outputs[0]);
}

bool UpsampleNearest3DAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[0], output_size_, scales_d_, scales_h_, scales_w_, outputs[0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(UpsampleNearest3D, UpsampleNearest3DAscend);
}  // namespace kernel
}  // namespace mindspore
