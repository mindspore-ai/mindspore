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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/upsample_linear1d_aclnn_kernel.h"

#include <tuple>
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>

#include "ir/tensor.h"
#include "mindapi/base/types.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace upsample_linear1d {
namespace {
const double UpsampleLinear1dEps = 1e-7;
const pyfloat DEFAULT_SCALE_VALUE = -1;
std::tuple<std::vector<int64_t>, double, bool> UpsampleLinear1DGenerate(const std::vector<KernelTensor *> &inputs,
                                                                        const std::vector<KernelTensor *> &outputs) {
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> output_size{output_shape.begin() + kIndex2, output_shape.end()};

  bool align_corners = inputs[kIndex3]->GetValueWithCheck<bool>();

  std::vector<pyfloat> scales{DEFAULT_SCALE_VALUE};
  if (inputs[kIndex2]->GetType()->type_id() != kMetaTypeNone) {
    scales = inputs[kIndex2]->GetValueWithCheck<std::vector<pyfloat>>();
  }

  // Python float obj would be parsed by float32 number, which should be parsed
  // to double number according to PyTorch. For example, python scale is 2.6,
  // but the last scale we got in c++ is 2.5999999046325684,
  // which caused aclnn verification to fail.
  double scale_l = scales.at(0) != DEFAULT_SCALE_VALUE ? (static_cast<double>(scales.at(0)) + UpsampleLinear1dEps)
                                                       : DEFAULT_SCALE_VALUE;

  return std::make_tuple(std::move(output_size), scale_l, align_corners);
}
}  // namespace

void UpsampleLinear1DAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  auto params = UpsampleLinear1DGenerate(inputs, outputs);
  output_size_ = std::get<kIndex0>(params);
  scales_l_ = std::get<kIndex1>(params);
  align_corners_ = std::get<kIndex2>(params);
  GetWorkspaceForResize(inputs[kIndex0], output_size_, align_corners_, scales_l_, outputs[kIndex0]);
}

bool UpsampleLinear1DAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], output_size_, align_corners_, scales_l_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(UpsampleLinear1D, UpsampleLinear1DAscend);
}  // namespace upsample_linear1d
}  // namespace kernel
}  // namespace mindspore
