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

#include "plugin/device/ascend/kernel/bisheng/wkv_grad_bisheng_kernel.h"
#include <algorithm>
#include "plugin/device/ascend/kernel/bisheng/bisheng_op_info.h"

namespace mindspore::kernel {
namespace {
constexpr size_t kWKVInputsNum = 5;
constexpr size_t kWKVOutputsNum = 4;
constexpr size_t kKIndex = 2;
}  // namespace

bool WKVGradBishengKernel::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  MS_EXCEPTION_IF_NULL(base_operator->GetPrim());
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kWKVOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  func_name_ = func_name_list_.at(index);
  return true;
}

template <typename T>
bool WKVGradBishengKernel::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs, void *stream) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kWKVInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kWKVOutputsNum, kernel_name_);
  MS_EXCEPTION_IF_NULL(stream);
  return true;
}

size_t WKVGradWorkSpaceFunc(const BiShengKernelArgs &args) {
  const auto &input_shapes = args.input_shapes;
  constexpr size_t shape_len = 3;
  if (input_shapes.size() < shape_len || input_shapes[kKIndex].size() < shape_len) {
    MS_LOG(EXCEPTION) << "Add op must have output shapes.";
  }
  auto size =
    LongToSize(input_shapes[kKIndex][0] * input_shapes[kKIndex][1] * input_shapes[kKIndex][kKIndex] * sizeof(float));
  MS_LOG(INFO) << size;
  return size;
}

int WKVGradTilingFunc(const BiShengKernelArgs &args, std::vector<uint8_t> *tiling_data) {
  MS_EXCEPTION_IF_NULL(tiling_data);
  const auto &input_shapes = args.input_shapes;
  constexpr size_t shape_len = 3;
  if (input_shapes.size() < shape_len || input_shapes[kKIndex].size() < shape_len) {
    MS_LOG(EXCEPTION) << "Add op must have output shapes.";
  }
  std::vector<int64_t> wkv_attr = {input_shapes[kKIndex][0], input_shapes[kKIndex][1], input_shapes[kKIndex][kKIndex]};
  TilingPacking::PackTiling(tiling_data, wkv_attr);
  return 0;
}

REG(WKVGradBishengKernel)
  .OpName("WKVGrad")
  .Input(0, "w")
  .Input(1, "u")
  .Input(2, "k")
  .Input(3, "v")
  .Input(4, "gy")
  .Output(0, "gw")
  .Output(1, "gu")
  .Output(2, "gk")
  .Output(3, "gv")
  .WorkSpace(&WKVGradWorkSpaceFunc)
  .WorkSpace(&WKVGradWorkSpaceFunc)
  .WorkSpace(&WKVGradWorkSpaceFunc)
  .DataTypeFormat({F32_Default, F32_Default, F32_Default, F32_Default, F32_Default, F32_Default, F32_Default,
                   F32_Default, F32_Default},
                  &WKVGradBishengKernel::LaunchKernel<float>, "_ZTSZ12wkv_backwardE19vbsRwkvFp32Backward")
  .Tiling(&WKVGradTilingFunc)
  .End();
}  // namespace mindspore::kernel
