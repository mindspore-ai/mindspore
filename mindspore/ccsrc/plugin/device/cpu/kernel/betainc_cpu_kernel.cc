/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/betainc_cpu_kernel.h"
#include <algorithm>
#include <memory>
#include "unsupported/Eigen/CXX11/Tensor"
#include "mindspore/core/ops/betainc.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBetaincInputsNum = 3;
constexpr size_t kBetaincOutputsNum = 1;
}  // namespace

bool BetaincCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::make_shared<ops::Betainc>(base_operator->GetPrim());
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast Betainc ops failed!";
    return false;
  }
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int BetaincCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    return ret;
  }
  input0_shape_ = inputs[kIndex0]->GetShapeVector();
  input1_shape_ = inputs[kIndex1]->GetShapeVector();
  input2_shape_ = inputs[kIndex2]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  if (!IsSameShape(input0_shape_, input1_shape_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of 'b' should be same with the shape of 'a', "
                  << "but got the shape of 'b': " << input1_shape_ << " and 'a': " << input0_shape_;
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(input0_shape_, input2_shape_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of 'x' should be same with the shape of 'a', "
                  << "but got the shape of 'x': " << input2_shape_ << " and 'a': " << input0_shape_;
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(input0_shape_, output_shape_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of output should be same with the shape of the 'a', "
                  << "but got the shape of the output: " << output_shape_ << " and 'a': " << input0_shape_;
    return KRET_RESIZE_FAILED;
  }
  return 0;
}

template <typename T>
inline T ScalarBetainc(T a, T b, T x) {
  return Eigen::numext::betainc(a, b, x);
}

template <typename T>
bool BetaincCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &workspace,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBetaincInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBetaincOutputsNum, kernel_name_);
  T *input0 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input1 = reinterpret_cast<T *>(inputs[1]->addr);
  T *input2 = reinterpret_cast<T *>(inputs[2]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  auto total = inputs[0]->size / sizeof(T);
  auto task = [&input0, &input1, &input2, &output](std::int64_t begin, std::int64_t end) {
    for (std::int64_t i = begin; i < end; i++) {
      output[i] = ScalarBetainc(input0[i], input1[i], input2[i]);
    }
  };
  ParallelLaunchAutoSearch(task, total, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, BetaincCpuKernelMod::KernelRunFunc>> &BetaincCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, BetaincCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &BetaincCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &BetaincCpuKernelMod::LaunchKernel<double>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Betainc, BetaincCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
