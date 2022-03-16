/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ELTWISE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ELTWISE_CPU_KERNEL_H_

#include <memory>
#include <vector>
#include <map>
#include <string>
#include "plugin/device/cpu/kernel/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr auto kElu = "Elu";
constexpr auto kReLU = "ReLU";
constexpr auto kReLU6 = "ReLU6";
constexpr auto kExp = "Exp";
constexpr auto kLog = "Log";
constexpr auto kSigmoid = "Sigmoid";
constexpr auto kTanh = "Tanh";
constexpr auto kSoftplus = "Softplus";
constexpr auto kUnKnown = "UnKnown";
class EltWiseCpuKernelMod : public MKLCpuKernelMod {
 public:
  EltWiseCpuKernelMod() = default;
  explicit EltWiseCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~EltWiseCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::map<std::string, std::vector<KernelAttr>> support_list_map = {
      {kElu, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
      {kReLU, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
      {kReLU6, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
      {kExp, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
      {kLog, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
      {kSigmoid, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
      {kTanh, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
      {kSoftplus, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}}};
    auto iter = support_list_map.find(kernel_type_);
    if (iter == support_list_map.end()) {
      MS_LOG(EXCEPTION) << "Does not support " << kernel_type_ << "!";
    }
    return iter->second;
  }

 private:
  dnnl::eltwise_forward::desc GetForwardEltwiseDesc(const dnnl::memory::desc src_desc);

  dnnl::prop_kind dnnl_forward_{dnnl::prop_kind::forward_training};

  std::string kernel_type_{kUnKnown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ELTWISE_CPU_KERNEL_H_
