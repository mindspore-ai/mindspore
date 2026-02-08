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
#ifndef MINDSPORE_CCSRC_KERNEL_GRAPH_KERNEL_KERNEL_PACKET_KERNEL_PACKET_INFER_FUNCTOR_H_
#define MINDSPORE_CCSRC_KERNEL_GRAPH_KERNEL_KERNEL_PACKET_KERNEL_PACKET_INFER_FUNCTOR_H_

#include <string>
#include <vector>
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/set_infershape_functor.h"
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_packet/kernel_packet_kernel_mod.h"

namespace mindspore {
namespace kernel {
class RUNTIME_HARDWARE_EXPORT KernelPacketInfer : public graphkernel::SymbolEngineInfer {
 public:
  KernelPacketInfer(const std::string &name, const FuncGraphPtr &fg, KernelPacketKernelMod *kernelmod)
      : SymbolEngineInfer(name, fg->symbol_engine(), fg->output()->abstract()), kernel_mod_holder_(kernelmod) {}
  ~KernelPacketInfer() override = default;
  MS_DECLARE_PARENT(KernelPacketInfer, SymbolEngineInfer)
  BaseShapePtr InferShape(const AbstractBasePtrList &args) override;
  void SetInnerInputNum(size_t n) { inner_inputs_abstract_.resize(n, nullptr); }
  void SetInnerInput(size_t i, const AbstractBasePtr &abs) { inner_inputs_abstract_[i] = abs; }

 protected:
  void SaveSymbolicValue();
  std::vector<AbstractBasePtr> inner_inputs_abstract_;
  KernelPacketKernelMod::InferResultSetter kernel_mod_holder_;
};

}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_GRAPH_KERNEL_KERNEL_PACKET_KERNEL_PACKET_INFER_FUNCTOR_H_
