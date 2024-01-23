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
#ifndef MINDSPORE_CCSRC_KERNEL_GRAPH_KERNEL_KERNEL_PACKET_KERNEL_PACKET_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_KERNEL_GRAPH_KERNEL_KERNEL_PACKET_KERNEL_PACKET_KERNEL_MOD_H_

#include <vector>
#include <utility>
#include <tuple>
#include <string>

#include "include/common/utils/anfalgo.h"
#include "kernel/kernel.h"
#include "base/base.h"
#include "kernel/framework_utils.h"
#include "mindspore/core/symbolic_shape/symbol_engine.h"

namespace mindspore {
constexpr auto kAttrKernelPacketNode = "kernel_packet_node";

namespace kernel {
using MemcpyAsyncFunc = std::function<bool(void *, const void *, size_t, void *)>;
/// \brief Kernel Mod of subgraph into which shape calc is clustered
class BACKEND_EXPORT KernelPacketKernelMod : public KernelMod {
 public:
  explicit KernelPacketKernelMod(const MemcpyAsyncFunc &memcpy_async) : memcpy_async_(memcpy_async) {}
  ~KernelPacketKernelMod() override = default;

  bool Init(const CNodePtr &node);

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  using AddressArgs = std::tuple<std::vector<KernelTensor *>, std::vector<KernelTensor *>, std::vector<KernelTensor *>>;

  AddressArgs GetLaunchArgs(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspaces,
                            const std::vector<KernelTensor *> &outputs);

  HashMap<size_t, size_t> input_map_;                         // Map inner_kernel's input index to outer input index
  HashMap<size_t, size_t> input_workspace_map_;               // Map inner kernel's input index to workspace index
  HashMap<size_t, common::KernelWithIndex> input_shape_map_;  // Map inner kernel's input index to node's output
  HashMap<size_t, ShapeVector> shape_cache_;  // cache shape of inner kernel's input, key is inner kernel's input index
  std::vector<size_t> workspace_;
  SymbolEnginePtr symbol_engine_;

  KernelModPtr real_kernel_mod_;
  std::string real_node_name_;
  size_t real_node_input_num_;
  std::vector<KernelTensorPtr> inputs_cache_;
  MemcpyAsyncFunc memcpy_async_;
};

inline CNodePtr GetKernelPacketRealNode(const AnfNodePtr &kernelpacket) {
  auto func_graph = common::AnfAlgo::GetNodeAttr<FuncGraphPtr>(kernelpacket, kAttrFuncGraph);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto real_node = func_graph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_node);
  return real_node;
}
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_GRAPH_KERNEL_KERNEL_PACKET_KERNEL_PACKET_KERNEL_MOD_H_
