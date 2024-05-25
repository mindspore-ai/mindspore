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
#include <unordered_map>

#include "include/common/utils/anfalgo.h"
#include "kernel/kernel.h"
#include "base/base.h"
#include "kernel/framework_utils.h"

namespace mindspore {
namespace kernel {
class KernelPacketKernelMod;
class KernelPacketInfer;
class BACKEND_EXPORT KernelPacketInitializer {
 public:
  static bool InitKernel(const CNodePtr &real_node, const KernelModPtr &real_kernel_mod,
                         KernelPacketKernelMod *packet_kernel_mod, KernelPacketInfer *infer);
};

/// \brief Kernel Mod of subgraph into which host ops are clustered
class BACKEND_EXPORT KernelPacketKernelMod : public KernelMod {
 public:
  friend class KernelPacketInfer;
  friend class KernelPacketInitializer;

  KernelPacketKernelMod() = default;
  ~KernelPacketKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &) override { return true; }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  virtual bool CopyHostToDevice(void *dst, const void *src, size_t size, void *stream) = 0;
  void AllocWorkspace(size_t i, size_t data_size);
  using AddressArgs = std::tuple<std::vector<KernelTensor *>, std::vector<KernelTensor *>>;
  AddressArgs GetLaunchArgs(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspaces,
                            void *stream_ptr);

  // Cache the inner_kernel's input KernelTensors
  std::vector<KernelTensorPtr> inputs_cache_;
  // Map inner_kernel's input index to outer input index
  std::unordered_map<size_t, size_t> input_map_;
  // Map inner_kernel's input index to workspace index
  std::unordered_map<size_t, size_t> input_workspace_map_;
  // Cache the host data.
  std::vector<ValuePtr> host_value_cache_;
  std::vector<const void *> host_data_cache_;
  KernelModPtr real_kernel_mod_;
  std::string real_node_debug_str_;
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
