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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SYMBOL_ENGINE_EXTENDER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SYMBOL_ENGINE_EXTENDER_H_

#include <string>
#include "include/backend/optimizer/pass.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore::graphkernel {
constexpr auto kAttrKernelPacketNode = "kernel_packet_node";

// Extend kernel to a bigger subgraph using a symbol engine,
// to include all the nodes that do shape calc for the kernel.
class SymbolEngineExtender : public opt::Pass {
 public:
  SymbolEngineExtender() : Pass("symbol_engine_extender") {}
  ~SymbolEngineExtender() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};

class ConvertCallToPrim : public opt::Pass {
 public:
  explicit ConvertCallToPrim(const std::string &fg_name, const std::string &prim_name)
      : Pass("convert_call_to_prim"), fg_name_(fg_name), prim_name_(prim_name) {}
  ~ConvertCallToPrim() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  std::string fg_name_;
  std::string prim_name_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SYMBOL_ENGINE_EXTENDER_H_
