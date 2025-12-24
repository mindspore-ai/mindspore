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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_CONVERT_BFLOAT_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_CONVERT_BFLOAT_H_

#include <string>
#include <utility>
#include <vector>
#include "include/backend/common/pass_manager/pass.h"

namespace mindspore::graphkernel {
class ConvertDataType : public opt::Pass {
 public:
  explicit ConvertDataType(const std::string &name) : Pass(name) {}
  virtual ~ConvertDataType() = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  virtual bool Process(const FuncGraphPtr &func_graph) = 0;
};

/**
 * @brief Add Cast for op's inputs if the input data type is bfloat16
 * @example
 *   sub_graph(p0: bfloat16, p1: bfloat16) {
 *     %0 = Op(p0, p1)
 *     return %0
 *   }
 *   ---------->
 *   sub_graph(p0: bfloat16, p1: bfloat16) {
 *     %0 = Cast(p0, float32)
 *     %1 = Cast(p1, float32)
 *     %2 = Op(%0, %1)
 *     %3 = Cast(%2, bfloat16)
 *     return %3
 *   }
 */
class ConvertBFloat16 : public ConvertDataType {
 public:
  ConvertBFloat16() : ConvertDataType("convert_bfloat16") {}
  ~ConvertBFloat16() override = default;

 private:
  AnfNodePtr GetCastedInput(const AnfNodePtr &input_node, TypeId dst_type, const FuncGraphPtr &func_graph);
  void CastInput(const CNodePtr &cnode, size_t input_idx, const FuncGraphPtr &func_graph);
  void GetKeepBF16Nodes(const FuncGraphPtr &func_graph);
  bool Process(const FuncGraphPtr &func_graph) override;

  HashMap<AnfNodePtr, AnfNodePtr> cast_nodes_;
  // (keep_bf16_node, {node_user, input_idx}), node_user's input[input_idx] is keep_bf16_node
  HashMap<AnfNodePtr, std::vector<std::pair<CNodePtr, size_t>>> keep_bf16_nodes_;
  CNodePtr last_node_;
};

/**
 * @brief Add Cast for op's input and output if data type is not same
 * @example
 *   sub_graph() {
 *     %0{Tensor(2,)fp16} = Mul(p0{Tensor(2,)fp16}, p1{Tensor()fp32})
 *     %1{Tensor(2,)fp16} = Abs(%0{Tensor(2,)fp16})
 *     return %1
 *   }
 *   ---------->
 *   sub_graph() {
 *     %0{Tensor(2,)fp32} = Cast(p0{Tensor(2,)fp16})
 *     %1{Tensor(2,)fp32} = Mul(%0{Tensor(2,)fp32}, p1{Tensor()fp32})
 *     %2{Tensor(2,)fp16} = Cast(%1{Tensor(2,)fp32})
 *     %3{Tensor(2,)fp16} = Abs(%2{Tensor(2,)fp16})
 *     return %3
 *   }
 */
class ConvertMixType : public ConvertDataType {
 public:
  ConvertMixType() : ConvertDataType("convert_mix_type") {}
  ~ConvertMixType() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  void CastInput(const CNodePtr &cnode, size_t input_idx, TypeId cur_type, TypeId target_type);
  void CastOutput(const CNodePtr &cnode, size_t output_idx, TypeId cur_type, TypeId target_type);
  bool ProcessBinary(const CNodePtr &cnode);
  bool Process(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_CONVERT_BFLOAT_H_
