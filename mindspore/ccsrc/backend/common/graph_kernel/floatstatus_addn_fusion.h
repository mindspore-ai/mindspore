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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_FLOATSTATUS_ADDN_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_FLOATSTATUS_ADDN_FUSION_H_

#include <string>
#include "include/backend/optimizer/pass.h"
#include "backend/common/graph_kernel/inplace_assign_builder.h"

namespace mindspore::graphkernel {
/**
 * @brief Expand FloatStatus+AddN
 * @example
 *   %1 = op1
 *   %2 = op2
 *   %3 = FloatStatus(%1)
 *   %4 = FloatStatus(%2)
 *   %5 = AddN(%3, %4)
 *   %6 = op3(%5)
 *   ---------->
 *   subgraph1(){
 *      %1 = BroadCastTo(shape=[1],value=0)
 *      return %1
 *    }
 *   -----
 *   subgraph2(a,b){
 *      %1 = IsNan(a)
 *      %2 = IsInf(a)
 *      %3 = LogicalOr(%1, %2)
 *      %4 = ElemAny(%3)
 *      %5 = Assign(b, %4)
 *      return %5
 *    }
 *   -----
 *   maingraph{
 *   %0 = subgraph1()
 *   %1 = op1
 *   %2 = op2
 *   %3 = subgraph2(%1, %0)
 *   %4 = subgraph2(%2, %0)
 *   %5 = MakeTuple(%3, %4)
 *   %6 = Depend(%0, %5)
 *   %7 = op3(%6)
 *   return %7
 *    }
 */
class FloatStatusAddNFusion : public InplaceAssignBuilder {
 public:
  explicit FloatStatusAddNFusion(const std::string &name = "floatstatus_addn_fusion") : InplaceAssignBuilder(name) {}
  ~FloatStatusAddNFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  void ProcessFloatStatusAddN(const FuncGraphPtr &main_graph, const CNodePtr &addn, const FuncGraphManagerPtr &mng);
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_FLOATSTATUS_ADDN_FUSION_H_
