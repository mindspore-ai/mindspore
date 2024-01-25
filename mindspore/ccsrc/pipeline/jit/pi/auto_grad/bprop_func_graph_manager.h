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

#ifndef MINDSPORE_PI_JIT_BPROP_FUNC_GRAPH_MANAGERER_H_
#define MINDSPORE_PI_JIT_BPROP_FUNC_GRAPH_MANAGERER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include "ir/func_graph.h"

namespace mindspore {
namespace pijit {
namespace grad {
class BpropFuncGraphManager {
 public:
  BpropFuncGraphManager() {}
  virtual ~BpropFuncGraphManager() = default;

  FuncGraphPtr PrimBpropGraphPass(const FuncGraphPtr &prim_grad_graph);
  FuncGraphPtr GetAccumulateGraph(const ValuePtr &dout, const ValuePtr &factor);
  FuncGraphPtr GetPrimBpropGraph(const PrimitivePtr &prim, const ValuePtrList &inputs, const ValuePtr &out,
                                 const ValuePtr &dout);
  FuncGraphPtr GetPrimBpropGraph(const PrimitivePtr &prim, const abstract::AbstractBasePtrList &args_abs);
  FuncGraphPtr GetFuncGraphBpropGraph(const FuncGraphPtr &forward_graph, const ValuePtrList &inputs,
                                      const ValuePtr &out, const ValuePtr &dout);

 private:
  std::map<std::string, FuncGraphPtr> prim_to_bprop_;
  std::map<FuncGraphPtr, FuncGraphPtr> func_graph_to_bprop_;
};

using BpropFuncGraphManagerPtr = std::shared_ptr<BpropFuncGraphManager>;
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_BPROP_FUNC_GRAPH_MANAGERER_H_
