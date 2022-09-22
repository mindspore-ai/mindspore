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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_PROPAGATE_QUANT_PARAM_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_PROPAGATE_QUANT_PARAM_PASS_H_

#include <unordered_map>
#include <memory>
#include <utility>
#include <vector>
#include <list>
#include <map>
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "tools/converter/quantizer/quant_param_holder.h"

namespace mindspore::lite::quant {
using mindspore::lite::QuantParamHolderPtr;
struct DependNodes {
  std::vector<AnfNodePtr> forwards;
  std::vector<CNodePtr> backwards;
};
class PropagateQuantParamPass {
 public:
  explicit PropagateQuantParamPass(const FuncGraphPtr &func_graph) : func_graph_(func_graph) {}
  ~PropagateQuantParamPass() = default;
  int Propagate();

 private:
  int FindNodeDepends(const std::list<CNodePtr> &nodes, std::map<CNodePtr, DependNodes> *node_depends);
  int PropagateSelf(const CNodePtr &cnode, bool forward);
  int ForwardPropagate(const std::list<CNodePtr> &nodes);
  int BackwardPropagate(const std::list<CNodePtr> &nodes);
  int BackwardPerNode(const CNodePtr &post_cnode, const CNodePtr &cnode, size_t curr_index);
  bool CheckValidQuantParams(const std::vector<schema::QuantParamT> quant_params);
  int ForwardTupleGetItem(const CNodePtr &cnode);
  int BackwardMultipleOutput(const CNodePtr &cnode, const DependNodes &depend_nodes, size_t index);

 private:
  FuncGraphPtr func_graph_ = nullptr;
  // Key:CNodePtr
  // Value:(Forward,Backward)
  std::map<CNodePtr, DependNodes> node_depends_;
};
}  // namespace mindspore::lite::quant

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_PROPAGATE_QUANT_PARAM_PASS_H_
