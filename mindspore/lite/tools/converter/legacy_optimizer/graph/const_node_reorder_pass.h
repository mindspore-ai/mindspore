/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_GRAPH_CONST_NODE_REORDERED_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_GRAPH_CONST_NODE_REORDERED_PASS_H_

#include <memory>
#include <vector>
#include "mindspore/lite/tools/converter/optimizer.h"
#include "tools/common/graph_util.h"

namespace mindspore {
namespace lite {
// The goal of this pass is to minimize runtime memory usage. Mainly for the case where the constant is not folded, such
// as weight quant on the fly mode. The idea is that for const output node, such as QuantDtypeCast, we want to put it in
// the closet place to the node that uses it in the execution order. So when this node is finished executing, we can
// free its memory immediately.
class ConstNodeReorderPass : public GraphPass {
 public:
  ConstNodeReorderPass() = default;

  ~ConstNodeReorderPass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override;

 private:
  bool IsConstNode(const std::unique_ptr<schema::CNodeT> &node, const schema::MetaGraphT &graph);
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_GRAPH_CONST_NODE_REORDERED_PASS_H_
