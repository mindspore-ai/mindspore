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

#ifndef MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_COMPILE_RESULT_BUILDER_H_
#define MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_COMPILE_RESULT_BUILDER_H_
#include <string>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>
#include "src/extendrt/graph_compiler/compile_result.h"
#include "src/extendrt/tensor.h"
#include "abstract/abstract_value.h"
#include "ir/anf.h"
#include "include/api/status.h"

namespace mindspore {
namespace infer {
class CompileResultBuilder {
 public:
  explicit CompileResultBuilder(Format graph_format) : graph_format_(graph_format) {}
  ~CompileResultBuilder() = default;
  CompileResultPtr Build(const GraphSegmentPtr &graph_segment, const AnfNodePtrList &inputs,
                         const AnfNodePtrList &outputs);

 private:
  // build
  StatusCode BuildInputs(const AnfNodePtrList &inputs);
  StatusCode BuildNodes(const GraphSegmentPtr &graph_segment);
  StatusCode BuildOutputs(const AnfNodePtrList &outputs);
  StatusCode OptimizeGraph();
  // methods about node
  StatusCode CreateAndAppendNode(const CNodePtr &cnode);
  StatusCode AppendInputCNodeToInputs(const CNodePtr &cnode, const CompileNode *compile_node);
  StatusCode AppendInputParameterToInputs(const ParameterPtr &param_node, const CompileNode *compile_node);
  StatusCode AppendInputValueNodeToInputs(const ValueNodePtr &value_node, const CompileNode *compile_node);
  // methods about tensor
  static StatusCode CreateTensorsFromAbstract(const mindspore::abstract::AbstractBasePtr &abstract,
                                              std::vector<Tensor *> *results);
  StatusCode BuildNodeOutputTensor(const CNodePtr &cnode, const CompileNode *compile_node);
  // methods about optimize
  StatusCode RemoveSeqGetItemNode();
  StatusCode RemoveMakeSeqNode();
  StatusCode RemoveDependNode();
  // Replace `index`th output tensor of `node` by `dst_tensor` tensor.
  void IsolateTensor(Tensor *dst_tensor, const CompileNode *node, size_t index);

 private:
  CompileResultPtr graph_ = nullptr;
  Format graph_format_{DEFAULT_FORMAT};
  std::set<std::string> input_names_{};
};
}  // namespace infer
}  // namespace mindspore

#endif
