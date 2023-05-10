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
#include "backend/graph_optimizer_test_framework.h"
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <deque>
#include <algorithm>
#include <set>
#include "common/common_test.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore::test {
void RunPass(const FuncGraphPtr &graph, const std::vector<opt::PassPtr> &passes) {
  UT_CHECK_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  for (const auto &pass : passes) {
    UT_CHECK_NULL(pass);
    pm->AddPass(pass);
  }
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
}

ConstructGraph::ConstructGraph() : graph_(std::make_shared<session::KernelGraph>()) { graph_->set_graph_id(0); }

const std::shared_ptr<session::KernelGraph> &ConstructGraph::GetGraph() const { return graph_; }

ParameterPtr ConstructGraph::NewInput(const std::string &name, const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(graph_);
  auto new_param = std::make_shared<Parameter>(graph_);
  new_param->set_name(name);
  new_param->set_abstract(abs);
  new_param = graph_->NewParameter(new_param);
  return new_param;
}

ParameterPtr ConstructGraph::NewScalarInput(const std::string &name, const TypePtr &type) {
  auto abs = std::make_shared<abstract::AbstractScalar>(type);
  return NewInput(name, abs);
}

ParameterPtr ConstructGraph::NewTensorInput(const std::string &name, const TypePtr &type, const ShapeVector &shape) {
  auto abs = std::make_shared<abstract::AbstractTensor>(type, shape);
  return NewInput(name, abs);
}

ParameterPtr ConstructGraph::NewTupleInput(const std::string &name,
                                           const std::vector<std::pair<TypePtr, ShapeVector>> &pairs) {
  AbstractBasePtrList list;
  for (const auto &[type, shape] : pairs) {
    auto abs = std::make_shared<abstract::AbstractTensor>(type, shape);
    list.emplace_back(std::move(abs));
  }
  auto abs = std::make_shared<abstract::AbstractTuple>(std::move(list), nullptr);
  return NewInput(name, abs);
}

ParameterPtr ConstructGraph::NewListInput(const std::string &name,
                                          const std::vector<std::pair<TypePtr, ShapeVector>> &pairs) {
  AbstractBasePtrList list;
  for (const auto &[type, shape] : pairs) {
    auto abs = std::make_shared<abstract::AbstractTensor>(type, shape);
    list.emplace_back(std::move(abs));
  }
  auto abs = std::make_shared<abstract::AbstractList>(std::move(list), nullptr);
  return NewInput(name, abs);
}

CNodePtr ConstructGraph::NewCNode(const std::string &prim_name, const std::vector<AnfNodePtr> &inputs,
                                  const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  MS_EXCEPTION_IF_NULL(graph_);
  auto prim = std::make_shared<Primitive>(prim_name);
  prim->SetAttrs(attrs);
  auto value_node = std::make_shared<ValueNode>(prim);
  std::vector<AnfNodePtr> new_inputs = {value_node};
  new_inputs.insert(new_inputs.end(), inputs.begin(), inputs.end());
  auto cnode = graph_->NewCNode(new_inputs);
  AbstractBasePtrList args;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(args),
                 [](const AnfNodePtr &node) -> abstract::AbstractBasePtr { return node->abstract(); });
  opt::CppInferShape(prim, args, cnode);
  return cnode;
}

void ConstructGraph::SetOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph_);
  graph_->set_output(node, true);
}
}  // namespace mindspore::test
