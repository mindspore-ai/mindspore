/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#ifndef UT_CPP_COMMON_GRAPH_OPTIMIZER_TEST_FRAMEWORK_
#define UT_CPP_COMMON_GRAPH_OPTIMIZER_TEST_FRAMEWORK_

#include "include/backend/optimizer/optimizer.h"
#include "include/backend/kernel_graph.h"

namespace mindspore::test {
void RunPass(const FuncGraphPtr &graph, const std::vector<opt::PassPtr> &passes);

class ConstructGraph {
 public:
  ConstructGraph();
  const std::shared_ptr<session::KernelGraph> &GetGraph() const;
  ParameterPtr NewScalarInput(const std::string &name, const TypePtr &type);
  ParameterPtr NewTensorInput(const std::string &name, const TypePtr &type, const ShapeVector &shape);
  ParameterPtr NewTupleInput(const std::string &name, const std::vector<std::pair<TypePtr, ShapeVector>> &pairs);
  ParameterPtr NewListInput(const std::string &name, const std::vector<std::pair<TypePtr, ShapeVector>> &pairs);

  ValueNodePtr NewValueNode(const ValuePtr &value);
  CNodePtr NewCNodeWithoutInfer(const std::string &prim_name, const std::vector<AnfNodePtr> &inputs,
                                const mindspore::HashMap<std::string, ValuePtr> &attrs = {});
  CNodePtr NewCNode(const std::string &prim_name, const std::vector<AnfNodePtr> &inputs,
                    const mindspore::HashMap<std::string, ValuePtr> &attrs = {});
  CNodePtr NewCNodeWithBuildInfo(const std::string &prim_name, const std::vector<AnfNodePtr> &inputs,
                                 const mindspore::HashMap<std::string, ValuePtr> &attrs = {});
  void SetOutput(const AnfNodePtr &node);
  void SetGeneralBuildInfo(const AnfNodePtr &node);

 private:
  ParameterPtr NewInput(const std::string &name, const AbstractBasePtr &abs);

  std::shared_ptr<session::KernelGraph> graph_;
};
}  // namespace mindspore::test

#define UT_CHECK_NULL(pointer) ASSERT_NE(pointer, nullptr)

#endif  // UT_CPP_COMMON_GRAPH_OPTIMIZER_TEST_FRAMEWORK_
