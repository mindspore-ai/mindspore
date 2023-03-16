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

#include "common/common_test.h"
#include "mindspore/core/ops/core_ops.h"
#define private public
#include "frontend/parallel/graph_util/graph_splitter.h"

namespace mindspore {
namespace parallel {
using namespace abstract;
class GraphSplitterTest : public UT::Common {
 public:
  GraphSplitterTest() = default;
  virtual ~GraphSplitterTest() = default;

  void SetUp() {
    func_graph_ = std::make_shared<FuncGraph>();
    MS_EXCEPTION_IF_NULL(func_graph_);
    splitter_ = std::make_shared<GraphSplitter>(func_graph_, 0, "MS_WORKER");
    MS_EXCEPTION_IF_NULL(splitter_);
  }

  void TearDown() {}

  AbstractBasePtr CreateAbs();

  FuncGraphPtr func_graph_{nullptr};
  std::shared_ptr<GraphSplitter> splitter_{nullptr};
};

AbstractBasePtr GraphSplitterTest::CreateAbs() {
  ShapeVector shp = {3, 3};
  auto ele = std::make_shared<AbstractScalar>(kValueAny, kFloat64);
  MS_EXCEPTION_IF_NULL(ele);
  auto abs = std::make_shared<AbstractTensor>(ele, std::make_shared<Shape>(shp));
  MS_EXCEPTION_IF_NULL(abs);
  return abs;
}

/// Feature: Distributed runtime.
/// Description: Test whether a value node is created from a real node.
/// Expectation: A value node is created, with the shape the same as the origin node.
TEST_F(GraphSplitterTest, TestCreateFakeValueNode) {
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimAdd)};
  auto origin_node = func_graph_->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(origin_node);

  origin_node->set_abstract(CreateAbs());

  ShapeVector shp = {3, 3};
  auto fake_value_node1 = CreateFakeValueNode(true, origin_node, false);
  ASSERT_EQ(fake_value_node1->abstract()->cast<abstract::AbstractTensorPtr>()->shape()->shape(), shp);

  shp = {1};
  auto fake_value_node2 = CreateFakeValueNode(true, origin_node, true);
  ASSERT_EQ(fake_value_node2->abstract()->cast<abstract::AbstractTensorPtr>()->shape()->shape(), shp);
}

/// Feature: Distributed runtime.
/// Description: Test generating inter-process communication edges for a node with inputs from other processes.
/// Expectation: Edges are successfully created with correct peer nodes.
TEST_F(GraphSplitterTest, TestGenerateInterProcessOpsForNodeInputs) {
  std::vector<AnfNodePtr> input1_node_inputs{NewValueNode(prim::kPrimMatMul)};
  auto input1 = func_graph_->NewCNode(input1_node_inputs);
  MS_EXCEPTION_IF_NULL(input1);
  input1->set_abstract(CreateAbs());
  splitter_->node_labels_[input1].rank_id = 1;
  splitter_->node_labels_[input1].ms_role = "MS_SERVER";

  std::vector<AnfNodePtr> input2_node_inputs{NewValueNode(prim::kPrimMatMul)};
  auto input2 = func_graph_->NewCNode(input2_node_inputs);
  MS_EXCEPTION_IF_NULL(input2);
  input2->set_abstract(CreateAbs());
  splitter_->node_labels_[input2].rank_id = 2;
  splitter_->node_labels_[input2].ms_role = "MS_PSERVER";

  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimAdd), input1, input2};
  auto split_node = func_graph_->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(split_node);
  splitter_->node_labels_[split_node].rank_id = 0;
  splitter_->node_labels_[split_node].ms_role = "MS_WORKER";

  auto comm_edges = splitter_->GenerateInterProcessOpsForNodeInputs(split_node);

  ASSERT_EQ(comm_edges.size(), 2);

  const auto &edge1 = comm_edges.begin()->first;
  std::string expected_edge =
    input1->fullname_with_scope() + "_1_MS_SERVER" + "->" + split_node->fullname_with_scope() + "_0_MS_WORKER";
  ASSERT_EQ(edge1.to_string(), expected_edge);
}
}  // namespace parallel
}  // namespace mindspore
