/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include <memory>

#include "abstract/abstract_value.h"
#include "common/common_test.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "primitive/framework_ops.h"
#include "utils/hyper_offload/hyper_offload_utils.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {

class TestHyperOffloadViewInplace : public UT::Common {
 public:
  TestHyperOffloadViewInplace() = default;
};

// Feature: HyperOffload Utils
// Description: Test IsViewNode to check if a node is marked as view node
// Expectation: Returns true when node has is_view user data, false otherwise
TEST_F(TestHyperOffloadViewInplace, TestIsViewNode) {
  auto funcGraph = std::make_shared<FuncGraph>();
  auto node = funcGraph->add_parameter();

  EXPECT_FALSE(IsViewNode(node));

  node->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  EXPECT_TRUE(IsViewNode(node));
}

// Feature: HyperOffload Utils
// Description: Test IsInplaceNode to check if a node is marked as inplace node
// Expectation: Returns true when node has is_inplace user data, false otherwise
TEST_F(TestHyperOffloadViewInplace, TestIsInplaceNode) {
  auto funcGraph = std::make_shared<FuncGraph>();
  auto node = funcGraph->add_parameter();

  EXPECT_FALSE(IsInplaceNode(node));

  node->set_user_data<bool>("is_inplace", std::make_shared<bool>(true));
  EXPECT_TRUE(IsInplaceNode(node));
}

// Feature: HyperOffload Utils
// Description: Test GetRealDataNodeWithIndex to get the original data node from view nodes
// Expectation: Returns the original data node by tracing through view nodes
TEST_F(TestHyperOffloadViewInplace, TestGetRealDataNodeWithIndex) {
  auto funcGraph = std::make_shared<FuncGraph>();
  auto dataNode = funcGraph->add_parameter();

  auto viewPrimitive = std::make_shared<Primitive>("ViewPrim");
  auto viewNode1 = funcGraph->NewCNode({NewValueNode(viewPrimitive), dataNode});
  viewNode1->set_user_data<bool>("is_view", std::make_shared<bool>(true));

  auto viewNode2 = funcGraph->NewCNode({NewValueNode(viewPrimitive), viewNode1});
  viewNode2->set_user_data<bool>("is_view", std::make_shared<bool>(true));

  EXPECT_EQ(GetRealDataNodeWithIndex(viewNode2).first, dataNode);
  EXPECT_EQ(GetRealDataNodeWithIndex(viewNode1).first, dataNode);
  EXPECT_EQ(GetRealDataNodeWithIndex(dataNode).first, dataNode);
}

// Feature: HyperOffload Utils
// Description: Test GetRealDataNodeWithIndex with tuple output and TupleGetItem
// Expectation: Returns the original data node
TEST_F(TestHyperOffloadViewInplace, TestGetRealDataNodeWithIndexTupleOutput) {
  auto funcGraph = std::make_shared<FuncGraph>();
  auto dataNode = funcGraph->add_parameter();

  auto viewPrimitive = std::make_shared<Primitive>("ViewPrim");
  auto viewNode = funcGraph->NewCNode({NewValueNode(viewPrimitive), dataNode});
  viewNode->set_user_data<bool>("is_view", std::make_shared<bool>(true));

  auto tupleGetItemPrimitive = prim::kPrimTupleGetItem;
  auto tupleGetItemNode =
    funcGraph->NewCNode({NewValueNode(tupleGetItemPrimitive), viewNode, NewValueNode(static_cast<int64_t>(0))});

  EXPECT_EQ(GetRealDataNodeWithIndex(tupleGetItemNode).first, dataNode);
}

// Feature: HyperOffload Utils
// Description: Test IsValidOffloadNode when view and inplace conflict exists
// Expectation: Returns false when both view and inplace users exist for the same node
TEST_F(TestHyperOffloadViewInplace, TestIsValidOffloadNodeConflict) {
  auto funcGraph = std::make_shared<FuncGraph>();
  auto tensorAbstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1});

  auto inputNode = funcGraph->add_parameter();
  auto matmulPrimitive = std::make_shared<Primitive>("MatMul");
  auto computeNode = funcGraph->NewCNode({NewValueNode(matmulPrimitive), inputNode, inputNode});
  computeNode->set_abstract(tensorAbstract);

  auto viewPrimitive = std::make_shared<Primitive>("View");
  auto viewUserNode = funcGraph->NewCNode({NewValueNode(viewPrimitive), computeNode});
  viewUserNode->set_user_data<bool>("is_view", std::make_shared<bool>(true));

  auto inplacePrimitive = std::make_shared<Primitive>("Inplace");
  auto inplaceUserNode = funcGraph->NewCNode({NewValueNode(inplacePrimitive), computeNode});
  inplaceUserNode->set_user_data<bool>("is_inplace", std::make_shared<bool>(true));

  UserInfo userInfo;
  userInfo.node = {computeNode, 0};
  userInfo.users.push_back({viewUserNode, 1});
  userInfo.users.push_back({inplaceUserNode, 1});

  EXPECT_FALSE(IsValidOffloadNode(userInfo));
}

// Feature: HyperOffload Utils
// Description: Test IsValidOffloadNode when only view user exists
// Expectation: Returns true when only view user exists without conflict
TEST_F(TestHyperOffloadViewInplace, TestIsValidOffloadNodeNoConflict) {
  auto funcGraph = std::make_shared<FuncGraph>();
  auto tensorAbstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1});
  auto inputNode = funcGraph->add_parameter();
  auto matmulPrimitive = std::make_shared<Primitive>("MatMul");
  auto computeNode = funcGraph->NewCNode({NewValueNode(matmulPrimitive), inputNode, inputNode});
  computeNode->set_abstract(tensorAbstract);

  auto viewPrimitive = std::make_shared<Primitive>("View");
  auto viewUserNode = funcGraph->NewCNode({NewValueNode(viewPrimitive), computeNode});
  viewUserNode->set_user_data<bool>("is_view", std::make_shared<bool>(true));

  UserInfo userInfo;
  userInfo.node = {computeNode, 0};
  userInfo.users.push_back({viewUserNode, 1});

  EXPECT_TRUE(IsValidOffloadNode(userInfo));
}

// Feature: HyperOffload Utils
// Description: Test CollectAllNodeUsers to collect all users of a node including view nodes
// Expectation: Returns all users including view nodes in the execution order
TEST_F(TestHyperOffloadViewInplace, TestCollectAllNodeUsersWithView) {
  auto funcGraph = std::make_shared<FuncGraph>();
  auto parameterNode = funcGraph->add_parameter();
  auto tensorAbstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1});
  parameterNode->set_abstract(tensorAbstract);

  auto absPrimitive = std::make_shared<Primitive>("Abs");
  auto inputNode = funcGraph->NewCNode({NewValueNode(absPrimitive), parameterNode});
  inputNode->set_abstract(tensorAbstract);

  auto viewPrimitive = std::make_shared<Primitive>("ViewPrim");
  auto viewNode = funcGraph->NewCNode({NewValueNode(viewPrimitive), inputNode});
  viewNode->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  viewNode->set_abstract(tensorAbstract);

  auto addPrimitive = std::make_shared<Primitive>("Add");
  auto addNode = funcGraph->NewCNode({NewValueNode(addPrimitive), viewNode});
  addNode->set_abstract(tensorAbstract);

  CNodePtrList executionOrder = {inputNode, viewNode, addNode};
  auto userInfos = CollectAllNodeUsers(executionOrder);

  bool foundInputNode = false;
  for (const auto &userInfo : userInfos) {
    if (userInfo.node.first == inputNode) {
      foundInputNode = true;
      bool hasAddUser = false;
      for (const auto &user : userInfo.users) {
        if (user.first == addNode) {
          hasAddUser = true;
        }
      }
      EXPECT_TRUE(hasAddUser);
    }
  }
  EXPECT_TRUE(foundInputNode);
}

// Feature: HyperOffload Utils
// Description: Test CollectAllNodeUsers to skip view nodes when collecting users
// Expectation: Returns only non-view nodes as users
TEST_F(TestHyperOffloadViewInplace, TestCollectAllNodeUsersSkipViewNode) {
  auto funcGraph = std::make_shared<FuncGraph>();
  auto parameterNode = funcGraph->add_parameter();
  auto tensorAbstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1});
  parameterNode->set_abstract(tensorAbstract);

  auto absPrimitive = std::make_shared<Primitive>("Abs");
  auto inputNode = funcGraph->NewCNode({NewValueNode(absPrimitive), parameterNode});
  inputNode->set_abstract(tensorAbstract);

  auto viewPrimitive = std::make_shared<Primitive>("ViewPrim");
  auto viewNode = funcGraph->NewCNode({NewValueNode(viewPrimitive), inputNode});
  viewNode->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  viewNode->set_abstract(tensorAbstract);

  CNodePtrList executionOrder = {inputNode, viewNode};
  auto userInfos = CollectAllNodeUsers(executionOrder);

  EXPECT_FALSE(userInfos.empty());
  EXPECT_EQ(userInfos.size(), 1);
  EXPECT_EQ(userInfos[0].node.first, inputNode);
  EXPECT_EQ(userInfos[0].users.size(), 1);
  EXPECT_EQ(userInfos[0].users[0].first, viewNode);
}

// Feature: HyperOffload Utils
// Description: Test CollectAllNodeUsers with tuple producer outputs
// Expectation: Tuple producer outputs are collected as distinct KernelWithIndex entries only in peak strategy mode
TEST_F(TestHyperOffloadViewInplace, test_collect_all_node_users_tuple_producer_outputs) {
  auto funcGraph = std::make_shared<FuncGraph>();
  auto tensorAbstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1});
  auto tupleAbstract =
    std::make_shared<abstract::AbstractTuple>(abstract::AbstractBasePtrList{tensorAbstract, tensorAbstract});

  auto sourcePrimitive = std::make_shared<Primitive>("TupleSource");
  auto sourceNode = funcGraph->NewCNode({NewValueNode(sourcePrimitive)});
  sourceNode->set_abstract(tupleAbstract);

  auto tupleGetItemPrimitive = prim::kPrimTupleGetItem;
  auto getItemNode0 =
    funcGraph->NewCNode({NewValueNode(tupleGetItemPrimitive), sourceNode, NewValueNode(static_cast<int64_t>(0))});
  getItemNode0->set_abstract(tensorAbstract);
  auto getItemNode1 =
    funcGraph->NewCNode({NewValueNode(tupleGetItemPrimitive), sourceNode, NewValueNode(static_cast<int64_t>(1))});
  getItemNode1->set_abstract(tensorAbstract);

  auto userPrimitive0 = std::make_shared<Primitive>("User0");
  auto userNode0 = funcGraph->NewCNode({NewValueNode(userPrimitive0), getItemNode0});
  userNode0->set_abstract(tensorAbstract);
  auto userPrimitive1 = std::make_shared<Primitive>("User1");
  auto userNode1 = funcGraph->NewCNode({NewValueNode(userPrimitive1), getItemNode1});
  userNode1->set_abstract(tensorAbstract);

  CNodePtrList executionOrder = {sourceNode, userNode0, userNode1};
  auto userInfos = CollectAllNodeUsers(executionOrder);
  const bool peakStrategyEnabled = HyperOffloadConfig::GetInstance().enable_hyper_offload_peak_strategy();

  bool foundOutput0 = false;
  bool foundOutput1 = false;
  size_t sourceNodeEntryCount = 0;
  for (const auto &userInfo : userInfos) {
    if (userInfo.node.first != sourceNode) {
      continue;
    }
    ++sourceNodeEntryCount;
    if (userInfo.node.second == 0) {
      foundOutput0 = true;
      ASSERT_EQ(userInfo.users.size(), 1);
      EXPECT_EQ(userInfo.users[0].first, userNode0);
    } else if (userInfo.node.second == 1) {
      foundOutput1 = true;
      ASSERT_EQ(userInfo.users.size(), 1);
      EXPECT_EQ(userInfo.users[0].first, userNode1);
    }
  }

  if (peakStrategyEnabled) {
    EXPECT_EQ(sourceNodeEntryCount, 2);
    EXPECT_TRUE(foundOutput0);
    EXPECT_TRUE(foundOutput1);
  } else {
    EXPECT_EQ(sourceNodeEntryCount, 0);
    EXPECT_FALSE(foundOutput0);
    EXPECT_FALSE(foundOutput1);
  }
}

// Feature: HyperOffload Utils
// Description: Test CollectAllNodeUsers with tuple producer event output
// Expectation: Event tuple outputs are not collected as HyperOffload candidates
TEST_F(TestHyperOffloadViewInplace, test_collect_all_node_users_skips_event_tuple_output) {
  auto funcGraph = std::make_shared<FuncGraph>();
  auto tensorAbstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1});
  auto eventAbstract =
    std::make_shared<abstract::AbstractEvent>(std::make_shared<Event>(0), std::make_shared<EventType>(), 0);
  auto tupleAbstract =
    std::make_shared<abstract::AbstractTuple>(abstract::AbstractBasePtrList{eventAbstract, tensorAbstract});

  auto sourcePrimitive = std::make_shared<Primitive>("TupleSource");
  auto sourceNode = funcGraph->NewCNode({NewValueNode(sourcePrimitive)});
  sourceNode->set_abstract(tupleAbstract);

  auto getEventNode =
    funcGraph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), sourceNode, NewValueNode(static_cast<int64_t>(0))});
  getEventNode->set_abstract(eventAbstract);
  auto recvNode = funcGraph->NewCNode({NewValueNode(prim::kPrimStreamRecv), getEventNode});
  recvNode->set_abstract(eventAbstract);

  auto getTensorNode =
    funcGraph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), sourceNode, NewValueNode(static_cast<int64_t>(1))});
  getTensorNode->set_abstract(tensorAbstract);
  auto userPrimitive = std::make_shared<Primitive>("TensorUser");
  auto userNode = funcGraph->NewCNode({NewValueNode(userPrimitive), getTensorNode});
  userNode->set_abstract(tensorAbstract);

  CNodePtrList executionOrder = {sourceNode, recvNode, userNode};
  auto userInfos = CollectAllNodeUsers(executionOrder);
  const bool peakStrategyEnabled = HyperOffloadConfig::GetInstance().enable_hyper_offload_peak_strategy();

  bool foundEventOutput = false;
  bool foundTensorOutput = false;
  for (const auto &userInfo : userInfos) {
    if (userInfo.node.first != sourceNode) {
      continue;
    }
    foundEventOutput = foundEventOutput || userInfo.node.second == 0;
    foundTensorOutput = foundTensorOutput || userInfo.node.second == 1;
  }

  EXPECT_FALSE(foundEventOutput);
  EXPECT_EQ(foundTensorOutput, peakStrategyEnabled);
}

// Feature: HyperOffload Utils
// Description: Test CollectAllNodeUsers with nested view nodes and TupleGetItem
// Expectation: Correctly identifies target users and skips TupleGetItem users
TEST_F(TestHyperOffloadViewInplace, test_collect_all_node_users_nested_view_tuple_get_item) {
  auto funcGraph = std::make_shared<FuncGraph>();
  auto tensorAbstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1});

  auto sourcePrimitive = std::make_shared<Primitive>("SourceOp");
  auto sourceNode = funcGraph->NewCNode({NewValueNode(sourcePrimitive)});
  sourceNode->set_abstract(tensorAbstract);

  auto viewPrimitive1 = std::make_shared<Primitive>("View1");
  auto viewNode1 = funcGraph->NewCNode({NewValueNode(viewPrimitive1), sourceNode});
  viewNode1->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  viewNode1->set_abstract(tensorAbstract);

  auto viewPrimitive2 = std::make_shared<Primitive>("View2");
  auto viewNode2 = funcGraph->NewCNode({NewValueNode(viewPrimitive2), viewNode1});
  viewNode2->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  auto tupleAbstract = std::make_shared<abstract::AbstractTuple>(abstract::AbstractBasePtrList{tensorAbstract});
  viewNode2->set_abstract(tupleAbstract);

  auto tupleGetItemPrimitive = prim::kPrimTupleGetItem;
  auto getItemNode =
    funcGraph->NewCNode({NewValueNode(tupleGetItemPrimitive), viewNode2, NewValueNode(static_cast<int64_t>(0))});
  getItemNode->set_abstract(tensorAbstract);

  auto viewPrimitive3 = std::make_shared<Primitive>("View3");
  auto viewNode3 = funcGraph->NewCNode({NewValueNode(viewPrimitive3), getItemNode});
  viewNode3->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  viewNode3->set_abstract(tensorAbstract);

  auto targetPrimitive = std::make_shared<Primitive>("TargetOp");
  auto targetNode = funcGraph->NewCNode({NewValueNode(targetPrimitive), viewNode3});
  targetNode->set_abstract(tensorAbstract);

  CNodePtrList executionOrder = {sourceNode, viewNode1, viewNode2, viewNode3, targetNode};
  auto userInfos = CollectAllNodeUsers(executionOrder);

  bool foundTargetUser = false;
  bool foundGetItemUser = false;
  for (const auto &userInfo : userInfos) {
    if (userInfo.node.first == sourceNode) {
      for (const auto &user : userInfo.users) {
        if (user.first == targetNode) {
          foundTargetUser = true;
        }
        if (user.first == getItemNode) {
          foundGetItemUser = true;
        }
      }
    }
  }
  EXPECT_TRUE(foundTargetUser);
  EXPECT_FALSE(foundGetItemUser);
}

}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
