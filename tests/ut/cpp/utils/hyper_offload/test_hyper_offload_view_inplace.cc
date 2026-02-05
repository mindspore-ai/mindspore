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

#include "common/common_test.h"
#include "utils/hyper_offload/hyper_offload_utils.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {

class TestHyperOffloadViewInplace : public UT::Common {
 public:
  TestHyperOffloadViewInplace() = default;
};

// Feature: HyperOffload Utils
// Description: Test IsViewNode with user_data "is_view".
// Expectation: Returns true when user_data is set.
TEST_F(TestHyperOffloadViewInplace, TestIsViewNode) {
  auto fg = std::make_shared<FuncGraph>();
  auto node = fg->add_parameter();
  
  EXPECT_FALSE(IsViewNode(node));
  
  node->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  EXPECT_TRUE(IsViewNode(node));
}

// Feature: HyperOffload Utils
// Description: Test IsInplaceNode with user_data "is_inplace".
// Expectation: Returns true when user_data is set.
TEST_F(TestHyperOffloadViewInplace, TestIsInplaceNode) {
  auto fg = std::make_shared<FuncGraph>();
  auto node = fg->add_parameter();
  
  EXPECT_FALSE(IsInplaceNode(node));
  
  node->set_user_data<bool>("is_inplace", std::make_shared<bool>(true));
  EXPECT_TRUE(IsInplaceNode(node));
}

// Feature: HyperOffload Utils
// Description: Test GetRealDataNodeWithIndex with nested view nodes.
// Expectation: Returns the original data node.
TEST_F(TestHyperOffloadViewInplace, TestGetRealDataNodeWithIndex) {
  auto fg = std::make_shared<FuncGraph>();
  auto data_node = fg->add_parameter();
  
  // Create ViewNode1 taking data_node as input
  auto prim_view = std::make_shared<Primitive>("ViewPrim");
  auto view_node1 = fg->NewCNode({NewValueNode(prim_view), data_node});
  view_node1->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  
  // Create ViewNode2 taking view_node1 as input
  auto view_node2 = fg->NewCNode({NewValueNode(prim_view), view_node1});
  view_node2->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  
  // GetRealDataNodeWithIndex should traverse back
  EXPECT_EQ(GetRealDataNodeWithIndex(view_node2).first, data_node);
  EXPECT_EQ(GetRealDataNodeWithIndex(view_node1).first, data_node);
  EXPECT_EQ(GetRealDataNodeWithIndex(data_node).first, data_node);
}

// Feature: HyperOffload Utils
// Description: Test GetRealDataNodeWithIndex with tuple output and TupleGetItem.
// Expectation: Returns the original data node.
TEST_F(TestHyperOffloadViewInplace, TestGetRealDataNodeWithIndex_TupleOutput) {
  auto fg = std::make_shared<FuncGraph>();
  auto data_node = fg->add_parameter();
  
  // Create ViewNode returning tuple
  auto prim_view = std::make_shared<Primitive>("ViewPrim");
  auto view_node = fg->NewCNode({NewValueNode(prim_view), data_node});
  view_node->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  
  // Create TupleGetItem(view_node, 0)
  auto tuple_get_item_prim = prim::kPrimTupleGetItem;
  auto get_item_node = fg->NewCNode({NewValueNode(tuple_get_item_prim), view_node, NewValueNode(static_cast<int64_t>(0))});
  
  // GetRealDataNodeWithIndex(get_item_node) should traverse back through TupleGetItem to View, then to data_node
  EXPECT_EQ(GetRealDataNodeWithIndex(get_item_node).first, data_node);
}

// Feature: HyperOffload Utils
// Description: Test IsValidOffloadNode when both view and inplace users exist.
// Expectation: Returns false due to conflict.
TEST_F(TestHyperOffloadViewInplace, TestIsValidOffloadNode_Conflict) {
  auto fg = std::make_shared<FuncGraph>();
  auto abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1});
  
  auto input = fg->add_parameter();
  auto prim = std::make_shared<Primitive>("MatMul");
  auto cnode = fg->NewCNode({NewValueNode(prim), input, input});
  cnode->set_abstract(abs);
  
  // Create users
  auto prim_view = std::make_shared<Primitive>("View");
  auto view_user = fg->NewCNode({NewValueNode(prim_view), cnode});
  view_user->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  
  auto prim_inplace = std::make_shared<Primitive>("Inplace");
  auto inplace_user = fg->NewCNode({NewValueNode(prim_inplace), cnode});
  inplace_user->set_user_data<bool>("is_inplace", std::make_shared<bool>(true));
  
  UserInfo user_info;
  user_info.node = {cnode, 0};
  // Users list needs to contain pairs of (Node, index)
  user_info.users.push_back({view_user, 1});
  user_info.users.push_back({inplace_user, 1});
  
  // Should return false because of conflict
  EXPECT_FALSE(IsValidOffloadNode(user_info));
}

// Feature: HyperOffload Utils
// Description: Test IsValidOffloadNode when only view user exists.
// Expectation: Returns true (assuming other conditions met).
TEST_F(TestHyperOffloadViewInplace, TestIsValidOffloadNode_NoConflict) {
    auto fg = std::make_shared<FuncGraph>();
    auto abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1});
    auto input = fg->add_parameter();
    auto prim = std::make_shared<Primitive>("MatMul");
    auto cnode = fg->NewCNode({NewValueNode(prim), input, input});
    cnode->set_abstract(abs);

    auto prim_view = std::make_shared<Primitive>("View");
    auto view_user = fg->NewCNode({NewValueNode(prim_view), cnode});
    view_user->set_user_data<bool>("is_view", std::make_shared<bool>(true));

    UserInfo user_info;
    user_info.node = {cnode, 0};
    user_info.users.push_back({view_user, 1});

    // Should return true (assuming other checks pass)
    EXPECT_TRUE(IsValidOffloadNode(user_info));
}

// Feature: HyperOffload Utils
// Description: Test CollectAllNodeUsers with view node in input chain.
// Expectation: The original data node is identified as the source.
TEST_F(TestHyperOffloadViewInplace, TestCollectAllNodeUsers_WithView) {
  auto fg = std::make_shared<FuncGraph>();
  auto p1 = fg->add_parameter();
  auto abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1});
  p1->set_abstract(abs);

  // Create a CNode as the source of data, because CollectAllNodeUsers ignores Parameters.
  auto prim_abs = std::make_shared<Primitive>("Abs");
  auto input_cnode = fg->NewCNode({NewValueNode(prim_abs), p1});
  input_cnode->set_abstract(abs);

  // input_cnode -> View -> Add
  auto prim_view = std::make_shared<Primitive>("ViewPrim");
  auto view_node = fg->NewCNode({NewValueNode(prim_view), input_cnode});
  view_node->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  view_node->set_abstract(abs);

  auto prim_add = std::make_shared<Primitive>("Add");
  auto add_node = fg->NewCNode({NewValueNode(prim_add), view_node});
  add_node->set_abstract(abs);

  CNodePtrList nodes = {input_cnode, view_node, add_node};
  auto user_infos = CollectAllNodeUsers(nodes);

  // We expect to find input_cnode in user_infos, with add_node as its user.
  bool found_input = false;
  for (const auto& info : user_infos) {
    if (info.node.first == input_cnode) {
      found_input = true;
      bool has_add = false;
      for (const auto& u : info.users) {
        if (u.first == add_node) has_add = true;
      }
      EXPECT_TRUE(has_add);
    }
  }
  EXPECT_TRUE(found_input);
}

// Feature: HyperOffload Utils
// Description: Test CollectAllNodeUsers skips view nodes as users.
// Expectation: View nodes are not ignored as users.
TEST_F(TestHyperOffloadViewInplace, TestCollectAllNodeUsers_SkipViewNode) {
  auto fg = std::make_shared<FuncGraph>();
  auto p1 = fg->add_parameter();
  auto abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1});
  p1->set_abstract(abs);

  // Use a CNode input to ensure the test passes because of IsViewNode check,
  // not because of the Parameter check.
  auto prim_abs = std::make_shared<Primitive>("Abs");
  auto input_cnode = fg->NewCNode({NewValueNode(prim_abs), p1});
  input_cnode->set_abstract(abs);

  auto prim_view = std::make_shared<Primitive>("ViewPrim");
  auto view_node = fg->NewCNode({NewValueNode(prim_view), input_cnode});
  view_node->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  view_node->set_abstract(abs);

  CNodePtrList nodes = {input_cnode, view_node};
  auto user_infos = CollectAllNodeUsers(nodes);

  // user_infos should contain users of abs, should not contain users of view.
  EXPECT_FALSE(user_infos.empty());
  EXPECT_EQ(user_infos.size(), 1);
  EXPECT_EQ(user_infos[0].node.first, input_cnode);
  EXPECT_EQ(user_infos[0].users.size(), 1);
  EXPECT_EQ(user_infos[0].users[0].first, view_node);
}

// Feature: HyperOffload Utils
// Description: Test CollectAllNodeUsers with View + View(Tuple) + GetItem + View.
// Expectation: The original data node is identified as the source for the final user.
TEST_F(TestHyperOffloadViewInplace, TestCollectAllNodeUsers_NestedViewTupleGetItem) {
  auto fg = std::make_shared<FuncGraph>();
  auto abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1});

  // Source Node A
  auto prim_a = std::make_shared<Primitive>("SourceOp");
  auto node_a = fg->NewCNode({NewValueNode(prim_a)});
  node_a->set_abstract(abs);

  // View1: Node A -> View1
  auto prim_view1 = std::make_shared<Primitive>("View1");
  auto view1 = fg->NewCNode({NewValueNode(prim_view1), node_a});
  view1->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  view1->set_abstract(abs);

  // View2: View1 -> View2 (Outputs Tuple)
  auto prim_view2 = std::make_shared<Primitive>("View2");
  auto view2 = fg->NewCNode({NewValueNode(prim_view2), view1});
  view2->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  // AbstractTuple for view2
  auto abs_tuple = std::make_shared<abstract::AbstractTuple>(abstract::AbstractBasePtrList{abs});
  view2->set_abstract(abs_tuple);

  // GetItem: View2 -> GetItem
  auto prim_getitem = prim::kPrimTupleGetItem;
  auto getitem = fg->NewCNode({NewValueNode(prim_getitem), view2, NewValueNode(static_cast<int64_t>(0))});
  getitem->set_abstract(abs);

  // View3: GetItem -> View3
  auto prim_view3 = std::make_shared<Primitive>("View3");
  auto view3 = fg->NewCNode({NewValueNode(prim_view3), getitem});
  view3->set_user_data<bool>("is_view", std::make_shared<bool>(true));
  view3->set_abstract(abs);

  // Node B: View3 -> Node B
  auto prim_b = std::make_shared<Primitive>("TargetOp");
  auto node_b = fg->NewCNode({NewValueNode(prim_b), view3});
  node_b->set_abstract(abs);

  CNodePtrList nodes = {node_a, view1, view2, view3, node_b};
  auto user_infos = CollectAllNodeUsers(nodes);

  // Verification: Node B should be a user of Node A
  bool found_link = false;
  bool found_getitem = false;
  for (const auto& info : user_infos) {
    if (info.node.first == node_a) {
      for (const auto& u : info.users) {
        if (u.first == node_b) found_link = true;
        if (u.first == getitem) found_getitem = true;
      }
    }
  }
  EXPECT_TRUE(found_link);
  EXPECT_FALSE(found_getitem);
}
} // namespace hyper_offload
} // namespace utils
} // namespace mindspore
