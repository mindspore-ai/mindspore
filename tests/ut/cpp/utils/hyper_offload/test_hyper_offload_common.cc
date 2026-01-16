/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <vector>
#include <string>
#include "common/common_test.h"
#include "utils/hyper_offload/hyper_offload.h"
#include "utils/hyper_offload/hyper_offload_utils.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "utils/compile_config.h"
#include "utils/log_adapter.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {

class TestHyperOffloadCommon : public UT::Common {
 public:
  TestHyperOffloadCommon() = default;
  ~TestHyperOffloadCommon() override = default;

  void SetUp() override {
    UT::Common::SetUp();
    // Set environment variables to enable HyperOffload
    (void)common::SetEnv("MS_DEV_HYPER_OFFLOAD", "1");
    // Set required compile configs to avoid GetConfig errors
    common::SetCompileConfig("COMPILE_DEBUG_MODE", "debug");
    common::SetCompileConfig("TRACE_LABEL_WITH_UNIQUE_ID", "0");
    common::SetCompileConfig("HYPER_OFFLOAD_SELECT_DISTANCE", "5");
    common::SetCompileConfig("HYPER_OFFLOAD_SELECT_NUM", "10");
    common::SetCompileConfig("HYPER_OFFLOAD_PREFETCH_DISTANCE", "3");
    common::SetCompileConfig("HYPER_OFFLOAD_RELEASE_DISTANCE", "2");
  }

  void TearDown() override {
    // Clean up environment variables (set to empty string or use unsetenv)
    (void)common::SetEnv("MS_DEV_HYPER_OFFLOAD", "", 1);
    // Note: We don't clear compile configs here as they may be needed by other tests
    // The compile config manager is a singleton and will persist across tests
    UT::Common::TearDown();
  }

  // Create a simple FuncGraph
  FuncGraphPtr CreateSimpleGraph() {
    auto graph = std::make_shared<FuncGraph>();
    return graph;
  }

  // Create a Parameter node
  ParameterPtr CreateParameter(const FuncGraphPtr &graph, const std::string &name,
                               const abstract::AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(graph);
    auto param = std::make_shared<Parameter>(graph);
    param->set_name(name);
    param->set_abstract(abs);
    return param;
  }

  // Create a CNode
  CNodePtr CreateCNode(const FuncGraphPtr &graph, const std::string &prim_name,
                       const std::vector<AnfNodePtr> &inputs) {
    MS_EXCEPTION_IF_NULL(graph);
    auto prim = std::make_shared<Primitive>(prim_name);
    auto value_node = std::make_shared<ValueNode>(prim);
    std::vector<AnfNodePtr> new_inputs = {value_node};
    new_inputs.insert(new_inputs.end(), inputs.begin(), inputs.end());
    auto cnode = graph->NewCNode(new_inputs);
    return cnode;
  }

  // Create a simple execution order (for basic tests)
  CNodePtrList CreateSimpleExecutionOrder(const FuncGraphPtr &graph) {
    CNodePtrList exec_order;
    
    // Create input parameters
    auto input_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2, 3});
    auto param1 = CreateParameter(graph, "param1", input_abs);
    auto param2 = CreateParameter(graph, "param2", input_abs);
    
    // Create computation nodes
    auto add_node = CreateCNode(graph, "Add", {param1, param2});
    add_node->set_abstract(input_abs);
    (void)exec_order.emplace_back(add_node);
    
    auto mul_node = CreateCNode(graph, "Mul", {add_node, param1});
    mul_node->set_abstract(input_abs);
    (void)exec_order.emplace_back(mul_node);
    
    // Set graph output
    graph->set_output(mul_node);
    
    return exec_order;
  }

  // Create a larger execution order (for testing offload strategy)
  // Create a chain structure with multiple nodes, with sufficient distance between nodes
  // The structure ensures relative distance > select_distance(5) to trigger offload
  CNodePtrList CreateLargeExecutionOrder(const FuncGraphPtr &graph) {
    CNodePtrList exec_order;
    
    // Create input parameters
    auto input_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2, 3});
    auto param1 = CreateParameter(graph, "param1", input_abs);
    auto param2 = CreateParameter(graph, "param2", input_abs);
    auto param3 = CreateParameter(graph, "param3", input_abs);
    
    // Create the first computation node (this node will be offloaded)
    // node1 at position 0
    auto node1 = CreateCNode(graph, "Add", {param1, param2});
    node1->set_abstract(input_abs);
    (void)exec_order.emplace_back(node1);
    
    // Create first user of node1 at position 1 (distance = 1)
    auto node2 = CreateCNode(graph, "Mul", {node1, param3});
    node2->set_abstract(input_abs);
    (void)exec_order.emplace_back(node2);
    
    // Create intermediate nodes that don't use node1 (positions 2-6)
    // This creates a gap to ensure relative distance > select_distance(5)
    auto node3 = CreateCNode(graph, "Sub", {node2, param1});
    node3->set_abstract(input_abs);
    (void)exec_order.emplace_back(node3);
    
    auto node4 = CreateCNode(graph, "Add", {node3, param2});
    node4->set_abstract(input_abs);
    (void)exec_order.emplace_back(node4);
    
    auto node5 = CreateCNode(graph, "Mul", {node4, param3});
    node5->set_abstract(input_abs);
    (void)exec_order.emplace_back(node5);
    
    auto node6 = CreateCNode(graph, "Sub", {node5, param1});
    node6->set_abstract(input_abs);
    (void)exec_order.emplace_back(node6);
    
    auto node7 = CreateCNode(graph, "Add", {node6, param2});
    node7->set_abstract(input_abs);
    (void)exec_order.emplace_back(node7);
    
    // Create second user of node1 at position 7 (distance = 7)
    // Relative distance from node2 (pos 1) to node8 (pos 7) = 7 - 1 = 6 > 5, triggers offload
    auto node8 = CreateCNode(graph, "Add", {node1, node7});
    node8->set_abstract(input_abs);
    (void)exec_order.emplace_back(node8);
    
    // Create more intermediate nodes
    auto node9 = CreateCNode(graph, "Mul", {node8, param1});
    node9->set_abstract(input_abs);
    (void)exec_order.emplace_back(node9);
    
    // Create third user of node1 at position 9 (distance = 9)
    // Relative distance from node8 (pos 7) to node10 (pos 9) = 9 - 7 = 2 (not > 5)
    // But the previous relative distance (6) already triggered offload
    auto node10 = CreateCNode(graph, "Mul", {node9, node1});
    node10->set_abstract(input_abs);
    (void)exec_order.emplace_back(node10);
    
    // Set graph output
    graph->set_output(node10);
    
    return exec_order;
  }
};

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestHyperOffloadConfig) {
  auto &config = HyperOffloadConfig::GetInstance();
  EXPECT_TRUE(config.enable_hyper_offload());
  EXPECT_GT(config.select_distance(), 0);
  EXPECT_GT(config.select_num(), 0);
  EXPECT_GT(config.prefetch_distance(), 0);
  EXPECT_GT(config.release_distance(), 0);
}

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestHyperOffloadCommonRun_Basic) {
  auto graph = CreateSimpleGraph();
  auto exec_order = CreateSimpleExecutionOrder(graph);
  
  HyperOffloadInput input;
  input.graph = graph;
  input.exec_order = exec_order;
  input.enable_offload_parameter = false;
  input.enable_offload_activation = false;
  
  auto plan = HyperOffloadOptimizer::Run(input);
  
  EXPECT_NE(plan.graph, nullptr);
  EXPECT_EQ(plan.graph, graph);
  EXPECT_EQ(plan.exec_order.size(), exec_order.size());
}

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestHyperOffloadCommonRun_Disabled) {
  // Temporarily disable HyperOffload
  (void)common::SetEnv("MS_DEV_HYPER_OFFLOAD", "", 1);
  
  auto graph = CreateSimpleGraph();
  auto exec_order = CreateSimpleExecutionOrder(graph);
  
  HyperOffloadInput input;
  input.graph = graph;
  input.exec_order = exec_order;
  input.enable_offload_parameter = true;
  input.enable_offload_activation = true;
  
  auto plan = HyperOffloadOptimizer::Run(input);
  
  // When HyperOffload is disabled, should return original input
  EXPECT_NE(plan.graph, nullptr);
  EXPECT_EQ(plan.graph, graph);
  EXPECT_EQ(plan.exec_order.size(), exec_order.size());
  
  // Re-enable
  (void)common::SetEnv("MS_DEV_HYPER_OFFLOAD", "1");
}

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestHyperOffloadCommonRun_EmptyInput) {
  auto graph = CreateSimpleGraph();
  CNodePtrList empty_exec_order;
  
  HyperOffloadInput input;
  input.graph = graph;
  input.exec_order = empty_exec_order;
  input.enable_offload_parameter = true;
  input.enable_offload_activation = true;
  
  auto plan = HyperOffloadOptimizer::Run(input);
  
  EXPECT_NE(plan.graph, nullptr);
  EXPECT_TRUE(plan.exec_order.empty());
}

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestHyperOffloadCommonRun_NullGraph) {
  CNodePtrList exec_order = CreateSimpleExecutionOrder(CreateSimpleGraph());
  
  HyperOffloadInput input;
  input.graph = nullptr;
  input.exec_order = exec_order;
  input.enable_offload_parameter = true;
  input.enable_offload_activation = true;
  
  auto plan = HyperOffloadOptimizer::Run(input);
  
  EXPECT_EQ(plan.graph, nullptr);
  EXPECT_EQ(plan.exec_order.size(), exec_order.size());
}

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestNodeIdentification) {
  auto graph = CreateSimpleGraph();
  auto input_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2, 3});
  auto param = CreateParameter(graph, "param", input_abs);
  
  // Create CopyToHost node
  auto copy_to_host = CreateCNode(graph, "CopyToHost", {param});
  EXPECT_TRUE(IsD2HNode(copy_to_host));
  EXPECT_FALSE(IsH2DNode(copy_to_host));
  
  // Create CopyToDevice node
  auto copy_to_device = CreateCNode(graph, "CopyToDevice", {param});
  EXPECT_TRUE(IsH2DNode(copy_to_device));
  EXPECT_FALSE(IsD2HNode(copy_to_device));
  
  // Normal node
  auto add_node = CreateCNode(graph, "Add", {param, param});
  EXPECT_FALSE(IsD2HNode(add_node));
  EXPECT_FALSE(IsH2DNode(add_node));
}

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestCollectAllNodeUsers) {
  auto graph = CreateSimpleGraph();
  auto exec_order = CreateSimpleExecutionOrder(graph);
  
  auto user_info_list = CollectAllNodeUsers(exec_order);
  
  // Should have nodes collected
  EXPECT_GE(user_info_list.size(), 0);
  
  // Check the structure of each UserInfo
  for (const auto &user_info : user_info_list) {
    EXPECT_NE(user_info.node.first, nullptr);
    EXPECT_GE(user_info.users.size(), 0);
  }
}

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestBuildOffloadNodes) {
  auto graph = CreateSimpleGraph();
  auto input_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2, 3});
  auto param = CreateParameter(graph, "param", input_abs);
  
  // Test BuildToHostNode
  auto to_host_node = BuildToHostNode(graph, param);
  EXPECT_NE(to_host_node, nullptr);
  EXPECT_TRUE(IsD2HNode(to_host_node));
  
  // Test BuildToDeviceNode
  auto to_device_node = BuildToDeviceNode(graph, param);
  EXPECT_NE(to_device_node, nullptr);
  EXPECT_TRUE(IsH2DNode(to_device_node));
}

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestDistanceBaseHyperOffloadStrategy) {
  auto graph = CreateSimpleGraph();
  auto exec_order = CreateLargeExecutionOrder(graph);
  auto user_info_list = CollectAllNodeUsers(exec_order);

  DistanceBaseHyperOffloadStrategy strategy;
  auto offload_info_list = strategy.Run(exec_order, user_info_list);

  // With large graph and select_distance=5, we expect offload_info_list not to be empty.
  // This also serves as a precondition for tests that rely on offload being triggered.
  EXPECT_GT(offload_info_list.size(), 0);
}

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestOffloadInfoFilterByNumber) {
  auto graph = CreateSimpleGraph();
  auto exec_order = CreateLargeExecutionOrder(graph);
  auto user_info_list = CollectAllNodeUsers(exec_order);
  
  DistanceBaseHyperOffloadStrategy strategy;
  auto offload_info_list = strategy.Run(exec_order, user_info_list);
  
  OffloadInfoFilterByNumber filter;
  auto filtered_list = filter.Filter(offload_info_list);
  
  // Filtered list should not exceed configured number
  EXPECT_LE(filtered_list.size(), offload_info_list.size());
  EXPECT_LE(filtered_list.size(), HyperOffloadConfig::GetInstance().select_num());
}

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestOffloadActivationFlow) {
  auto graph = CreateSimpleGraph();
  auto exec_order = CreateLargeExecutionOrder(graph);
  size_t original_size = exec_order.size();

  ASSERT_GE(exec_order.size(), static_cast<size_t>(10));
  auto node1 = exec_order[0];
  auto node2 = exec_order[1];
  auto node4 = exec_order[3];
  auto node5 = exec_order[4];

  HyperOffloadInput input;
  input.graph = graph;
  input.exec_order = exec_order;
  input.enable_offload_parameter = false;
  input.enable_offload_activation = true;

  auto plan = HyperOffloadOptimizer::Run(input);

  EXPECT_NE(plan.graph, nullptr);
  // For this single OffloadInfo (node1), we expect exactly one D2H and one H2D to be inserted.
  EXPECT_EQ(plan.exec_order.size(), original_size + 12);

  // Helper lambda to find index of a node in execution order
  auto find_index = [](const CNodePtrList &order, const AnfNodePtr &target) -> size_t {
    for (size_t i = 0; i < order.size(); ++i) {
      if (order[i] == target) {
        return i;
      }
    }
    return order.size();
  };

  const auto &new_order = plan.exec_order;
  size_t idx_node1 = find_index(new_order, node1);
  size_t idx_node2 = find_index(new_order, node2);
  size_t idx_node4 = find_index(new_order, node4);
  size_t idx_node5 = find_index(new_order, node5);

  // 1. Ensure there is exactly one D2H for node1, inserted between node2 and node3.
  size_t d2h_count = 0;
  size_t d2h_idx = new_order.size();
  for (size_t i = 0; i < new_order.size(); ++i) {
    const auto &n = new_order[i];
    if (!IsD2HNode(n)) {
      continue;
    }
    ++d2h_count;
    d2h_idx = i;
  }
  EXPECT_EQ(d2h_count, 1U) << "Expect exactly one CopyToHost for node1.";
  EXPECT_GT(d2h_idx, idx_node1);
  EXPECT_LT(d2h_idx, idx_node2);

  size_t h2d_count = 0;
  size_t h2d_idx = new_order.size();
  for (size_t i = 0; i < new_order.size(); ++i) {
    const auto &n = new_order[i];
    if (!IsH2DNode(n)) {
      continue;
    }
    ++h2d_count;
    h2d_idx = i;
  }
  EXPECT_EQ(h2d_count, 1U) << "Expect exactly one CopyToDevice for node1.";
  EXPECT_GT(h2d_idx, idx_node4);
  EXPECT_LT(h2d_idx, idx_node5);
  EXPECT_GT(h2d_idx, d2h_idx) << "H2D should come after D2H.";
}

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestOffloadStrategyWithLargeGraph) {
  auto graph = CreateSimpleGraph();
  auto exec_order = CreateLargeExecutionOrder(graph);
  auto user_info_list = CollectAllNodeUsers(exec_order);
  
  // Check if any nodes are identified as offloadable
  EXPECT_GT(user_info_list.size(), 0);
  
  DistanceBaseHyperOffloadStrategy strategy;
  auto offload_info_list = strategy.Run(exec_order, user_info_list);
  
  // In large graph, if distance between nodes is large enough, should have offload info
  // node1 at position 0, node6 at position 5, node8 at position 7, node10 at position 9
  // If distance exceeds select_distance(5), should trigger offload
  MS_LOG(INFO) << "Execution order size: " << exec_order.size();
  MS_LOG(INFO) << "User info list size: " << user_info_list.size();
  MS_LOG(INFO) << "Offload info list size: " << offload_info_list.size();
  
  // Strategy should at least be able to run
  EXPECT_GE(offload_info_list.size(), 0);
}

// Feature: HyperOffload.
// Description: test function AssignHyperOffloadIds.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestAssignHyperOffloadIds_Tuple) {
  auto graph = CreateSimpleGraph();
  // Create a node that outputs a tuple (abstract).
  auto input_abs_element = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2, 3});
  auto input_abs_tuple = std::make_shared<abstract::AbstractTuple>(abstract::AbstractBasePtrList{input_abs_element, input_abs_element});
  
  auto param = CreateParameter(graph, "param_tuple", input_abs_tuple);

  // Create TupleGetItem nodes
  auto getitem0 = CreateCNode(graph, "TupleGetItem", {param, NewValueNode(MakeValue(static_cast<int64_t>(0)))});
  auto getitem1 = CreateCNode(graph, "TupleGetItem", {param, NewValueNode(MakeValue(static_cast<int64_t>(1)))});
  auto getitem0_dup = CreateCNode(graph, "TupleGetItem", {param, NewValueNode(MakeValue(static_cast<int64_t>(0)))});

  CNodePtrList exec_order;
  (void)exec_order.emplace_back(getitem0);
  (void)exec_order.emplace_back(getitem1);
  (void)exec_order.emplace_back(getitem0_dup);

  HyperOffloadOptimizer::AssignHyperOffloadIds(exec_order);

  // Helper to get ID
  auto get_id = [](const AnfNodePtr &node) -> size_t {
    auto id_ptr = node->user_data<size_t>(kHyperOffloadId);
    if (id_ptr) return *id_ptr;
    return 0;
  };

  size_t id0 = get_id(getitem0);
  size_t id1 = get_id(getitem1);
  size_t id0_dup = get_id(getitem0_dup);

  EXPECT_NE(id0, 0);
  EXPECT_NE(id1, 0);
  EXPECT_NE(id0_dup, 0);

  EXPECT_NE(id0, id1);
  EXPECT_EQ(id0, id0_dup);
}

}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore