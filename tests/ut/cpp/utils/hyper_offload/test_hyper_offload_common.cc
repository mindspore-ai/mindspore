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
#include "utils/hyper_offload/distance_base_hyper_offload_strategy.h"
#include "utils/hyper_offload/hyper_offload_event.h"
#include "utils/hyper_offload/hyper_offload_operation.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "utils/compile_config.h"
#include "utils/log_adapter.h"
#include "include/utils/anfalgo.h"
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
    // Disable peak strategy to use distance-based strategy
    (void)common::SetEnv("MS_DEV_HYPER_OFFLOAD_PEAK_STRATEGY", "0");
    // Set required compile configs to avoid GetConfig errors
    common::SetCompileConfig("COMPILE_DEBUG_MODE", "debug");
    common::SetCompileConfig("TRACE_LABEL_WITH_UNIQUE_ID", "0");
    common::SetCompileConfig("HYPER_OFFLOAD_SELECT_DISTANCE", "5");
    common::SetCompileConfig("HYPER_OFFLOAD_SELECT_NUM", "10");
    common::SetCompileConfig("HYPER_OFFLOAD_PREFETCH_DISTANCE", "3");
    common::SetCompileConfig("HYPER_OFFLOAD_RELEASE_DISTANCE", "2");
    common::SetCompileConfig("HYPER_OFFLOAD_MAX_DEVICE_MEMORY", "0");
    common::SetCompileConfig("HYPER_OFFLOAD_SCORE_WEIGHT_SIZE", "1.0");
    common::SetCompileConfig("HYPER_OFFLOAD_SCORE_WEIGHT_COVER_START", "1.0");
    common::SetCompileConfig("HYPER_OFFLOAD_SCORE_WEIGHT_COVER_END", "1.0");
    common::SetCompileConfig("HYPER_OFFLOAD_SCORE_WEIGHT_NEIGHBOR_IMPACT", "1.0");
    // Add missing compile configs that are accessed by hyper_offload code
    common::SetCompileConfig("HYPER_OFFLOAD_MEMORY_BLOCK_PATH", "");
    common::SetCompileConfig("HYPER_OFFLOAD_DYNAMIC_SHAPE_PLACEHOLDER_BYTES", "134217728");
    common::SetCompileConfig("HYPER_OFFLOAD_PEAK_MODELED_HEADROOM_BYTES", "4294967296");
    common::SetCompileConfig("HYPER_OFFLOAD_D2H_BANDWIDTH_GBPS", "20.0");
    common::SetCompileConfig("HYPER_OFFLOAD_H2D_BANDWIDTH_GBPS", "55.0");
    common::SetCompileConfig("HYPER_OFFLOAD_DEFAULT_KERNEL_TIME_US", "400.0");
  }

  void TearDown() override {
    // Clean up environment variables (set to empty string or use unsetenv)
    (void)common::SetEnv("MS_DEV_HYPER_OFFLOAD", "", 1);
    (void)common::SetEnv("MS_DEV_HYPER_OFFLOAD_PEAK_STRATEGY", "", 1);
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
  CNodePtr CreateCNode(const FuncGraphPtr &graph, const std::string &prim_name, const std::vector<AnfNodePtr> &inputs) {
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

  HyperOffloadOptimizer optimizer(input);
  auto plan = optimizer.Run();

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

  HyperOffloadOptimizer optimizer(input);
  auto plan = optimizer.Run();

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

  HyperOffloadOptimizer optimizer(input);
  auto plan = optimizer.Run();

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

  HyperOffloadOptimizer optimizer(input);
  auto plan = optimizer.Run();

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

  HyperOffloadOptimizer optimizer(input);
  auto plan = optimizer.Run();

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
// Description: Inserted CopyToHost / CopyToDevice must agree on stream binding and must not inherit compute
//              primitive stream_id via full CopyNodeAttrs (which can split transfer across streams).
// Expectation: Paired D2H and H2D have identical primitive stream_id state; neither inherits source stream_id.
TEST_F(TestHyperOffloadCommon, TestH2DAndD2HShareTransferStreamAssignment) {
  auto graph = CreateSimpleGraph();
  auto exec_order = CreateLargeExecutionOrder(graph);
  ASSERT_FALSE(exec_order.empty()) << "exec_order should not be empty";

  auto node1 = exec_order[0];
  constexpr auto kStreamIdAttr = "stream_id";
  auto src_prim = common::AnfAlgo::GetCNodePrimitive(node1);
  MS_EXCEPTION_IF_NULL(src_prim);
  (void)src_prim->set_attr(kStreamIdAttr, MakeValue(static_cast<int64_t>(7)));
  ASSERT_NE(src_prim->GetAttr(kStreamIdAttr), nullptr) << "Sanity: source node should carry stream_id";

  HyperOffloadInput input;
  input.graph = graph;
  input.exec_order = exec_order;
  input.enable_offload_parameter = false;
  input.enable_offload_activation = true;

  HyperOffloadOptimizer optimizer(input);
  auto plan = optimizer.Run();
  EXPECT_NE(plan.graph, nullptr);

  constexpr size_t kDataInputIndex = 1;
  CNodePtr d2h_node = nullptr;
  for (const auto &n : plan.exec_order) {
    if (!IsD2HNode(n) || n->inputs().size() <= kDataInputIndex) {
      continue;
    }
    if (n->input(kDataInputIndex) == node1) {
      d2h_node = n;
      break;
    }
  }
  ASSERT_NE(d2h_node, nullptr) << "D2H node for node1 should exist";

  CNodePtr h2d_node = nullptr;
  for (const auto &n : plan.exec_order) {
    if (!IsH2DNode(n) || n->inputs().size() <= kDataInputIndex) {
      continue;
    }
    if (n->input(kDataInputIndex) == d2h_node) {
      h2d_node = n;
      break;
    }
  }
  ASSERT_NE(h2d_node, nullptr) << "H2D node consuming D2H output should exist";

  auto prim_stream_attr = [kStreamIdAttr](const CNodePtr &cn) -> ValuePtr {
    auto prim = common::AnfAlgo::GetCNodePrimitive(cn);
    if (prim == nullptr) {
      return nullptr;
    }
    return prim->GetAttr(kStreamIdAttr);
  };

  auto d2h_stream = prim_stream_attr(d2h_node);
  auto h2d_stream = prim_stream_attr(h2d_node);
  EXPECT_EQ(d2h_stream, h2d_stream) << "D2H and H2D must share the same primitive stream_id binding";
  EXPECT_EQ(d2h_stream, nullptr)
    << "Transfer nodes must not inherit compute stream_id; allocator uses same default stream";
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
  auto input_abs_tuple =
    std::make_shared<abstract::AbstractTuple>(abstract::AbstractBasePtrList{input_abs_element, input_abs_element});

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

// Feature: HyperOffload.
// Description: test enable_offload=false prevents node from being offloaded.
// Expectation: Node with enable_offload=false should not be offloaded even if it meets offload conditions.
TEST_F(TestHyperOffloadCommon, TestEnableOffloadFalse) {
  auto graph = CreateSimpleGraph();
  auto exec_order = CreateLargeExecutionOrder(graph);

  ASSERT_FALSE(exec_order.empty()) << "exec_order should not be empty";
  auto node1 = exec_order[0];
  auto cnode1 = node1->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode1);
  cnode1->AddAttr("enable_offload", MakeValue(false));

  auto enable_opt = GetNodeEnableOffload(node1);
  ASSERT_TRUE(enable_opt.has_value());
  EXPECT_FALSE(enable_opt.value()) << "enable_offload should be false";

  auto user_info_list = CollectAllNodeUsers(exec_order);
  DistanceBaseHyperOffloadStrategy strategy;
  auto offload_info_list = strategy.Run(exec_order, user_info_list);

  bool node1_found = false;
  for (const auto &offload_info : offload_info_list) {
    if (offload_info.data_node.first == node1) {
      node1_found = true;
      break;
    }
  }
  EXPECT_FALSE(node1_found) << "Node with enable_offload=false should not be offloaded";
}

// Feature: HyperOffload.
// Description: test node prefetch_distance attribute is prioritized over global config.
// Expectation: When node has prefetch_distance attribute, it should be used instead of global config.
TEST_F(TestHyperOffloadCommon, TestNodePrefetchDistancePriority) {
  auto graph = CreateSimpleGraph();
  auto exec_order = CreateLargeExecutionOrder(graph);

  ASSERT_FALSE(exec_order.empty()) << "exec_order should not be empty";
  constexpr size_t custom_prefetch_distance = 5;
  auto node1 = exec_order[0];
  auto cnode1 = node1->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode1);
  cnode1->AddAttr("prefetch_distance", MakeValue(custom_prefetch_distance));

  auto prefetch_opt = GetNodePrefetchDistance(node1);
  ASSERT_TRUE(prefetch_opt.has_value());
  EXPECT_EQ(prefetch_opt.value(), custom_prefetch_distance) << "Node prefetch_distance should match set value";

  HyperOffloadInput input;
  input.graph = graph;
  input.exec_order = exec_order;
  input.enable_offload_parameter = false;
  input.enable_offload_activation = true;

  HyperOffloadOptimizer optimizer(input);
  auto plan = optimizer.Run();

  EXPECT_NE(plan.graph, nullptr);
  EXPECT_GE(plan.exec_order.size(), exec_order.size());

  auto find_h2d_for_node1 = [&](const CNodePtrList &order) -> size_t {
    for (size_t i = 0; i < order.size(); ++i) {
      if (IsH2DNode(order[i])) {
        constexpr size_t data_index = 1;
        auto input_node = order[i]->input(data_index);
        if (IsD2HNode(input_node)) {
          auto d2h_node = input_node->cast<CNodePtr>();
          if (d2h_node->input(data_index) == node1) {
            return i;
          }
        }
      }
    }
    return order.size();
  };

  size_t h2d_idx = find_h2d_for_node1(plan.exec_order);
  EXPECT_NE(h2d_idx, plan.exec_order.size()) << "H2D node for node1 should exist";

  auto find_node1_idx = [&](const CNodePtrList &order) -> size_t {
    for (size_t i = 0; i < order.size(); ++i) {
      if (order[i] == node1) {
        return i;
      }
    }
    return order.size();
  };

  size_t node1_idx = find_node1_idx(plan.exec_order);
  EXPECT_NE(node1_idx, plan.exec_order.size()) << "node1 should exist in execution order";

  if (h2d_idx < node1_idx) {
    size_t actual_distance = node1_idx - h2d_idx;
    EXPECT_LE(actual_distance, custom_prefetch_distance)
      << "H2D node should be moved ahead according to prefetch_distance";
  }
}

// Feature: HyperOffload.
// Description: test node release_distance attribute is prioritized over global config.
// Expectation: When node has release_distance attribute, it should be used instead of global config.
TEST_F(TestHyperOffloadCommon, TestNodeReleaseDistancePriority) {
  auto graph = CreateSimpleGraph();
  auto exec_order = CreateLargeExecutionOrder(graph);

  ASSERT_FALSE(exec_order.empty()) << "exec_order should not be empty";
  constexpr size_t custom_release_distance = 5;
  auto node1 = exec_order[0];
  auto cnode1 = node1->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode1);
  cnode1->AddAttr("release_distance", MakeValue(custom_release_distance));

  auto release_opt = GetNodeReleaseDistance(node1);
  ASSERT_TRUE(release_opt.has_value());
  EXPECT_EQ(release_opt.value(), custom_release_distance) << "Node release_distance should match set value";

  HyperOffloadInput input;
  input.graph = graph;
  input.exec_order = exec_order;
  input.enable_offload_parameter = false;
  input.enable_offload_activation = true;

  HyperOffloadOptimizer optimizer(input);
  auto plan = optimizer.Run();

  EXPECT_NE(plan.graph, nullptr);
  EXPECT_GT(plan.exec_order.size(), exec_order.size())
    << "Event nodes should be added when release_distance attribute is set";

  auto find_d2h_for_node1 = [&](const CNodePtrList &order) -> size_t {
    for (size_t i = 0; i < order.size(); ++i) {
      if (IsD2HNode(order[i])) {
        constexpr size_t data_index = 1;
        if (order[i]->input(data_index) == node1) {
          return i;
        }
      }
    }
    return order.size();
  };

  size_t d2h_idx = find_d2h_for_node1(plan.exec_order);
  EXPECT_NE(d2h_idx, plan.exec_order.size()) << "D2H node for node1 should exist";

  size_t d2h_original_idx = 0;
  for (size_t i = 0; i < exec_order.size(); ++i) {
    if (exec_order[i] == node1) {
      d2h_original_idx = i;
      break;
    }
  }

  size_t target_idx = d2h_original_idx + custom_release_distance;
  if (target_idx >= exec_order.size()) {
    target_idx = exec_order.size() - 1;
  }

  auto find_recv_completion = [&](const CNodePtrList &order, size_t start_idx) -> size_t {
    for (size_t i = start_idx; i < order.size(); ++i) {
      if (IsPrimitiveCNode(order[i], prim::kPrimReceive)) {
        auto event_type = order[i]->user_data<EventNodeType>(kHyperOffloadEventNodeType);
        if (event_type && *event_type == kSendRecvAfterD2H) {
          constexpr size_t data_index = 1;
          auto input_node = order[i]->input(data_index);
          if (input_node == node1) {
            return i;
          }
        }
      }
    }
    return order.size();
  };

  size_t recv_idx = find_recv_completion(plan.exec_order, d2h_idx);
  EXPECT_NE(recv_idx, plan.exec_order.size()) << "Recv completion node should exist";

  auto find_target_node_idx = [&](const CNodePtrList &order, size_t target_original_idx) -> size_t {
    if (target_original_idx >= exec_order.size()) {
      return order.size();
    }
    auto target_node = exec_order[target_original_idx];
    for (size_t i = 0; i < order.size(); ++i) {
      if (order[i] == target_node) {
        return i;
      }
    }
    return order.size();
  };

  size_t target_node_idx = find_target_node_idx(plan.exec_order, target_idx);
  if (target_node_idx != plan.exec_order.size()) {
    EXPECT_GE(recv_idx, target_node_idx)
      << "Recv completion should be inserted after target node according to release_distance";
  }
}

// Feature: HyperOffload.
// Description: test function HyperOffload for operations with tuple output.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestHyperOffloadTupleOutput) {
  auto graph = CreateSimpleGraph();
  CNodePtrList exec_order;

  // 1. Create parameters for BatchNorm
  auto input_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2, 3});
  auto x = CreateParameter(graph, "x", input_abs);
  auto scale = CreateParameter(graph, "scale", input_abs);
  auto bias = CreateParameter(graph, "bias", input_abs);
  auto mean = CreateParameter(graph, "mean", input_abs);
  auto variance = CreateParameter(graph, "variance", input_abs);

  // 2. Create BatchNorm node (Tuple output)
  // Inputs: x, scale, bias, mean, variance
  auto batch_norm = CreateCNode(graph, "BatchNorm", {x, scale, bias, mean, variance});

  // BatchNorm outputs tuple of 5 tensors: y, batch_mean, batch_var, reserve_space_1, reserve_space_2
  abstract::AbstractBasePtrList abs_list;
  for (int i = 0; i < 5; ++i) {
    abs_list.push_back(input_abs);
  }
  auto tuple_abs = std::make_shared<abstract::AbstractTuple>(abs_list);
  batch_norm->set_abstract(tuple_abs);
  (void)exec_order.emplace_back(batch_norm);

  // 3. Create TupleGetItem nodes
  // We use the first output (y)
  auto getitem0 = CreateCNode(graph, "TupleGetItem", {batch_norm, NewValueNode(MakeValue(static_cast<int64_t>(0)))});
  getitem0->set_abstract(input_abs);

  auto getitem1 = CreateCNode(graph, "TupleGetItem", {batch_norm, NewValueNode(MakeValue(static_cast<int64_t>(1)))});
  getitem1->set_abstract(input_abs);

  // 4. Create users with large enough distance
  // User 1 of getitem0 (position: immediate)
  auto user1 = CreateCNode(graph, "Sub", {getitem0, x});
  user1->set_abstract(input_abs);
  (void)exec_order.emplace_back(user1);

  // Create gap nodes
  auto last_node = user1;
  for (int i = 0; i < 8; ++i) {
    auto gap_node = CreateCNode(graph, "Add", {last_node, x});
    gap_node->set_abstract(input_abs);
    (void)exec_order.emplace_back(gap_node);
    last_node = gap_node;
  }

  // User 2 of getitem0 (position: distant)
  auto user2 = CreateCNode(graph, "Add", {getitem0, last_node});
  user2->set_abstract(input_abs);
  (void)exec_order.emplace_back(user2);

  auto user3 = CreateCNode(graph, "Add", {getitem1, user2});
  user3->set_abstract(input_abs);
  (void)exec_order.emplace_back(user3);

  graph->set_output(user3);

  HyperOffloadInput input;
  input.graph = graph;
  input.exec_order = exec_order;
  input.enable_offload_parameter = false;
  input.enable_offload_activation = true;

  HyperOffloadOptimizer optimizer(input);
  auto plan = optimizer.Run();

  EXPECT_NE(plan.graph, nullptr);
  EXPECT_EQ(plan.exec_order.size(), exec_order.size());
}

// Feature: HyperOffload.
// Description: test function HyperOffload.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestD2HEventInjectionWithLongLivedData) {
  auto graph = CreateSimpleGraph();
  auto input_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2, 3});
  auto param = CreateParameter(graph, "data", input_abs);
  // Use a CNode as the source of data, because FindLastUserIndex only considers CNodes.
  auto data_node = CreateCNode(graph, "Abs", {param});
  data_node->set_abstract(input_abs);

  // 1. Create D2H Node (CopyToHost)
  auto d2h_node = CreateCNode(graph, "CopyToHost", {data_node});

  // 2. Create User Node (uses data_node)
  auto user_node = CreateCNode(graph, "Add", {data_node, data_node});

  // 3. Construct Execution Order
  // Index 0: d2h_node
  // Index 1..10: placeholders
  // Index 11: user_node
  CNodePtrList exec_order;
  (void)exec_order.emplace_back(d2h_node);

  for (int i = 0; i < 10; ++i) {
    auto no_op = CreateCNode(graph, "NoOp", {});
    (void)exec_order.emplace_back(no_op);
  }
  (void)exec_order.emplace_back(user_node);

  ASSERT_EQ(exec_order.size(), 12);
  ASSERT_EQ(exec_order[0], d2h_node);
  ASSERT_EQ(exec_order[11], user_node);

  // 4. Setup Operations
  HyperOffloadOperations operations;
  operations.Init(exec_order.size());
  // Add d2h_node to operations so Injector finds it
  operations.AddOperationAfter(0, d2h_node);

  // 5. Config (release_distance = 2 from SetUp)
  // node_index = 0.
  // release_distance = 2.
  // last_user_index = 11.
  // target_index should be max(0+2, 11) = 11.
  // Since we insert AFTER target_index, it should be appended to operations[11].

  // 6. Run Injector
  HyperOffloadEventInjector injector(graph, exec_order, &operations);
  injector.Run();

  // 7. Verify
  const auto &ops = operations.GetOperations();

  // Helper to check for FreeData by name (avoiding pointer equality issues across TUs)
  auto is_free_data = [](const CNodePtr &node) {
    if (!node || node->inputs().empty()) return false;
    auto prim = GetValueNode<PrimitivePtr>(node->input(0));
    return prim && prim->name() == "FreeData";
  };

  // Check index 2 (original target without fix)
  // It should NOT contain FreeData (unless index 2 is also index 11, which is not)
  bool found_free_at_2 = false;
  for (const auto &node : ops[2].operations) {
    if (is_free_data(node)) {
      found_free_at_2 = true;
      break;
    }
  }
  EXPECT_FALSE(found_free_at_2) << "FreeData should not be inserted at index 2 (too early)";

  // Check index 11 (new target)
  // It SHOULD contain FreeData
  bool found_free_at_11 = false;
  for (const auto &node : ops[11].operations) {
    if (is_free_data(node)) {
      found_free_at_11 = true;
      break;
    }
  }
  EXPECT_TRUE(found_free_at_11) << "FreeData should be inserted at index 11 (after last user)";
}

// Feature: HyperOffload.
// Description: H2D must be placed after D2H-side device FreeData when both are attached to the same slot.
// Expectation: Final order is D2H -> FreeData(device) -> H2D -> FreeData(host).
TEST_F(TestHyperOffloadCommon, TestH2DAfterD2HDeviceFreeDataInSameSlot) {
  auto graph = CreateSimpleGraph();
  auto input_abs = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2, 3});
  auto param = CreateParameter(graph, "data", input_abs);
  auto data_node = CreateCNode(graph, "Abs", {param});
  data_node->set_abstract(input_abs);
  data_node->AddAttr("release_distance", MakeValue(static_cast<size_t>(0)));

  auto d2h_node = CreateCNode(graph, "CopyToHost", {data_node});
  d2h_node->set_abstract(input_abs);
  auto h2d_node = CreateCNode(graph, "CopyToDevice", {d2h_node});
  h2d_node->set_abstract(input_abs);
  auto h2d_user = CreateCNode(graph, "Add", {h2d_node, param});
  h2d_user->set_abstract(input_abs);

  CNodePtrList exec_order;
  (void)exec_order.emplace_back(data_node);
  (void)exec_order.emplace_back(h2d_user);
  graph->set_output(h2d_user);

  HyperOffloadOperations operations;
  operations.Init(exec_order.size());
  operations.AddOperationAfter(0, d2h_node);
  operations.AddOperationAfter(0, h2d_node);

  HyperOffloadEventInjector injector(graph, exec_order, &operations);
  injector.Run();

  CNodePtrList final_order;
  const auto &pre_ops = operations.GetPreOperations().operations;
  final_order.insert(final_order.end(), pre_ops.begin(), pre_ops.end());
  const auto &op_slots = operations.GetOperations();
  for (size_t i = 0; i < exec_order.size(); ++i) {
    final_order.emplace_back(exec_order[i]);
    const auto &slot_ops = op_slots[i].operations;
    final_order.insert(final_order.end(), slot_ops.begin(), slot_ops.end());
  }

  const size_t not_found = final_order.size();
  size_t d2h_idx = not_found;
  size_t device_free_idx = not_found;
  size_t h2d_idx = not_found;
  size_t host_free_idx = not_found;

  for (size_t i = 0; i < final_order.size(); ++i) {
    const auto &node = final_order[i];
    if (node == d2h_node) {
      d2h_idx = i;
      continue;
    }
    if (node == h2d_node) {
      h2d_idx = i;
      continue;
    }
    if (!IsPrimitiveCNode(node, kFreeDataPrimitive)) {
      continue;
    }
    auto free_type = GetFreeOrAssignNodeType(node);
    ASSERT_TRUE(free_type.has_value());
    if (*free_type == kDeviceType) {
      device_free_idx = i;
    } else if (*free_type == kHostType) {
      host_free_idx = i;
    }
  }

  ASSERT_NE(d2h_idx, not_found) << "D2H should exist in the final execution order";
  ASSERT_NE(device_free_idx, not_found) << "FreeData(device) should exist in the final execution order";
  ASSERT_NE(h2d_idx, not_found) << "H2D should exist in the final execution order";
  ASSERT_NE(host_free_idx, not_found) << "FreeData(host) should exist in the final execution order";
  EXPECT_LT(d2h_idx, device_free_idx);
  EXPECT_LT(device_free_idx, h2d_idx);
  EXPECT_LT(h2d_idx, host_free_idx);
}

// Feature: HyperOffload.
// Description: test function FindLastUserIndex for tuple scene.
// Expectation: Correct result.
TEST_F(TestHyperOffloadCommon, TestFindLastUserIndex_Tuple) {
  auto graph = CreateSimpleGraph();
  auto input_abs_element = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{2, 3});
  auto input_abs_tuple =
    std::make_shared<abstract::AbstractTuple>(abstract::AbstractBasePtrList{input_abs_element, input_abs_element});

  auto param = CreateParameter(graph, "param", input_abs_element);
  // Use a CNode as the source of tuple data, because FindLastUserIndex only considers CNodes.
  auto tuple_source = CreateCNode(graph, "BatchNorm", {param, param, param, param, param});
  tuple_source->set_abstract(input_abs_tuple);

  // Create TupleGetItem nodes
  auto getitem0 = CreateCNode(graph, "TupleGetItem", {tuple_source, NewValueNode(MakeValue(static_cast<int64_t>(0)))});
  getitem0->set_abstract(input_abs_element);
  auto getitem1 = CreateCNode(graph, "TupleGetItem", {tuple_source, NewValueNode(MakeValue(static_cast<int64_t>(1)))});
  getitem1->set_abstract(input_abs_element);
  auto getitem0_dup =
    CreateCNode(graph, "TupleGetItem", {tuple_source, NewValueNode(MakeValue(static_cast<int64_t>(0)))});
  getitem0_dup->set_abstract(input_abs_element);

  // Users of getitem0 and getitem1
  auto user1 = CreateCNode(graph, "Add", {getitem0, getitem0});          // uses getitem0, index 0
  auto user2 = CreateCNode(graph, "Add", {getitem1, getitem1});          // uses getitem1, index 1
  auto user3 = CreateCNode(graph, "Add", {getitem0_dup, getitem0_dup});  // uses getitem0_dup, index 2

  CNodePtrList exec_order;
  (void)exec_order.emplace_back(user1);
  (void)exec_order.emplace_back(user2);
  (void)exec_order.emplace_back(user3);

  // Find last user of getitem0 starting from index 0.
  // It should be user3 because getitem0_dup is equivalent to getitem0.
  size_t last_idx = FindLastUserIndex(exec_order, getitem0, 0);
  EXPECT_EQ(last_idx, 2U);

  // Find last user of getitem1 starting from index 0.
  // It should be user2.
  size_t last_idx1 = FindLastUserIndex(exec_order, getitem1, 0);
  EXPECT_EQ(last_idx1, 1U);
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
