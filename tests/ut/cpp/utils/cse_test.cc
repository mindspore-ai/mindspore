
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"

#include "ir/anf.h"
#include "ir/visitor.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/opt.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/arithmetic_simplify.h"
#include "pipeline/jit/action.h"

#include "include/common/debug/draw.h"
#include "frontend/operator/ops.h"
#include "include/common/utils/cse.h"
#include "include/common/utils/convert_utils.h"
#include "frontend/optimizer/cse_pass.h"
namespace mindspore {
namespace opt {
class TestCSE : public UT::Common {
 public:
  TestCSE() : getPyFun("gtest_input.cse.cse_test", true) {}
  virtual void SetUp() {}
  virtual void TearDown() {}

 public:
  UT::PyFuncGraphFetcher getPyFun;
};

size_t GetFuncGraphCallCount(const FuncGraphPtr &fg) {
  auto nodes = TopoSort(fg->return_node());
  return std::accumulate(nodes.cbegin(), nodes.cend(), 0, [](size_t sum, const AnfNodePtr &node) {
    if (!node->isa<CNode>()) {
      return sum;
    }
    auto cnode = node->cast<CNodePtr>();
    auto input0 = cnode->input(0);
    if (!IsValueNode<FuncGraph>(input0)) {
      return sum;
    }
    return sum + 1;
  });
}

// Feature: CSE.
// Description: test function HasHiddenSideEffect.
// Expectation: Correct result of checking a node is a hidden side effect node.
TEST_F(TestCSE, TestHasHiddenSideEffect) {
  CSE cse_instance;
  FuncGraphPtr normal_call_node_graph =
    getPyFun.CallAndParseRet("test_has_hidden_side_effect", "root_graph_normal_call");
  ASSERT_TRUE(nullptr != normal_call_node_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  std::vector<AbstractBasePtr> args_spec;
  normal_call_node_graph = pipeline::Renormalize(res, normal_call_node_graph, args_spec);
  (void)cse_instance.Cse(normal_call_node_graph, normal_call_node_graph->manager());
  // Expect cse matched
  auto call_node_count = GetFuncGraphCallCount(normal_call_node_graph);
  ASSERT_EQ(call_node_count, 1);

  FuncGraphPtr hidden_effect_node_call_graph =
    getPyFun.CallAndParseRet("test_has_hidden_side_effect", "root_graph_hidden_side_effect_call");
  ASSERT_TRUE(nullptr != hidden_effect_node_call_graph);
  hidden_effect_node_call_graph = pipeline::Renormalize(res, hidden_effect_node_call_graph, args_spec);
  (void)cse_instance.Cse(hidden_effect_node_call_graph, hidden_effect_node_call_graph->manager());
  // Expect cse not matched.
  call_node_count = GetFuncGraphCallCount(hidden_effect_node_call_graph);
  ASSERT_EQ(call_node_count, 2);
}
}  // namespace opt
}  // namespace mindspore
