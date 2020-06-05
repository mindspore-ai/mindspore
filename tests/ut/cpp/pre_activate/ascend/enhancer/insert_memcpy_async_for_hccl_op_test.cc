/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "session/anf_runtime_algorithm.h"
#include "operator/ops.h"
#include "ir/tensor.h"
#include "debug/anf_ir_dump.h"
#include "utils/utils.h"
#include "kernel/kernel_build_info.h"
#include "pre_activate/common/optimizer.h"
#define private public
#define protected public
#include "pre_activate/ascend/enhancer/insert_memcpy_async_for_hccl_op.h"
#undef private
#undef protected
namespace mindspore {
namespace opt {
class TestHWInsertMemcpyForHccl : public BackendCommon {
 public:
  TestHWInsertMemcpyForHccl() : get_py_fun_("gtest_input.pre_activate.insert_memcpy_async_for_hccl_op", true) {}
  ~TestHWInsertMemcpyForHccl() override = default;

 public:
  UT::PyFuncGraphFetcher get_py_fun_;
};

class MockInsertMemcpyForHcclKernelQuery : public KernelQuery {
 public:
  MockInsertMemcpyForHcclKernelQuery() = default;
  ~MockInsertMemcpyForHcclKernelQuery() override = default;
  bool IsTbeRef(const AnfNodePtr &node) override {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      return false;
    }
    auto name = AnfAlgo::GetCNodeName(cnode);
    return name == "ApplyMomentum";
  }
};

TEST_F(TestHWInsertMemcpyForHccl, test_cond1) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_insert_memcpy_async_for_hccl_op_cond1", "before1");
  ASSERT_TRUE(g != nullptr);
  std::vector<int> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::InsertMemcpyAsyncForHcclOp>();
  pass->kernel_query_ = std::make_shared<MockInsertMemcpyForHcclKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_insert_memcpy_async_for_hccl_op_cond1", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWInsertMemcpyForHccl, test_cond1_no_insert) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_insert_memcpy_async_for_hccl_op_cond1", "before2");
  ASSERT_TRUE(g != nullptr);
  std::vector<int> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);
  auto origin_graph = std::make_shared<session::KernelGraph>(*kg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::InsertMemcpyAsyncForHcclOp>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kg);

  EXPECT_TRUE(CheckEqualGraph(origin_graph, new_graph));
}

TEST_F(TestHWInsertMemcpyForHccl, test_cond2) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_insert_memcpy_async_for_hccl_op_cond2", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::InsertMemcpyAsyncForHcclOp>();
  pass->kernel_query_ = std::make_shared<MockInsertMemcpyForHcclKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_insert_memcpy_async_for_hccl_op_cond2", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWInsertMemcpyForHccl, test_cond3) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_insert_memcpy_async_for_hccl_op_cond3", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract, x_abstract, x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::InsertMemcpyAsyncForHcclOp>();
  pass->kernel_query_ = std::make_shared<MockInsertMemcpyForHcclKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_insert_memcpy_async_for_hccl_op_cond3", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWInsertMemcpyForHccl, test_cond4) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_insert_memcpy_async_for_hccl_op_cond4", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract, x_abstract, x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::InsertMemcpyAsyncForHcclOp>();
  pass->kernel_query_ = std::make_shared<MockInsertMemcpyForHcclKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_insert_memcpy_async_for_hccl_op_cond4", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
