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

#include "plugin/device/ascend/optimizer/ir_fission/unsorted_segment_sum_d_fission.h"
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "include/common/debug/anf_ir_dump.h"
#include "plugin/device/ascend/optimizer/mindir/ascend_vm_op_adapter.h"

namespace mindspore {
namespace opt {
class TestHWUnsortedSegmentSumFission : public BackendCommon {
 public:
  TestHWUnsortedSegmentSumFission() : get_py_fun_("gtest_input.pre_activate.unsorted_segment_sum_fission", true) {}
  ~TestHWUnsortedSegmentSumFission() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWUnsortedSegmentSumFission, test_fission) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_unsorted_segment_sum_fission", "before1");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{3, 39, 1};
  std::vector<int64_t> shp_y{3};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, shp_y);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AscendVmOpAdapter>());
  pm->AddPass(std::make_shared<opt::UnsortedSegmentSumDFission>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_unsorted_segment_sum_fission", "after1");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWUnsortedSegmentSumFission, test_no_fission) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_unsorted_segment_sum_fission", "before2");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{3, 39, 2};
  std::vector<int64_t> shp_y{3};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, shp_y);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AscendVmOpAdapter>());
  pm->AddPass(std::make_shared<opt::UnsortedSegmentSumDFission>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_unsorted_segment_sum_fission", "after2");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
