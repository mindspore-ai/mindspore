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

#include "backend/optimizer/ascend/ir_fission/batch_norm_grad_infer_fission.h"
#include "backend/optimizer/ascend/mindir/bn_grad_unify_mindir.h"
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"

namespace mindspore {
namespace opt {
class TestHWBatchNormGradInferFission : public BackendCommon {
 public:
  TestHWBatchNormGradInferFission()
      : get_py_fun_("gtest_input.pre_activate.batch_norm_grad_infer_fission_test", true) {}
  ~TestHWBatchNormGradInferFission() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWBatchNormGradInferFission, test_batch_norm_grad_infer_fission) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_batch_norm_grad_infer_fission", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{32, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 6; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::BatchNormGradUnifyMindIR>());
  pm->AddPass(std::make_shared<opt::BatchNormGradInferFission>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_batch_norm_grad_infer_fission", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWBatchNormGradInferFission, test_batch_norm_grad_infer_no_fission1) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_batch_norm_grad_infer_fission", "before_is_training");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{32, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 6; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::BatchNormGradUnifyMindIR>());
  pm->AddPass(std::make_shared<opt::BatchNormGradInferFission>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);
  EXPECT_TRUE(CheckEqualGraph(kg, new_graph));
}

TEST_F(TestHWBatchNormGradInferFission, test_batch_norm_grad_infer_no_fission2) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_batch_norm_grad_infer_fission", "before_output3_not_null");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{32, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 6; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::BatchNormGradUnifyMindIR>());
  pm->AddPass(std::make_shared<opt::BatchNormGradInferFission>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);
  EXPECT_TRUE(CheckEqualGraph(kg, new_graph));
}
}  // namespace opt
}  // namespace mindspore
