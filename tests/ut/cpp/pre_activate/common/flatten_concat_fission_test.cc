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

#include "backend/common/pass/flatten_concat_fission.h"
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"

namespace mindspore {
namespace opt {
class TestHWFlattenConcatFission : public BackendCommon {
 public:
  TestHWFlattenConcatFission() : get_py_fun_("gtest_input.pre_activate.flatten_concat_fission_test", true) {}
  ~TestHWFlattenConcatFission() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

/// Feature: FlattenConcatFission
/// Description: FlattenConcat->Flatten+Concat
/// Expectation: pass
TEST_F(TestHWFlattenConcatFission, test_flatten_concat_fission) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("flatten_concat_fission_graph", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> input_shape{32, 64, 112, 112};
  auto fp16_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, input_shape);
  auto fp32_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, input_shape);
  AbstractBasePtrList args_spec_list{fp16_abstract, fp32_abstract, fp16_abstract, fp32_abstract, fp16_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::FlattenConcatFission>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);
  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("flatten_concat_fission_graph", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
