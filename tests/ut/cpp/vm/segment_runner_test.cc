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
#include <algorithm>

#include "common/common_test.h"

#include "primitive/comparison_ops.h"
#include "primitive/arithmetic_ops.h"
#include "primitive/framework_ops.h"
#include "common/py_func_graph_fetcher.h"
#include "ir/manager.h"
#include "utils/log_adapter.h"
#include "ir/func_graph_cloner.h"
#include "frontend/jit/ps/parse/parse.h"
#include "ir/graph_utils.h"
#include "frontend/jit/ps/resource.h"
#include "mindspore/ccsrc/utils/ir_dump/draw.h"
#include "frontend/operator/ops.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "ir/tensor.h"
#include "utils/convert_utils.h"
#include "include/utils/convert_utils_py.h"
#include "utils/log_adapter.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace compile {
using Tensor = tensor::Tensor;

class TestCompileSegmentRunner : public UT::Common {
 public:
  TestCompileSegmentRunner() : get_py_fun_("gtest_input.vm", true) { UT::InitPythonPath(); }

 protected:
  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestCompileSegmentRunner, test_RunOperation1) {
  VectorRef args({1});
  auto res = kernel::pyboost::PyBoostUtils::RunOperation(
    std::make_shared<PrimitivePy>(py::str(prim::kPrimIdentity->name()).cast<std::string>()), args);
  ASSERT_EQ(py::cast<int>(BaseRefToPyData(res)), 1);
}

TEST_F(TestCompileSegmentRunner, test_RunOperation2) {
  VectorRef args({1, 2});
  auto res = kernel::pyboost::PyBoostUtils::RunOperation(
    std::make_shared<PrimitivePy>(py::str(prim::kPrimScalarGt->name()).cast<std::string>()), args);
  ASSERT_EQ(py::cast<bool>(BaseRefToPyData(res)), false);
}
}  // namespace compile
}  // namespace mindspore
