/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ops/ops_func_impl/bias_add_grad.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/gen_ops_name.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct TestBiasAddGradParams {
  ShapeVector dout_shape;
  TypePtr dout_type;
  std::string data_format;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestBiasAddGrad : public TestOps, public testing::WithParamInterface<TestBiasAddGradParams> {};

TEST_P(TestBiasAddGrad, bias_add_grad_dyn_shape) {
  const auto &param = GetParam();
  auto dout = std::make_shared<abstract::AbstractTensor>(param.dout_type, param.dout_shape);
  ASSERT_NE(dout, nullptr);
  AbstractBasePtr format;
  if (param.data_format == "kValueAny") {
    format = std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64);
  } else {
    auto format_value = MakeValue<int64_t>(FormatStringToEnum(param.data_format));
    format = std::make_shared<abstract::AbstractScalar>(format_value, kInt64);
  }
  ASSERT_NE(format, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<BiasAddGradFuncImpl>(kNameBiasAddGrad, {dout, format}, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(TestBiasAddGrad, TestBiasAddGrad,
    testing::Values(TestBiasAddGradParams{{-1, -1, -1, 3}, kFloat32, "NHWC", {3}, kFloat32},
                    TestBiasAddGradParams{{-1, -1, -1, 3}, kFloat64, "kValueAny", {-1}, kFloat64},
                    TestBiasAddGradParams{{-1, -1, -1, 3}, kFloat32, "NCHW", {-1}, kFloat32},
                    TestBiasAddGradParams{{-1, -1, -1, -1}, kInt32, "NCHW", {-1}, kInt32},
                    TestBiasAddGradParams{{-1, -1, -1, -1, -1}, kFloat32, "NCDHW", {-1}, kFloat32},
                    TestBiasAddGradParams{{-2}, kFloat32, "NHWC", {-1}, kFloat32}));
}  // namespace ops
}  // namespace mindspore
