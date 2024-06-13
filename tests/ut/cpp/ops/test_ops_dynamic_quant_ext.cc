/**
* Copyright 2024 Huawei Technologies Co., Ltd
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
#include "ops/ops_func_impl/dynamic_quant_ext.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_name.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
#define I64(x) (static_cast<int64_t>((x)))
struct DynamicQuantExtShape {
  ShapeVector x_shape;
  ShapeVector smooth_scales_shape;
  ShapeVector scale_shape;
};

struct DynamicQuantExtType {
  TypePtr x_type;
  TypePtr out_type;
};

class TestDynamicQuantExt : public TestOps, public testing::WithParamInterface<std::tuple<DynamicQuantExtShape, DynamicQuantExtType>> {};

TEST_P(TestDynamicQuantExt, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  DynamicQuantExtFuncImpl dynamic_quant_ext_func_impl;
  auto primitive = std::make_shared<Primitive>("DynamicQuantExt");
  ASSERT_NE(primitive, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(type_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto smooth_scales = std::make_shared<abstract::AbstractTensor>(type_param.x_type, shape_param.smooth_scales_shape);
  ASSERT_NE(x, nullptr);
  std::vector<AbstractBasePtr> input_args = {x, smooth_scales};

  // expect output
  auto expect_shape = std::make_shared<abstract::TupleShape>(
    std::vector<BaseShapePtr>{
      std::make_shared<abstract::Shape>(shape_param.x_shape),
      std::make_shared<abstract::Shape>(shape_param.scale_shape)
    }
  );
  ASSERT_NE(expect_shape, nullptr);
  auto expect_dtype = std::make_shared<Tuple>(std::vector<TypePtr>{kInt8, kFloat32});
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = dynamic_quant_ext_func_impl.InferShape(primitive, input_args);
  auto out_dtype = dynamic_quant_ext_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto dynamic_quant_ext_shape_cases =
  testing::Values(DynamicQuantExtShape{{5, 4}, {4},{5}},
                  DynamicQuantExtShape{{16, 8, 3}, {3},{16, 8}},
                  DynamicQuantExtShape{{64}, {64},{}},
                  DynamicQuantExtShape{{-2}, {-2},{-2}},
                  DynamicQuantExtShape{{-1, 2, 3, -1}, {-1,}, {-1, 2, 3}});

auto dynamic_quant_ext_type_cases = testing::ValuesIn({
  DynamicQuantExtType{kInt16, kInt8},
  DynamicQuantExtType{kInt32, kInt8},
  DynamicQuantExtType{kInt64, kInt8},
  DynamicQuantExtType{kFloat16, kInt8},
  DynamicQuantExtType{kFloat32, kInt8},
  DynamicQuantExtType{kFloat64, kInt8},
});

INSTANTIATE_TEST_CASE_P(TestDynamicQuantExtGroup, TestDynamicQuantExt, testing::Combine(dynamic_quant_ext_shape_cases, dynamic_quant_ext_type_cases));
}  // namespace ops
}  // namespace mindspore
