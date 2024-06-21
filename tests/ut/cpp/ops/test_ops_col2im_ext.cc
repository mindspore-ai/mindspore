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
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/col2im_ext.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore::ops {
struct Col2ImExtParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ValuePtr output_size;
  ValuePtr kernel_size;
  ValuePtr dilation;
  ValuePtr padding;
  ValuePtr stride;
  ShapeVector out_shape;
};

class TestCol2ImExt : public TestOps, public testing::WithParamInterface<Col2ImExtParams> {};

TEST_P(TestCol2ImExt, dyn_shape) {
  // inputs
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto output_size = param.output_size->ToAbstract();
  auto kernel_size = param.kernel_size->ToAbstract();
  auto dilation = param.dilation->ToAbstract();
  auto padding = param.padding->ToAbstract();
  auto stride = param.stride->ToAbstract();
  std::vector<AbstractBasePtr> input_args{input, output_size, kernel_size, dilation, padding, stride};

  // expect
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.input_type);

  // run
  auto infer_impl = std::make_shared<Col2ImExtFuncImpl>();
  auto prim = std::make_shared<Primitive>("Col2ImExt");
  (void)infer_impl->CheckValidation(prim, input_args);
  auto out_dtype = infer_impl->InferType(prim, input_args);
  TypeCompare(out_dtype, expect_type);
  auto out_shape = infer_impl->InferShape(prim, input_args);
  ShapeCompare(out_shape, expect_shape);
}

INSTANTIATE_TEST_CASE_P(TestCol2ImExt, TestCol2ImExt,
                        testing::Values(Col2ImExtParams({{-2},
                                                         kFloat32,
                                                         CreatePyIntTuple({4, 5}),
                                                         CreatePyIntTuple({2, 2}),
                                                         CreatePyIntTuple({1, 1}),
                                                         CreatePyIntTuple({0, 0}),
                                                         CreatePyIntTuple({1, 1}),
                                                         {-2}}),

                                        Col2ImExtParams({{2, -1, 12},
                                                         kFloat32,
                                                         CreatePyIntTuple({4, 5}),
                                                         CreatePyIntTuple({2, 2}),
                                                         CreatePyIntTuple({1, 1}),
                                                         CreatePyIntTuple({0, 0}),
                                                         CreatePyIntTuple({1, 1}),
                                                         {2, -1, 4, 5}}),

                                        Col2ImExtParams({{2, 12, 12},
                                                         kFloat32,
                                                         CreatePyIntTuple({4, 5}),
                                                         CreatePyIntTuple({kValueAny, 2}),
                                                         CreatePyIntTuple({1, 1}),
                                                         CreatePyIntTuple({0, 0}),
                                                         CreatePyIntTuple({1, 1}),
                                                         {2, -1, 4, 5}}),
                                        Col2ImExtParams({{2, 12, 12},
                                                         kFloat32,
                                                         CreatePyIntTuple({4, 5}),
                                                         CreatePyIntTuple({kValueAny, kValueAny}),
                                                         CreatePyIntTuple({1, 1}),
                                                         CreatePyIntTuple({0, 0}),
                                                         CreatePyIntTuple({1, 1}),
                                                         {2, -1, 4, 5}}),
                                        Col2ImExtParams({{2, 12, 12},
                                                         kFloat32,
                                                         CreatePyIntTuple({4, 5}),
                                                         kValueAny,
                                                         CreatePyIntTuple({1, 1}),
                                                         CreatePyIntTuple({0, 0}),
                                                         CreatePyIntTuple({1, 1}),
                                                         {2, -1, 4, 5}}),

                                        Col2ImExtParams({{12, 12},
                                                         kFloat32,
                                                         CreatePyIntTuple({kValueAny, 5}),
                                                         CreatePyIntTuple({2, 2}),
                                                         CreatePyIntTuple({1, 1}),
                                                         CreatePyIntTuple({0, 0}),
                                                         CreatePyIntTuple({1, 1}),
                                                         {3, -1, 5}}),
                                        Col2ImExtParams({{12, 12},
                                                         kFloat32,
                                                         CreatePyIntTuple({kValueAny, kValueAny}),
                                                         CreatePyIntTuple({2, 2}),
                                                         CreatePyIntTuple({1, 1}),
                                                         CreatePyIntTuple({0, 0}),
                                                         CreatePyIntTuple({1, 1}),
                                                         {3, -1, -1}}),
                                        Col2ImExtParams({{12, 12},
                                                         kFloat32,
                                                         kValueAny,
                                                         CreatePyIntTuple({2, 2}),
                                                         CreatePyIntTuple({1, 1}),
                                                         CreatePyIntTuple({0, 0}),
                                                         CreatePyIntTuple({1, 1}),
                                                         {3, -1, -1}}),

                                        Col2ImExtParams({{-1, -1},
                                                         kFloat32,
                                                         CreatePyIntTuple({4, 5}),
                                                         CreatePyIntTuple({2, 2}),
                                                         CreatePyIntTuple({1, 1}),
                                                         CreatePyIntTuple({0, 0}),
                                                         CreatePyIntTuple({1, 1}),
                                                         {-1, 4, 5}})));
}  // namespace mindspore::ops
