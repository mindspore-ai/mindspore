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
#include "ops/grad/layer_norm_grad.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "utils/ms_context.h"
#include "ops/test_ops.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
struct LayerNormGradParams {
  ShapeVector dy_shape;
  TypePtr dy_type;
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector var_shape;
  TypePtr var_type;
  ShapeVector mean_shape;
  TypePtr mean_type;
  ShapeVector gamma_shape;
  TypePtr gamma_type;
  int64_t begin_norm_axis;
  int64_t begin_params_axis;
  ShapeVector output_shape;
  TypePtr output_type;
  ShapeVector out_gamma_shape;
  TypePtr out_gamma_type;
  ShapeVector out_gamma2_shape;
  TypePtr out_gamma2_type;
};

class TestLayerNormGrad : public TestOps, public testing::WithParamInterface<LayerNormGradParams> {};

TEST_P(TestLayerNormGrad, test_ops_layernormgrad) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto gamma = std::make_shared<abstract::AbstractTensor>(param.gamma_type, param.gamma_shape);
  auto mean = std::make_shared<abstract::AbstractTensor>(param.mean_type, param.mean_shape);
  auto var = std::make_shared<abstract::AbstractTensor>(param.var_type, param.var_shape);
  auto dy = std::make_shared<abstract::AbstractTensor>(param.dy_type, param.dy_shape);

  ASSERT_NE(x, nullptr);
  ASSERT_NE(gamma, nullptr);

  auto prim = std::make_shared<Primitive>(kNameLayerNormGrad);
  prim->set_attr("begin_norm_axis", MakeValue<int64_t>(param.begin_norm_axis));
  prim->set_attr("begin_params_axis", MakeValue<int64_t>(param.begin_params_axis));

  auto output = std::make_shared<abstract::AbstractTensor>(param.output_type, param.output_shape);
  auto out_gamma = std::make_shared<abstract::AbstractTensor>(param.out_gamma_type, param.out_gamma_shape);
  auto out_gamma2 = std::make_shared<abstract::AbstractTensor>(param.out_gamma2_type, param.out_gamma2_shape);

  AbstractBasePtrList abstract_list{output, out_gamma, out_gamma2};
  auto expect = std::make_shared<abstract::AbstractTuple>(abstract_list);

  auto out_abstract = opt::CppInferShapeAndType(prim, {dy, x, var, mean, gamma});

  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(TestLayerNormGradGroup, TestLayerNormGrad,
                        testing::Values(LayerNormGradParams{{2, 3, 4},
                                                            kFloat32,
                                                            {2, 3, 4},
                                                            kFloat32,
                                                            {2, 3, 1},
                                                            kFloat32,
                                                            {2, 3, 1},
                                                            kFloat32,
                                                            {4},
                                                            kFloat32,
                                                            2,
                                                            2,
                                                            {2, 3, 4},
                                                            kFloat32,
                                                            {4},
                                                            kFloat32,
                                                            {4},
                                                            kFloat32},
                                        LayerNormGradParams{{2, 3, 4},
                                                            kFloat16,
                                                            {2, 3, 4},
                                                            kFloat16,
                                                            {2, 1, 1},
                                                            kFloat32,
                                                            {2, 1, 1},
                                                            kFloat32,
                                                            {3, 4},
                                                            kFloat32,
                                                            1,
                                                            1,
                                                            {2, 3, 4},
                                                            kFloat16,
                                                            {3, 4},
                                                            kFloat32,
                                                            {3, 4},
                                                            kFloat32}));
}  // namespace ops
}  // namespace mindspore