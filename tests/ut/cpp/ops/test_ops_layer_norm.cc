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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ops/layer_norm.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "utils/ms_context.h"
#include "ops/test_ops.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {

struct LayerNormParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ShapeVector gamm_shape;
  TypePtr gamm_type;
  ShapeVector beta_shape;
  TypePtr beta_type;
  int64_t begin_norm_axis;
  int64_t begin_params_axis;
  float epsilon;
  ShapeVector output_shape;
  TypePtr output_type;
  ShapeVector mean_shape;
  TypePtr mean_type;
  ShapeVector var_shape;
  TypePtr var_type;
};

class TestLayerNorm : public TestOps, public testing::WithParamInterface<LayerNormParams> {
 public:
  TestLayerNorm() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_P(TestLayerNorm, test_ops_layernorm) {
  const auto &param = GetParam();
  auto prim = std::make_shared<Primitive>(kNameLayerNorm);
  prim->set_attr("begin_norm_axis", MakeValue<int64_t>(param.begin_norm_axis));
  prim->set_attr("begin_params_axis", MakeValue<int64_t>(param.begin_params_axis));
  prim->set_attr("epsilon", MakeValue<float>(param.epsilon));

  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto gamm = std::make_shared<abstract::AbstractTensor>(param.gamm_type, param.gamm_shape);
  auto beta = std::make_shared<abstract::AbstractTensor>(param.beta_type, param.beta_shape);

  auto output = std::make_shared<abstract::AbstractTensor>(param.output_type, param.output_shape);
  auto mean = std::make_shared<abstract::AbstractTensor>(param.mean_type, param.mean_shape);
  auto var = std::make_shared<abstract::AbstractTensor>(param.var_type, param.var_shape);

  AbstractBasePtrList abstract_list{output, mean, var};
  auto expect = std::make_shared<abstract::AbstractTuple>(abstract_list);

  auto out_abstract = opt::CppInferShapeAndType(prim, {input, gamm, beta});

  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(TestLayerNormGroup, TestLayerNorm,
                        testing::Values(LayerNormParams{{4, 3, 2, 2},
                                                        kFloat32,
                                                        {3, 2, 2},
                                                        kFloat32,
                                                        {3, 2, 2},
                                                        kFloat32,
                                                        1,
                                                        1,
                                                        1e-7,
                                                        {4, 3, 2, 2},
                                                        kFloat32,
                                                        {4, 1, 1, 1},
                                                        kFloat32,
                                                        {4, 1, 1, 1},
                                                        kFloat32},
                                        LayerNormParams{{4, 3, 2, 2},
                                                        kFloat16,
                                                        {3, 2, 2},
                                                        kFloat32,
                                                        {3, 2, 2},
                                                        kFloat32,
                                                        1,
                                                        1,
                                                        1e-7,
                                                        {4, 3, 2, 2},
                                                        kFloat16,
                                                        {4, 1, 1, 1},
                                                        kFloat32,
                                                        {4, 1, 1, 1},
                                                        kFloat32}));

}  // namespace ops
}  // namespace mindspore