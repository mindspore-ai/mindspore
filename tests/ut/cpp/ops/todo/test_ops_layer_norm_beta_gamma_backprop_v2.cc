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
#include "ops/layer_norm_beta_gamma_backprop_v2.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class TestLayerNormBetaGammaBackpropV2 : public UT::Common {
 public:
  TestLayerNormBetaGammaBackpropV2() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestLayerNormBetaGammaBackpropV2, test_ops_layer_norm_beta_gamma_backprop_v2_1) {
  auto ln_back = std::make_shared<LayerNormBetaGammaBackpropV2>();
  ln_back->Init(std::vector<int64_t>{1024});
  auto shape_gamma = ln_back->get_shape_gamma();
  EXPECT_EQ(shape_gamma.size(), 1);
  EXPECT_EQ(shape_gamma[0], 1024);
  auto input_dy = TensorConstructUtils::CreateOnesTensor(kFloat16, std::vector<int64_t>{1, 128, 1});
  auto input_res_gamma = TensorConstructUtils::CreateOnesTensor(kFloat32, std::vector<int64_t>{1, 128, 1});
  MS_EXCEPTION_IF_NULL(input_dy);
  MS_EXCEPTION_IF_NULL(input_res_gamma);
  AbstractBasePtrList inputs = {input_dy->ToAbstract(), input_res_gamma->ToAbstract()};
  auto abstract = ln_back->Infer(inputs);
  MS_EXCEPTION_IF_NULL(abstract);
  EXPECT_EQ(abstract->isa<abstract::AbstractTuple>(), true);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::TupleShape>(), true);
  auto shape = shape_ptr->cast<abstract::TupleShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  EXPECT_EQ(shape_vec.size(), 2);
  auto shape1 = shape_vec[0]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape1.size(), 1);
  EXPECT_EQ(shape1[0], 1024);
  auto shape2 = shape_vec[1]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape2.size(), 1);
  EXPECT_EQ(shape2[0], 1024);
  auto type_ptr = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type = type_ptr->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(type);
  auto type_vec = type->elements();
  MS_EXCEPTION_IF_NULL(type_vec[0]);
  auto data_type = type_vec[0]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type);
  EXPECT_EQ(data_type->type_id(), kNumberTypeFloat16);
  MS_EXCEPTION_IF_NULL(type_vec[1]);
  auto data1_type = type_vec[1]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data1_type);
  EXPECT_EQ(data1_type->type_id(), kNumberTypeFloat16);
}
}  // namespace ops
}  // namespace mindspore
