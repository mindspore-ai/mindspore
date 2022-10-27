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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ops/fake_quant_param.h"
#include "ir/dtype/type.h"
#include "mindapi/ir/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "mindapi/ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_id.h"
#include "ops/primitive_c.h"

namespace mindspore {
namespace ops {
class TestFakeQuantParam : public UT::Common {
 public:
  TestFakeQuantParam() {}
  void SetUp() {}
  void TearDown() {}
};

/// Feature: setter and getter of per-layer FakeQuantParam operation.
/// Description: call setter and getter of FakeQuantParam operation and compare result of getter with argument of
/// setter.
/// Expectation: success.
TEST_F(TestFakeQuantParam, test_attr_perlayer) {
  auto ops = std::make_shared<FakeQuantParam>();
  ops->Init(kQuantDataTypeInt4, kAttrKeyLinearQuantAlgoName, false);
  auto quant_dtype = ops->get_quant_dtype();
  EXPECT_EQ(quant_dtype, kQuantDataTypeInt4);
  auto algo_name = ops->get_quant_algo_name();
  EXPECT_EQ(algo_name, kAttrKeyLinearQuantAlgoName);
  auto perchannel = ops->get_is_perchannel();
  EXPECT_EQ(perchannel, false);

  bool has_error = false;
  try {
    ops->set_scale(1.0, 1);
  } catch (...) {
    has_error = true;
  }
  EXPECT_EQ(has_error, true);

  ops->set_scale(1.0);
  auto scale = ops->get_scale();
  EXPECT_EQ(scale, 1.0);

  ops->set_zero_point(1);
  auto zp = ops->get_zero_point();
  EXPECT_EQ(zp, 1);

  ops->set_quant_param("slb-rate", api::MakeValue<float>(1.0));
  auto slb_rate_value = ops->get_quant_param("slb-rate");
  EXPECT_EQ(slb_rate_value->isa<api::FP32Imm>(), true);
  auto slb_rate_imm = slb_rate_value->cast<api::FP32ImmPtr>();
  auto slb_rate = slb_rate_imm->value();
  EXPECT_EQ(slb_rate, 1.0);
}

/// Feature: setter and getter of per-channel FakeQuantParam operation.
/// Description: call setter and getter of FakeQuantParam operation and compare result of getter with argument of
/// setter.
/// Expectation: success.
TEST_F(TestFakeQuantParam, test_attr_perchannel) {
  auto ops = std::make_shared<FakeQuantParam>();
  ops->Init(kQuantDataTypeInt7, kAttrKeyLinearQuantAlgoName, true);
  auto quant_dtype = ops->get_quant_dtype();
  EXPECT_EQ(quant_dtype, kQuantDataTypeInt7);
  auto algo_name = ops->get_quant_algo_name();
  EXPECT_EQ(algo_name, kAttrKeyLinearQuantAlgoName);
  auto perchannel = ops->get_is_perchannel();
  EXPECT_EQ(perchannel, true);

  bool has_error = false;
  try {
    ops->set_scale(1.0, 1);
  } catch (...) {
    has_error = true;
  }
  EXPECT_EQ(has_error, true);

  ops->set_scale(1.0);
  auto scale = ops->get_scale();
  EXPECT_EQ(scale, 1.0);

  ops->set_zero_point(1);
  auto zp = ops->get_zero_point();
  EXPECT_EQ(zp, 1);

  ops->set_quant_param("slb-rate", api::MakeValue<float>(1.0));
  auto slb_rate_value = ops->get_quant_param("slb-rate");
  EXPECT_EQ(slb_rate_value->isa<api::FP32Imm>(), true);
  auto slb_rate_imm = slb_rate_value->cast<api::FP32ImmPtr>();
  auto slb_rate = slb_rate_imm->value();
  EXPECT_EQ(slb_rate, 1.0);
}

/// Feature: infer-shape and infer-type function of FakeQuantParam operation.
/// Description: call Infer function of FakeQuantParam operation and check the result.
/// Expectation: success.
TEST_F(TestFakeQuantParam, test_infer_shape) {
  auto ops = std::make_shared<FakeQuantParam>();
  ops->Init(kQuantDataTypeInt7, kAttrKeyLinearQuantAlgoName, false);
  ops->set_scale(1.0);
  ops->set_zero_point(1);

  auto input_x = TensorConstructUtils::CreateOnesTensor(kFloat32, std::vector<int64_t>{32, 3, 224, 224});
  MS_EXCEPTION_IF_NULL(input_x);
  const auto &infer_fn_map = OpPrimCRegister::GetInstance().GetPrimCMap();
  const auto &infer_fn_iter = infer_fn_map.find("FakeQuantParam");
  EXPECT_NE(infer_fn_iter, infer_fn_map.end());
  const auto &infer_fn = infer_fn_iter->second();
  auto ops_abstract = infer_fn->Infer({input_x->ToAbstract()});
  MS_EXCEPTION_IF_NULL(ops_abstract);
  EXPECT_EQ(ops_abstract->isa<abstract::AbstractTensor>(), true);
  auto shape_ptr = ops_abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::Shape>(), true);
  auto conv_shape = shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(conv_shape);
  auto shape_vec = conv_shape->shape();
  auto type = ops_abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  EXPECT_EQ(type->isa<TensorType>(), true);
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto elem_type = tensor_type->element();
  EXPECT_EQ(elem_type->type_id(), kNumberTypeFloat32);
  EXPECT_EQ(shape_vec.size(), 4);
  EXPECT_EQ(shape_vec[0], 32);
  EXPECT_EQ(shape_vec[1], 3);
  EXPECT_EQ(shape_vec[2], 224);
  EXPECT_EQ(shape_vec[3], 224);
}
}  // namespace ops
}  // namespace mindspore
