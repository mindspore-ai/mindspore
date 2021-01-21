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
#include "ops/grad/batch_norm_grad.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class TestBatchNormGrad : public UT::Common {
 public:
  TestBatchNormGrad() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestBatchNormGrad, test_ops_batch_norm_grad1) {
  auto batch_norm_grad = std::make_shared<BatchNormGrad>();
  batch_norm_grad->Init();
  EXPECT_EQ(batch_norm_grad->get_is_training(), false);
  EXPECT_EQ((int64_t)(batch_norm_grad->get_epsilon() - 1e-05), 0);
  auto input0 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{2});
  auto input1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{2});
  auto input2 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1});
  auto input3 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1});
  auto input4 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1});
  auto abstract = batch_norm_grad->Infer(
    {input0->ToAbstract(), input1->ToAbstract(), input2->ToAbstract(), input3->ToAbstract(), input4->ToAbstract()});
  MS_EXCEPTION_IF_NULL(abstract);
  EXPECT_EQ(abstract->isa<abstract::AbstractTuple>(), true);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::TupleShape>(), true);
  auto shape = shape_ptr->cast<abstract::TupleShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  EXPECT_EQ(shape_vec.size(), 5);
  auto shape0 = shape_vec[0]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape0.size(), 1);
  EXPECT_EQ(shape0[0], 2);
  auto shape1 = shape_vec[1]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape1.size(), 1);
  EXPECT_EQ(shape1[0], 1);
  auto shape2 = shape_vec[2]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape2.size(), 1);
  EXPECT_EQ(shape2[0], 1);
  auto shape3 = shape_vec[3]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape3.size(), 1);
  EXPECT_EQ(shape3[0], 1);
  auto shape4 = shape_vec[4]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape4.size(), 1);
  EXPECT_EQ(shape4[0], 1);
  auto type_ptr = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type = type_ptr->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(type);
  auto type_vec = type->elements();
  MS_EXCEPTION_IF_NULL(type_vec[0]);
  auto data_type0 = type_vec[0]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type0);
  EXPECT_EQ(data_type0->type_id(), kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(type_vec[1]);
  auto data_type1 = type_vec[1]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type1);
  EXPECT_EQ(data_type1->type_id(), kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(type_vec[2]);
  auto data_type2 = type_vec[2]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type2);
  EXPECT_EQ(data_type2->type_id(), kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(type_vec[3]);
  auto data_type3 = type_vec[3]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type3);
  EXPECT_EQ(data_type3->type_id(), kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(type_vec[4]);
  auto data_type4 = type_vec[4]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type4);
  EXPECT_EQ(data_type4->type_id(), kNumberTypeFloat32);
}
}  // namespace ops
}  // namespace mindspore
