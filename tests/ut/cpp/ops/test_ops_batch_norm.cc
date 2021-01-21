/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/batch_norm.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
namespace mindspore {
namespace ops {
class TestBatchNorm : public UT::Common {
 public:
  TestBatchNorm() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestBatchNorm, test_batch_norm) {
  auto batch = std::make_shared<BatchNorm>();
  batch->Init();
  auto input_x = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{2, 2});
  auto scale = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{2});
  auto bias = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{2});
  auto mean = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{2});
  auto variance = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{2});
  auto abstract = batch->Infer(
    {input_x->ToAbstract(), scale->ToAbstract(), bias->ToAbstract(), mean->ToAbstract(), variance->ToAbstract()});
  MS_EXCEPTION_IF_NULL(abstract);
  EXPECT_EQ(abstract->isa<abstract::AbstractTuple>(), true);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::TupleShape>(), true);
  auto shape = shape_ptr->cast<abstract::TupleShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  EXPECT_EQ(shape_vec.size(), 5);
  auto shape1 = shape_vec[0]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape1.size(), 2);
  EXPECT_EQ(shape1[0], 2);
  EXPECT_EQ(shape1[1], 2);
  auto shape2 = shape_vec[1]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape2.size(), 1);
  EXPECT_EQ(shape2[0], 2);
  auto shape3 = shape_vec[2]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape3.size(), 1);
  EXPECT_EQ(shape3[0], 2);
  auto shape4 = shape_vec[3]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape4.size(), 1);
  EXPECT_EQ(shape4[0], 2);
  auto shape5 = shape_vec[4]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape5.size(), 1);
  EXPECT_EQ(shape5[0], 2);
  auto type_ptr = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type = type_ptr->cast<TuplePtr>();
  auto type_vec = type->elements();
  EXPECT_EQ(type_vec.size(), 5);
  MS_EXCEPTION_IF_NULL(type_vec[0]);
  auto data_type1 = type_vec[0]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type1);
  EXPECT_EQ(data_type1->type_id(), kNumberTypeFloat16);
  MS_EXCEPTION_IF_NULL(type_vec[1]);
  auto data_type2 = type_vec[1]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type2);
  EXPECT_EQ(data_type2->type_id(), kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(type_vec[2]);
  auto data_type3 = type_vec[2]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type3);
  EXPECT_EQ(data_type3->type_id(), kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(type_vec[3]);
  auto data_type4 = type_vec[3]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type4);
  EXPECT_EQ(data_type4->type_id(), kNumberTypeFloat16);
  MS_EXCEPTION_IF_NULL(type_vec[4]);
  auto data_type5 = type_vec[4]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type5);
  EXPECT_EQ(data_type5->type_id(), kNumberTypeFloat16);
}
}  // namespace ops
}  // namespace mindspore
