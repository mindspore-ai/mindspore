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
#include "ops/concat.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class TestConcat : public UT::Common {
 public:
  TestConcat() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestConcat, test_ops_concat1) {
  auto concat = std::make_shared<Concat>();
  concat->Init(1);
  auto tensor_x1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{3, 2, 7, 7});
  auto tensor_x2 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{3, 3, 7, 7});
  auto tensor_x3 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{3, 4, 7, 7});
  MS_EXCEPTION_IF_NULL(tensor_x1);
  MS_EXCEPTION_IF_NULL(tensor_x2);
  MS_EXCEPTION_IF_NULL(tensor_x3);
  auto input_tuple = std::make_shared<ValueTuple>(std::vector<ValuePtr>{tensor_x1, tensor_x2, tensor_x3});
  auto abstract = concat->Infer({input_tuple->ToAbstract()});
  MS_EXCEPTION_IF_NULL(abstract);
  EXPECT_EQ(abstract->isa<abstract::AbstractTensor>(), true);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::Shape>(), true);
  auto shape = shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  auto type = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  EXPECT_EQ(type->isa<TensorType>(), true);
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  EXPECT_EQ(data_type->type_id(), kNumberTypeFloat32);
  EXPECT_EQ(shape_vec.size(), 4);
  EXPECT_EQ(shape_vec[0], 3);
  EXPECT_EQ(shape_vec[1], 9);
  EXPECT_EQ(shape_vec[2], 7);
  EXPECT_EQ(shape_vec[3], 7);
}

TEST_F(TestConcat, test_ops_concat2) {
  auto concat = std::make_shared<Concat>();
  concat->Init(2);
  auto tensor_x1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{3, 4, 5});
  auto tensor_x2 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{3, 4, 2});
  auto tensor_x3 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{3, 4, 3});
  MS_EXCEPTION_IF_NULL(tensor_x1);
  MS_EXCEPTION_IF_NULL(tensor_x2);
  MS_EXCEPTION_IF_NULL(tensor_x3);
  auto input_tuple = std::make_shared<ValueTuple>(std::vector<ValuePtr>{tensor_x1, tensor_x2, tensor_x3});
  auto abstract = concat->Infer({input_tuple->ToAbstract()});
  MS_EXCEPTION_IF_NULL(abstract);
  EXPECT_EQ(abstract->isa<abstract::AbstractTensor>(), true);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::Shape>(), true);
  auto shape = shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  auto type = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  EXPECT_EQ(type->isa<TensorType>(), true);
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  EXPECT_EQ(data_type->type_id(), kNumberTypeFloat16);
  EXPECT_EQ(shape_vec.size(), 3);
  EXPECT_EQ(shape_vec[0], 3);
  EXPECT_EQ(shape_vec[1], 4);
  EXPECT_EQ(shape_vec[2], 10);
}
}  // namespace ops
}  // namespace mindspore
