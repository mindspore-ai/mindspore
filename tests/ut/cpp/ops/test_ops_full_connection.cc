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
#include "ops/fusion/full_connection.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class TestFullConnection : public UT::Common {
 public:
  TestFullConnection() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestFullConnection, test_full_connection_1) {
  auto op = std::make_shared<FullConnection>();
  bool has_bias = false;
  bool use_axis = false;
  int64_t axis = 3;
  op->Init(has_bias, axis, use_axis, NO_ACTIVATION);
  auto tensor_1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{2, 3});
  auto tensor_2 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{2, 3});
  auto abstract = op->Infer({tensor_1->ToAbstract(), tensor_2->ToAbstract()});
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
  EXPECT_EQ(shape_vec.size(), 2);
  EXPECT_EQ(shape_vec[0], 2);
  EXPECT_EQ(shape_vec[1], 2);
}

TEST_F(TestFullConnection, test_full_connection_2) {
  auto op = std::make_shared<FullConnection>();
  bool has_bias = true;
  bool use_axis = false;
  int64_t axis = 1;
  op->Init(has_bias, axis, use_axis, NO_ACTIVATION);
  auto tensor_1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{2, 3});
  auto tensor_2 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{2, 3});
  auto tensor_3 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{2, 2});
  auto abstract = op->Infer({tensor_1->ToAbstract(), tensor_2->ToAbstract(), tensor_3->ToAbstract()});
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
  EXPECT_EQ(shape_vec.size(), 2);
  EXPECT_EQ(shape_vec[0], 2);
  EXPECT_EQ(shape_vec[1], 2);
}

TEST_F(TestFullConnection, test_full_connection_3) {
  auto op = std::make_shared<FullConnection>();
  bool has_bias = false;
  bool use_axis = true;
  int64_t axis = 1;
  op->Init(has_bias, axis, use_axis, NO_ACTIVATION);
  auto tensor_1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{2, 3});
  auto tensor_2 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{2, 3});
  auto abstract = op->Infer({tensor_1->ToAbstract(), tensor_2->ToAbstract()});
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
  EXPECT_EQ(shape_vec.size(), 2);
  EXPECT_EQ(shape_vec[0], 2);
  EXPECT_EQ(shape_vec[1], 2);
}
}  // namespace ops
}  // namespace mindspore
