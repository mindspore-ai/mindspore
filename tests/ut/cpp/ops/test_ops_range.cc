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
#include "ops/range.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class TestRange : public UT::Common {
 public:
  TestRange() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestRange, test_ops_range1) {
  auto range = std::make_shared<Range>();
  range->Init(1, 3, 34, 4);
  EXPECT_EQ(range->get_d_type(), 1);
  EXPECT_EQ(range->get_start(), 3);
  EXPECT_EQ(range->get_limit(), 34);
  EXPECT_EQ(range->get_delta(), 4);
  range->set_d_type(1);
  range->set_start(3);
  range->set_limit(34);
  range->set_delta(4);
  auto abstract = range->Infer({});
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
  EXPECT_EQ(data_type->type_id(), kNumberTypeInt32);
  EXPECT_EQ(shape_vec.size(), 1);
  EXPECT_EQ(shape_vec[0], 8);
  EXPECT_EQ(range->get_d_type(), 1);
  EXPECT_EQ(range->get_start(), 3);
  EXPECT_EQ(range->get_limit(), 34);
  EXPECT_EQ(range->get_delta(), 4);
}

TEST_F(TestRange, test_ops_range2) {
  auto range = std::make_shared<Range>();
  range->Init(1, 1, 1, 1);
  EXPECT_EQ(range->get_d_type(), 1);
  EXPECT_EQ(range->get_start(), 1);
  EXPECT_EQ(range->get_limit(), 1);
  EXPECT_EQ(range->get_delta(), 1);
  range->set_d_type(1);
  range->set_start(1);
  range->set_limit(1);
  range->set_delta(1);
  auto tensor_x1 = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, std::vector<int64_t>{1});
  auto tensor_x2 = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, std::vector<int64_t>{1});
  auto tensor_x3 = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, std::vector<int64_t>{1});
  MS_EXCEPTION_IF_NULL(tensor_x1);
  MS_EXCEPTION_IF_NULL(tensor_x2);
  MS_EXCEPTION_IF_NULL(tensor_x3);
  auto data_x1 = tensor_x1->data_c();
  MS_EXCEPTION_IF_NULL(data_x1);
  auto val_x1 = reinterpret_cast<float *>(data_x1);
  *val_x1 = 1.0;
  auto data_x2 = tensor_x2->data_c();
  MS_EXCEPTION_IF_NULL(data_x2);
  auto val_x2 = reinterpret_cast<float *>(data_x2);
  *val_x2 = 42.0;
  auto data_x3 = tensor_x3->data_c();
  MS_EXCEPTION_IF_NULL(data_x3);
  auto val_x3 = reinterpret_cast<float *>(data_x3);
  *val_x3 = 3.0;
  auto abstract = range->Infer({tensor_x1->ToAbstract(), tensor_x2->ToAbstract(), tensor_x3->ToAbstract()});
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
  EXPECT_EQ(shape_vec.size(), 1);
  EXPECT_EQ(shape_vec[0], 14);
  EXPECT_EQ(range->get_d_type(), 1);
  EXPECT_EQ(range->get_start(), 1);
  EXPECT_EQ(range->get_limit(), 1);
  EXPECT_EQ(range->get_delta(), 1);
}
}  // namespace ops
}  // namespace mindspore
