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
#include "ops/softmax.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class TestSoftMax : public UT::Common {
 public:
  TestSoftMax() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestSoftMax, test_ops_softmax1) {
  auto softmax = std::make_shared<Softmax>();
  std::vector<std::int64_t> init_data = {-1};
  softmax->Init(-1);
  EXPECT_EQ(softmax->get_axis(), init_data);
  softmax->set_axis(init_data);
  EXPECT_EQ(softmax->get_axis(), init_data);
  auto input1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{1, 2, 3, 4, 5});
  MS_EXCEPTION_IF_NULL(input1);
  auto abstract = softmax->Infer({input1->ToAbstract()});
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
  EXPECT_EQ(shape_vec.size(), 5);
  EXPECT_EQ(shape_vec[0], 1);
}

TEST_F(TestSoftMax, test_ops_softmax2) {
  auto softmax = std::make_shared<Softmax>();
  std::vector<std::int64_t> init_data = {-1};
  softmax->Init(-1);
  EXPECT_EQ(softmax->get_axis(), init_data);
  softmax->set_axis(init_data);
  EXPECT_EQ(softmax->get_axis(), init_data);
  auto input1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1, 2, 3, 4, 5});
  MS_EXCEPTION_IF_NULL(input1);
  auto abstract = softmax->Infer({input1->ToAbstract()});
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
  EXPECT_EQ(shape_vec.size(), 5);
  EXPECT_EQ(shape_vec[0], 1);
}

TEST_F(TestSoftMax, test_ops_softmax3) {
  auto softmax = std::make_shared<Softmax>();
  std::vector<std::int64_t> init_data = {-1};
  softmax->Init(-1);
  EXPECT_EQ(softmax->get_axis(), init_data);
  softmax->set_axis(init_data);
  EXPECT_EQ(softmax->get_axis(), init_data);
  auto input1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat64, std::vector<int64_t>{1, 2, 3, 4, 5});
  MS_EXCEPTION_IF_NULL(input1);
  auto abstract = softmax->Infer({input1->ToAbstract()});
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
  EXPECT_EQ(data_type->type_id(), kNumberTypeFloat64);
  EXPECT_EQ(shape_vec.size(), 5);
  EXPECT_EQ(shape_vec[0], 1);
}
}  // namespace ops
}  // namespace mindspore
