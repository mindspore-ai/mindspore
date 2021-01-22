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
#include "ops/grad/binary_cross_entropy_grad.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class TestBinaryCrossEntropyGrad : public UT::Common {
 public:
  TestBinaryCrossEntropyGrad() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestBinaryCrossEntropyGrad, test_ops_binary_cross_entropy_grad1) {
  auto binary_cross_entropy_grad = std::make_shared<BinaryCrossEntropyGrad>();
  binary_cross_entropy_grad->Init(MEAN);
  EXPECT_EQ(binary_cross_entropy_grad->get_reduction(), MEAN);
  binary_cross_entropy_grad->set_reduction(MEAN);
  EXPECT_EQ(binary_cross_entropy_grad->get_reduction(), MEAN);
  auto input1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{1, 2, 3, 4, 5});
  auto input2 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{1, 2, 3, 4, 5});
  auto input3 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{1, 2, 3, 4, 5});
  auto input4 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat16, std::vector<int64_t>{1, 2, 3, 4, 5});
  auto abstract = binary_cross_entropy_grad->Infer(
    {input1->ToAbstract(), input2->ToAbstract(), input3->ToAbstract(), input4->ToAbstract()});
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

TEST_F(TestBinaryCrossEntropyGrad, test_ops_binary_cross_entropy_grad2) {
  auto binary_cross_entropy_grad = std::make_shared<BinaryCrossEntropyGrad>();
  binary_cross_entropy_grad->Init(MEAN);
  EXPECT_EQ(binary_cross_entropy_grad->get_reduction(), MEAN);
  binary_cross_entropy_grad->set_reduction(MEAN);
  EXPECT_EQ(binary_cross_entropy_grad->get_reduction(), MEAN);
  auto input1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1, 2, 3, 4, 5});
  auto input2 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1, 2, 3, 4, 5});
  auto input3 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1, 2, 3, 4, 5});
  auto input4 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1, 2, 3, 4, 5});
  auto abstract = binary_cross_entropy_grad->Infer(
    {input1->ToAbstract(), input2->ToAbstract(), input3->ToAbstract(), input4->ToAbstract()});
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
}  // namespace ops
}  // namespace mindspore
