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
#include "ops/rfft.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
namespace mindspore {
namespace ops {
class TestRfft : public UT::Common {
 public:
  TestRfft() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestRfft, test_ops_rfft) {
  auto rfft = std::make_shared<Rfft>();
  rfft->Init(2);
  EXPECT_EQ(rfft->get_fft_length(), 2);
  rfft->set_fft_length(2);
  auto tensor_x = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1, 3});
  MS_EXCEPTION_IF_NULL(tensor_x);
  auto rfft_abstract = rfft->Infer({tensor_x->ToAbstract()});
  MS_EXCEPTION_IF_NULL(rfft_abstract);
  EXPECT_EQ(rfft_abstract->isa<abstract::AbstractTensor>(), true);
  auto shape_ptr = rfft_abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::Shape>(), true);
  auto rfft_shape = shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(rfft_shape);
  auto shape_vec = rfft_shape->shape();
  auto type = rfft_abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  EXPECT_EQ(type->isa<TensorType>(), true);
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto elem_type = tensor_type->element();
  EXPECT_EQ(elem_type->type_id(), kNumberTypeComplex64);
  EXPECT_EQ(shape_vec.size(), 3);
  EXPECT_EQ(shape_vec[0], 1);
  EXPECT_EQ(shape_vec[1], 2);
  EXPECT_EQ(shape_vec[2], 2);
  EXPECT_EQ(rfft->get_fft_length(), 2);
}

}  // namespace ops
}  // namespace mindspore
