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
#include "ops/conv2d.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
namespace mindspore {
namespace ops {
class TestConv2d : public UT::Common {
 public:
  TestConv2d() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestConv2d, test_ops_conv2d) {
  auto conv_2d = std::make_shared<Conv2D>();
  conv_2d->Init(64, {7, 7});
  std::vector<int64_t> kernel_size = conv_2d->get_kernel_size();
  for (auto item : kernel_size) {
    EXPECT_EQ(item, 7);
  }
  std::vector<int64_t> stride = conv_2d->get_stride();
  for (auto item : stride) {
    EXPECT_EQ(item, 1);
  }
  std::vector<int64_t> dilation = conv_2d->get_dilation();
  for (auto item : dilation) {
    EXPECT_EQ(item, 1);
  }
  EXPECT_EQ(conv_2d->get_pad_mode(), VALID);
  std::vector<int64_t> pad = conv_2d->get_pad();
  for (auto item : pad) {
    EXPECT_EQ(item, 0);
  }
  EXPECT_EQ(conv_2d->get_mode(), 1);
  EXPECT_EQ(conv_2d->get_group(), 1);
  EXPECT_EQ(conv_2d->get_out_channel(), 64);
  EXPECT_EQ(conv_2d->get_format(), NCHW);
  auto tensor_x = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{32, 3, 224, 224});
  auto tensor_w = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{64, 3, 7, 7});
  MS_EXCEPTION_IF_NULL(tensor_x);
  MS_EXCEPTION_IF_NULL(tensor_w);
  auto conv_abstract = conv_2d->Infer({tensor_x->ToAbstract(), tensor_w->ToAbstract()});
  MS_EXCEPTION_IF_NULL(conv_abstract);
  EXPECT_EQ(conv_abstract->isa<abstract::AbstractTensor>(), true);
  auto shape_ptr = conv_abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::Shape>(), true);
  auto conv_shape = shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(conv_shape);
  auto shape_vec = conv_shape->shape();
  auto type = conv_abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  EXPECT_EQ(type->isa<TensorType>(), true);
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto elem_type = tensor_type->element();
  EXPECT_EQ(elem_type->type_id(), kNumberTypeFloat32);
  EXPECT_EQ(shape_vec.size(), 4);
  EXPECT_EQ(shape_vec[0], 32);
  EXPECT_EQ(shape_vec[1], 64);
  EXPECT_EQ(shape_vec[2], 218);
  EXPECT_EQ(shape_vec[3], 218);
}

}  // namespace ops
}  // namespace mindspore
