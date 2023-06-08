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
#include "ops/strided_slice.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void SetTensorData(void *data, std::vector<T> num) {
  MS_EXCEPTION_IF_NULL(data);
  auto tensor_data = reinterpret_cast<T *>(data);
  MS_EXCEPTION_IF_NULL(tensor_data);
  for (size_t index = 0; index < num.size(); ++index) {
    *tensor_data = num[index];
  }
}
}  // namespace
class TestStridedSlice : public UT::Common {
 public:
  TestStridedSlice() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestStridedSlice, test_ops_stridedslice1) {
  auto stridedslice = std::make_shared<StridedSlice>();
  stridedslice->Init(0, 0, 0, 0, 0);
  EXPECT_EQ(stridedslice->get_begin_mask(), 0);
  EXPECT_EQ(stridedslice->get_end_mask(), 0);
  EXPECT_EQ(stridedslice->get_ellipsis_mask(), 0);
  EXPECT_EQ(stridedslice->get_new_axis_mask(), 0);
  EXPECT_EQ(stridedslice->get_shrink_axis_mask(), 0);
  auto tensor_x = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{3, 3, 3});
  auto begin = MakeValue(std::vector<int64_t>{1, 0, 0});
  auto end = MakeValue(std::vector<int64_t>{2, 1, 3});
  auto strides = MakeValue(std::vector<int64_t>{1, 1, 1});
  MS_EXCEPTION_IF_NULL(tensor_x);
  MS_EXCEPTION_IF_NULL(begin);
  MS_EXCEPTION_IF_NULL(end);
  MS_EXCEPTION_IF_NULL(strides);
  auto abstract =
    stridedslice->Infer({tensor_x->ToAbstract(), begin->ToAbstract(), end->ToAbstract(), strides->ToAbstract()});
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
  EXPECT_EQ(shape_vec.size(), 3);
  EXPECT_EQ(shape_vec[0], 1);
  EXPECT_EQ(shape_vec[1], 1);
  EXPECT_EQ(shape_vec[2], 3);
}
/*
TEST_F(TestStridedSlice, test_ops_stridedslice2) {
auto stridedslice = std::make_shared<StridedSlice>();
stridedslice->Init(0, 0, 0, 0, 0);
EXPECT_EQ(stridedslice->get_begin_mask(), 0);
EXPECT_EQ(stridedslice->get_end_mask(), 0);
EXPECT_EQ(stridedslice->get_ellipsis_mask(), 0);
EXPECT_EQ(stridedslice->get_new_axis_mask(), 0);
EXPECT_EQ(stridedslice->get_shrink_axis_mask(), 0);
auto tensor_x = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{3,3,3});
auto begin = MakeValue(std::vector<int64_t>{1,0,0});
auto end = MakeValue(std::vector<int64_t>{2,2,3});
auto strides =MakeValue(std::vector<int64_t>{1,1,1});
MS_EXCEPTION_IF_NULL(tensor_x);
MS_EXCEPTION_IF_NULL(begin);
MS_EXCEPTION_IF_NULL(end);
MS_EXCEPTION_IF_NULL(strides);
auto abstract =
stridedslice->Infer({tensor_x->ToAbstract(),begin->ToAbstract(),end->ToAbstract(),strides->ToAbstract()});
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
EXPECT_EQ(shape_vec.size(), 3);
EXPECT_EQ(shape_vec[0], 1);
EXPECT_EQ(shape_vec[1], 2);
EXPECT_EQ(shape_vec[2], 3);
}

TEST_F(TestStridedSlice, test_ops_stridedslice3) {
auto stridedslice = std::make_shared<StridedSlice>();
stridedslice->Init(0, 0, 0, 0, 0);
EXPECT_EQ(stridedslice->get_begin_mask(), 0);
EXPECT_EQ(stridedslice->get_end_mask(), 0);
EXPECT_EQ(stridedslice->get_ellipsis_mask(), 0);
EXPECT_EQ(stridedslice->get_new_axis_mask(), 0);
EXPECT_EQ(stridedslice->get_shrink_axis_mask(), 0);
auto tensor_x = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{3,3,3});
auto begin = MakeValue(std::vector<int64_t>{1,0,0});
auto end = MakeValue(std::vector<int64_t>{2,-3,3});
auto strides =MakeValue(std::vector<int64_t>{1,-1,1});
MS_EXCEPTION_IF_NULL(tensor_x);
MS_EXCEPTION_IF_NULL(begin);
MS_EXCEPTION_IF_NULL(end);
MS_EXCEPTION_IF_NULL(strides);
auto abstract =
stridedslice->Infer({tensor_x->ToAbstract(),begin->ToAbstract(),end->ToAbstract(),strides->ToAbstract()});
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
EXPECT_EQ(shape_vec.size(), 3);
EXPECT_EQ(shape_vec[0], 1);
EXPECT_EQ(shape_vec[1], 2);
EXPECT_EQ(shape_vec[2], 3);
}

TEST_F(TestStridedSlice, test_ops_stridedslice4) {
auto stridedslice = std::make_shared<StridedSlice>();
stridedslice->Init(0, 0, 0, 0, 0);
EXPECT_EQ(stridedslice->get_begin_mask(), 0);
EXPECT_EQ(stridedslice->get_end_mask(), 0);
EXPECT_EQ(stridedslice->get_ellipsis_mask(), 0);
EXPECT_EQ(stridedslice->get_new_axis_mask(), 0);
EXPECT_EQ(stridedslice->get_shrink_axis_mask(), 0);

auto tensor_x = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{5});
auto begin = MakeValue(std::vector<int64_t>{1});
auto end = MakeValue(std::vector<int64_t>{-2});
auto strides =MakeValue(std::vector<int64_t>{1});
MS_EXCEPTION_IF_NULL(tensor_x);
MS_EXCEPTION_IF_NULL(begin);
MS_EXCEPTION_IF_NULL(end);
MS_EXCEPTION_IF_NULL(strides);
auto abstract =
stridedslice->Infer({tensor_x->ToAbstract(),begin->ToAbstract(),end->ToAbstract(),strides->ToAbstract()});
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
EXPECT_EQ(shape_vec[0], 2);
}*/
}  // namespace ops
}  // namespace mindspore