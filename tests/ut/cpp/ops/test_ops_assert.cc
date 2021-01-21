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
#include "ops/assert.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {

namespace {
template <typename T>
void SetTensorData(void *data, T num, size_t data_length) {
  MS_EXCEPTION_IF_NULL(data);
  auto tensor_data = reinterpret_cast<T *>(data);
  MS_EXCEPTION_IF_NULL(tensor_data);
  for (size_t index = 0; index < data_length; ++index) {
    *tensor_data = num;
    ++tensor_data;
  }
}
}  // namespace

class TestAssert : public UT::Common {
 public:
  TestAssert() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestAssert, test_ops_assert1) {
  auto assert = std::make_shared<Assert>();
  assert->Init(3);
  EXPECT_EQ(assert->get_summarize(), 3);
  std::vector<ValuePtr> inputs_ = {TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1})};
  auto condition = MakeValue(std::vector<bool>{true});
  auto inputs = std::make_shared<ValueTuple>(inputs_);
  auto abstract = assert->Infer({condition->ToAbstract(), inputs->ToAbstract()});
  MS_EXCEPTION_IF_NULL(abstract);
  EXPECT_EQ(abstract->isa<abstract::AbstractTensor>(), true);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::Shape>(), true);
  auto shape = shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  EXPECT_EQ(shape_vec.size(), 1);
  EXPECT_EQ(shape_vec[0], 1);
  auto type = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  EXPECT_EQ(type->isa<TensorType>(), true);
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  EXPECT_EQ(data_type->type_id(), kNumberTypeInt32);
}

TEST_F(TestAssert, test_ops_assert2) {
  auto assert = std::make_shared<Assert>();
  assert->Init(3);
  EXPECT_EQ(assert->get_summarize(), 3);
  std::vector<ValuePtr> inputs_ = {TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1})};
  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeBool, std::vector<int64_t>{1});
  MS_EXCEPTION_IF_NULL(tensor);
  auto mem_size = IntToSize(tensor->ElementsNum());
  SetTensorData<bool>(tensor->data_c(), true, mem_size);
  auto inputs = std::make_shared<ValueTuple>(inputs_);
  auto abstract = assert->Infer({tensor->ToAbstract(), inputs->ToAbstract()});
  MS_EXCEPTION_IF_NULL(abstract);
  EXPECT_EQ(abstract->isa<abstract::AbstractTensor>(), true);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::Shape>(), true);
  auto shape = shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  EXPECT_EQ(shape_vec.size(), 1);
  EXPECT_EQ(shape_vec[0], 1);
  auto type = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  EXPECT_EQ(type->isa<TensorType>(), true);
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  EXPECT_EQ(data_type->type_id(), kNumberTypeInt32);
}

}  // namespace ops
}  // namespace mindspore
