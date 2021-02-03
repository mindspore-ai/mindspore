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
#include "ops/merge.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {

class TestMerge : public UT::Common {
 public:
  TestMerge() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestMerge, test_ops_merge1) {
  auto merge = std::make_shared<Merge>();
  merge->Init();
  auto input_x = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{2, 4});
  auto input_y = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{2, 4});
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_y);
  std::vector<ValuePtr> inputs_ = {input_x, input_y};
  auto input = std::make_shared<ValueTuple>(inputs_);
  auto abstract = merge->Infer({input->ToAbstract()});
  MS_EXCEPTION_IF_NULL(abstract);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::TupleShape>(), true);
  auto shape = shape_ptr->cast<abstract::TupleShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  EXPECT_EQ(shape_vec.size(), 2);
  auto shape1 = shape_vec[0]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape1.size(), 2);
  EXPECT_EQ(shape1[0], 2);
  EXPECT_EQ(shape1[1], 4);
  auto shape2 = shape_vec[1]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape2.size(), 1);
  EXPECT_EQ(shape2[0], 1);
  auto type_ptr = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type = type_ptr->cast<TuplePtr>();
  auto type_vec = type->elements();
  MS_EXCEPTION_IF_NULL(type_vec[0]);
  auto data_type1 = type_vec[0]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type1);
  EXPECT_EQ(data_type1->type_id(), kNumberTypeFloat32);
  auto data_type2 = type_vec[1]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type2);
  EXPECT_EQ(data_type2->type_id(), kNumberTypeInt32);
}

}  // namespace ops
}  // namespace mindspore
