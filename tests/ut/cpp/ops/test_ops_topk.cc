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
#include "ops/topk.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class TestTopK : public UT::Common {
 public:
  TestTopK() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestTopK, test_topk) {
  auto topk = std::make_shared<TopK>();
  bool sorted = true;
  topk->Init(sorted);
  EXPECT_EQ(topk->get_sorted(), true);
  auto tensor_x = std::make_shared<tensor::Tensor>(kNumberTypeFloat16, std::vector<int64_t>{5});
  MS_EXCEPTION_IF_NULL(tensor_x);
  auto tensor_x_data = reinterpret_cast<int *>(tensor_x->data_c());
  *tensor_x_data = 1;
  tensor_x_data++;
  *tensor_x_data = 2;
  tensor_x_data++;
  *tensor_x_data = 3;
  tensor_x_data++;
  *tensor_x_data = 4;
  tensor_x_data++;
  *tensor_x_data = 5;
  tensor_x_data++;
  auto k = MakeValue(3);
  MS_EXCEPTION_IF_NULL(k);
  auto abstract = topk->Infer({tensor_x->ToAbstract(), k->ToAbstract()});
  MS_EXCEPTION_IF_NULL(abstract);
  EXPECT_EQ(abstract->isa<abstract::AbstractTuple>(), true);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::TupleShape>(), true);
  auto shape = shape_ptr->cast<abstract::TupleShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  EXPECT_EQ(shape_vec.size(), 2);
  auto shape1 = shape_vec[0]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape1.size(), 1);
  EXPECT_EQ(shape1[0], 3);
  auto shape2 = shape_vec[1]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape2.size(), 1);
  EXPECT_EQ(shape2[0], 3);
  auto type_ptr = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type = type_ptr->cast<TuplePtr>();
  auto type_vec = type->elements();
  EXPECT_EQ(type_vec.size(), 2);
  MS_EXCEPTION_IF_NULL(type_vec[0]);
  auto data_type1 = type_vec[0]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type1);
  EXPECT_EQ(data_type1->type_id(), kNumberTypeFloat16);
  auto data_type2 = type_vec[1]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data_type2);
  EXPECT_EQ(data_type2->type_id(), kNumberTypeInt32);
}
}  // namespace ops
}  // namespace mindspore
