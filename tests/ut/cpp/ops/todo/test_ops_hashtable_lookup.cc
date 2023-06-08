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
#include "ops/hashtable_lookup.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {

class TestHashtableLookup : public UT::Common {
 public:
  TestHashtableLookup() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestHashtableLookup, test_ops_hashtable_lookup1) {
  auto hashtable_lookup = std::make_shared<HashtableLookup>();
  hashtable_lookup->Init();
  auto inputs0 = TensorConstructUtils::CreateOnesTensor(kNumberTypeInt32, std::vector<int64_t>{4, 3});
  auto inputs1 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1});
  auto inputs2 = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1});
  MS_EXCEPTION_IF_NULL(inputs0);
  MS_EXCEPTION_IF_NULL(inputs1);
  MS_EXCEPTION_IF_NULL(inputs2);
  auto abstract = hashtable_lookup->Infer({inputs0->ToAbstract(), inputs1->ToAbstract(), inputs2->ToAbstract()});
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
  EXPECT_EQ(shape1.size(), 0);
  auto shape2 = shape_vec[1]->cast<abstract::ShapePtr>()->shape();
  EXPECT_EQ(shape2.size(), 1);
  EXPECT_EQ(shape2[0], 4);
  auto type_ptr = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type = type_ptr->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(type);
  auto type_vec = type->elements();
  MS_EXCEPTION_IF_NULL(type_vec[0]);
  auto data0_type = type_vec[0]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data0_type);
  EXPECT_EQ(data0_type->type_id(), kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(type_vec[1]);
  auto data1_type = type_vec[1]->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(data1_type);
  EXPECT_EQ(data1_type->type_id(), kNumberTypeInt8);
}

}  // namespace ops
}  // namespace mindspore
