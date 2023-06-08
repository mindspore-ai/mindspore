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
#include "ops/gather.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class TestGather : public UT::Common {
 public:
  TestGather() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestGather, test_gather) {
  auto gather = std::make_shared<Gather>();
  gather->Init();
  auto tensor_x = std::make_shared<tensor::Tensor>(kNumberTypeInt32, std::vector<int64_t>{2, 2});
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
  auto index = std::make_shared<tensor::Tensor>(kNumberTypeInt32, std::vector<int64_t>{2, 2});
  MS_EXCEPTION_IF_NULL(index);
  auto index_data = reinterpret_cast<int *>(index->data_c());
  *index_data = 0;
  index_data++;
  *index_data = 0;
  index_data++;
  *index_data = 1;
  index_data++;
  *index_data = 0;
  index_data++;
  auto dim = MakeValue(1);
  MS_EXCEPTION_IF_NULL(dim);
  auto abstract = gather->Infer({tensor_x->ToAbstract(), dim->ToAbstract(), index->ToAbstract()});
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
  EXPECT_EQ(shape_vec.size(), 2);
  EXPECT_EQ(shape_vec[0], 2);
  EXPECT_EQ(shape_vec[1], 2);
}
}  // namespace ops
}  // namespace mindspore
