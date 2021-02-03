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
#include "ops/crop.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class TestCrop : public UT::Common {
 public:
  TestCrop() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestCrop, test_ops_crop1) {
  auto crop = std::make_shared<Crop>();
  crop->Init(1, std::vector<int64_t>{1, 1, 1, 1});
  std::vector<int64_t> ret = crop->get_offsets();
  EXPECT_EQ(crop->get_axis(), 1);
  for (auto item : ret) {
    EXPECT_EQ(item, 1);
  }
  auto tensor_x1 = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, std::vector<int64_t>{2, 2});
  auto tensor_x2 = std::make_shared<tensor::Tensor>(kNumberTypeInt32, std::vector<int64_t>{1});
  MS_EXCEPTION_IF_NULL(tensor_x1);
  MS_EXCEPTION_IF_NULL(tensor_x2);
  auto tensor_x1_data = reinterpret_cast<float *>(tensor_x1->data_c());
  *tensor_x1_data = 1.0;
  tensor_x1_data++;
  *tensor_x1_data = 2.0;
  tensor_x1_data++;
  *tensor_x1_data = 3.0;
  tensor_x1_data++;
  *tensor_x1_data = 4.0;
  tensor_x1_data++;
  auto tensor_x2_data = reinterpret_cast<int *>(tensor_x2->data_c());
  *tensor_x2_data = 1;
  auto abstract = crop->Infer({tensor_x1->ToAbstract(), tensor_x2->ToAbstract()});
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
  EXPECT_EQ(shape_vec[0], 1);
}
}  // namespace ops
}  // namespace mindspore