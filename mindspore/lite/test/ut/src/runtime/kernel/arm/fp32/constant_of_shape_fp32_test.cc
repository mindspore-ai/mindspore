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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/base/constant_of_shape.h"
#include "src/kernel_registry.h"
#include "src/lite_kernel.h"

namespace mindspore {
class TestConstantOfShapeFp32 : public mindspore::CommonTest {
 public:
  TestConstantOfShapeFp32() {}
};

int ConstantOfShapeTestInit(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_, float *a_ptr,
                            std::vector<int> a_shape) {
  auto in_t = new lite::Tensor(kNumberTypeInt32, a_shape, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  in_t->MallocData();
  memcpy(in_t->MutableData(), a_ptr, sizeof(float) * in_t->ElementsNum());
  inputs_->push_back(in_t);

  std::vector<int> c_shape(in_t->ElementsNum());
  for (unsigned int i = 0; i < c_shape.size(); ++i) {
    c_shape[i] = a_ptr[i];
  }
  auto out_t = new lite::Tensor(kNumberTypeFloat, c_shape, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  out_t->MallocData();
  outputs_->push_back(out_t);

  return out_t->ElementsNum();
}

TEST_F(TestConstantOfShapeFp32, Simple) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto param = new ConstantOfShapeParameter();
  param->value_.f32_value_ = 1;
  param->data_type_ = kNumberTypeFloat32;
  float a[] = {1, 2, 3, 4};
  std::vector<int> a_shape = {4, 1, 1, 1};
  // std::vector<int> c_shape = {2, 2, 2, 1};
  int total_size = ConstantOfShapeTestInit(&inputs_, &outputs_, a, a_shape);
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 4;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::ConstantOfShapeCPUKernel *op =
    new kernel::ConstantOfShapeCPUKernel(reinterpret_cast<OpParameter *>(param), inputs_, outputs_, ctx);
  op->Init();
  op->Run();
  float correct[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float *output = reinterpret_cast<float *>(outputs_[0]->MutableData());
  for (int i = 0; i < 8; ++i) printf("%f ", output[i]);
  printf("\n");
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct, total_size, 0.0001));
  delete op;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
}
}  // namespace mindspore
