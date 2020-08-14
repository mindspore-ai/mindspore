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

#include <iostream>
#include <memory>
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/reverse_sequence.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {
class TestReverseSequenceFp32 : public mindspore::CommonTest {
 public:
  TestReverseSequenceFp32() {}
};

TEST_F(TestReverseSequenceFp32, BatchLessSeq) {
  lite::tensor::Tensor in_tensor0(kNumberTypeFloat32, {2, 3, 4, 2});
  lite::tensor::Tensor in_tensor1(kNumberTypeInt32, {3});
  lite::tensor::Tensor out_tensor(kNumberTypeFloat32, {2, 3, 4, 2});
  float input_data0[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
  int input_data1[] = {2, 3, 4};
  float output_data[2 * 3 * 4 * 2] = {0};
  in_tensor0.SetData(input_data0);
  in_tensor1.SetData(input_data1);
  out_tensor.SetData(output_data);
  std::vector<lite::tensor::Tensor *> inputs = {&in_tensor0, &in_tensor1};
  std::vector<lite::tensor::Tensor *> outputs = {&out_tensor};

  ReverseSequenceParameter parameter = {0};
  parameter.batch_axis_ = 1;
  parameter.seq_axis_ = 2;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_ReverseSequence};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::Context>();
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc, nullptr);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect[] = {2,  3,  0,  1,  4,  5,  6,  7,  12, 13, 10, 11, 8,  9,  14, 15, 22, 23, 20, 21, 18, 19, 16, 17,
                    26, 27, 24, 25, 28, 29, 30, 31, 36, 37, 34, 35, 32, 33, 38, 39, 46, 47, 44, 45, 42, 43, 40, 41};
  EXPECT_EQ(out_tensor.ElementsNum(), 2 * 3 * 4 * 2);

  for (int i = 0; i < 2 * 3 * 4 * 2; i++) {
    EXPECT_EQ(output_data[i], expect[i]);
  }

  in_tensor0.SetData(nullptr);
  in_tensor1.SetData(nullptr);
  out_tensor.SetData(nullptr);
}

TEST_F(TestReverseSequenceFp32, BatchGreaterSeq) {
  lite::tensor::Tensor in_tensor0(kNumberTypeFloat32, {2, 3, 4, 2});
  lite::tensor::Tensor in_tensor1(kNumberTypeInt32, {4});
  lite::tensor::Tensor out_tensor(kNumberTypeFloat32, {2, 3, 4, 2});
  float input_data0[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
  int input_data1[] = {2, 3, 3, 2};
  float output_data[2 * 3 * 4 * 2] = {0};
  in_tensor0.SetData(input_data0);
  in_tensor1.SetData(input_data1);
  out_tensor.SetData(output_data);
  std::vector<lite::tensor::Tensor *> inputs = {&in_tensor0, &in_tensor1};
  std::vector<lite::tensor::Tensor *> outputs = {&out_tensor};

  ReverseSequenceParameter parameter = {0};
  parameter.batch_axis_ = 2;
  parameter.seq_axis_ = 1;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_ReverseSequence};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::Context>();
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc, nullptr);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect[] = {8,  9,  18, 19, 20, 21, 14, 15, 0,  1,  10, 11, 12, 13, 6,  7,  16, 17, 2,  3,  4,  5,  22, 23,
                    32, 33, 42, 43, 44, 45, 38, 39, 24, 25, 34, 35, 36, 37, 30, 31, 40, 41, 26, 27, 28, 29, 46, 47};
  EXPECT_EQ(out_tensor.ElementsNum(), 2 * 3 * 4 * 2);

  for (int i = 0; i < 2 * 3 * 4 * 2; i++) {
    EXPECT_EQ(output_data[i], expect[i]);
  }

  in_tensor0.SetData(nullptr);
  in_tensor1.SetData(nullptr);
  out_tensor.SetData(nullptr);
}

TEST_F(TestReverseSequenceFp32, BatchSeqNotAdjacent) {
  lite::tensor::Tensor in_tensor0(kNumberTypeFloat32, {2, 3, 4, 2});
  lite::tensor::Tensor in_tensor1(kNumberTypeInt32, {2});
  lite::tensor::Tensor out_tensor(kNumberTypeFloat32, {2, 3, 4, 2});
  float input_data0[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
  int input_data1[] = {2, 4};
  float output_data[2 * 3 * 4 * 2] = {0};
  in_tensor0.SetData(input_data0);
  in_tensor1.SetData(input_data1);
  out_tensor.SetData(output_data);
  std::vector<lite::tensor::Tensor *> inputs = {&in_tensor0, &in_tensor1};
  std::vector<lite::tensor::Tensor *> outputs = {&out_tensor};

  ReverseSequenceParameter parameter = {0};
  parameter.batch_axis_ = 0;
  parameter.seq_axis_ = 2;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_ReverseSequence};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  EXPECT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::Context>();
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(&parameter), ctx.get(), desc, nullptr);
  EXPECT_NE(kernel, nullptr);

  auto ret = kernel->Run();
  EXPECT_EQ(0, ret);

  float expect[] = {2,  3,  0,  1,  4,  5,  6,  7,  10, 11, 8,  9,  12, 13, 14, 15, 18, 19, 16, 17, 20, 21, 22, 23,
                    30, 31, 28, 29, 26, 27, 24, 25, 38, 39, 36, 37, 34, 35, 32, 33, 46, 47, 44, 45, 42, 43, 40, 41};
  EXPECT_EQ(out_tensor.ElementsNum(), 2 * 3 * 4 * 2);

  for (int i = 0; i < 2 * 3 * 4 * 2; i++) {
    EXPECT_EQ(output_data[i], expect[i]);
  }

  in_tensor0.SetData(nullptr);
  in_tensor1.SetData(nullptr);
  out_tensor.SetData(nullptr);
}
}  // namespace mindspore
