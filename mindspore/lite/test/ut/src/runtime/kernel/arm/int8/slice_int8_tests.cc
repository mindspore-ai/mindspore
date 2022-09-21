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
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "mindspore/lite/src/litert/kernel/cpu/int8/slice_int8.h"
#include "mindspore/lite/src/litert/kernel_registry.h"

namespace mindspore {
class TestSliceInt8 : public mindspore::CommonTest {
 public:
  TestSliceInt8() {}
};

TEST_F(TestSliceInt8, SliceInt8) {
  lite::Tensor in_tensor(kNumberTypeInt8, {1, 3, 2, 3});
  lite::Tensor out_tensor(kNumberTypeInt8, {1, 2, 2, 3});

  int8_t input_data[] = {105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  int8_t output_data[12];
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  lite::Tensor begin_tensor(kNumberTypeInt32, {4});
  int begin_data[4] = {0, 1, 0, 0};
  begin_tensor.set_data(begin_data);
  lite::Tensor size_tensor(kNumberTypeInt32, {4});
  int size_data[4] = {1, 2, 2, 3};
  size_tensor.set_data(size_data);

  const lite::LiteQuantParam quant_in0 = {0.00784314f, 0};  // -1.0--1.0 -> 0--255
  const lite::LiteQuantParam quant_out = {0.00784314f, 0};
  in_tensor.AddQuantParam(quant_in0);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor, &begin_tensor, &size_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  SliceParameter *parameter = new (std::nothrow) SliceParameter;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_SliceFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect0[12] = {16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(output_data[i], expect0[i]);
  }

  in_tensor.set_data(nullptr);
  begin_tensor.set_data(nullptr);
  size_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  delete kernel;
}

TEST_F(TestSliceInt8, Slice5D) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 1, 3, 2, 3});
  lite::Tensor out_tensor(kNumberTypeInt8, {1, 1, 2, 2, 3});

  int8_t input_data[] = {105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68,
                         105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  int8_t output_data[12];
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  lite::Tensor begin_tensor(kNumberTypeInt32, {5});
  int begin_data[5] = {0, 0, 1, 0, 0};
  begin_tensor.set_data(begin_data);
  lite::Tensor size_tensor(kNumberTypeInt32, {5});
  int size_data[5] = {1, 1, 2, 2, 3};
  size_tensor.set_data(size_data);

  const lite::LiteQuantParam quant_in0 = {0.00784314f, 0};  // -1.0--1.0 -> 0--255
  const lite::LiteQuantParam quant_out = {0.00784314f, 0};
  in_tensor.AddQuantParam(quant_in0);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor, &begin_tensor, &size_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  SliceParameter *parameter = new (std::nothrow) SliceParameter;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_SliceFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect0[12] = {16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(output_data[i], expect0[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  begin_tensor.set_data(nullptr);
  size_tensor.set_data(nullptr);
  delete kernel;
}

TEST_F(TestSliceInt8, Slice6D) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 1, 1, 3, 2, 3});
  lite::Tensor out_tensor(kNumberTypeInt8, {1, 1, 1, 2, 2, 3});

  int8_t input_data[] = {105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68,
                         105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  int8_t output_data[12];
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  lite::Tensor begin_tensor(kNumberTypeInt32, {6});
  int begin_data[6] = {0, 0, 0, 1, 0, 0};
  begin_tensor.set_data(begin_data);
  lite::Tensor size_tensor(kNumberTypeInt32, {6});
  int size_data[6] = {1, 1, 1, 2, 2, 3};
  size_tensor.set_data(size_data);

  const lite::LiteQuantParam quant_in0 = {0.00784314f, 0};  // -1.0--1.0 -> 0--255
  const lite::LiteQuantParam quant_out = {0.00784314f, 0};
  in_tensor.AddQuantParam(quant_in0);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor, &begin_tensor, &size_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  SliceParameter *parameter = new (std::nothrow) SliceParameter;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_SliceFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect0[12] = {16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(output_data[i], expect0[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  begin_tensor.set_data(nullptr);
  size_tensor.set_data(nullptr);
  delete kernel;
}

TEST_F(TestSliceInt8, Slice7D) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 1, 1, 1, 3, 2, 3});
  lite::Tensor out_tensor(kNumberTypeInt8, {1, 1, 1, 1, 2, 2, 3});

  int8_t input_data[] = {105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68,
                         105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  int8_t output_data[12];
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);
  lite::Tensor begin_tensor(kNumberTypeInt32, {7});
  int begin_data[7] = {0, 0, 0, 0, 1, 0, 0};
  begin_tensor.set_data(begin_data);
  lite::Tensor size_tensor(kNumberTypeInt32, {7});
  int size_data[7] = {1, 1, 1, 1, 2, 2, 3};
  size_tensor.set_data(size_data);

  const lite::LiteQuantParam quant_in0 = {0.00784314f, 0};  // -1.0--1.0 -> 0--255
  const lite::LiteQuantParam quant_out = {0.00784314f, 0};
  in_tensor.AddQuantParam(quant_in0);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor, &begin_tensor, &size_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  SliceParameter *parameter = new (std::nothrow) SliceParameter;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_SliceFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect0[12] = {16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(output_data[i], expect0[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  begin_tensor.set_data(nullptr);
  size_tensor.set_data(nullptr);
  delete kernel;
}

TEST_F(TestSliceInt8, Slice8D) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 1, 1, 1, 1, 3, 2, 3});
  lite::Tensor out_tensor(kNumberTypeInt8, {1, 1, 1, 1, 1, 1, 2, 3});

  lite::Tensor begin_tensor(kNumberTypeInt32, {8});
  int begin_data[8] = {1, 0, 0, 0, 0, 1, 0, 0};
  begin_tensor.set_data(begin_data);
  lite::Tensor size_tensor(kNumberTypeInt32, {8});
  int size_data[8] = {1, 1, 1, 1, 1, 2, 2, 3};
  size_tensor.set_data(size_data);

  int8_t input_data[] = {105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68,
                         105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  int8_t output_data[12];
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);

  const lite::LiteQuantParam quant_in0 = {0.00784314f, 0};  // -1.0--1.0 -> 0--255
  const lite::LiteQuantParam quant_out = {0.00784314f, 0};
  in_tensor.AddQuantParam(quant_in0);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor, &begin_tensor, &size_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  SliceParameter *parameter = new (std::nothrow) SliceParameter;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_SliceFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect0[12] = {16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(output_data[i], expect0[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  begin_tensor.set_data(nullptr);
  size_tensor.set_data(nullptr);
  delete kernel;
}

TEST_F(TestSliceInt8, SliceDiffQuantArgs) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 1, 1, 1, 1, 3, 2, 3});
  lite::Tensor out_tensor(kNumberTypeInt8, {1, 1, 1, 1, 1, 1, 2, 3});

  lite::Tensor begin_tensor(kNumberTypeInt32, {8});
  int begin_data[8] = {1, 0, 0, 0, 0, 1, 0, 0};
  begin_tensor.set_data(begin_data);
  lite::Tensor size_tensor(kNumberTypeInt32, {8});
  int size_data[8] = {1, 1, 1, 1, 1, 2, 2, 3};
  size_tensor.set_data(size_data);

  int8_t input_data[] = {105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68,
                         105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  int8_t output_data[12];
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);

  const lite::LiteQuantParam quant_in0 = {0.00784314f, 0};  // -1.0--1.0 -> 0--255
  const lite::LiteQuantParam quant_out = {0.01568628f, 0};
  in_tensor.AddQuantParam(quant_in0);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor, &begin_tensor, &size_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  SliceParameter *parameter = new (std::nothrow) SliceParameter;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_SliceFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect0[12] = {8, 23, 34, -25, -58, 53, -49, 60, 52, 41, -57, 34};
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(output_data[i], expect0[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  begin_tensor.set_data(nullptr);
  size_tensor.set_data(nullptr);
  delete kernel;
}

TEST_F(TestSliceInt8, SliceSingleThread) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 1, 1, 1, 1, 3, 2, 3});
  lite::Tensor out_tensor(kNumberTypeInt8, {1, 1, 1, 1, 1, 1, 2, 3});

  lite::Tensor begin_tensor(kNumberTypeInt32, {8});
  int begin_data[8] = {1, 0, 0, 0, 0, 1, 0, 0};
  begin_tensor.set_data(begin_data);
  lite::Tensor size_tensor(kNumberTypeInt32, {8});
  int size_data[8] = {1, 1, 1, 1, 1, 2, 2, 3};
  size_tensor.set_data(size_data);

  int8_t input_data[] = {105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68,
                         105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  int8_t output_data[12];
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);

  const lite::LiteQuantParam quant_in0 = {0.00784314f, 0};  // -1.0--1.0 -> 0--255
  const lite::LiteQuantParam quant_out = {0.00784314f, 0};
  in_tensor.AddQuantParam(quant_in0);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor, &begin_tensor, &size_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  SliceParameter *parameter = new (std::nothrow) SliceParameter;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_SliceFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 1;

  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect0[12] = {16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(output_data[i], expect0[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  begin_tensor.set_data(nullptr);
  size_tensor.set_data(nullptr);
  delete kernel;
}

TEST_F(TestSliceInt8, Slice4Thread) {
  lite::Tensor in_tensor(kNumberTypeInt8, {2, 1, 1, 1, 1, 3, 2, 3});
  lite::Tensor out_tensor(kNumberTypeInt8, {1, 1, 1, 1, 1, 1, 2, 3});

  lite::Tensor begin_tensor(kNumberTypeInt32, {8});
  int begin_data[8] = {1, 0, 0, 0, 0, 1, 0, 0};
  begin_tensor.set_data(begin_data);
  lite::Tensor size_tensor(kNumberTypeInt32, {8});
  int size_data[8] = {1, 1, 1, 1, 1, 2, 2, 3};
  size_tensor.set_data(size_data);

  int8_t input_data[] = {105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68,
                         105, 35, -27, 0, -63, 99, 16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  int8_t output_data[12];
  in_tensor.set_data(input_data);
  out_tensor.set_data(output_data);

  const lite::LiteQuantParam quant_in0 = {0.00784314f, 0};  // -1.0--1.0 -> 0--255
  const lite::LiteQuantParam quant_out = {0.00784314f, 0};
  in_tensor.AddQuantParam(quant_in0);
  out_tensor.AddQuantParam(quant_out);

  std::vector<lite::Tensor *> inputs = {&in_tensor, &begin_tensor, &size_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  SliceParameter *parameter = new (std::nothrow) SliceParameter;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_SliceFusion};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 4;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(parameter), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int8_t expect0[12] = {16, 45, 67, -49, -115, 106, -98, 119, 103, 81, -114, 68};
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(output_data[i], expect0[i]);
  }

  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
  begin_tensor.set_data(nullptr);
  size_tensor.set_data(nullptr);
  delete kernel;
}
}  // namespace mindspore
