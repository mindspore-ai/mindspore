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
#include <memory>
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/tensor.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "nnacl/fp32/reduce_fp32.h"

namespace mindspore {
using mindspore::lite::QuantArg;
using mindspore::lite::Tensor;
using mindspore::schema::ReduceMode;
using mindspore::schema::ReduceMode_ReduceMax;
using mindspore::schema::ReduceMode_ReduceMean;
using mindspore::schema::ReduceMode_ReduceMin;
using mindspore::schema::ReduceMode_ReduceProd;
using mindspore::schema::ReduceMode_ReduceSum;
using mindspore::schema::ReduceMode_ReduceSumSquare;

class TestReduceInt8 : public mindspore::CommonTest {
 public:
  TestReduceInt8() = default;
  void Prepare(const std::vector<int> &in_shape, const std::vector<int> &out_shape, int8_t *input_data,
               int8_t *output_data, ReduceMode mode, const int *axes, const int num_axes);
  void TearDown() override;

 public:
  int thread_num_ = 1;

  ReduceParameter param_ = {};
  Tensor in_tensor_;
  Tensor out_tensor_;
  std::vector<Tensor *> inputs{&in_tensor_};
  std::vector<Tensor *> outputs{&out_tensor_};
  kernel::KernelKey desc_ = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_ReduceFusion};
  kernel::KernelCreator creator_ = nullptr;
  lite::InnerContext ctx_ = lite::InnerContext();
  kernel::LiteKernel *kernel_ = nullptr;
  const QuantArg quant_in_ = {0.005f, 5};
  const QuantArg quant_out_ = {0.01f, 1};
  float err_tol_ = 0.05;
};

void TestReduceInt8::TearDown() {
  in_tensor_.set_data(nullptr);
  out_tensor_.set_data(nullptr);
}

void TestReduceInt8::Prepare(const std::vector<int> &in_shape, const std::vector<int> &out_shape, int8_t *input_data,
                             int8_t *output_data, ReduceMode mode, const int *axes, const int num_axes) {
  in_tensor_.set_data_type(kNumberTypeInt8);
  in_tensor_.set_shape(in_shape);
  in_tensor_.set_data(input_data);
  in_tensor_.AddQuantParam(quant_in_);

  out_tensor_.set_data_type(kNumberTypeInt8);
  out_tensor_.set_shape(out_shape);
  out_tensor_.set_data(output_data);
  out_tensor_.AddQuantParam(quant_out_);

  param_.mode_ = static_cast<int>(mode);
  param_.num_axes_ = num_axes;
  memcpy(param_.axes_, axes, num_axes * sizeof(int));

  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc_);

  ctx_.thread_num_ = thread_num_;
  ASSERT_EQ(lite::RET_OK, ctx_.Init());
  kernel_ = creator_(inputs, outputs, reinterpret_cast<OpParameter *>(&param_), &ctx_, desc_);
}

TEST_F(TestReduceInt8, Mean) {
  /* 2 4 4 3 NHWC */
  int8_t input_data[96] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                           20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                           40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                           60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  int8_t output_data[32] = {0};
  int axes[] = {3};
  int num_axes = 1;
  std::vector<int> input_shape = {2, 4, 4, 3};
  std::vector<int> output_shape = {2, 4, 4, 1};
  int output_size = 32;
  int8_t correct[] = {-1, 1,  2,  3,  5,  7,  8,  10, 11, 12, 14, 16, 17, 19, 20, 22,
                      23, 25, 26, 28, 29, 30, 32, 34, 35, 37, 38, 40, 41, 43, 44, 46};

  thread_num_ = 2;
  Prepare(input_shape, output_shape, input_data, output_data, ReduceMode_ReduceMean, axes, num_axes);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  err_tol_ = 0.09375;
  CompareOutputInt8(output_data, correct, output_size, err_tol_);
}

TEST_F(TestReduceInt8, MeanAllAxis) {
  /* 2*4*4*3 NHWC */
  int8_t input_data[96] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                           20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                           40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                           60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  int8_t output_data[1] = {0};
  int axes[] = {0};
  int num_axes = 0;
  std::vector<int> input_shape = {2, 4, 4, 3};
  std::vector<int> output_shape = {1};
  int output_size = 1;
  int8_t correct[] = {22};
  thread_num_ = 2;
  Prepare(input_shape, output_shape, input_data, output_data, ReduceMode_ReduceMean, axes, num_axes);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  err_tol_ = 1.0f;
  CompareOutputInt8(output_data, correct, output_size, err_tol_);
}

TEST_F(TestReduceInt8, Sum) {
  /* 2*4*4*3 NHWC */
  int8_t input_data[96] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                           20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                           40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                           60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  int8_t output_data[32] = {0};
  int axes[] = {-1};
  int num_axes = 1;
  std::vector<int> input_shape = {2, 4, 4, 3};
  std::vector<int> output_shape = {2, 4, 4, 1};
  int output_size = 32;
  int8_t correct[] = {-5, -1, 4,  9,  13, 18, 22, 27, 31,  36,  40,  45,  49,  54,  58,  63,
                      67, 72, 76, 81, 85, 90, 94, 99, 103, 107, 112, 117, 121, 126, 127, 127};
  thread_num_ = 2;
  Prepare(input_shape, output_shape, input_data, output_data, ReduceMode_ReduceSum, axes, num_axes);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  err_tol_ = 0.0625f;
  CompareOutputInt8(output_data, correct, output_size, err_tol_);
}

TEST_F(TestReduceInt8, SumAllAxis) {
  /* 2*4*4*3 NHWC */
  int8_t input_data[96] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  };
  int8_t output_data[1] = {0};
  int axes[] = {0, 1, 2, 3};
  int num_axes = 4;
  std::vector<int> input_shape = {2, 4, 4, 3};
  std::vector<int> output_shape = {1};
  int output_size = 1;
  int8_t correct[] = {-47};
  thread_num_ = 2;
  Prepare(input_shape, output_shape, input_data, output_data, ReduceMode_ReduceSum, axes, num_axes);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputInt8(output_data, correct, output_size, err_tol_);
}

TEST_F(TestReduceInt8, Max) {
  /* 2*4*4*3 NHWC */
  int8_t input_data[96] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                           20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                           40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                           60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  int8_t output_data[32] = {0};
  int axes[] = {3};
  int num_axes = 1;
  std::vector<int> input_shape = {2, 4, 4, 3};
  std::vector<int> output_shape = {2, 4, 4, 1};
  int output_size = 32;
  int8_t correct[] = {-1, 1,  3,  4,  6,  7,  9,  10, 12, 13, 15, 16, 18, 19, 21, 22,
                      24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 46};
  thread_num_ = 2;
  Prepare(input_shape, output_shape, input_data, output_data, ReduceMode_ReduceMax, axes, num_axes);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputInt8(output_data, correct, output_size, err_tol_);
}

TEST_F(TestReduceInt8, MaxAll) {
  /* 2*4*4*3 NHWC */
  int8_t input_data[96] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                           20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                           40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                           60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  int8_t output_data[1] = {0};
  int axes[] = {0, 1, 2, 3};
  int num_axes = 4;
  std::vector<int> input_shape = {2, 4, 4, 3};
  std::vector<int> output_shape = {1};
  int output_size = 1;
  int8_t correct[] = {46};
  thread_num_ = 2;
  Prepare(input_shape, output_shape, input_data, output_data, ReduceMode_ReduceMax, axes, num_axes);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputInt8(output_data, correct, output_size, err_tol_);
}

TEST_F(TestReduceInt8, Min) {
  /* 2*4*4*3 NHWC */
  int8_t input_data[96] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                           20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                           40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                           60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  int8_t output_data[32] = {0};
  int axes[] = {3};
  int num_axes = 1;
  std::vector<int> input_shape = {2, 4, 4, 3};
  std::vector<int> output_shape = {2, 4, 4, 1};
  int output_size = 32;
  int8_t correct[] = {-2, 0,  2,  3,  5,  6,  8,  9,  11, 12, 14, 15, 17, 18, 20, 21,
                      23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45};
  thread_num_ = 2;
  Prepare(input_shape, output_shape, input_data, output_data, ReduceMode_ReduceMin, axes, num_axes);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputInt8(output_data, correct, output_size, err_tol_);
}

TEST_F(TestReduceInt8, MinAll) {
  /* 2*4*4*3 NHWC */
  int8_t input_data[96] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                           20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                           40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                           60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  int8_t output_data[1] = {0};
  int axes[] = {0};
  int num_axes = 0;
  std::vector<int> input_shape = {2, 4, 4, 3};
  std::vector<int> output_shape = {1};
  int output_size = 1;
  int8_t correct[] = {-2};
  thread_num_ = 2;
  Prepare(input_shape, output_shape, input_data, output_data, ReduceMode_ReduceMin, axes, num_axes);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputInt8(output_data, correct, output_size, err_tol_);
}

TEST_F(TestReduceInt8, Prod) {
  /* 2*4*4*3 NHWC */
  int8_t input_data[96] = {105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105,
                           105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105,
                           105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105,
                           105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105,
                           105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105,
                           105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105};
  int8_t output_data[32] = {0};
  int axes[] = {3};
  int num_axes = 1;
  std::vector<int> input_shape = {2, 4, 4, 3};
  std::vector<int> output_shape = {2, 4, 4, 1};
  int output_size = 32;
  int8_t correct[] = {
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
  };
  thread_num_ = 2;
  Prepare(input_shape, output_shape, input_data, output_data, ReduceMode_ReduceProd, axes, num_axes);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputInt8(output_data, correct, output_size, err_tol_);
}

TEST_F(TestReduceInt8, Prod2Axis) {
  /* 2*4*4*3 NHWC */
  int8_t input_data[12] = {105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105};
  int8_t output_data[8] = {0};
  int axes[] = {2, 3};
  int num_axes = 2;
  std::vector<int> input_shape = {1, 2, 2, 3};
  std::vector<int> output_shape = {1, 2};
  int output_size = 2;
  int8_t correct[] = {3, 3};
  thread_num_ = 1;
  Prepare(input_shape, output_shape, input_data, output_data, ReduceMode_ReduceProd, axes, num_axes);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputInt8(output_data, correct, output_size, err_tol_);
}

TEST_F(TestReduceInt8, SumSquare) {
  /* 2*4*4*3 NHWC */

  int8_t input_data[96] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                           20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                           40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                           60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  int8_t output_data[32] = {0};
  int axes[] = {3};
  int num_axes = 1;
  std::vector<int> input_shape = {2, 4, 4, 3};
  std::vector<int> output_shape = {2, 4, 4, 1};
  int output_size = 32;
  int8_t correct[] = {1,  1,  1,  1,  1,  2,  2,  3,  4,  5,  6,  7,  9,  10, 12, 14,
                      16, 18, 20, 22, 25, 27, 30, 33, 36, 39, 42, 45, 49, 53, 56, 60};
  thread_num_ = 1;
  Prepare(input_shape, output_shape, input_data, output_data, ReduceMode_ReduceSumSquare, axes, num_axes);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputInt8(output_data, correct, output_size, err_tol_);
}

TEST_F(TestReduceInt8, SumSquare2Axis) {
  /* 2*4*4*3 NHWC */
  int8_t input_data[12] = {105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105};
  int8_t output_data[8] = {0};
  int axes[] = {3, 2};
  int num_axes = 2;
  std::vector<int> input_shape = {1, 2, 2, 3};
  std::vector<int> output_shape = {1, 2};
  int output_size = 2;
  int8_t correct[] = {114, 114};
  thread_num_ = 1;
  Prepare(input_shape, output_shape, input_data, output_data, ReduceMode_ReduceSumSquare, axes, num_axes);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputInt8(output_data, correct, output_size, err_tol_);
}

}  // namespace mindspore
