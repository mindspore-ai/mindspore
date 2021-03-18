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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/nnacl/fp32/reduce_fp32.h"
#include "schema/inner/model_generated.h"
#include "src/tensor.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/runtime/allocator.h"

using mindspore::lite::Tensor;
using mindspore::schema::ReduceMode;
using mindspore::schema::ReduceMode_ReduceASum;
using mindspore::schema::ReduceMode_ReduceMax;
using mindspore::schema::ReduceMode_ReduceMean;
using mindspore::schema::ReduceMode_ReduceMin;
using mindspore::schema::ReduceMode_ReduceProd;
using mindspore::schema::ReduceMode_ReduceSum;
using mindspore::schema::ReduceMode_ReduceSumSquare;

namespace mindspore {

class TestReduceFp32 : public mindspore::CommonTest {
 public:
  TestReduceFp32() = default;

  void Prepare(const std::vector<int> &in_shape, const std::vector<int> &out_shape, float *input_data,
               float *output_data, ReduceMode mode, const int *axes, const int num_axes, bool reduce_to_end,
               float coeff);
  void TearDown() override;

 public:
  int tid_ = 0;
  int thread_num_ = 1;
  float err_tol = 1e-5;
  ReduceParameter param_ = {};
  Tensor in_tensor_;
  Tensor out_tensor_;
  std::vector<Tensor *> inputs{&in_tensor_};
  std::vector<Tensor *> outputs{&out_tensor_};
  kernel::KernelKey desc_ = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_ReduceFusion};
  kernel::KernelCreator creator_ = nullptr;
  lite::InnerContext *ctx_ = nullptr;
  kernel::LiteKernel *kernel_ = nullptr;
};

void TestReduceFp32::TearDown() {
  delete ctx_;
  in_tensor_.set_data(nullptr);
  out_tensor_.set_data(nullptr);
}

void TestReduceFp32::Prepare(const std::vector<int> &in_shape, const std::vector<int> &out_shape, float *input_data,
                             float *output_data, ReduceMode mode, const int *axes, const int num_axes,
                             bool reduce_to_end, float coeff) {
  in_tensor_.set_data_type(kNumberTypeFloat32);
  in_tensor_.set_shape(in_shape);
  in_tensor_.set_data(input_data);

  out_tensor_.set_data_type(kNumberTypeFloat32);
  out_tensor_.set_shape(out_shape);
  out_tensor_.set_data(output_data);

  param_.mode_ = static_cast<int>(mode);
  param_.num_axes_ = num_axes;
  memcpy(param_.axes_, axes, num_axes * sizeof(int));
  param_.reduce_to_end_ = reduce_to_end;
  param_.coeff = coeff;

  ctx_ = new (std::nothrow) lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx_->Init());
  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc_);
  if (ctx_->allocator == nullptr) {
    ctx_->allocator = Allocator::Create();
  }
  ctx_->thread_num_ = thread_num_;
  kernel_ = creator_(inputs, outputs, reinterpret_cast<OpParameter *>(&param_), ctx_, desc_);
}

TEST_F(TestReduceFp32, Mean1) {
  /* 2 4 4 3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                       66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{2, 1, 4, 3};
  int axes[1] = {1};
  int axis_num = 1;
  float out[24] = {0};
  bool reduce_to_end = false;
  float coeff = 1.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceMean, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 24;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

// thread num 2 reduce_to_end
TEST_F(TestReduceFp32, Mean2) {
  /* 2 4 4 3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[2] = {47.0, 143.0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{2, 1, 1, 1};
  int axes[1] = {1};
  int axis_num = 1;
  float out[24] = {0};
  bool reduce_to_end = true;
  float coeff = 2.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceMean, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 2;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

// thread num 1
TEST_F(TestReduceFp32, Mean3) {
  /* 2 4 4 3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                       66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{2, 1, 4, 3};
  int axes[1] = {1};
  int axis_num = 1;
  float out[24] = {0};
  bool reduce_to_end = false;
  float coeff = 2.0f;
  thread_num_ = 1;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceMean, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 2;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

TEST_F(TestReduceFp32, MeanAllAxis) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[1] = {47.5};
  float out[1] = {0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{1, 1, 1, 1};
  int axes[4] = {0, 1, 2, 3};
  int axis_num = 4;
  bool reduce_to_end = false;
  float coeff = 0.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceMean, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 1;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

TEST_F(TestReduceFp32, Sum) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {72.0,  76.0,  80.0,  84.0,  88.0,  92.0,  96.0,  100.0, 104.0, 108.0, 112.0, 116.0,
                       264.0, 268.0, 272.0, 276.0, 280.0, 284.0, 288.0, 292.0, 296.0, 300.0, 304.0, 308.0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{2, 1, 4, 3};
  int axes[1] = {1};
  int axis_num = 1;
  float out[24] = {0};
  bool reduce_to_end = false;
  float coeff = 1.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceSum, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 24;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

// sum reduce_to_end
TEST_F(TestReduceFp32, Sum2) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[32] = {6.0,   24.0,  42.0,  60.0,  78.0,  96.0,  114.0, 132.0, 150.0, 168.0, 186.0,
                       204.0, 222.0, 240.0, 258.0, 276.0, 294.0, 312.0, 330.0, 348.0, 366.0, 384.0,
                       402.0, 420.0, 438.0, 456.0, 474.0, 492.0, 510.0, 528.0, 546.0, 564.0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{2, 4, 4, 1};
  int axes[1] = {-1};
  int axis_num = 1;
  float out[32] = {0};
  bool reduce_to_end = true;
  float coeff = 2.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceSum, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 32;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

TEST_F(TestReduceFp32, Sum3) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[32] = {3.0,   12.0,  21.0,  30.0,  39.0,  48.0,  57.0,  66.0,  75.0,  84.0,  93.0,
                       102.0, 111.0, 120.0, 129.0, 138.0, 147.0, 156.0, 165.0, 174.0, 183.0, 192.0,
                       201.0, 210.0, 219.0, 228.0, 237.0, 246.0, 255.0, 264.0, 273.0, 282.0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{2, 4, 4, 1};
  int axes[1] = {-1};
  int axis_num = 1;
  float out[32] = {0};
  bool reduce_to_end = false;
  float coeff = 0.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceSum, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 32;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

TEST_F(TestReduceFp32, SumAllAxis) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[1] = {4560};
  float out[1] = {0};
  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{1, 1, 1, 1};
  int axes[4] = {0};
  int axis_num = 4;
  bool reduce_to_end = true;
  float coeff = 1.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceSum, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 1;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

TEST_F(TestReduceFp32, Max) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                       84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{2, 1, 4, 3};
  int axes[1] = {1};
  int axis_num = 1;
  float out[24] = {0};
  bool reduce_to_end = false;
  float coeff = 1.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceMax, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 24;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

TEST_F(TestReduceFp32, Min) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0,
                       48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{2, 1, 4, 3};
  int axes[1] = {1};
  int axis_num = 1;
  float out[24] = {0};
  bool reduce_to_end = false;
  float coeff = 1.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceMin, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 24;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

TEST_F(TestReduceFp32, Prod) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {0.0,        12025.0,    27664.0,    47385.0,    71680.0,    101065.0,   136080.0,   177289.0,
                       225280.0,   280665.0,   344080.0,   416185.0,   17418240.0, 18546744.0, 19728400.0, 20964824.0,
                       22257664.0, 23608584.0, 25019280.0, 26491464.0, 28026880.0, 29627288.0, 31294480.0, 33030264.0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{2, 1, 4, 3};
  int axes[1] = {1};
  int axis_num = 1;
  float out[24] = {0};
  bool reduce_to_end = false;
  float coeff = 1.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceProd, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 24;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

TEST_F(TestReduceFp32, SumSquare) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[8] = {1012.0, 7636.0, 21172.0, 41620.0, 68980.0, 103252.0, 144436.0, 192532.0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{2, 4, 1, 1};
  int axes[1] = {2};
  int axis_num = 1;
  float out[8] = {0};
  bool reduce_to_end = true;
  float coeff = 2.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceSumSquare, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 8;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

TEST_F(TestReduceFp32, SumSquare2) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[32] = {10.0,    100.0,   298.0,   604.0,   1018.0,  1540.0,  2170.0,  2908.0,
                       3754.0,  4708.0,  5770.0,  6940.0,  8218.0,  9604.0,  11098.0, 12700.0,
                       14410.0, 16228.0, 18154.0, 20188.0, 22330.0, 24580.0, 26938.0, 29404.0,
                       31978.0, 34660.0, 37450.0, 40348.0, 43354.0, 46468.0, 49690.0, 53020.0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{2, 4, 4, 1};
  int axes[1] = {3};
  int axis_num = 1;
  float out[32] = {0};
  bool reduce_to_end = true;
  float coeff = 2.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceSumSquare, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 32;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}

TEST_F(TestReduceFp32, ASum) {
  /* 2*4*4*3 NHWC */
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[32] = {3.0,   12.0,  21.0,  30.0,  39.0,  48.0,  57.0,  66.0,  75.0,  84.0,  93.0,
                       102.0, 111.0, 120.0, 129.0, 138.0, 147.0, 156.0, 165.0, 174.0, 183.0, 192.0,
                       201.0, 210.0, 219.0, 228.0, 237.0, 246.0, 255.0, 264.0, 273.0, 282.0};

  std::vector<int> in_shape{2, 4, 4, 3};
  std::vector<int> out_shape{2, 4, 4, 1};
  int axes[1] = {3};
  int axis_num = 1;
  float out[32] = {0};
  bool reduce_to_end = true;
  float coeff = 1.0f;
  thread_num_ = 2;

  Prepare(in_shape, out_shape, in, out, ReduceMode_ReduceASum, axes, axis_num, reduce_to_end, coeff);
  kernel_->Run();

  int output_size = 32;
  ASSERT_EQ(0, CompareOutputData(out, correct, output_size, err_tol));
}
}  // namespace mindspore
