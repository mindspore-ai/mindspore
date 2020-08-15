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
#include "common/common_test.h"
#include "nnacl/resize_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {

class TestResizeNearestNeighborFp32 : public mindspore::CommonTest {
 public:
  TestResizeNearestNeighborFp32() = default;
  void Prepare(const std::vector<int> &input_shape, const std::vector<int> &output_shape, float *input_data,
               float *output_data, const bool align_corners, const int thread_num);

  void TearDown() override;

 public:
  float err_tol = 1e-5;
  lite::tensor::Tensor in_tensor_;
  lite::tensor::Tensor out_tensor_;
  std::vector<lite::tensor::Tensor *> inputs_{&in_tensor_};
  std::vector<lite::tensor::Tensor *> outputs_{&out_tensor_};
  ResizeParameter param_ = {{}};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Resize};
  lite::Context ctx_ = lite::Context();
  kernel::KernelCreator creator_ = nullptr;
  kernel::LiteKernel *kernel_ = nullptr;
};

void TestResizeNearestNeighborFp32::TearDown() {
  in_tensor_.SetData(nullptr);
  out_tensor_.SetData(nullptr);
}

void TestResizeNearestNeighborFp32::Prepare(const std::vector<int> &input_shape, const std::vector<int> &output_shape,
                                            float *input_data, float *output_data, const bool align_corners,
                                            const int thread_num) {
  in_tensor_.set_data_type(kNumberTypeFloat32);
  in_tensor_.set_shape(input_shape);
  out_tensor_.set_data_type(kNumberTypeFloat32);
  out_tensor_.set_shape(output_shape);
  in_tensor_.SetData(input_data);
  out_tensor_.SetData(output_data);

  ResizeParameter param_ = {
    {}, static_cast<int>(schema::ResizeMethod_NEAREST_NEIGHBOR), output_shape[1], output_shape[2], align_corners};
  desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Resize};
  ctx_ = lite::Context();
  ctx_.thread_num_ = thread_num;
  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator_, nullptr);
  kernel_ = creator_(inputs_, outputs_, reinterpret_cast<OpParameter *>(&param_), &ctx_, desc, nullptr);
  ASSERT_NE(kernel_, nullptr);
}
// 1*1 -> 1*1
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest1) {
  float input_data[] = {1.0};
  float output_data[1] = {0};
  std::vector<int> input_shape = {1, 1, 1, 1};
  std::vector<int> output_shape = {1, 1, 1, 1};
  std::vector<float> expect = {1.0};
  size_t output_size = 1;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2 -> 1*1
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest2) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[1] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 1, 1, 1};
  std::vector<float> expect = {0.0};
  size_t output_size = 1;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2 -> 1*2
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest3) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[2] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 1, 2, 1};
  std::vector<float> expect = {0.0, 1.0};
  size_t output_size = 2;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2 -> 2*1
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest4) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[2] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 2, 1, 1};
  std::vector<float> expect = {0.0, 2.0};
  size_t output_size = 2;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2 -> 2*2
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest5) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[4] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 2, 2, 1};
  std::vector<float> expect = {0.0, 1.0, 2.0, 3.0};
  size_t output_size = 4;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2 -> 1*4
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest6) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[4] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 1, 4, 1};
  std::vector<float> expect = {0.0, 0.0, 1.0, 1.0};
  size_t output_size = 4;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2 -> 4*1
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest7) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[4] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 4, 1, 1};
  std::vector<float> expect = {0.0, 0.0, 2.0, 2.0};
  size_t output_size = 4;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2 -> 2*4
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest8) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[8] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 2, 4, 1};
  std::vector<float> expect = {0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0};
  size_t output_size = 8;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2 -> 4*2
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest9) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[8] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 4, 2, 1};
  std::vector<float> expect = {0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0};
  size_t output_size = 8;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2 -> 3*3
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest10) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[9] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 3, 3, 1};
  std::vector<float> expect = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0};
  size_t output_size = 9;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2 -> 4*4
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest11) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[16] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 4, 4, 1};
  std::vector<float> expect = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0};
  size_t output_size = 16;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2*2*5 -> 2*4*4*5
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest12) {
  float input_data[] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
                        28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  float output_data[160] = {0};
  std::vector<int> input_shape = {2, 2, 2, 5};
  std::vector<int> output_shape = {2, 4, 4, 5};
  std::vector<float> expect = {
    0.0,  1.0,  2.0,  3.0,  4.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,  6.0,  7.0,
    8.0,  9.0,  0.0,  1.0,  2.0,  3.0,  4.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,
    6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 20.0, 21.0, 22.0, 23.0, 24.0, 20.0, 21.0, 22.0,
    23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 30.0,
    31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0, 30.0, 31.0, 32.0, 33.0,
    34.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  size_t output_size = 160;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2*2*5 -> 2*4*4*5 thread_num 2
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest13) {
  float input_data[] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
                        28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  float output_data[160] = {0};
  std::vector<int> input_shape = {2, 2, 2, 5};
  std::vector<int> output_shape = {2, 4, 4, 5};
  std::vector<float> expect = {
    0.0,  1.0,  2.0,  3.0,  4.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,  6.0,  7.0,
    8.0,  9.0,  0.0,  1.0,  2.0,  3.0,  4.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,
    6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 20.0, 21.0, 22.0, 23.0, 24.0, 20.0, 21.0, 22.0,
    23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 30.0,
    31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0, 30.0, 31.0, 32.0, 33.0,
    34.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  size_t output_size = 160;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 2);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}

// 2*2*2*5 -> 2*4*4*5 thread_num 4
TEST_F(TestResizeNearestNeighborFp32, ResizeNearestNeighborTest14) {
  float input_data[] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
                        28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  float output_data[160] = {0};
  std::vector<int> input_shape = {2, 2, 2, 5};
  std::vector<int> output_shape = {2, 4, 4, 5};
  std::vector<float> expect = {
    0.0,  1.0,  2.0,  3.0,  4.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,  6.0,  7.0,
    8.0,  9.0,  0.0,  1.0,  2.0,  3.0,  4.0,  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,
    6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 20.0, 21.0, 22.0, 23.0, 24.0, 20.0, 21.0, 22.0,
    23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 30.0,
    31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0, 30.0, 31.0, 32.0, 33.0,
    34.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  size_t output_size = 160;
  bool align_corners = false;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 4);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  CompareOutputData(output_data, expect.data(), output_size, err_tol);
}
}  // namespace mindspore
