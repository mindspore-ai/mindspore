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
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"
#include "mindspore/lite/src/tensor.h"
#include "nnacl/resize_parameter.h"
#include "schema/ops_generated.h"
using mindspore::schema::Format_NHWC;

namespace mindspore {

class TestResizeBilinearFp32 : public mindspore::CommonTest {
 public:
  TestResizeBilinearFp32() = default;
  void Prepare(const std::vector<int> &input_shape, const std::vector<int> &output_shape, float *input_data,
               float *output_data, const bool align_corners, const int thread_num);

  void TearDown() override;

 public:
  float err_tol = 1e-5;
  lite::Tensor in_tensor_;
  lite::Tensor out_tensor_;
  std::vector<lite::Tensor *> inputs_{&in_tensor_};
  std::vector<lite::Tensor *> outputs_{&out_tensor_};
  ResizeParameter param_ = {{}};
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Resize};
  lite::InnerContext ctx_ = lite::InnerContext();
  kernel::KernelCreator creator_ = nullptr;
  kernel::LiteKernel *kernel_ = nullptr;
};

void TestResizeBilinearFp32::TearDown() {
  in_tensor_.set_data(nullptr);
  out_tensor_.set_data(nullptr);
}

void TestResizeBilinearFp32::Prepare(const std::vector<int> &input_shape, const std::vector<int> &output_shape,
                                     float *input_data, float *output_data, const bool align_corners,
                                     const int thread_num) {
  in_tensor_.set_data_type(kNumberTypeFloat32);
  in_tensor_.set_format(Format_NHWC);
  in_tensor_.set_shape(input_shape);
  out_tensor_.set_data_type(kNumberTypeFloat32);
  out_tensor_.set_shape(output_shape);
  in_tensor_.set_data(input_data);
  out_tensor_.set_data(output_data);

  ResizeParameter param_ = {
    {}, static_cast<int>(schema::ResizeMethod_LINEAR), output_shape[1], output_shape[2], align_corners};
  desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Resize};
  ctx_ = lite::InnerContext();
  ctx_.thread_num_ = thread_num;
  ASSERT_EQ(lite::RET_OK, ctx_.Init());
  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator_, nullptr);
  kernel_ = creator_(inputs_, outputs_, reinterpret_cast<OpParameter *>(&param_), &ctx_, desc);
  ASSERT_NE(kernel_, nullptr);
}

// 1*1 -> 1*1
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest1) {
  float input_data[] = {1.0f};
  float output_data[1] = {0};
  std::vector<int> input_shape = {1, 1, 1, 1};
  std::vector<int> output_shape = {1, 1, 1, 1};
  std::vector<float> expect = {1.0};
  bool align_corners = false;
  auto output_size = 1;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2 -> 1*1
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest2) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[1] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 1, 1, 1};
  std::vector<float> expect = {0.0};
  bool align_corners = false;
  int output_size = 1;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2 -> 1*2
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest3) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[2] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 1, 2, 1};
  std::vector<float> expect = {0.0, 1.0};
  bool align_corners = false;
  auto output_size = 2;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2 -> 2*1
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest4) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[2] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 2, 1, 1};
  std::vector<float> expect = {0.0, 2.0};
  bool align_corners = false;
  auto output_size = 2;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2 -> 2*2
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest5) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[4] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 2, 2, 1};
  std::vector<float> expect = {0.0, 1.0, 2.0, 3.0};
  bool align_corners = false;
  auto output_size = 4;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2 -> 1*4
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest6) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[4] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 1, 4, 1};
  std::vector<float> expect = {0.0, 0.5, 1.0, 1.0};
  bool align_corners = false;
  auto output_size = 4;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2 -> 4*1
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest7) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[4] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 4, 1, 1};
  std::vector<float> expect = {0.0, 1.0, 2.0, 2.0};
  bool align_corners = false;
  auto output_size = 4;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2 -> 2*4
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest8) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[8] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 2, 4, 1};
  std::vector<float> expect = {0.0, 0.5, 1.0, 1.0, 2.0, 2.5, 3.0, 3.0};
  bool align_corners = false;
  auto output_size = 8;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2 -> 4*2
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest9) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[8] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 4, 2, 1};
  std::vector<float> expect = {0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 3.0};
  bool align_corners = false;
  auto output_size = 8;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2 -> 3*3
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest10) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[9] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 3, 3, 1};
  std::vector<float> expect = {0.0, 0.6666667, 1.0, 1.3333334, 2.0, 2.3333335, 2.0, 2.6666667, 3.0};
  bool align_corners = false;

  auto output_size = 9;
  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2 -> 4*4
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest11) {
  float input_data[] = {0.0, 1.0, 2.0, 3.0};
  float output_data[16] = {0};
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 4, 4, 1};
  std::vector<float> expect = {0.0, 0.5, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 2.0, 2.5, 3.0, 3.0};
  bool align_corners = false;

  auto output_size = 16;
  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2*2*5 -> 2*4*4*5
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest12) {
  float input_data[] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
                        28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  float output_data[160] = {0};
  std::vector<int> input_shape = {2, 2, 2, 5};
  std::vector<int> output_shape = {2, 4, 4, 5};
  std::vector<float> expect = {
    0.0,  1.0,  2.0,  3.0,  4.0,  2.5,  3.5,  4.5,  5.5,  6.5,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,  6.0,  7.0,
    8.0,  9.0,  5.0,  6.0,  7.0,  8.0,  9.0,  7.5,  8.5,  9.5,  10.5, 11.5, 10.0, 11.0, 12.0, 13.0, 14.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 12.5, 13.5, 14.5, 15.5, 16.5, 15.0, 16.0, 17.0, 18.0,
    19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 10.0, 11.0, 12.0, 13.0, 14.0, 12.5, 13.5, 14.5, 15.5, 16.5, 15.0, 16.0,
    17.0, 18.0, 19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 22.5, 23.5, 24.5, 25.5, 26.5,
    25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 27.5, 28.5, 29.5,
    30.5, 31.5, 30.0, 31.0, 32.0, 33.0, 34.0, 30.0, 31.0, 32.0, 33.0, 34.0, 30.0, 31.0, 32.0, 33.0, 34.0, 32.5,
    33.5, 34.5, 35.5, 36.5, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0, 30.0, 31.0, 32.0, 33.0,
    34.0, 32.5, 33.5, 34.5, 35.5, 36.5, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  bool align_corners = false;
  auto output_size = 160;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2*2*5 -> 2*4*4*5 align corners
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest13) {
  float input_data[] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
                        28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  float output_data[160] = {0};
  std::vector<int> input_shape = {2, 2, 2, 5};
  std::vector<int> output_shape = {2, 4, 4, 5};
  std::vector<float> expect = {
    0.0,       1.0,       2.0,       3.0,       4.0,       1.6666667, 2.6666667, 3.6666667, 4.666667,  5.666667,
    3.3333335, 4.3333335, 5.3333335, 6.3333335, 7.3333335, 5.0,       6.0,       7.0,       8.0,       9.0,
    3.3333335, 4.3333335, 5.3333335, 6.3333335, 7.3333335, 5.0,       6.0,       7.0,       8.0,       9.0,
    6.666667,  7.666667,  8.666667,  9.666667,  10.666667, 8.333334,  9.333334,  10.333334, 11.333334, 12.333334,
    6.666667,  7.666667,  8.666667,  9.666667,  10.666667, 8.333334,  9.333334,  10.333334, 11.333334, 12.333334,
    10.0,      11.0,      12.0,      13.0,      14.0,      11.666667, 12.666667, 13.666667, 14.666667, 15.666667,
    10.0,      11.0,      12.0,      13.0,      14.0,      11.666667, 12.666667, 13.666667, 14.666667, 15.666667,
    13.333334, 14.333334, 15.333334, 16.333334, 17.333334, 15.0,      16.0,      17.0,      18.0,      19.0,
    20.0,      21.0,      22.0,      23.0,      24.0,      21.666666, 22.666666, 23.666666, 24.666666, 25.666666,
    23.333334, 24.333334, 25.333334, 26.333334, 27.333334, 25.0,      26.0,      27.0,      28.0,      29.0,
    23.333334, 24.333334, 25.333334, 26.333334, 27.333334, 25.0,      26.0,      27.0,      28.0,      29.0,
    26.666666, 27.666666, 28.666666, 29.666666, 30.666666, 28.333334, 29.333334, 30.333334, 31.333334, 32.333332,
    26.666668, 27.666668, 28.666668, 29.666668, 30.666668, 28.333332, 29.333334, 30.333334, 31.333334, 32.333336,
    30.0,      31.0,      32.0,      33.0,      34.0,      31.666668, 32.666668, 33.666668, 34.666668, 35.666668,
    30.0,      31.0,      32.0,      33.0,      34.0,      31.666666, 32.666668, 33.666668, 34.666668, 35.666668,
    33.333332, 34.333332, 35.333332, 36.333332, 37.333332, 35.0,      36.0,      37.0,      38.0,      39.0};
  bool align_corners = true;
  auto output_size = 160;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 1);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2*2*5 -> 2*4*4*5 thread_num 2
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest14) {
  float input_data[] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
                        28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  float output_data[160] = {0};
  std::vector<int> input_shape = {2, 2, 2, 5};
  std::vector<int> output_shape = {2, 4, 4, 5};
  std::vector<float> expect = {
    0.0,  1.0,  2.0,  3.0,  4.0,  2.5,  3.5,  4.5,  5.5,  6.5,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,  6.0,  7.0,
    8.0,  9.0,  5.0,  6.0,  7.0,  8.0,  9.0,  7.5,  8.5,  9.5,  10.5, 11.5, 10.0, 11.0, 12.0, 13.0, 14.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 12.5, 13.5, 14.5, 15.5, 16.5, 15.0, 16.0, 17.0, 18.0,
    19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 10.0, 11.0, 12.0, 13.0, 14.0, 12.5, 13.5, 14.5, 15.5, 16.5, 15.0, 16.0,
    17.0, 18.0, 19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 22.5, 23.5, 24.5, 25.5, 26.5,
    25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 27.5, 28.5, 29.5,
    30.5, 31.5, 30.0, 31.0, 32.0, 33.0, 34.0, 30.0, 31.0, 32.0, 33.0, 34.0, 30.0, 31.0, 32.0, 33.0, 34.0, 32.5,
    33.5, 34.5, 35.5, 36.5, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0, 30.0, 31.0, 32.0, 33.0,
    34.0, 32.5, 33.5, 34.5, 35.5, 36.5, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  bool align_corners = false;
  auto output_size = 160;
  int thread_num = 2;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, thread_num);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 2*2*2*5 -> 2*4*4*5 thread_num 4
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest15) {
  float input_data[] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
                        28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  float output_data[160] = {0};
  std::vector<int> input_shape = {2, 2, 2, 5};
  std::vector<int> output_shape = {2, 4, 4, 5};
  std::vector<float> expect = {
    0.0,  1.0,  2.0,  3.0,  4.0,  2.5,  3.5,  4.5,  5.5,  6.5,  5.0,  6.0,  7.0,  8.0,  9.0,  5.0,  6.0,  7.0,
    8.0,  9.0,  5.0,  6.0,  7.0,  8.0,  9.0,  7.5,  8.5,  9.5,  10.5, 11.5, 10.0, 11.0, 12.0, 13.0, 14.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0, 12.5, 13.5, 14.5, 15.5, 16.5, 15.0, 16.0, 17.0, 18.0,
    19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 10.0, 11.0, 12.0, 13.0, 14.0, 12.5, 13.5, 14.5, 15.5, 16.5, 15.0, 16.0,
    17.0, 18.0, 19.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 22.5, 23.5, 24.5, 25.5, 26.5,
    25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 25.0, 26.0, 27.0, 28.0, 29.0, 27.5, 28.5, 29.5,
    30.5, 31.5, 30.0, 31.0, 32.0, 33.0, 34.0, 30.0, 31.0, 32.0, 33.0, 34.0, 30.0, 31.0, 32.0, 33.0, 34.0, 32.5,
    33.5, 34.5, 35.5, 36.5, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0, 30.0, 31.0, 32.0, 33.0,
    34.0, 32.5, 33.5, 34.5, 35.5, 36.5, 35.0, 36.0, 37.0, 38.0, 39.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  bool align_corners = false;

  auto output_size = 160;
  std::vector<float> output(output_size, 0.0);
  int thread_num = 4;
  Prepare(input_shape, output_shape, input_data, output_data, align_corners, thread_num);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}

// 5*5 -> 2*2
TEST_F(TestResizeBilinearFp32, ResizeBilinearTest16) {
  float input_data[] = {
    0.0,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,   10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
    16.0,  17.0,  18.0,  19.0,  20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,  30.0,  31.0,
    32.0,  33.0,  34.0,  35.0,  36.0,  37.0,  38.0,  39.0,  40.0,  41.0,  42.0,  43.0,  44.0,  45.0,  46.0,  47.0,
    48.0,  49.0,  50.0,  51.0,  52.0,  53.0,  54.0,  55.0,  56.0,  57.0,  58.0,  59.0,  60.0,  61.0,  62.0,  63.0,
    64.0,  65.0,  66.0,  67.0,  68.0,  69.0,  70.0,  71.0,  72.0,  73.0,  74.0,  75.0,  76.0,  77.0,  78.0,  79.0,
    80.0,  81.0,  82.0,  83.0,  84.0,  85.0,  86.0,  87.0,  88.0,  89.0,  90.0,  91.0,  92.0,  93.0,  94.0,  95.0,
    96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
    112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0};
  float output_data[20] = {0};
  std::vector<int> input_shape = {1, 5, 5, 5};
  std::vector<int> output_shape = {1, 2, 2, 5};
  std::vector<float> expect = {0.0,  1.0,  2.0,  3.0,  4.0,  12.5, 13.5, 14.5, 15.5, 16.5,
                               62.5, 63.5, 64.5, 65.5, 66.5, 75.0, 76.0, 77.0, 78.0, 79.0};
  bool align_corners = false;
  auto output_size = 20;

  Prepare(input_shape, output_shape, input_data, output_data, align_corners, 2);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol));
}
}  // namespace mindspore
