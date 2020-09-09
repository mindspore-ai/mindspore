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
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/common/utils.h"
#include "nnacl/strided_slice.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {
class TestStridedSliceFp32 : public mindspore::CommonTest {
 public:
  TestStridedSliceFp32() {}
};

void InitStridedSliceParam(StridedSliceParameter *param, const std::vector<int> &in_shape,
                           const std::vector<int> &begins, const std::vector<int> &ends,
                           const std::vector<int> &strides, const int begins_length, const int in_shape_length) {
  for (auto i = 0; i < begins_length; i++) {
    param->in_shape_[i] = in_shape[i];
    param->begins_[i] = begins[i];
    param->ends_[i] = ends[i];
    param->strides_[i] = strides[i];
  }
  param->num_axes_ = begins_length;
  param->in_shape_length_ = in_shape_length;
}

TEST_F(TestStridedSliceFp32, StridedSlice1) {
  // prepare stage
  auto strided_slice_param = new StridedSliceParameter();
  std::vector<int> in_shape{1, 2, 4};
  std::vector<int> begins{0, 0, 0};
  std::vector<int> ends{1, 2, 4};
  std::vector<int> strides{1, 2, 2};
  int length = 3;
  int in_shape_length = 3;
  InitStridedSliceParam(strided_slice_param, in_shape, begins, ends, strides, length, in_shape_length);

  float input_data[8] = {0.2390374, 0.92039955, 0.05051243, 0.49574447, 0.8355223, 0.02647042, 0.08811307, 0.4566604};

  float correct[2] = {0.2390374, 0.05051243};

  float output_data[2];

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;

  // warm up loop
  for (int i = 0; i < 3; i++) {
    DoStridedSlice(input_data, output_data, strided_slice_param);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    DoStridedSlice(input_data, output_data, strided_slice_param);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  printf("==================output data=================\n");
  std::cout << output_data[0] << " , " << output_data[1];
  std::cout << std::endl;
  printf("==================corret data=================\n");
  std::cout << correct[0] << " , " << correct[1];
  std::cout << std::endl;

  CompareOutputData(output_data, correct, 2, 0.00001);

  delete strided_slice_param;
  MS_LOG(INFO) << "Teststrided_sliceFp32 passed";
}

TEST_F(TestStridedSliceFp32, StridedSlice2) {
  // prepare stage
  auto strided_slice_param = new StridedSliceParameter();
  std::vector<int> in_shape{1, 3, 3};
  std::vector<int> begins{0, 0, 0};
  std::vector<int> ends{2, 4, 4};
  std::vector<int> strides{1, 1, 1};
  int length = 3;
  int in_shape_length = 3;
  InitStridedSliceParam(strided_slice_param, in_shape, begins, ends, strides, length, in_shape_length);

  float input_data[9] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  float correct[9] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  float output_data[9];

  // runtime part
  DoStridedSlice(input_data, output_data, strided_slice_param);

  CompareOutputData(output_data, correct, 9, 0.00001);

  delete strided_slice_param;
}

TEST_F(TestStridedSliceFp32, StridedSlice3) {
  // prepare stage
  auto strided_slice_param = new StridedSliceParameter();
  std::vector<int> in_shape{1, 2, 4};
  std::vector<int> begins{0, 0, 0};
  std::vector<int> ends{1, 2, 4};
  std::vector<int> strides{1, 2, 2};
  int length = 3;
  int in_shape_length = 3;
  InitStridedSliceParam(strided_slice_param, in_shape, begins, ends, strides, length, in_shape_length);

  // int input_shape[3] = {1, 2, 4};
  std::vector<int> input_shape = {1, 2, 4};
  float input_data[8] = {0.2390374, 0.92039955, 0.05051243, 0.49574447, 0.8355223, 0.02647042, 0.08811307, 0.4566604};

  float correct[2] = {0.2390374, 0.05051243};
  float output_data[2];
  // int out_shape[3] = {1, 1, 2};
  std::vector<int> output_shape = {1, 1, 2};

  lite::Tensor input_tensor;
  input_tensor.SetData(input_data);
  input_tensor.set_shape(input_shape);
  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = &input_tensor;

  std::vector<lite::Tensor *> outputs_tensor;
  lite::Tensor output_tensor;
  outputs_tensor.push_back(&output_tensor);
  output_tensor.SetData(output_data);
  output_tensor.set_data_type(input_tensor.data_type());
  output_tensor.set_shape(output_shape);

  lite::Context *ctx = new lite::Context();
  ctx->thread_num_ = 2;
  strided_slice_param->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_StridedSlice};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(strided_slice_param), ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  kernel->Run();
  delete ctx;

  CompareOutputData(output_data, correct, 2, 0.000001);
  input_tensor.SetData(nullptr);
  output_tensor.SetData(nullptr);
}

TEST_F(TestStridedSliceFp32, StridedSlice4) {
  // prepare stage
  auto strided_slice_param = new StridedSliceParameter();
  std::vector<int> in_shape{5, 5};
  std::vector<int> begins{-5, -3};
  std::vector<int> ends{-1, -1};
  std::vector<int> strides{2, 1};
  int length = 2;
  int in_shape_length = 2;
  InitStridedSliceParam(strided_slice_param, in_shape, begins, ends, strides, length, in_shape_length);

  std::vector<int> input_shape = {5, 5};
  float input_data[25] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

  float correct[4] = {2.0, 3.0, 12.0, 13.0};
  float output_data[4];
  std::vector<int> output_shape = {2, 2};

  lite::Tensor input_tensor;
  input_tensor.SetData(input_data);
  input_tensor.set_shape(input_shape);
  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = &input_tensor;

  std::vector<lite::Tensor *> outputs_tensor;
  lite::Tensor output_tensor;
  outputs_tensor.push_back(&output_tensor);
  output_tensor.SetData(output_data);
  output_tensor.set_data_type(input_tensor.data_type());
  output_tensor.set_shape(output_shape);

  lite::Context *ctx = new lite::Context();
  ctx->thread_num_ = 2;
  strided_slice_param->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_StridedSlice};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(strided_slice_param), ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  kernel->Run();
  delete ctx;

  CompareOutputData(output_data, correct, 4, 0.000001);
  input_tensor.SetData(nullptr);
  output_tensor.SetData(nullptr);
}

TEST_F(TestStridedSliceFp32, StridedSlice5) {
  // prepare stage
  auto strided_slice_param = new StridedSliceParameter();
  std::vector<int> in_shape{5, 5, 5};
  std::vector<int> begins{1, -1, 1};
  std::vector<int> ends{4, -3, 3};
  std::vector<int> strides{1, -1, 1};
  int length = 3;
  int in_shape_length = 3;
  InitStridedSliceParam(strided_slice_param, in_shape, begins, ends, strides, length, in_shape_length);

  std::vector<int> input_shape = {5, 5, 5};
  float input_data[125] = {
    0.0,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,   10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
    16.0,  17.0,  18.0,  19.0,  20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,  30.0,  31.0,
    32.0,  33.0,  34.0,  35.0,  36.0,  37.0,  38.0,  39.0,  40.0,  41.0,  42.0,  43.0,  44.0,  45.0,  46.0,  47.0,
    48.0,  49.0,  50.0,  51.0,  52.0,  53.0,  54.0,  55.0,  56.0,  57.0,  58.0,  59.0,  60.0,  61.0,  62.0,  63.0,
    64.0,  65.0,  66.0,  67.0,  68.0,  69.0,  70.0,  71.0,  72.0,  73.0,  74.0,  75.0,  76.0,  77.0,  78.0,  79.0,
    80.0,  81.0,  82.0,  83.0,  84.0,  85.0,  86.0,  87.0,  88.0,  89.0,  90.0,  91.0,  92.0,  93.0,  94.0,  95.0,
    96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
    112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0};

  float correct[12] = {46.0, 47.0, 41.0, 42.0, 71.0, 72.0, 66.0, 67.0, 96.0, 97.0, 91.0, 92.0};
  float output_data[12];
  std::vector<int> output_shape = {3, 2, 2};

  lite::Tensor input_tensor;
  input_tensor.SetData(input_data);
  input_tensor.set_shape(input_shape);
  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = &input_tensor;

  std::vector<lite::Tensor *> outputs_tensor;
  lite::Tensor output_tensor;
  outputs_tensor.push_back(&output_tensor);
  output_tensor.SetData(output_data);
  output_tensor.set_data_type(input_tensor.data_type());
  output_tensor.set_shape(output_shape);

  lite::Context *ctx = new lite::Context();
  ctx->thread_num_ = 2;
  strided_slice_param->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_StridedSlice};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(strided_slice_param), ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  kernel->Run();
  delete ctx;

  CompareOutputData(output_data, correct, 12, 0.000001);
  input_tensor.SetData(nullptr);
  output_tensor.SetData(nullptr);
}

TEST_F(TestStridedSliceFp32, StridedSlice6) {
  // prepare stage
  auto strided_slice_param = new StridedSliceParameter();
  std::vector<int> in_shape{5, 5, 5};
  std::vector<int> begins{-5, -3, 1};
  std::vector<int> ends{-1, -1, 3};
  std::vector<int> strides{2, 1, 1};
  int length = 3;
  int in_shape_length = 3;
  InitStridedSliceParam(strided_slice_param, in_shape, begins, ends, strides, length, in_shape_length);

  std::vector<int> input_shape = {5, 5, 5};
  float input_data[125] = {
    0.0,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,   10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
    16.0,  17.0,  18.0,  19.0,  20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,  30.0,  31.0,
    32.0,  33.0,  34.0,  35.0,  36.0,  37.0,  38.0,  39.0,  40.0,  41.0,  42.0,  43.0,  44.0,  45.0,  46.0,  47.0,
    48.0,  49.0,  50.0,  51.0,  52.0,  53.0,  54.0,  55.0,  56.0,  57.0,  58.0,  59.0,  60.0,  61.0,  62.0,  63.0,
    64.0,  65.0,  66.0,  67.0,  68.0,  69.0,  70.0,  71.0,  72.0,  73.0,  74.0,  75.0,  76.0,  77.0,  78.0,  79.0,
    80.0,  81.0,  82.0,  83.0,  84.0,  85.0,  86.0,  87.0,  88.0,  89.0,  90.0,  91.0,  92.0,  93.0,  94.0,  95.0,
    96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
    112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0};

  float correct[8] = {11.0, 12.0, 16.0, 17.0, 61.0, 62.0, 66.0, 67.0};
  float output_data[8];
  std::vector<int> output_shape = {2, 2, 2};

  lite::Tensor input_tensor;
  input_tensor.SetData(input_data);
  input_tensor.set_shape(input_shape);
  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = &input_tensor;

  std::vector<lite::Tensor *> outputs_tensor;
  lite::Tensor output_tensor;
  outputs_tensor.push_back(&output_tensor);
  output_tensor.SetData(output_data);
  output_tensor.set_data_type(input_tensor.data_type());
  output_tensor.set_shape(output_shape);

  lite::Context *ctx = new lite::Context();
  ctx->thread_num_ = 2;
  strided_slice_param->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_StridedSlice};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(strided_slice_param), ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  kernel->Run();
  delete ctx;

  CompareOutputData(output_data, correct, 8, 0.000001);
  input_tensor.SetData(nullptr);
  output_tensor.SetData(nullptr);
}

TEST_F(TestStridedSliceFp32, StridedSlice7) {
  // prepare stage
  auto strided_slice_param = new StridedSliceParameter();
  std::vector<int> in_shape{2, 3};
  std::vector<int> begins{-1, 1};
  std::vector<int> ends{2, 2};
  std::vector<int> strides{1, 1};
  int length = 2;
  int in_shape_length = 2;
  InitStridedSliceParam(strided_slice_param, in_shape, begins, ends, strides, length, in_shape_length);

  std::vector<int> input_shape = {2, 3};
  float input_data[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};

  float correct[1] = {4.0};
  float output_data[1];
  std::vector<int> output_shape = {1, 1};

  lite::Tensor input_tensor;
  input_tensor.SetData(input_data);
  input_tensor.set_shape(input_shape);
  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = &input_tensor;

  std::vector<lite::Tensor *> outputs_tensor;
  lite::Tensor output_tensor;
  outputs_tensor.push_back(&output_tensor);
  output_tensor.SetData(output_data);
  output_tensor.set_data_type(input_tensor.data_type());
  output_tensor.set_shape(output_shape);

  lite::Context *ctx = new lite::Context();
  ctx->thread_num_ = 2;
  strided_slice_param->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_StridedSlice};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(strided_slice_param), ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  kernel->Run();
  delete ctx;

  CompareOutputData(output_data, correct, 1, 0.000001);
  input_tensor.SetData(nullptr);
  output_tensor.SetData(nullptr);
}

TEST_F(TestStridedSliceFp32, StridedSlice8) {
  // prepare stage
  auto strided_slice_param = new StridedSliceParameter();
  std::vector<int> in_shape{5, 5, 5};
  std::vector<int> begins{-2, -3};
  std::vector<int> ends{-1, -2};
  std::vector<int> strides{1, 1};
  int length = 2;
  int in_shape_length = 3;
  InitStridedSliceParam(strided_slice_param, in_shape, begins, ends, strides, length, in_shape_length);

  std::vector<int> input_shape = {5, 5, 5};
  float input_data[125] = {
    0.0,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,   10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
    16.0,  17.0,  18.0,  19.0,  20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,  30.0,  31.0,
    32.0,  33.0,  34.0,  35.0,  36.0,  37.0,  38.0,  39.0,  40.0,  41.0,  42.0,  43.0,  44.0,  45.0,  46.0,  47.0,
    48.0,  49.0,  50.0,  51.0,  52.0,  53.0,  54.0,  55.0,  56.0,  57.0,  58.0,  59.0,  60.0,  61.0,  62.0,  63.0,
    64.0,  65.0,  66.0,  67.0,  68.0,  69.0,  70.0,  71.0,  72.0,  73.0,  74.0,  75.0,  76.0,  77.0,  78.0,  79.0,
    80.0,  81.0,  82.0,  83.0,  84.0,  85.0,  86.0,  87.0,  88.0,  89.0,  90.0,  91.0,  92.0,  93.0,  94.0,  95.0,
    96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
    112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0};

  float correct[5] = {85.0, 86.0, 87.0, 88.0, 89.0};
  float output_data[5];
  std::vector<int> output_shape = {1, 1, 5};

  lite::Tensor input_tensor;
  input_tensor.SetData(input_data);
  input_tensor.set_shape(input_shape);
  std::vector<lite::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = &input_tensor;

  std::vector<lite::Tensor *> outputs_tensor;
  lite::Tensor output_tensor;
  outputs_tensor.push_back(&output_tensor);
  output_tensor.SetData(output_data);
  output_tensor.set_data_type(input_tensor.data_type());
  output_tensor.set_shape(output_shape);

  lite::Context *ctx = new lite::Context();
  ctx->thread_num_ = 2;
  strided_slice_param->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_StridedSlice};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(strided_slice_param), ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  kernel->Run();
  delete ctx;

  CompareOutputData(output_data, correct, 5, 0.000001);
  input_tensor.SetData(nullptr);
  output_tensor.SetData(nullptr);
}

}  // namespace mindspore
