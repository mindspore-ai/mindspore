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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/space_to_depth.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {
class TestSpaceToDepthOpenCL : public mindspore::CommonTest {
 public:
  TestSpaceToDepthOpenCL() {}
};

void RunTestCaseSpaceToDepth(const std::vector<int> &shape_in, const std::vector<int> &shape_out, void *input_data,
                             void *output_data, bool enable_fp16, int block_size) {
  auto ocl_runtime = lite::opencl::OpenCLRuntimeWrapper().GetInstance();
  ocl_runtime->Init();
  size_t dtype_size = enable_fp16 ? sizeof(float16_t) : sizeof(float);
  ocl_runtime->SetFp16Enable(enable_fp16);
  auto allocator = ocl_runtime->GetAllocator();
  auto param = static_cast<SpaceToDepthParameter *>(malloc(sizeof(SpaceToDepthParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "param_ptr create error.";
    return;
  }
  param->block_size_ = block_size;
  auto tensor_x_ptr = std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32),
                                                     shape_in, schema::Format_NHWC);
  auto tensor_x = tensor_x_ptr.get();
  if (tensor_x == nullptr) {
    MS_LOG(ERROR) << "tensor_x create error.";
    return;
  }
  auto tensor_out_ptr =
    std::make_unique<lite::Tensor>(TypeId(enable_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32), shape_out);
  auto tensor_out = tensor_out_ptr.get();
  if (tensor_out == nullptr) {
    MS_LOG(ERROR) << "tensor_out create error.";
    return;
  }
  std::vector<lite::Tensor *> inputs{tensor_x};
  std::vector<lite::Tensor *> outputs{tensor_out};
  auto arith_kernel = kernel::OpenCLKernelCreator<kernel::SpaceToDepthOpenCLKernel>(
    inputs, outputs, reinterpret_cast<OpParameter *>(param), nullptr, kernel::KernelKey(), nullptr);
  if (arith_kernel == nullptr) {
    MS_LOG(ERROR) << "arith_kernel create error.";
    return;
  }

  inputs[0]->MallocData(allocator);

  std::vector<kernel::LiteKernel *> kernels{arith_kernel};
  auto pGraph_ptr = std::make_unique<kernel::SubGraphOpenCLKernel>(inputs, outputs, kernels, kernels, kernels);
  auto pGraph = pGraph_ptr.get();
  if (pGraph == nullptr) {
    MS_LOG(ERROR) << "pGraph create error.";
    return;
  }
  pGraph->Init();
  memcpy(inputs[0]->MutableData(), input_data, inputs[0]->ElementsNum() * dtype_size);
  pGraph->Run();

  if (enable_fp16) {
    CompareOutput(outputs[0]->MutableData(), output_data, outputs[0]->ElementsNum(), static_cast<float16_t>(1e-3),
                  2e-2);
  } else {
    CompareOutput(outputs[0]->MutableData(), output_data, outputs[0]->ElementsNum(), static_cast<float>(1e-5));
  }
  for (auto t : inputs) {
    t->set_data(nullptr);
  }
  for (auto t : outputs) {
    t->set_data(nullptr);
  }

  MS_LOG(INFO) << "Test SpaceToDepth passed";
}

TEST_F(TestSpaceToDepthOpenCL, AlignTest1Fp32) {
  std::vector<int> shape_in = {1, 2, 2, 4};
  std::vector<int> shape_out = {1, 1, 1, 16};
  std::vector<float> input_data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                   9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  std::vector<float> output_data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                    9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

  RunTestCaseSpaceToDepth(shape_in, shape_out, input_data.data(), output_data.data(), false, 2);
}

TEST_F(TestSpaceToDepthOpenCL, AlignTest1Fp16) {
  std::vector<int> shape_in = {1, 2, 2, 4};
  std::vector<int> shape_out = {1, 1, 1, 16};
  std::vector<float16_t> input_data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  std::vector<float16_t> output_data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

  RunTestCaseSpaceToDepth(shape_in, shape_out, input_data.data(), output_data.data(), true, 2);
}

TEST_F(TestSpaceToDepthOpenCL, AlignTest2Fp32) {
  std::vector<int> shape_in = {1, 4, 4, 4};
  std::vector<int> shape_out = {1, 2, 2, 16};
  std::vector<float> input_data = {
    0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
    32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,
    48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f};
  std::vector<float> output_data = {
    0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
    8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
    32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
    40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f};

  RunTestCaseSpaceToDepth(shape_in, shape_out, input_data.data(), output_data.data(), false, 2);
}

TEST_F(TestSpaceToDepthOpenCL, AlignTest2Fp16) {
  std::vector<int> shape_in = {1, 4, 4, 4};
  std::vector<int> shape_out = {1, 2, 2, 16};
  std::vector<float16_t> input_data = {
    0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
    32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,
    48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f};
  std::vector<float16_t> output_data = {
    0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
    8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
    32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
    40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f};

  RunTestCaseSpaceToDepth(shape_in, shape_out, input_data.data(), output_data.data(), true, 2);
}

TEST_F(TestSpaceToDepthOpenCL, AlignTest3Fp32) {
  std::vector<int> shape_in = {1, 6, 6, 4};
  std::vector<int> shape_out = {1, 2, 2, 36};
  std::vector<float> input_data = {
    0.0f,   1.0f,   2.0f,   3.0f,   4.0f,   5.0f,   6.0f,   7.0f,   8.0f,   9.0f,   10.0f,  11.0f,  12.0f,  13.0f,
    14.0f,  15.0f,  16.0f,  17.0f,  18.0f,  19.0f,  20.0f,  21.0f,  22.0f,  23.0f,  24.0f,  25.0f,  26.0f,  27.0f,
    28.0f,  29.0f,  30.0f,  31.0f,  32.0f,  33.0f,  34.0f,  35.0f,  36.0f,  37.0f,  38.0f,  39.0f,  40.0f,  41.0f,
    42.0f,  43.0f,  44.0f,  45.0f,  46.0f,  47.0f,  48.0f,  49.0f,  50.0f,  51.0f,  52.0f,  53.0f,  54.0f,  55.0f,
    56.0f,  57.0f,  58.0f,  59.0f,  60.0f,  61.0f,  62.0f,  63.0f,  64.0f,  65.0f,  66.0f,  67.0f,  68.0f,  69.0f,
    70.0f,  71.0f,  72.0f,  73.0f,  74.0f,  75.0f,  76.0f,  77.0f,  78.0f,  79.0f,  80.0f,  81.0f,  82.0f,  83.0f,
    84.0f,  85.0f,  86.0f,  87.0f,  88.0f,  89.0f,  90.0f,  91.0f,  92.0f,  93.0f,  94.0f,  95.0f,  96.0f,  97.0f,
    98.0f,  99.0f,  100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f, 110.0f, 111.0f,
    112.0f, 113.0f, 114.0f, 115.0f, 116.0f, 117.0f, 118.0f, 119.0f, 120.0f, 121.0f, 122.0f, 123.0f, 124.0f, 125.0f,
    126.0f, 127.0f, 128.0f, 129.0f, 130.0f, 131.0f, 132.0f, 133.0f, 134.0f, 135.0f, 136.0f, 137.0f, 138.0f, 139.0f,
    140.0f, 141.0f, 142.0f, 143.0f};
  std::vector<float> output_data = {
    0.0f,   1.0f,   2.0f,   3.0f,   4.0f,   5.0f,   6.0f,   7.0f,   8.0f,   9.0f,   10.0f,  11.0f,  24.0f,  25.0f,
    26.0f,  27.0f,  28.0f,  29.0f,  30.0f,  31.0f,  32.0f,  33.0f,  34.0f,  35.0f,  48.0f,  49.0f,  50.0f,  51.0f,
    52.0f,  53.0f,  54.0f,  55.0f,  56.0f,  57.0f,  58.0f,  59.0f,  12.0f,  13.0f,  14.0f,  15.0f,  16.0f,  17.0f,
    18.0f,  19.0f,  20.0f,  21.0f,  22.0f,  23.0f,  36.0f,  37.0f,  38.0f,  39.0f,  40.0f,  41.0f,  42.0f,  43.0f,
    44.0f,  45.0f,  46.0f,  47.0f,  60.0f,  61.0f,  62.0f,  63.0f,  64.0f,  65.0f,  66.0f,  67.0f,  68.0f,  69.0f,
    70.0f,  71.0f,  72.0f,  73.0f,  74.0f,  75.0f,  76.0f,  77.0f,  78.0f,  79.0f,  80.0f,  81.0f,  82.0f,  83.0f,
    96.0f,  97.0f,  98.0f,  99.0f,  100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 120.0f, 121.0f,
    122.0f, 123.0f, 124.0f, 125.0f, 126.0f, 127.0f, 128.0f, 129.0f, 130.0f, 131.0f, 84.0f,  85.0f,  86.0f,  87.0f,
    88.0f,  89.0f,  90.0f,  91.0f,  92.0f,  93.0f,  94.0f,  95.0f,  108.0f, 109.0f, 110.0f, 111.0f, 112.0f, 113.0f,
    114.0f, 115.0f, 116.0f, 117.0f, 118.0f, 119.0f, 132.0f, 133.0f, 134.0f, 135.0f, 136.0f, 137.0f, 138.0f, 139.0f,
    140.0f, 141.0f, 142.0f, 143.0f};

  RunTestCaseSpaceToDepth(shape_in, shape_out, input_data.data(), output_data.data(), false, 3);
}

TEST_F(TestSpaceToDepthOpenCL, NotAlignTest1Fp32) {
  std::vector<int> shape_in = {1, 2, 2, 1};
  std::vector<int> shape_out = {1, 1, 1, 4};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> output_data = {0.0f, 1.0f, 2.0f, 3.0f};

  RunTestCaseSpaceToDepth(shape_in, shape_out, input_data.data(), output_data.data(), false, 2);
}

TEST_F(TestSpaceToDepthOpenCL, NotAlignTest1Fp16) {
  std::vector<int> shape_in = {1, 2, 2, 1};
  std::vector<int> shape_out = {1, 1, 1, 4};
  std::vector<float16_t> input_data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float16_t> output_data = {0.0f, 1.0f, 2.0f, 3.0f};

  RunTestCaseSpaceToDepth(shape_in, shape_out, input_data.data(), output_data.data(), true, 2);
}

TEST_F(TestSpaceToDepthOpenCL, NotAlignTest2Fp32) {
  std::vector<int> shape_in = {1, 2, 2, 3};
  std::vector<int> shape_out = {1, 1, 1, 12};
  std::vector<float> input_data = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
  };
  std::vector<float> output_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

  RunTestCaseSpaceToDepth(shape_in, shape_out, input_data.data(), output_data.data(), false, 2);
}

TEST_F(TestSpaceToDepthOpenCL, NotAlignTest3Fp32) {
  std::vector<int> shape_in = {1, 4, 4, 3};
  std::vector<int> shape_out = {1, 2, 2, 12};
  std::vector<float> input_data = {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                                   12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
                                   24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
                                   36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f};
  std::vector<float> output_data = {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
                                    6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
                                    24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f,
                                    30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f};

  RunTestCaseSpaceToDepth(shape_in, shape_out, input_data.data(), output_data.data(), false, 2);
}

TEST_F(TestSpaceToDepthOpenCL, NotAlignTest4Fp32) {
  std::vector<int> shape_in = {1, 6, 6, 6};
  std::vector<int> shape_out = {1, 2, 2, 54};
  std::vector<float> input_data = {
    0.0f,   1.0f,   2.0f,   3.0f,   4.0f,   5.0f,   6.0f,   7.0f,   8.0f,   9.0f,   10.0f,  11.0f,  12.0f,  13.0f,
    14.0f,  15.0f,  16.0f,  17.0f,  18.0f,  19.0f,  20.0f,  21.0f,  22.0f,  23.0f,  24.0f,  25.0f,  26.0f,  27.0f,
    28.0f,  29.0f,  30.0f,  31.0f,  32.0f,  33.0f,  34.0f,  35.0f,  36.0f,  37.0f,  38.0f,  39.0f,  40.0f,  41.0f,
    42.0f,  43.0f,  44.0f,  45.0f,  46.0f,  47.0f,  48.0f,  49.0f,  50.0f,  51.0f,  52.0f,  53.0f,  54.0f,  55.0f,
    56.0f,  57.0f,  58.0f,  59.0f,  60.0f,  61.0f,  62.0f,  63.0f,  64.0f,  65.0f,  66.0f,  67.0f,  68.0f,  69.0f,
    70.0f,  71.0f,  72.0f,  73.0f,  74.0f,  75.0f,  76.0f,  77.0f,  78.0f,  79.0f,  80.0f,  81.0f,  82.0f,  83.0f,
    84.0f,  85.0f,  86.0f,  87.0f,  88.0f,  89.0f,  90.0f,  91.0f,  92.0f,  93.0f,  94.0f,  95.0f,  96.0f,  97.0f,
    98.0f,  99.0f,  100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f, 110.0f, 111.0f,
    112.0f, 113.0f, 114.0f, 115.0f, 116.0f, 117.0f, 118.0f, 119.0f, 120.0f, 121.0f, 122.0f, 123.0f, 124.0f, 125.0f,
    126.0f, 127.0f, 128.0f, 129.0f, 130.0f, 131.0f, 132.0f, 133.0f, 134.0f, 135.0f, 136.0f, 137.0f, 138.0f, 139.0f,
    140.0f, 141.0f, 142.0f, 143.0f, 144.0f, 145.0f, 146.0f, 147.0f, 148.0f, 149.0f, 150.0f, 151.0f, 152.0f, 153.0f,
    154.0f, 155.0f, 156.0f, 157.0f, 158.0f, 159.0f, 160.0f, 161.0f, 162.0f, 163.0f, 164.0f, 165.0f, 166.0f, 167.0f,
    168.0f, 169.0f, 170.0f, 171.0f, 172.0f, 173.0f, 174.0f, 175.0f, 176.0f, 177.0f, 178.0f, 179.0f, 180.0f, 181.0f,
    182.0f, 183.0f, 184.0f, 185.0f, 186.0f, 187.0f, 188.0f, 189.0f, 190.0f, 191.0f, 192.0f, 193.0f, 194.0f, 195.0f,
    196.0f, 197.0f, 198.0f, 199.0f, 200.0f, 201.0f, 202.0f, 203.0f, 204.0f, 205.0f, 206.0f, 207.0f, 208.0f, 209.0f,
    210.0f, 211.0f, 212.0f, 213.0f, 214.0f, 215.0f};
  std::vector<float> output_data = {
    0.0f,   1.0f,   2.0f,   3.0f,   4.0f,   5.0f,   6.0f,   7.0f,   8.0f,   9.0f,   10.0f,  11.0f,  12.0f,  13.0f,
    14.0f,  15.0f,  16.0f,  17.0f,  36.0f,  37.0f,  38.0f,  39.0f,  40.0f,  41.0f,  42.0f,  43.0f,  44.0f,  45.0f,
    46.0f,  47.0f,  48.0f,  49.0f,  50.0f,  51.0f,  52.0f,  53.0f,  72.0f,  73.0f,  74.0f,  75.0f,  76.0f,  77.0f,
    78.0f,  79.0f,  80.0f,  81.0f,  82.0f,  83.0f,  84.0f,  85.0f,  86.0f,  87.0f,  88.0f,  89.0f,  18.0f,  19.0f,
    20.0f,  21.0f,  22.0f,  23.0f,  24.0f,  25.0f,  26.0f,  27.0f,  28.0f,  29.0f,  30.0f,  31.0f,  32.0f,  33.0f,
    34.0f,  35.0f,  54.0f,  55.0f,  56.0f,  57.0f,  58.0f,  59.0f,  60.0f,  61.0f,  62.0f,  63.0f,  64.0f,  65.0f,
    66.0f,  67.0f,  68.0f,  69.0f,  70.0f,  71.0f,  90.0f,  91.0f,  92.0f,  93.0f,  94.0f,  95.0f,  96.0f,  97.0f,
    98.0f,  99.0f,  100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f, 110.0f, 111.0f,
    112.0f, 113.0f, 114.0f, 115.0f, 116.0f, 117.0f, 118.0f, 119.0f, 120.0f, 121.0f, 122.0f, 123.0f, 124.0f, 125.0f,
    144.0f, 145.0f, 146.0f, 147.0f, 148.0f, 149.0f, 150.0f, 151.0f, 152.0f, 153.0f, 154.0f, 155.0f, 156.0f, 157.0f,
    158.0f, 159.0f, 160.0f, 161.0f, 180.0f, 181.0f, 182.0f, 183.0f, 184.0f, 185.0f, 186.0f, 187.0f, 188.0f, 189.0f,
    190.0f, 191.0f, 192.0f, 193.0f, 194.0f, 195.0f, 196.0f, 197.0f, 126.0f, 127.0f, 128.0f, 129.0f, 130.0f, 131.0f,
    132.0f, 133.0f, 134.0f, 135.0f, 136.0f, 137.0f, 138.0f, 139.0f, 140.0f, 141.0f, 142.0f, 143.0f, 162.0f, 163.0f,
    164.0f, 165.0f, 166.0f, 167.0f, 168.0f, 169.0f, 170.0f, 171.0f, 172.0f, 173.0f, 174.0f, 175.0f, 176.0f, 177.0f,
    178.0f, 179.0f, 198.0f, 199.0f, 200.0f, 201.0f, 202.0f, 203.0f, 204.0f, 205.0f, 206.0f, 207.0f, 208.0f, 209.0f,
    210.0f, 211.0f, 212.0f, 213.0f, 214.0f, 215.0f};

  RunTestCaseSpaceToDepth(shape_in, shape_out, input_data.data(), output_data.data(), false, 3);
}
}  // namespace mindspore
