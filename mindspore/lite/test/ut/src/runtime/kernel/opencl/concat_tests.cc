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
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/concat.h"

int DivideRoundUp(int n, int div) {
  int q = n / div;
  return n % div == 0 ? q : q + 1;
}
void printfNode(float *result, const std::vector<int> &tempNode) {
  for (int i = 0; i < tempNode[0]; i++) {
    for (int j = 0; j < tempNode[1]; j++) {
      for (int k = 0; k < tempNode[2]; k++) {
        for (int w = 0; w < tempNode[3]; w++) {
          std::cout
            << result[i * tempNode[2] * tempNode[1] * tempNode[3] + j * tempNode[2] * tempNode[3] + k * tempNode[3] + w]
            << "  ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void ConcatComputeByCPU_2input_dim4_axis3(float *input0, float *input1, float *output, std::vector<int> input_shape0,
                                          std::vector<int> input_shape1, std::vector<int> output_shape,
                                          const int axis) {
  int postion, index0 = 0, index1 = 0;
  for (int i = 0; i < output_shape[0]; i++) {
    for (int j = 0; j < output_shape[1]; j++) {
      for (int k = 0; k < output_shape[2]; k++) {
        postion = i * output_shape[1] * output_shape[2] * output_shape[3] + j * output_shape[2] * output_shape[3] +
                  k * output_shape[3];
        for (int w = 0; w < output_shape[3]; w++) {
          if (w < input_shape0[3] + input_shape1[3]) {
            output[postion++] = (w < input_shape0[3]) ? input0[index0++] : input1[index1++];
          } else {
            for (int ind = input_shape0[3] + input_shape1[3]; ind < output_shape[3]; ind++) {
              output[postion++] = 0;
            }
          }
        }
      }
    }
  }
}
void ConcatComputeByCPU_3input_dim4_axis3(float *input0, float *input1, float *input2, float *output,
                                          std::vector<int> input_shape0, std::vector<int> input_shape1,
                                          std::vector<int> input_shape2, std::vector<int> output_shape,
                                          const int axis) {
  int postion, index0 = 0, index1 = 0, index2 = 0;
  for (int i = 0; i < output_shape[0]; i++) {
    for (int j = 0; j < output_shape[1]; j++) {
      for (int k = 0; k < output_shape[2]; k++) {
        postion = i * output_shape[1] * output_shape[2] * output_shape[3] + j * output_shape[2] * output_shape[3] +
                  k * output_shape[3];
        for (int w = 0; w < output_shape[3]; w++) {
          if (w < input_shape0[3]) {
            int align = DivideRoundUp(input_shape0[3], 4) * 4;
            index0 = i * input_shape0[1] * input_shape0[2] * align + j * input_shape0[2] * align + k * align + w;
            output[postion++] = input0[index0];
          } else if (w >= input_shape0[3] && w < (input_shape0[3] + input_shape1[3])) {
            int align = DivideRoundUp(input_shape1[3], 4) * 4;
            index1 = i * input_shape1[1] * input_shape1[2] * align + j * input_shape1[2] * align + k * align + w -
                     input_shape0[3];
            output[postion++] = input1[index1];
          } else if ((input_shape0[3] + input_shape1[3]) <= w &&
                     w < (input_shape0[3] + input_shape1[3] + input_shape2[3])) {
            int align = DivideRoundUp(input_shape2[3], 4) * 4;
            index2 = i * input_shape2[1] * input_shape2[2] * align + j * input_shape2[2] * align + k * align + w -
                     input_shape0[3] - input_shape1[3];
            output[postion++] = input2[index2];
          } else {
            for (int ind = input_shape0[3] + input_shape1[3] + input_shape2[3]; ind < output_shape[3]; ind++) {
              output[postion++] = 0;
            }
            break;
          }
        }
      }
    }
  }
}

namespace mindspore {
class TestConcatOpenCL : public mindspore::CommonTest {
 public:
  TestConcatOpenCL() {}
};

template <typename T>
void CompareOutputData1(T *output_data, T *correct_data, int size, float err_bound) {
  for (size_t i = 0; i < size; i++) {
    T abs = fabs(output_data[i] - correct_data[i]);
    //          printf("i=%d %.3f %.3f\n", i, output_data[i], correct_data[i]);
    ASSERT_LE(abs, err_bound);
  }
}

TEST_F(TestConcatOpenCL, ConcatFp32_2input_dim4_axis3) {
  MS_LOG(INFO) << "begin test";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(INFO) << "init tensors";
  constexpr int INPUT_NUM = 2;
  //  std::array<std::vector<int>, INPUT_NUM> input_shapes = {
  //    std::vector<int>{1, 120, 120, 16}, std::vector<int>{1, 120, 120, 16},std::vector<int>{1, 120, 120, 96}};
  std::array<std::vector<int>, INPUT_NUM> input_shapes = {std::vector<int>{1, 32, 512, 48},
                                                          std::vector<int>{1, 32, 512, 48}};
  std::vector<int> output_shape = {1, 32, 512, 96};
  output_shape[3] = DivideRoundUp(output_shape[3], 4) * 4;
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = schema::NodeType_ValueNode;
  std::vector<lite::tensor::Tensor *> inputs;
  for (auto &shape : input_shapes) {
    inputs.push_back(new lite::tensor::Tensor(data_type, shape, schema::Format_NHWC, tensor_type));
  }
  auto *output_tensor = new lite::tensor::Tensor(data_type, output_shape, schema::Format_NHWC, tensor_type);
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};
  std::cout << "input_shapes size=: " << input_shapes.size() << std::endl;

  std::cout << "initialize tensors";
  auto param = new ConcatParameter();
  param->axis_ = 3;
  auto *concat_kernel = new kernel::ConcatOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  concat_kernel->Init();
  MS_LOG(INFO) << "initialize sub_graph";
  std::vector<kernel::LiteKernel *> kernels{concat_kernel};
  auto *sub_graph = new kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  // to do allocate memory for inputs and outputs
  for (auto &input_tensor : inputs) {
    input_tensor->MallocData(allocator);
  }
  sub_graph->Init();
  unsigned int seed = 123;
  MS_LOG(INFO) << "initialize input data";
  for (auto &input_tensor : inputs) {
    auto input_data = reinterpret_cast<float *>(input_tensor->Data());
    for (int i = 0; i < input_tensor->ElementsNum(); ++i) {
      input_data[i] = static_cast<float>(rand_r(&seed) % 10 + 1);
    }
  }

  // compute the result for CPU
  auto *input_data0 = reinterpret_cast<float *>(inputs[0]->Data());
  auto *input_data1 = reinterpret_cast<float *>(inputs[1]->Data());
  std::vector<float> output_data_cpu(output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]);
  if (inputs.size() == 2) {
    ConcatComputeByCPU_2input_dim4_axis3(input_data0, input_data1, output_data_cpu.data(), input_shapes[0],
                                         input_shapes[1], output_shape, param->axis_);
  }
  if (inputs.size() == 3) {
    auto *input_data2 = reinterpret_cast<float *>(inputs[2]->Data());
    ConcatComputeByCPU_3input_dim4_axis3(input_data0, input_data1, input_data2, output_data_cpu.data(), input_shapes[0],
                                         input_shapes[1], input_shapes[2], output_shape, param->axis_);
  }

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor->Data());
  CompareOutputData1(output_data_gpu, output_data_cpu.data(), output_tensor->ElementsNum(), 0.00001);
}
}  // namespace mindspore
