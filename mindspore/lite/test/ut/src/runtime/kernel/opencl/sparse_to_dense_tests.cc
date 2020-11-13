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
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/sparse_to_dense.h"
using mindspore::lite::Tensor;
using mindspore::schema::Format::Format_NHWC;
namespace mindspore {
class TestSparseToDenseOpenCLCI : public mindspore::CommonTest {
 public:
  TestSparseToDenseOpenCLCI() {}
};

TEST_F(TestSparseToDenseOpenCLCI, Fp32Dim2Scalar) {
  MS_LOG(INFO) << " begin test ";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto runtime = runtime_wrapper.GetInstance();
  runtime->Init();
  auto allocator = runtime->GetAllocator();

  MS_LOG(INFO) << " init tensors ";
  std::vector<int> input_shape1 = {6, 2};
  std::vector<int> input_shape2 = {2};
  std::vector<int> input_shape3 = {1};
  std::vector<int> input_shape4 = {1};
  float input_data1[] = {0, 0, 1, 2, 2, 3, 3, 6, 4, 7, 5, 9};
  float input_data2[] = {6, 10};
  float input_data3[] = {6.0};
  float input_data4[] = {0.0};
  float correctOutput[] = {6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6};
  auto data_type = kNumberTypeFloat32;
  std::vector<int> output_shape = {6, 10};
  auto in_tensor1 = Tensor(data_type, input_shape1, Format_NHWC, lite::Tensor::VAR);
  auto in_tensor2 = Tensor(data_type, input_shape2, Format_NHWC, lite::Tensor::CONST_TENSOR);
  auto in_tensor3 = Tensor(data_type, input_shape3, Format_NHWC, lite::Tensor::CONST_SCALAR);
  auto in_tensor4 = Tensor(data_type, input_shape4, Format_NHWC, lite::Tensor::CONST_SCALAR);
  auto output_tensor = Tensor(data_type, output_shape, Format_NHWC, lite::Tensor::VAR);
  // allocate memory for weights
  in_tensor2.MallocData();
  in_tensor3.MallocData();
  in_tensor4.MallocData();
  std::vector<lite::Tensor *> inputs{&in_tensor1, &in_tensor2, &in_tensor3, &in_tensor4};
  std::vector<lite::Tensor *> outputs{&output_tensor};
  // initialize weights
  memcpy(inputs[1]->data_c(), input_data2, sizeof(input_data2));
  memcpy(inputs[2]->data_c(), input_data3, sizeof(input_data3));
  memcpy(inputs[3]->data_c(), input_data4, sizeof(input_data4));
  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<SparseToDenseParameter *>(malloc(sizeof(SparseToDenseParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new ActivationParameter failed ";
    return;
  }

  auto *sparse_to_dense_kernel =
    new (std::nothrow) kernel::SparseToDenseOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (sparse_to_dense_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::SparseToDenseOpenCLKernel failed ";
    delete param;
    return;
  }
  sparse_to_dense_kernel->Init();
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{sparse_to_dense_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel({&in_tensor1}, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    delete param;
    delete sparse_to_dense_kernel;
    return;
  }
  // to do allocate memory for inputs
  in_tensor1.MallocData(allocator);
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data1, sizeof(input_data1));

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor.data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correctOutput, output_tensor.ElementsNum(), 0.0001));
  delete sub_graph;
}

TEST_F(TestSparseToDenseOpenCLCI, Fp32Dim2Vector) {
  MS_LOG(INFO) << " begin test ";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto runtime = runtime_wrapper.GetInstance();
  runtime->Init();
  auto allocator = runtime->GetAllocator();

  MS_LOG(INFO) << " init tensors ";
  std::vector<int> input_shape1 = {6, 2};
  std::vector<int> input_shape2 = {2};
  std::vector<int> input_shape3 = {6};
  std::vector<int> input_shape4 = {1};
  float input_data1[] = {0, 0, 1, 2, 2, 3, 3, 6, 4, 7, 5, 9};
  float input_data2[] = {6, 10};
  float input_data3[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  float input_data4[] = {0.0};
  float correctOutput[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6};
  auto data_type = kNumberTypeFloat32;
  std::vector<int> output_shape = {6, 10};
  auto in_tensor1 = Tensor(data_type, input_shape1, Format_NHWC, lite::Tensor::VAR);
  auto in_tensor2 = Tensor(data_type, input_shape2, Format_NHWC, lite::Tensor::CONST_TENSOR);
  auto in_tensor3 = Tensor(data_type, input_shape3, Format_NHWC, lite::Tensor::CONST_TENSOR);
  auto in_tensor4 = Tensor(data_type, input_shape4, Format_NHWC, lite::Tensor::CONST_SCALAR);
  auto output_tensor = Tensor(data_type, output_shape, Format_NHWC, lite::Tensor::VAR);
  // allocate memory for weights
  in_tensor2.MallocData();
  in_tensor3.MallocData();
  in_tensor4.MallocData();
  std::vector<lite::Tensor *> inputs{&in_tensor1, &in_tensor2, &in_tensor3, &in_tensor4};
  std::vector<lite::Tensor *> outputs{&output_tensor};
  // initialize weights
  memcpy(inputs[1]->data_c(), input_data2, sizeof(input_data2));
  memcpy(inputs[2]->data_c(), input_data3, sizeof(input_data3));
  memcpy(inputs[3]->data_c(), input_data4, sizeof(input_data4));
  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<SparseToDenseParameter *>(malloc(sizeof(SparseToDenseParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new ActivationParameter failed ";
    return;
  }

  auto *sparse_to_dense_kernel =
    new (std::nothrow) kernel::SparseToDenseOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (sparse_to_dense_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::SparseToDenseOpenCLKernel failed ";
    delete param;
    return;
  }
  sparse_to_dense_kernel->Init();
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{sparse_to_dense_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel({&in_tensor1}, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    delete param;
    delete sparse_to_dense_kernel;
    return;
  }
  // to do allocate memory for inputs
  in_tensor1.MallocData(allocator);
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data1, sizeof(input_data1));

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor.data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correctOutput, output_tensor.ElementsNum(), 0.0001));
  delete sub_graph;
}

TEST_F(TestSparseToDenseOpenCLCI, Fp32Dim2Shape1Vector) {
  MS_LOG(INFO) << " begin test ";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto runtime = runtime_wrapper.GetInstance();
  runtime->Init();
  auto allocator = runtime->GetAllocator();

  MS_LOG(INFO) << " init tensors ";
  std::vector<int> input_shape1 = {6, 1};
  std::vector<int> input_shape2 = {1};
  std::vector<int> input_shape3 = {6};
  std::vector<int> input_shape4 = {1};
  float input_data1[] = {0, 2, 3, 6, 7, 9};
  float input_data2[] = {10};
  float input_data3[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  float input_data4[] = {0.0};
  float correctOutput[] = {1, 0, 2, 3, 0, 0, 4, 5, 0, 6};
  auto data_type = kNumberTypeFloat32;
  std::vector<int> output_shape = {10};
  auto in_tensor1 = Tensor(data_type, input_shape1, Format_NHWC, lite::Tensor::VAR);
  auto in_tensor2 = Tensor(data_type, input_shape2, Format_NHWC, lite::Tensor::CONST_TENSOR);
  auto in_tensor3 = Tensor(data_type, input_shape3, Format_NHWC, lite::Tensor::CONST_TENSOR);
  auto in_tensor4 = Tensor(data_type, input_shape4, Format_NHWC, lite::Tensor::CONST_SCALAR);
  auto output_tensor = Tensor(data_type, output_shape, Format_NHWC, lite::Tensor::VAR);
  // allocate memory for weights
  in_tensor2.MallocData();
  in_tensor3.MallocData();
  in_tensor4.MallocData();
  std::vector<lite::Tensor *> inputs{&in_tensor1, &in_tensor2, &in_tensor3, &in_tensor4};
  std::vector<lite::Tensor *> outputs{&output_tensor};
  // initialize weights
  memcpy(inputs[1]->data_c(), input_data2, sizeof(input_data2));
  memcpy(inputs[2]->data_c(), input_data3, sizeof(input_data3));
  memcpy(inputs[3]->data_c(), input_data4, sizeof(input_data4));
  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<SparseToDenseParameter *>(malloc(sizeof(SparseToDenseParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new ActivationParameter failed ";
    return;
  }

  auto *sparse_to_dense_kernel =
    new (std::nothrow) kernel::SparseToDenseOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (sparse_to_dense_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::SparseToDenseOpenCLKernel failed ";
    delete param;
    return;
  }
  sparse_to_dense_kernel->Init();
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{sparse_to_dense_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel({&in_tensor1}, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    delete param;
    delete sparse_to_dense_kernel;
    return;
  }
  // to do allocate memory for inputs
  in_tensor1.MallocData(allocator);
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data1, sizeof(input_data1));

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor.data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correctOutput, output_tensor.ElementsNum(), 0.0001));
  delete sub_graph;
}

TEST_F(TestSparseToDenseOpenCLCI, Fp32Dim2Shape1Scalar) {
  MS_LOG(INFO) << " begin test ";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto runtime = runtime_wrapper.GetInstance();
  runtime->Init();
  auto allocator = runtime->GetAllocator();

  MS_LOG(INFO) << " init tensors ";
  std::vector<int> input_shape1 = {7, 1};  // shape[1] = 1
  std::vector<int> input_shape2 = {1};
  std::vector<int> input_shape3 = {1};
  std::vector<int> input_shape4 = {1};
  float input_data1[] = {0, 1, 2, 3, 4, 5, 9};
  float input_data2[] = {10};
  float input_data3[] = {6.0};
  float input_data4[] = {0.0};
  float correctOutput[] = {6, 6, 6, 6, 6, 6, 0, 0, 0, 6};
  auto data_type = kNumberTypeFloat32;
  std::vector<int> output_shape = {10};
  auto in_tensor1 = Tensor(data_type, input_shape1, Format_NHWC, lite::Tensor::VAR);
  auto in_tensor2 = Tensor(data_type, input_shape2, Format_NHWC, lite::Tensor::CONST_TENSOR);
  auto in_tensor3 = Tensor(data_type, input_shape3, Format_NHWC, lite::Tensor::CONST_SCALAR);
  auto in_tensor4 = Tensor(data_type, input_shape4, Format_NHWC, lite::Tensor::CONST_SCALAR);
  auto output_tensor = Tensor(data_type, output_shape, Format_NHWC, lite::Tensor::VAR);
  // allocate memory for weights
  in_tensor2.MallocData();
  in_tensor3.MallocData();
  in_tensor4.MallocData();
  std::vector<lite::Tensor *> inputs{&in_tensor1, &in_tensor2, &in_tensor3, &in_tensor4};
  std::vector<lite::Tensor *> outputs{&output_tensor};
  // initialize weights
  memcpy(inputs[1]->data_c(), input_data2, sizeof(input_data2));
  memcpy(inputs[2]->data_c(), input_data3, sizeof(input_data3));
  memcpy(inputs[3]->data_c(), input_data4, sizeof(input_data4));
  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<SparseToDenseParameter *>(malloc(sizeof(SparseToDenseParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new ActivationParameter failed ";
    return;
  }

  auto *sparse_to_dense_kernel =
    new (std::nothrow) kernel::SparseToDenseOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (sparse_to_dense_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::SparseToDenseOpenCLKernel failed ";
    delete param;
    return;
  }
  sparse_to_dense_kernel->Init();
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{sparse_to_dense_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel({&in_tensor1}, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    delete param;
    delete sparse_to_dense_kernel;
    return;
  }
  // to do allocate memory for inputs
  in_tensor1.MallocData(allocator);
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data1, sizeof(input_data1));

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor.data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correctOutput, output_tensor.ElementsNum(), 0.0001));
  delete sub_graph;
}

TEST_F(TestSparseToDenseOpenCLCI, Fp32Dim1Scalar) {
  MS_LOG(INFO) << " begin test ";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto runtime = runtime_wrapper.GetInstance();
  runtime->Init();
  auto allocator = runtime->GetAllocator();
  MS_LOG(INFO) << " init tensors ";
  std::vector<int> input_shape1 = {6};
  std::vector<int> input_shape2 = {1};
  std::vector<int> input_shape3 = {1};
  std::vector<int> input_shape4 = {1};
  float input_data1[] = {1, 3, 4, 5, 6, 7};
  float input_data2[] = {10};
  float input_data3[] = {1.0};
  float input_data4[] = {2.0};
  float correctOutput[] = {2, 1, 2, 1, 1, 1, 1, 1, 2, 2};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = lite::Tensor::CONST_TENSOR;
  std::vector<int> output_shape = {10};
  auto in_tensor1 = Tensor(data_type, input_shape1, Format_NHWC, tensor_type);
  auto in_tensor2 = Tensor(data_type, input_shape2, Format_NHWC, tensor_type);
  auto in_tensor3 = Tensor(data_type, input_shape3, Format_NHWC, lite::Tensor::CONST_SCALAR);
  auto in_tensor4 = Tensor(data_type, input_shape4, Format_NHWC, tensor_type);
  auto output_tensor = Tensor(data_type, output_shape, Format_NHWC, tensor_type);
  // allocate memory for weights
  in_tensor2.MallocData();
  in_tensor3.MallocData();
  in_tensor4.MallocData();
  std::vector<lite::Tensor *> inputs{&in_tensor1, &in_tensor2, &in_tensor3, &in_tensor4};
  std::vector<lite::Tensor *> outputs{&output_tensor};
  // initialize weights
  memcpy(inputs[1]->data_c(), input_data2, sizeof(input_data2));
  memcpy(inputs[2]->data_c(), input_data3, sizeof(input_data3));
  memcpy(inputs[3]->data_c(), input_data4, sizeof(input_data4));
  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<SparseToDenseParameter *>(malloc(sizeof(SparseToDenseParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new ActivationParameter failed ";
    return;
  }

  auto *sparse_to_dense_kernel =
    new (std::nothrow) kernel::SparseToDenseOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (sparse_to_dense_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::SparseToDenseOpenCLKernel failed ";
    delete param;
    return;
  }
  sparse_to_dense_kernel->Init();
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{sparse_to_dense_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel({&in_tensor1}, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    delete param;
    delete sparse_to_dense_kernel;
    return;
  }
  // to do allocate memory for inputs
  in_tensor1.MallocData(allocator);
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data1, sizeof(input_data1));

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor.data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correctOutput, output_tensor.ElementsNum(), 0.0001));
  delete sub_graph;
}

TEST_F(TestSparseToDenseOpenCLCI, Fp32Dim1Vector) {
  MS_LOG(INFO) << " begin test ";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto runtime = runtime_wrapper.GetInstance();
  runtime->Init();
  auto allocator = runtime->GetAllocator();
  MS_LOG(INFO) << " init tensors ";
  std::vector<int> input_shape1 = {6};
  std::vector<int> input_shape2 = {1};
  std::vector<int> input_shape3 = {6};
  std::vector<int> input_shape4 = {1};
  float input_data1[] = {1, 3, 4, 5, 6, 7};
  float input_data2[] = {10};
  float input_data3[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  float input_data4[] = {2.0};
  float correctOutput[] = {2, 1, 2, 2, 3, 4, 5, 6, 2, 2};
  auto data_type = kNumberTypeFloat32;
  auto tensor_type = lite::Tensor::CONST_TENSOR;
  std::vector<int> output_shape = {10};
  auto in_tensor1 = Tensor(data_type, input_shape1, Format_NHWC, tensor_type);
  auto in_tensor2 = Tensor(data_type, input_shape2, Format_NHWC, tensor_type);
  auto in_tensor3 = Tensor(data_type, input_shape3, Format_NHWC, tensor_type);
  auto in_tensor4 = Tensor(data_type, input_shape4, Format_NHWC, tensor_type);
  auto output_tensor = Tensor(data_type, output_shape, Format_NHWC, tensor_type);
  // allocate memory for weights
  in_tensor2.MallocData();
  in_tensor3.MallocData();
  in_tensor4.MallocData();
  std::vector<lite::Tensor *> inputs{&in_tensor1, &in_tensor2, &in_tensor3, &in_tensor4};
  std::vector<lite::Tensor *> outputs{&output_tensor};
  // initialize weights
  memcpy(inputs[1]->data_c(), input_data2, sizeof(input_data2));
  memcpy(inputs[2]->data_c(), input_data3, sizeof(input_data3));
  memcpy(inputs[3]->data_c(), input_data4, sizeof(input_data4));
  MS_LOG(INFO) << " initialize tensors ";
  auto param = reinterpret_cast<SparseToDenseParameter *>(malloc(sizeof(SparseToDenseParameter)));
  if (param == nullptr) {
    MS_LOG(INFO) << " new ActivationParameter failed ";
    return;
  }

  auto *sparse_to_dense_kernel =
    new (std::nothrow) kernel::SparseToDenseOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  if (sparse_to_dense_kernel == nullptr) {
    MS_LOG(INFO) << " new kernel::SparseToDenseOpenCLKernel failed ";
    delete param;
    return;
  }
  sparse_to_dense_kernel->Init();
  MS_LOG(INFO) << " initialize sub_graph ";
  std::vector<kernel::LiteKernel *> kernels{sparse_to_dense_kernel};
  auto *sub_graph = new (std::nothrow) kernel::SubGraphOpenCLKernel({&in_tensor1}, outputs, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << " new kernel::SubGraphOpenCLKernel failed ";
    delete param;
    delete sparse_to_dense_kernel;
    return;
  }
  // to do allocate memory for inputs
  in_tensor1.MallocData(allocator);
  sub_graph->Init();
  MS_LOG(INFO) << " initialize input data ";
  memcpy(inputs[0]->data_c(), input_data1, sizeof(input_data1));

  std::cout << "==================output data================" << std::endl;
  sub_graph->Run();
  auto *output_data_gpu = reinterpret_cast<float *>(output_tensor.data_c());
  ASSERT_EQ(0, CompareOutputData(output_data_gpu, correctOutput, output_tensor.ElementsNum(), 0.0001));
  delete sub_graph;
}

}  // namespace mindspore
