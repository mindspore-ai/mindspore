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

#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/scale.h"

namespace mindspore {

template <class T>
static void BoardcaseScale(const T *in, const T scale, const T offset, T *out, const int size) {
  for (int i = 0; i < size; i++) {
    out[i] = in[i] * scale + offset;
  }
}

template <class T>
static void Scale(const T *in, const T *scale, T *offset, T *out, const int size) {
  for (int i = 0; i < size; i++) {
    out[i] = in[i] * scale[i] + offset[i];
  }
}

template <class T>
static bool DataCompare(const T *a, const T *b, const int size, const T accuracy = 1e-4) {
  for (int i = 0; i < size; i++) {
    auto diff = fabs(a[i] - b[i]);
    if (diff > accuracy) {
      MS_LOG(ERROR) << "compare failed at " << i << " exp " << a[i] << " bug got " << b[i];
      return false;
    }
  }
  return true;
}

template <class T>
static void InitData(void *data, const int size) {
  T *data_float = reinterpret_cast<T *>(data);
  static unsigned int seed = 123;
  for (int i = 0; i < size; i++) {
    data_float[i] = static_cast<int>(rand_r(&seed)) % 100;
  }
}

template <class T>
static void LogData(void *data, const int size, const std::string prefix) {
  std::cout << prefix;
  T *data_float = reinterpret_cast<T *>(data);
  for (int i = 0; i < size; i++) {
    std::cout << data_float[i] << ",";
  }
  std::cout << std::endl;
}

template <class T>
static void TestCase(const std::vector<int> &shape_a, const std::vector<int> &shape_b) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto allocator = ocl_runtime->GetAllocator();

  bool is_broadcast = shape_b.empty();
  auto format = schema::Format_NHWC4;

  auto data_type = kNumberTypeFloat32;
  if (sizeof(T) == 2) {
    data_type = kNumberTypeFloat16;
    ocl_runtime->SetFp16Enable(true);
  }
  lite::Tensor *tensor_in = new (std::nothrow) lite::Tensor(data_type, shape_a, format);
  lite::Tensor *tensor_scale = new (std::nothrow) lite::Tensor(data_type, shape_b, format);
  lite::Tensor *tensor_offset = new (std::nothrow) lite::Tensor(data_type, shape_b, format);
  lite::Tensor *tensor_out = new (std::nothrow) lite::Tensor(data_type, shape_a, format);
  if (tensor_in == nullptr || tensor_scale == nullptr || tensor_offset == nullptr) {
    MS_LOG(ERROR) << "Create tensor failed!";
    delete tensor_in;
    delete tensor_scale;
    delete tensor_offset;
    delete tensor_out;
    return;
  }

  int64_t element_num = tensor_in->ElementsC4Num();
  int64_t element_num_b = is_broadcast ? 1 : tensor_scale->ElementsC4Num();

  T *data_in = new (std::nothrow) T[element_num];
  T *data_scale = new (std::nothrow) T[element_num_b];
  T *data_offset = new (std::nothrow) T[element_num_b];
  T *data_out_cpu = new (std::nothrow) T[element_num];
  T *data_out_ocl = new (std::nothrow) T[element_num];
  if (data_in == nullptr || data_scale == nullptr || data_out_cpu == nullptr || data_out_ocl == nullptr) {
    MS_LOG(ERROR) << "Create buffer failed!";
    delete tensor_in;
    delete tensor_scale;
    delete tensor_offset;
    delete tensor_out;
    delete[] data_in;
    delete[] data_scale;
    delete[] data_offset;
    delete[] data_out_cpu;
    delete[] data_out_ocl;
    return;
  }

  InitData<T>(data_in, element_num);
  InitData<T>(data_scale, element_num_b);
  InitData<T>(data_offset, element_num_b);
  memset(data_out_ocl, 0, sizeof(T) * element_num);

  if (is_broadcast) {
    BoardcaseScale(data_in, static_cast<T *>(data_scale)[0], static_cast<T *>(data_offset)[0], data_out_cpu,
                   element_num);
  } else {
    Scale(data_in, data_scale, data_offset, data_out_cpu, element_num);
  }

  std::vector<lite::Tensor *> inputs = {tensor_in};
  if (!is_broadcast) {
    inputs.push_back(tensor_scale);
    inputs.push_back(tensor_offset);
  } else {
    tensor_scale->MallocData();
    tensor_offset->MallocData();
    memcpy(tensor_scale->data_c(), data_scale, sizeof(T));
    memcpy(tensor_offset->data_c(), data_offset, sizeof(T));
  }
  std::vector<lite::Tensor *> outputs = {tensor_out};

  ScaleParameter *param = new (std::nothrow) ScaleParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "Create parameter failed!";
    delete tensor_in;
    delete tensor_scale;
    delete tensor_offset;
    delete tensor_out;
    delete[] data_in;
    delete[] data_scale;
    delete[] data_offset;
    delete[] data_out_cpu;
    delete[] data_out_ocl;
    return;
  }
  param->axis_ = 0;
  param->op_parameter_.type_ = schema::PrimitiveType_Scale;

  std::vector<lite::Tensor *> scale_inputs = {tensor_in, tensor_scale, tensor_offset};
  lite::Context ctx;
  auto *scale_kernel =
    new (std::nothrow) kernel::ScaleOpenCLKernel(reinterpret_cast<OpParameter *>(param), scale_inputs, outputs, &ctx);
  if (scale_kernel == nullptr) {
    MS_LOG(ERROR) << "Create ScaleOpenCLKernel failed!";
    delete tensor_in;
    delete tensor_scale;
    delete tensor_offset;
    delete tensor_out;
    delete[] data_in;
    delete[] data_scale;
    delete[] data_offset;
    delete[] data_out_cpu;
    delete[] data_out_ocl;
    delete param;
    return;
  }
  scale_kernel->Init();

  tensor_in->MallocData(allocator);
  tensor_scale->MallocData(allocator);
  tensor_offset->MallocData(allocator);
  std::vector<kernel::LiteKernel *> kernels{scale_kernel};
  auto *kernel = new (std::nothrow) kernel::SubGraphOpenCLKernel(inputs, outputs, kernels, kernels, kernels);
  if (scale_kernel == nullptr) {
    MS_LOG(ERROR) << "Create SubGraphOpenCLKernel failed!";
    delete tensor_in;
    delete tensor_scale;
    delete tensor_offset;
    delete tensor_out;
    delete[] data_in;
    delete[] data_scale;
    delete[] data_offset;
    delete[] data_out_cpu;
    delete[] data_out_ocl;
    delete scale_kernel;
    return;
  }
  kernel->Init();

  memcpy(inputs[0]->data_c(), data_in, sizeof(T) * element_num);
  if (!is_broadcast) {
    memcpy(inputs[1]->data_c(), data_scale, sizeof(T) * element_num_b);
    memcpy(inputs[2]->data_c(), data_offset, sizeof(T) * element_num_b);
  }

  kernel->Run();

  memcpy(data_out_ocl, outputs[0]->data_c(), sizeof(T) * element_num);

  LogData<T>(data_in, 10, "Data input : ");
  LogData<T>(data_scale, tensor_scale->shape().empty() ? 1 : 10, "Data scale : ");
  LogData<T>(data_offset, tensor_offset->shape().empty() ? 1 : 10, "Data offset : ");
  LogData<T>(data_out_cpu, 10, "Expect compute : ");
  LogData<T>(outputs[0]->data_c(), 10, "OpenCL compute : ");
  bool cmp = DataCompare(data_out_cpu, data_out_ocl, element_num);
  MS_LOG(INFO) << "Compare " << (cmp ? "success!" : "failed!");
  EXPECT_EQ(true, cmp);

  // free
  delete[] data_in;
  delete[] data_scale;
  delete[] data_offset;
  delete[] data_out_cpu;
  delete[] data_out_ocl;

  delete kernel;
  delete scale_kernel;
  delete param;
  for (auto tensor : inputs) {
    delete tensor;
  }
  for (auto tensor : outputs) {
    delete tensor;
  }
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

class TestScaleOpenCL : public mindspore::CommonTest {
 public:
  TestScaleOpenCL() {}
};

TEST_F(TestScaleOpenCL, ElementFP32) {
  const std::vector<int> &shape_a = {1, 1024, 1024, 4};
  const std::vector<int> &shape_b = {1, 1024, 1024, 4};
  TestCase<float>(shape_a, shape_b);
}

TEST_F(TestScaleOpenCL, BroadcastFP32) {
  const std::vector<int> &shape_a = {1, 128, 128, 4};
  const std::vector<int> &shape_b = {};
  TestCase<float>(shape_a, shape_b);
}

TEST_F(TestScaleOpenCL, ElementFP16) {
  const std::vector<int> &shape_a = {1, 1024, 1024, 4};
  const std::vector<int> &shape_b = {1, 1024, 1024, 4};
  TestCase<float16_t>(shape_a, shape_b);
}

TEST_F(TestScaleOpenCL, BroadcastFP16) {
  const std::vector<int> &shape_a = {1, 128, 128, 4};
  const std::vector<int> &shape_b = {};
  TestCase<float16_t>(shape_a, shape_b);
}
}  // namespace mindspore
