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

#include <string>
#include "common/common_test.h"
#include "src/kernel_registry.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

using mindspore::kernel::LiteKernel;
using mindspore::kernel::SubGraphOpenCLKernel;
using mindspore::lite::KernelRegistry;
using mindspore::lite::Tensor;
using mindspore::schema::Format::Format_NHWC;

namespace mindspore {

void LoadTestData(void *dst, size_t dst_size, const std::string &file_path) {
  if (file_path.empty()) {
    memset(dst, 0x00, dst_size);
  } else {
    auto src_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &dst_size));
    if (src_data != nullptr) {
      memcpy(dst, src_data, dst_size);
    } else {
      MS_LOG(ERROR) << "read file empty.";
    }
  }
}

void TestMain(const std::vector<std::tuple<std::vector<int>, float *, Tensor::Category>> &input_infos,
              std::tuple<std::vector<int>, float *> output_info, OpParameter *op_parameter, bool fp16_enable,
              float atol, bool print_output) {
  MS_LOG(DEBUG) << "initialize OpenCLRuntime and OpenCLAllocator";
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto ocl_runtime = runtime_wrapper.GetInstance();
  EXPECT_TRUE(ocl_runtime->Init() == RET_OK);
  ocl_runtime->SetFp16Enable(fp16_enable);
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(DEBUG) << "create Tensors & init weight data";
  std::vector<Tensor> tensors;
  std::vector<Tensor *> kernel_inputs;
  std::vector<Tensor *> subgraph_inputs;
  std::map<Tensor *, float *> subgraph_inputs_data;
  for (auto input_info : input_infos) {
    const std::vector<int> &shape = std::get<0>(input_info);
    auto *input_data = std::get<1>(input_info);
    const Tensor::Category category = std::get<2>(input_info);
    tensors.emplace_back(kNumberTypeFloat32, shape, Format_NHWC, category);
    auto *new_tensor = &tensors.back();
    kernel_inputs.push_back(new_tensor);
    if (category != Tensor::Category::VAR) {
      memcpy(new_tensor->MutableData(), input_data, new_tensor->Size());
    } else {
      subgraph_inputs.push_back(new_tensor);
      subgraph_inputs_data[new_tensor] = input_data;
    }
  }
  const std::vector<int> &output_shape = std::get<0>(output_info);
  float *expect_data = std::get<1>(output_info);
  auto output = Tensor(kNumberTypeFloat32, output_shape, Format_NHWC, Tensor::Category::VAR);

  MS_LOG(DEBUG) << "create OpenCL Kernel";
  auto primitive_type = static_cast<schema::PrimitiveType>(op_parameter->type_);
  kernel::KernelKey key{kernel::kGPU, kernel_inputs.front()->data_type(), primitive_type};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  if (creator == nullptr) {
    std::cerr << "get kernel registry function error: " << schema::EnumNamePrimitiveType(primitive_type) << std::endl;
    free(op_parameter);
    FAIL();
  }
  auto *kernel = creator(kernel_inputs, {&output}, op_parameter, nullptr, key, nullptr);
  if (kernel == nullptr) {
    std::cerr << "call kernel registry function error: " << schema::EnumNamePrimitiveType(primitive_type) << std::endl;
    free(op_parameter);
    FAIL();
  }

  MS_LOG(DEBUG) << "create SubGraph & init input data";
  std::vector<LiteKernel *> kernels{kernel};
  auto sub_graph = new (std::nothrow) SubGraphOpenCLKernel(subgraph_inputs, {&output}, kernels, kernels, kernels);
  if (sub_graph == nullptr) {
    return;
  }
  for (auto input : subgraph_inputs) {
    EXPECT_TRUE(input->MallocData(allocator) == RET_OK);
  }
  EXPECT_TRUE(sub_graph->Init() == RET_OK);
  for (auto input : subgraph_inputs) {
    memcpy(input->data_c(), subgraph_inputs_data[input], input->Size());
  }

  MS_LOG(DEBUG) << "run SubGraph & compare result";
  EXPECT_TRUE(sub_graph->Run() == RET_OK);
  if (print_output) {
    for (int i = 0; i < output.ElementsNum(); ++i) {
      printf("%d: expect=%.3f output=%.3f\n", i, expect_data[i], reinterpret_cast<float *>(output.data_c())[i]);
    }
  }
  CommonTest::CompareOutputData(reinterpret_cast<float *>(output.data_c()), expect_data, output.ElementsNum(), atol);

  MS_LOG(DEBUG) << "release resources";
  delete sub_graph;
}

}  // namespace mindspore
