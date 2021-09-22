/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <arm_neon.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "src/custom_common.h"
#include "include/errorcode.h"
#include "include/registry/register_kernel_interface.h"
#include "include/registry/register_kernel.h"
#include "include/registry/opencl_runtime_wrapper.h"
#include "src/cl/arithmetic.cl.inc"
#include "include/api/data_type.h"
#include "include/schema/ops_generated.h"

#define UP_ROUND(x, y) (((x) + (y) - (1)) / (y) * (y))

namespace mindspore {
namespace custom_gpu_demo {
class CustomAddKernelGpu : public kernel::Kernel {
 public:
  CustomAddKernelGpu(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                     const schema::Primitive *primitive, const mindspore::Context *ctx,
                     const std::string &build_options, bool fp16_enable)
      : Kernel(inputs, outputs, primitive, ctx), build_options_(build_options), fp16_enable_(fp16_enable) {}
  ~CustomAddKernelGpu() override { FreeWeight(); }
  // Prepare will be called during graph compilation
  int Prepare() override {
    const std::string kernel_name_ = "ElementAdd";
    const std::string program_name = "Arithmetic";
    std::string source = arithmetic_source;
    if (opencl_runtime_.LoadSource(program_name, source) != kSuccess) {
      std::cerr << "Load source failed.";
      return lite::RET_ERROR;
    }
    std::vector<std::string> build_options_ext = {"-cl-mad-enable -cl-fast-relaxed-math -Werror"};

    build_options_ext.push_back(build_options_);
    if (opencl_runtime_.BuildKernel(&kernel_, program_name, kernel_name_, build_options_ext) != kSuccess) {
      std::cerr << "Build kernel failed.";
      return lite::RET_ERROR;
    }

    auto out_shape = custom_common::GpuTensorInfo(&outputs_[0], &opencl_runtime_);
    local_range_ = cl::NullRange;
    global_range_ = cl::NDRange(out_shape.width, out_shape.height);
    for (int i = 0; i < inputs_.size(); ++i) {
      auto &in_tensor = inputs_.at(i);
      custom_common::GpuTensorInfo in_shape = custom_common::GpuTensorInfo(&in_tensor, &opencl_runtime_);
      if (in_tensor.IsConst()) {
        std::vector<char> weight(in_shape.Image2DSize, 0);
        bool src_is_fp16 = in_tensor.DataType() == mindspore::DataType::kNumberTypeFloat16;
        PackNHWCToNHWC4(in_tensor.MutableData(), weight.data(), src_is_fp16, fp16_enable_, in_shape,
                        in_tensor.DataType());
        DataType dtype =
          fp16_enable_ ? mindspore::DataType::kNumberTypeFloat16 : mindspore::DataType::kNumberTypeFloat32;
        auto allocator = opencl_runtime_.GetAllocator();
        if (allocator == nullptr) {
          std::cerr << "GetAllocator fail.";
          FreeWeight();
          return lite::RET_ERROR;
        }
        auto weight_ptr = allocator->Malloc(in_shape.width, in_shape.height, dtype);
        if (weight_ptr == nullptr) {
          std::cerr << "Malloc fail.";
          FreeWeight();
          return lite::RET_ERROR;
        }
        weight_ptrs_.push_back(weight_ptr);
        // Use API to write GPU memory
        if (opencl_runtime_.WriteImage(weight_ptr, weight.data()) != kSuccess) {
          std::cerr << "WriteImage fail.";
          FreeWeight();
          return lite::RET_ERROR;
        }
      } else {
        weight_ptrs_.push_back(nullptr);
      }
    }

    int arg_idx = 3;
    cl_int2 output_shape{static_cast<int>(global_range_[0]), static_cast<int>(global_range_[1])};
    if (opencl_runtime_.SetKernelArg(kernel_, arg_idx, output_shape) != kSuccess) {
      std::cerr << "Set kernel arg" << arg_idx << "failed.";
      FreeWeight();
      return lite::RET_ERROR;
    }

    std::cout << kernel_name_ << " Init Done!" << std::endl;
    return lite::RET_OK;
  }

  // Execute is called to compute.
  int Execute() override {
    if (inputs_.size() != 2) {
      return lite::RET_PARAM_INVALID;
    }
    PreProcess();
    std::cout << this->name() << " Running!" << std::endl;
    auto input_0_ptr = weight_ptrs_[0] == nullptr ? inputs_[0].MutableData() : weight_ptrs_[0];
    auto input_1_ptr = weight_ptrs_[1] == nullptr ? inputs_[1].MutableData() : weight_ptrs_[1];
    int arg_idx = 0;
    if (opencl_runtime_.SetKernelArg(kernel_, arg_idx++, input_0_ptr) != kSuccess) {
      std::cerr << "Set kernel arg" << arg_idx - 1 << "failed.";
      return lite::RET_ERROR;
    }
    if (opencl_runtime_.SetKernelArg(kernel_, arg_idx++, input_1_ptr) != kSuccess) {
      std::cerr << "Set kernel arg" << arg_idx - 1 << "failed.";
      return lite::RET_ERROR;
    }
    if (opencl_runtime_.SetKernelArg(kernel_, arg_idx++, outputs_[0].MutableData()) != kSuccess) {
      std::cerr << "Set kernel arg" << arg_idx - 1 << "failed.";
      return lite::RET_ERROR;
    }
    if (opencl_runtime_.RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != kSuccess) {
      std::cerr << "Run kernel failed.";
      return lite::RET_ERROR;
    }

    return lite::RET_OK;
  }

  int CheckSpecs() {
    for (auto &tensor : inputs_) {
      if (tensor.DataType() != DataType::kNumberTypeFloat32 && tensor.DataType() != DataType::kNumberTypeFloat16) {
        std::cerr << "ArithmeticOpenCLKernel only support fp32/fp16 input";
        return lite::RET_ERROR;
      }
    }
    for (auto &tensor : outputs_) {
      if (tensor.DataType() != DataType::kNumberTypeFloat32 && tensor.DataType() != DataType::kNumberTypeFloat16) {
        std::cerr << "ArithmeticOpenCLKernel only support fp32/fp16 output";
        return lite::RET_ERROR;
      }
    }

    if (inputs_.size() != 2 || outputs_.size() != 1) {
      std::cerr << "in size: " << inputs_.size() << ", out size: " << outputs_.size();
      return lite::RET_ERROR;
    }

    return lite::RET_OK;
  }

  // Resize is used to update some parameters if current node can change along with inputs.
  int ReSize() override {
    if (custom_common::CheckOutputs(outputs_) == lite::RET_OK) {
      return lite::RET_OK;
    }
    auto status =
      registry::RegisterKernelInterface::GetKernelInterface({}, primitive_)->Infer(&inputs_, &outputs_, primitive_);
    if (status != kSuccess) {
      std::cerr << "infer failed." << std::endl;
      return lite::RET_ERROR;
    }
    auto ret = CheckSpecs();
    if (ret != lite::RET_OK) {
      std::cerr << "ReSize failed for check kernel specs!";
      return ret;
    }
    ret = Prepare();
    if (ret != lite::RET_OK) {
      std::cerr << "ReSize failed for kernel prepare!";
      return ret;
    }
    return lite::RET_OK;
  }

 private:
  std::string build_options_;
  bool fp16_enable_;
  cl::Kernel kernel_;
  cl::Event event_;
  cl::NDRange global_range_{cl::NullRange};
  cl::NDRange local_range_{cl::NullRange};
  std::vector<void *> weight_ptrs_;
  registry::opencl::OpenCLRuntimeWrapper opencl_runtime_;

  int PreProcess() {
    int ret = 0;
    ret = ReSize();
    if (ret != lite::RET_OK) {
      return ret;
    }
    for (auto i = 0; i < outputs_.size(); ++i) {
      auto *output = &outputs_.at(i);
      auto img_info = custom_common::GpuTensorInfo(output, &opencl_runtime_);
      auto allocator = output->allocator();
      if (allocator == nullptr) {
        std::cerr << "The output tensor of OpenCL kernel must have an allocator.";
        return lite::RET_ERROR;
      }
      auto data_ptr = allocator->Malloc(img_info.width, img_info.height, output->DataType());
      if (data_ptr == nullptr) {
        std::cerr << "Malloc data failed";
        return lite::RET_ERROR;
      }
      output->SetData(data_ptr);
    }
    return lite::RET_OK;
  }

  void FreeWeight() {
    auto allocator = opencl_runtime_.GetAllocator();
    if (allocator == nullptr) {
      std::cerr << "GetAllocator fail.";
      return;
    }
    for (auto &weight_ptr : weight_ptrs_) {
      if (weight_ptr != nullptr) {
        allocator->Free(weight_ptr);
        weight_ptr = nullptr;
      }
    }
  }
};

std::shared_ptr<kernel::Kernel> CustomAddCreator(const std::vector<MSTensor> &inputs,
                                                 const std::vector<MSTensor> &outputs,
                                                 const schema::Primitive *primitive, const mindspore::Context *ctx) {
  const std::string build_options = " -DFLT4=float4 -DWRITE_IMAGE=write_imagef -DREAD_IMAGE=read_imagef ";
  bool fp16_enable = false;

  std::cout << "using fp32 add.\n" << std::endl;
  return std::make_shared<CustomAddKernelGpu>(inputs, outputs, primitive, ctx, build_options, fp16_enable);
}

std::shared_ptr<kernel::Kernel> CustomAddFP16Creator(const std::vector<MSTensor> &inputs,
                                                     const std::vector<MSTensor> &outputs,
                                                     const schema::Primitive *primitive,
                                                     const mindspore::Context *ctx) {
  const std::string build_options = " -DFLT4=half4 -DWRITE_IMAGE=write_imageh -DREAD_IMAGE=read_imageh";
  bool fp16_enable = true;

  std::cout << "using fp16 add." << std::endl;
  return std::make_shared<CustomAddKernelGpu>(inputs, outputs, primitive, ctx, build_options, fp16_enable);
}
}  // namespace custom_gpu_demo
const auto kFloat32 = DataType::kNumberTypeFloat32;
const auto kFloat16 = DataType::kNumberTypeFloat16;
// Register custom “Custom_Add” operator
REGISTER_CUSTOM_KERNEL(GPU, Tutorial, kFloat32, Custom_Add, custom_gpu_demo::CustomAddCreator)
REGISTER_CUSTOM_KERNEL(GPU, Tutorial, kFloat16, Custom_Add, custom_gpu_demo::CustomAddFP16Creator)
using schema::PrimitiveType_AddFusion;
// Register the add operator to replace the internal add operator of MindSpore Lite
REGISTER_KERNEL(GPU, Tutorial, kFloat32, PrimitiveType_AddFusion, custom_gpu_demo::CustomAddCreator)
REGISTER_KERNEL(GPU, Tutorial, kFloat16, PrimitiveType_AddFusion, custom_gpu_demo::CustomAddFP16Creator)
}  // namespace mindspore
