/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/opencl/kernel/fill.h"
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Fill;
using mindspore::schema::PrimitiveType_Shape;

namespace mindspore::kernel {

int FillOpenCLKernel::RunFill() {
  auto allocator_ = ocl_runtime_->GetAllocator();
  auto param = reinterpret_cast<FillParameter *>(this->op_parameter_);
  default_ = param->num_dims_;
  std::vector<size_t> img_size;
  cl_float4 fill_value = {};
  fill_value.s[0] = fill_value.s[1] = fill_value.s[2] = fill_value.s[3] = default_;
  auto src_data = out_tensors_[0]->data_c();
  allocator_->GetImageSize(src_data, &img_size);
  auto src_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
  auto region = cl::array<cl::size_type, 3U>{img_size[0], img_size[1], 1};
  cl::Image2D *out_image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(src_data));
  ocl_runtime_->GetDefaultCommandQueue()->enqueueFillImage(*out_image, fill_value, src_origin, region);
  return RET_OK;
}

int FillOpenCLKernel::RunShape() {
  auto allocator_ = ocl_runtime_->GetAllocator();
  auto src_data = out_tensors_[0]->data_c();
  cl_float4 fill_value = {default_, default_, default_, default_};
  for (int i = 0; i < in_tensors_[0]->shape().size(); ++i) {
    fill_value.s[0] = in_tensors_[0]->shape()[i];
    size_t index = static_cast<size_t>(i);
    auto src_origin = cl::array<cl::size_type, 3U>{0, index, 0};
    auto region = cl::array<cl::size_type, 3U>{1, 1, 1};
    cl::Image2D *out_image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(src_data));
    ocl_runtime_->GetDefaultCommandQueue()->enqueueFillImage(*out_image, fill_value, src_origin, region);
  }
  return RET_OK;
}

int FillOpenCLKernel::Init() {
  auto param = this->op_parameter_;

  if (out_tensors_[0]->shape().size() > 4) {
    MS_LOG(ERROR) << " only support dim <= 4";
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape().size() > 1 && param->type_ == PrimitiveType_Fill) {
    MS_LOG(ERROR) << " fill only support dim = 1";
    return RET_ERROR;
  }
  return RET_OK;
}

int FillOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  auto param = this->op_parameter_;
  if (param->type_ == PrimitiveType_Fill) {
    RunFill();
  } else {
    RunShape();
  }

  return RET_OK;
}

kernel::LiteKernel *FillOpenCLKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                            const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                            const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) FillOpenCLKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << " new FillOpenCLKernel failed ";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << " Init kernel failed, name: fill ";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Fill, FillOpenCLKernelCreator);
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Shape, FillOpenCLKernelCreator);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Fill, FillOpenCLKernelCreator);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Shape, FillOpenCLKernelCreator);

}  // namespace mindspore::kernel
