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
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::schema::PrimitiveType_Fill;
using mindspore::schema::PrimitiveType_Shape;

namespace mindspore::kernel {
int FillOpenCLKernel::RunFill() {
  auto allocator_ = ocl_runtime_->GetAllocator();
  auto param = reinterpret_cast<FillParameter *>(this->op_parameter_);
  CHECK_NULL_RETURN(param);
  default_ = param->num_dims_;
  ImageSize img_size;
  cl_int4 fill_value = {};
  fill_value.s[0] = fill_value.s[1] = fill_value.s[2] = fill_value.s[3] = default_;
  auto src_data = out_tensors_[0]->data();
  CHECK_NULL_RETURN(src_data);
  if (allocator_->GetImageSize(src_data, &img_size) != RET_OK) {
    MS_LOG(ERROR) << "GetImageSize failed.";
    return RET_ERROR;
  }
  auto src_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
  auto region = cl::array<cl::size_type, 3U>{img_size.width, img_size.height, 1};
  cl::Image2D *out_image = allocator_->GetImage(src_data);
  if (ocl_runtime_->GetDefaultCommandQueue()->enqueueFillImage(*out_image, fill_value, src_origin, region) !=
      CL_SUCCESS) {
    MS_LOG(ERROR) << "enqueueFillImage failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int FillOpenCLKernel::RunShape() {
  auto allocator_ = ocl_runtime_->GetAllocator();
  CHECK_NULL_RETURN(allocator_);
  auto src_data = out_tensors_[0]->data();
  CHECK_NULL_RETURN(src_data);
  cl_int4 fill_value = {default_, default_, default_, default_};
  auto tensor_shape = in_tensors_[0]->shape();
  void *tensor_shape_data = tensor_shape.data();
  CHECK_NULL_RETURN(tensor_shape_data);
  for (int i = 0; i < tensor_shape.size(); ++i) {
    fill_value.s[i] = reinterpret_cast<int *>(tensor_shape_data)[i];
  }
  auto src_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
  auto region = cl::array<cl::size_type, 3U>{1, 1, 1};
  cl::Image2D *out_image = allocator_->GetImage(src_data);
  if (ocl_runtime_->GetDefaultCommandQueue()->enqueueFillImage(*out_image, fill_value, src_origin, region) !=
      CL_SUCCESS) {
    MS_LOG(ERROR) << "enqueueFillImage failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int FillOpenCLKernel::SetConstArgs() { return RET_OK; }

void FillOpenCLKernel::SetGlobalLocal() {}

int FillOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto param = this->op_parameter_;

  auto input = in_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  if (input->shape().size() > DIMENSION_1D && param->type_ == PrimitiveType_Fill) {
    MS_LOG(WARNING) << " fill only support dim = 1";
    return RET_ERROR;
  }
  auto output = out_tensors_.at(0);
  CHECK_NULL_RETURN(output);
  if (output->shape().size() > OUTPUT_TENSOR_SIZE_4) {
    MS_LOG(WARNING) << " only support dim <= 4";
    return RET_ERROR;
  }
  return RET_OK;
}

int FillOpenCLKernel::Prepare() { return RET_OK; }

int FillOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  auto param = this->op_parameter_;
  if (param->type_ == PrimitiveType_Fill) {
    return RunFill();
  } else {
    return RunShape();
  }

  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Fill, OpenCLKernelCreator<FillOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Shape, OpenCLKernelCreator<FillOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Fill, OpenCLKernelCreator<FillOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Shape, OpenCLKernelCreator<FillOpenCLKernel>);
}  // namespace mindspore::kernel
