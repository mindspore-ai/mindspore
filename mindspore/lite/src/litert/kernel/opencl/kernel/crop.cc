
/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/opencl/kernel/crop.h"
#include <map>
#include <string>
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/opencl/utils.h"
#include "src/litert/kernel/opencl/cl/crop.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Crop;

namespace mindspore::kernel {
namespace {
const std::map<int, std::string> CROP_SUPPORT_DTYPES = {
  {kNumberTypeFloat32, "fp32"},
  {kNumberTypeFloat16, "fp16"},
  {kNumberTypeInt32, "int32"},
};
}

int CropOpenCLKernel::CheckSpecsWithoutShape() {
  auto input_dtype = in_tensors_.front()->data_type();
  if (CROP_SUPPORT_DTYPES.find(input_dtype) == CROP_SUPPORT_DTYPES.end()) {
    MS_LOG(WARNING) << "input dtype must be float32/float16/int32";
    return RET_ERROR;
  }

  auto output_dtype = out_tensors_.front()->data_type();
  if (CROP_SUPPORT_DTYPES.find(output_dtype) == CROP_SUPPORT_DTYPES.end()) {
    MS_LOG(WARNING) << "output dtype must be float32/float16/int32";
    return RET_ERROR;
  }
  return RET_OK;
}

int CropOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }

  auto input_tensor = in_tensors_.front();
  if ((input_tensor->Channel() % C4NUM) != 0) {
    MS_LOG(WARNING) << "input channel must can be divided by 4";
    return RET_ERROR;
  }

  auto output_tensor = out_tensors_.front();
  if ((output_tensor->Channel() % C4NUM) != 0) {
    MS_LOG(WARNING) << "output channel must can be divided by 4";
    return RET_ERROR;
  }

  return RET_OK;
}

int CropOpenCLKernel::Prepare() {
  out_gpu_info_ = GpuTensorInfo(out_tensors_[0]);

  const std::string program_name = "crop_program";
  const std::string kernel_name = "crop";
  auto build_option_ext = CreateBuildOptionsExtByDType(this->registry_data_type());
  if (!ocl_runtime_->LoadSource(program_name, crop_source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_option_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  (void)SetGlobalLocal();
  return RET_OK;
}

void CropOpenCLKernel::RightShiftOffsetByAxis() {
  bzero(offset_, sizeof(int) * COMM_SHAPE_SIZE);
  for (int i = 0; i < crop_param_->offset_size_; i++) {
    int index = i + crop_param_->axis_;
    if ((index < 0) || (index >= COMM_SHAPE_SIZE)) {
      continue;
    }
    offset_[index] = crop_param_->offset_[i];
  }
}

int CropOpenCLKernel::SetConstArgs() {
  auto out_tensor = out_tensors_[0];
  cl_int4 cl_out_shape = {static_cast<int>(out_tensor->Batch()), static_cast<int>(out_tensor->Height()),
                          static_cast<int>(out_tensor->Width()), static_cast<int>(out_tensor->Channel())};
  auto in_tensor = in_tensors_[0];
  cl_int4 cl_in_shape = {static_cast<int>(in_tensor->Batch()), static_cast<int>(in_tensor->Height()),
                         static_cast<int>(in_tensor->Width()), static_cast<int>(in_tensor->Channel())};
  RightShiftOffsetByAxis();
  cl_int4 cl_offset = {offset_[0], offset_[1], offset_[2], offset_[3]};

  if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX2, cl_in_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "Set cl arg: in_shape failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX3, cl_out_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "Set cl arg: out_shape failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX4, cl_offset) != CL_SUCCESS) {
    MS_LOG(ERROR) << "Set cl arg: offset failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int CropOpenCLKernel::SetGlobalLocal() {
  global_size_ = {out_gpu_info_.width, out_gpu_info_.height};
  OpenCLKernel::AlignGlobalLocal(global_size_, {});
  return RET_OK;
}

int CropOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX0, in_tensors_.front()->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX1, out_tensors_.front()->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Crop, OpenCLKernelCreator<CropOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Crop, OpenCLKernelCreator<CropOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeInt32, PrimitiveType_Crop, OpenCLKernelCreator<CropOpenCLKernel>);
}  // namespace mindspore::kernel
