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
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include <utility>
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/opencl/kernel/gather.h"
#include "src/litert/kernel/opencl/cl/gather.cl.inc"
#include "src/litert/kernel/opencl/utils.h"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::kernel {
int GatherOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_3) {
    MS_LOG(WARNING) << "GatherOpenCLKernel only supports 3 input Tensor but get " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "GatherOpenCLKernel only supports 1 output Tensor but get " << out_tensors_.size();
    return RET_ERROR;
  }

  auto input_tensor = in_tensors_.at(FIRST_INPUT);
  auto indices_tensor = in_tensors_.at(SECOND_INPUT);
  auto axis_tensor = in_tensors_.at(THIRD_INPUT);

  int input_ndim = input_tensor->shape().size();
  if (input_ndim < 0 || input_ndim > DIMENSION_4D) {
    MS_LOG(WARNING) << "GatherOpenCLKernel only supports 1-4D input Tensor but get " << input_ndim << "D.";
    return RET_ERROR;
  }

  is_fp16_enabled_ = ocl_runtime_->GetFp16Enable();
  if (!indices_tensor->IsConst() && is_fp16_enabled_) {
    MS_LOG(WARNING) << "GatherOpenCLKernel not support indices = tensor and datatype = fp16";
    return RET_ERROR;
  }
  int indices_ndim = indices_tensor->shape().size();
  if (indices_ndim > DIMENSION_1D) {
    MS_LOG(WARNING) << "GatherOpenCLKernel only supports 1D or scalar indices Tensor but get " << indices_ndim << "D.";
    return RET_ERROR;
  }
  TypeId indices_dtype = indices_tensor->data_type();
  if (indices_dtype != kNumberTypeInt32 && indices_dtype != kNumberTypeInt64 && indices_dtype != kNumberTypeFloat32 &&
      indices_dtype != kNumberTypeFloat16) {
    MS_LOG(WARNING) << "GatherOpenCLKernel only supports Int32/Int64/Float32/Float16 indices Tensor.";
    return RET_ERROR;
  }

  if (CheckParamLikeTensor("Gather", "axis", axis_tensor, kNumberTypeInt32, {1}) != RET_OK) {
    return RET_ERROR;
  }
  if (axis_tensor->data() == nullptr) {
    MS_LOG(WARNING) << "GatherOpenCLKernel need Axis.";
    return RET_ERROR;
  }
  int cpu_axis = *reinterpret_cast<int *>(axis_tensor->data());
  if (cpu_axis < 0) {
    cpu_axis += input_ndim;
  }
  if ((cpu_axis < 0) || (cpu_axis >= input_ndim)) {
    MS_LOG(WARNING) << "axis is invalid: axis=" << cpu_axis << ".";
    return RET_ERROR;
  }

  auto ret = CpuAxis2GpuAxis(input_ndim, cpu_axis, &axis_);
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "convert cpu axis to gpu axis failed";
    return RET_ERROR;
  }

  return RET_OK;
}

int GatherOpenCLKernel::SetConstArgs() {
  auto input = GpuTensorInfo(in_tensors_.front());
  auto output = GpuTensorInfo(out_tensors_.front());
  int indices_num = in_tensors_.at(SECOND_INPUT)->ElementsNum();
  cl_int4 src_size = {static_cast<cl_int>(input.W), static_cast<cl_int>(input.H), static_cast<cl_int>(input.Slice),
                      static_cast<cl_int>(input.N)};
  cl_int4 dst_size = {static_cast<cl_int>(output.W), static_cast<cl_int>(output.H), static_cast<cl_int>(output.Slice),
                      static_cast<cl_int>(output.N)};
  int arg_cnt = 3;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, src_size) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, dst_size) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, indices_num) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt, axis_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherOpenCLKernel::SetGlobalLocal() {
  auto output = GpuTensorInfo(out_tensors_.front());
  local_size_ = {1, 1, 1};
  global_size_ = {output.W, output.N * output.H, output.Slice};
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
  return RET_OK;
}

int GatherOpenCLKernel::Prepare() {
  const std::string kernel_name = "gather";
  if (in_tensors_.at(0)->shape().size() == 1 && axis_ == 0) {
    axis_ = 3;
  }
  const std::string program_name = "gather";
  if (!ocl_runtime_->LoadSource(program_name, gather_source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  if (in_tensors_.at(0)->IsConst()) {
    is_input_tensor_const_ = true;
    ret = InitConstInput();
    if (ret != RET_OK) {
      return ret;
    }
  }

  if (in_tensors_.at(1)->IsConst()) {
    is_indices_tensor_const_ = false;
    ret = InitWeights();
    if (ret != RET_OK) {
      return ret;
    }
  }
  (void)SetGlobalLocal();
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int GatherOpenCLKernel::ConvertTensorToweight() {
  auto allocator = ocl_runtime_->GetAllocator();
  auto indices_tensor = in_tensors_.at(1);
  if (allocator->MapBuffer(indices_tensor->data(), CL_MAP_WRITE, nullptr, true) == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  auto indices_num = indices_tensor->ElementsNum();
  indices_data_ =
    reinterpret_cast<int32_t *>(allocator->Malloc(sizeof(int32_t) * indices_num, lite::opencl::MemType::BUF));
  if (indices_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  if (allocator->MapBuffer(indices_data_, CL_MAP_WRITE, nullptr, true) == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  if (indices_data_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  auto data_type = indices_tensor->data_type();
  auto data = indices_tensor->data();
  MS_ASSERT(data);
  if (data_type == kNumberTypeInt32) {
    for (int i = 0; i < indices_num; i++) {
      indices_data_[i] = reinterpret_cast<int32_t *>(data)[i];
    }
  } else {
    MS_LOG(ERROR) << "Gather Only supported The DataType Of Intensor1 is Int32  "
                  << " But Your type is :" << data_type;
    return RET_ERROR;
  }
  if (allocator->UnmapBuffer(indices_data_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  if (allocator->UnmapBuffer(indices_tensor->data()) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherOpenCLKernel::InitConstInput() {
  auto input_tensor = in_tensors_.at(0);

  auto allocator = ocl_runtime_->GetAllocator();
  auto fp16_enable = ocl_runtime_->GetFp16Enable();
  if (input_tensor->data_type() == kNumberTypeFloat32 || input_tensor->data_type() == kNumberTypeFloat16) {
    size_t FLT_size = fp16_enable ? sizeof(cl_half) : sizeof(cl_float);
    size_t FLT_type = fp16_enable ? CL_HALF_FLOAT : CL_FLOAT;

    GpuTensorInfo input_shape = GpuTensorInfo(input_tensor);
    std::vector<char> input_data(input_shape.ElementsC4Num * FLT_size, 0);
    bool input_dtype_flag = input_tensor->data_type() == kNumberTypeFloat16;

    if (input_tensor->format() == mindspore::NHWC) {
      PackNHWCToNHWC4(input_tensor->data(), input_data.data(), input_dtype_flag, fp16_enable, input_shape);
    } else if (input_tensor->format() == mindspore::NCHW) {
      PackNCHWToNHWC4(input_tensor->data(), input_data.data(), input_dtype_flag, fp16_enable, input_shape);
    } else {
      MS_LOG(ERROR) << "Unsupported format : " << input_tensor->format();
      return RET_ERROR;
    }

    ImageSize input_img_size{input_shape.width, input_shape.height, FLT_type};
    input_data_ = allocator->Malloc(input_img_size, input_data.data());
    if (input_data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "Unsupported data type : " << input_tensor->data_type();
    return RET_ERROR;
  }
  return RET_OK;
}

#ifdef ENABLE_FP16
int GatherOpenCLKernel::InitWeights() {
  auto indices_tensor = in_tensors_.at(1);
  auto indices_num = indices_tensor->ElementsNum();
  auto allocator = ocl_runtime_->GetAllocator();
  indices_data_ =
    reinterpret_cast<int32_t *>(allocator->Malloc(sizeof(int32_t) * indices_num, lite::opencl::MemType::BUF));
  if (indices_data_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }

  auto data_type = indices_tensor->data_type();
  auto data = indices_tensor->data();
  MS_ASSERT(data);
  if (data_type == kNumberTypeInt32) {
    for (int i = 0; i < indices_num; i++) {
      indices_data_[i] = reinterpret_cast<int32_t *>(data)[i];
    }
  } else if (data_type == kNumberTypeInt64) {
    for (int i = 0; i < indices_num; i++) {
      indices_data_[i] = reinterpret_cast<int64_t *>(data)[i];
    }
  } else if (data_type == kNumberTypeFloat32) {
    for (int i = 0; i < indices_num; i++) {
      indices_data_[i] = reinterpret_cast<float *>(data)[i];
    }
  } else if (data_type == kNumberTypeFloat16) {
    for (int i = 0; i < indices_num; i++) {
      indices_data_[i] = reinterpret_cast<float16_t *>(data)[i];
    }
  }
  return RET_OK;
}
#else
int GatherOpenCLKernel::InitWeights() {
  auto indices_tensor = in_tensors_.at(1);
  auto indices_num = indices_tensor->ElementsNum();
  auto allocator = ocl_runtime_->GetAllocator();
  indices_data_ =
    reinterpret_cast<int32_t *>(allocator->Malloc(sizeof(int32_t) * indices_num, lite::opencl::MemType::BUF));
  if (indices_data_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }

  auto data_type = indices_tensor->data_type();
  auto data = indices_tensor->data();
  MS_ASSERT(data);
  if (data_type == kNumberTypeInt32) {
    for (int i = 0; i < indices_num; i++) {
      indices_data_[i] = reinterpret_cast<int32_t *>(data)[i];
    }
  } else if (data_type == kNumberTypeInt64) {
    for (int i = 0; i < indices_num; i++) {
      indices_data_[i] = reinterpret_cast<int64_t *>(data)[i];
    }
  } else if (data_type == kNumberTypeFloat32) {
    for (int i = 0; i < indices_num; i++) {
      indices_data_[i] = reinterpret_cast<float *>(data)[i];
    }
  }
  return RET_OK;
}
#endif

int GatherOpenCLKernel::PreProcess() {
  if (!InferShapeDone()) {
    auto indices_tensor = in_tensors_[1];
    if (!indices_tensor->IsConst()) {
      if (!ocl_runtime_->SyncCommandQueue()) {
        MS_LOG(ERROR) << "SyncCommandQueue failed.";
        return RET_ERROR;
      }
      indices_tensor->MutableData();
    }
  }
  return OpenCLKernel::PreProcess();
}

int GatherOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  if (!is_input_tensor_const_) {
    input_data_ = in_tensors_.front()->data();
  }

  if (is_indices_tensor_const_) {
    int ret = ConvertTensorToweight();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ConvertTensorToweight failed.";
      return ret;
    }
  }
  if (ocl_runtime_->SetKernelArg(kernel_, 0, out_tensors_.front()->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, 1, input_data_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, C2NUM, indices_data_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Gather, OpenCLKernelCreator<GatherOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Gather, OpenCLKernelCreator<GatherOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeInt32, PrimitiveType_Gather, OpenCLKernelCreator<GatherOpenCLKernel>);
}  // namespace mindspore::kernel
