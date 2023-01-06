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

#include <set>
#include <string>
#include <map>
#include "nnacl/fp32/common_func_fp32.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/opencl/kernel/fullconnection.h"
#include "src/litert/kernel/opencl/utils.h"
#include "src/litert/kernel/opencl/cl/fullconnection.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_TANH;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {
int FullConnectionOpenCLKernel::CheckSpecs() {
  if ((in_tensors_.size() != INPUT_TENSOR_SIZE_2 && in_tensors_.size() != INPUT_TENSOR_SIZE_3) ||
      out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  if (param->a_transpose_) {
    MS_LOG(WARNING) << "fullconnection only support a_transpose_=false yet.";
    return RET_ERROR;
  }
  auto out_gpu_info = GpuTensorInfo(out_tensors_[0]);
  if (out_gpu_info.H != 1 || out_gpu_info.W != 1) {
    MS_LOG(WARNING) << "fullconnection only support 2d output shape or 4d output but H=W=1";
    return RET_ERROR;
  }
  // for fusion: ActivationType_TANH
  if (param->act_type_ != ActType_No && param->act_type_ != ActType_Relu && param->act_type_ != ActType_Relu6 &&
      param->act_type_ != ActType_Tanh && param->act_type_ != ActType_Sigmoid) {
    MS_LOG(WARNING) << "Unsupported activation type " << param->act_type_;
    return RET_ERROR;
  }
  N_ = out_gpu_info.N;
  CO_ = out_gpu_info.C;
  auto intensor_shape = GpuTensorInfo(in_tensors_[0]);
  int input_nhw = intensor_shape.N * intensor_shape.H * intensor_shape.W;
  if (input_nhw < N_) {
    MS_LOG(WARNING) << "Unsupported fullconnection shape";
  }
  if (!in_tensors_.at(kWeightIndex)->IsConst()) {
    weight_var_ = true;
    if (!param->b_transpose_) {
      MS_LOG(WARNING) << "If fullconnection input weight is not constant, b_transpose_ should be true.";
      return RET_ERROR;
    }
    if (in_tensors_.at(kWeightIndex)->shape().size() != DIMENSION_2D) {
      MS_LOG(WARNING) << "If fullconnection input weight is not constant, it should be 2d.";
      return RET_ERROR;
    }
    if (static_cast<int>(intensor_shape.C) != in_tensors_.at(kWeightIndex)->shape()[1]) {
      MS_LOG(WARNING) << "input weight is not constant, input channel should equal to weight in_channel.";
      return RET_ERROR;
    }
  }
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3 && !in_tensors_.at(DIMENSION_2D)->IsConst()) {
    MS_LOG(WARNING) << "FullConnection don't support non-constant bias yet.";
    return RET_ERROR;
  }
  CI_remainder_ = input_nhw / N_;
  return RET_OK;
}

int FullConnectionOpenCLKernel::Prepare() {
  auto param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  transposeA = param->a_transpose_;
  transposeB = param->b_transpose_;
  enable_fp16_ = ocl_runtime_->GetFp16Enable();

  std::string kernel_name = "FullConnection";
  if (weight_var_) {
    kernel_name = "FullConnectionWeightVar";
  }
  std::string source = fullconnection_source;
  const std::string program_name = "FullConnection";
  if (!ocl_runtime_->LoadSource(program_name, GetActDefines() + source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  ret = InitWeights();
  if (ret != RET_OK) {
    return ret;
  }
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  (void)SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int FullConnectionOpenCLKernel::InitWeights() {
  if (!weight_var_) {
    auto ret = InitFilter();
    if (ret != RET_OK) {
      return ret;
    }
  }
  return InitBias();
}  // namespace mindspore::kernel

#ifdef ENABLE_FP16
int FullConnectionOpenCLKernel::InitFilter() {
  auto allocator = ocl_runtime_->GetAllocator();
  auto intensor_shape = GpuTensorInfo(in_tensors_[0]);
  int co4 = UP_DIV(CO_, C4NUM);
  int nhw_remainder = intensor_shape.N * intensor_shape.H * intensor_shape.W / N_;
  size_t dtype_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  padWeight_ = allocator->Malloc(nhw_remainder * intensor_shape.Slice * co4 * C4NUM * C4NUM * dtype_size,
                                 lite::opencl::MemType::BUF);
  if (padWeight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  padWeight_ = allocator->MapBuffer(padWeight_, CL_MAP_WRITE, nullptr, true);
  if (padWeight_ == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  auto padWeightFp32 = reinterpret_cast<float *>(padWeight_);
  auto padWeightFp16 = reinterpret_cast<float16_t *>(padWeight_);
  memset(padWeight_, 0x00, nhw_remainder * intensor_shape.Slice * co4 * C4NUM * C4NUM * dtype_size);
  void *src_data = stored_weight_ == nullptr ? in_tensors_.at(kWeightIndex)->data() : stored_weight_;
  MS_ASSERT(src_data);
  auto originWeightFp32 = reinterpret_cast<float *>(src_data);
  auto originWeightFp16 = reinterpret_cast<float16_t *>(src_data);
  bool isModelFp16 = in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat16;

  // pad weight
  // HWCICO -> (HWCI4)(CO4)(4 from CO)(4 from CI)
  // if tranposeB, COHWCI -> (HWCI4)(CO4)(4 from CO)(4 from CI)
  int index = 0;
  for (int nhw = 0; nhw < nhw_remainder; nhw++) {
    for (size_t i = 0; i < intensor_shape.Slice; ++i) {
      for (int j = 0; j < co4; ++j) {
        for (int k = 0; k < C4NUM; ++k) {
          for (int l = 0; l < C4NUM; ++l) {
            size_t src_ci = i * C4NUM + l;
            int src_co = j * C4NUM + k;
            if (src_ci < intensor_shape.C && src_co < CO_) {
              int originId = (nhw * intensor_shape.C + src_ci) * CO_ + src_co;
              if (transposeB) {
                originId = src_co * intensor_shape.C * nhw_remainder + nhw * intensor_shape.C + src_ci;
              }
              if (enable_fp16_) {
                if (!isModelFp16) {
                  padWeightFp16[index++] = originWeightFp32[originId];
                } else {
                  padWeightFp16[index++] = originWeightFp16[originId];
                }
              } else {
                if (!isModelFp16) {
                  padWeightFp32[index++] = originWeightFp32[originId];
                } else {
                  padWeightFp32[index++] = originWeightFp16[originId];
                }
              }
            } else {
              index++;
            }
          }
        }
      }
    }
  }
  if (allocator->UnmapBuffer(padWeight_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  FreeStoredData(stored_weight_);
  return RET_OK;
}

int FullConnectionOpenCLKernel::InitBias() {
  // pad FC Bias
  auto allocator = ocl_runtime_->GetAllocator();
  int co4 = UP_DIV(CO_, C4NUM);
  size_t dtype_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  size_t im_dst_x, im_dst_y;
  im_dst_x = co4;
  im_dst_y = 1;
  size_t img_dtype = CL_FLOAT;
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  ImageSize img_size{im_dst_x, im_dst_y, img_dtype};
  bias_ = allocator->Malloc(img_size);
  if (bias_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  bias_ = allocator->MapBuffer(bias_, CL_MAP_WRITE, nullptr, true);
  if (bias_ == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  memset(bias_, 0x00, co4 * C4NUM * dtype_size);
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3) {
    void *src_data = stored_bias_ == nullptr ? in_tensors_.at(kBiasIndex)->data() : stored_bias_;
    MS_ASSERT(src_data);
    if (in_tensors_[kBiasIndex]->data_type() == kNumberTypeFloat32 && enable_fp16_) {
      for (int i = 0; i < CO_; i++) {
        reinterpret_cast<float16_t *>(bias_)[i] = reinterpret_cast<float *>(src_data)[i];
      }
    } else if (in_tensors_[kBiasIndex]->data_type() == kNumberTypeFloat16 && !enable_fp16_) {
      for (int i = 0; i < CO_; i++) {
        reinterpret_cast<float *>(bias_)[i] = reinterpret_cast<float16_t *>(src_data)[i];
      }
    } else {
      memcpy(bias_, src_data, CO_ * dtype_size);
    }
  }
  if (allocator->UnmapBuffer(bias_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  FreeStoredData(stored_bias_);
  return RET_OK;
}
#else
int FullConnectionOpenCLKernel::InitFilter() {
  auto allocator = ocl_runtime_->GetAllocator();
  auto intensor_shape = GpuTensorInfo(in_tensors_[0]);
  int co4 = UP_DIV(CO_, C4NUM);
  int nhw_remainder = intensor_shape.N * intensor_shape.H * intensor_shape.W / N_;
  size_t dtype_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  padWeight_ = allocator->Malloc(nhw_remainder * intensor_shape.Slice * co4 * C4NUM * C4NUM * dtype_size,
                                 lite::opencl::MemType::BUF);
  if (padWeight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  padWeight_ = allocator->MapBuffer(padWeight_, CL_MAP_WRITE, nullptr, true);
  if (padWeight_ == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  auto padWeight = reinterpret_cast<float *>(padWeight_);
  memset(padWeight_, 0x00, nhw_remainder * intensor_shape.Slice * co4 * C4NUM * C4NUM * dtype_size);
  void *src_data = stored_weight_ == nullptr ? in_tensors_.at(kWeightIndex)->data() : stored_weight_;
  MS_ASSERT(src_data);
  auto originWeight = reinterpret_cast<float *>(src_data);

  // pad weight
  // HWCICO -> (HWCI4)(CO4)(4 from CO)(4 from CI)
  // if tranposeB, COHWCI -> (HWCI4)(CO4)(4 from CO)(4 from CI)
  int index = 0;
  for (int nhw = 0; nhw < nhw_remainder; nhw++) {
    for (size_t i = 0; i < intensor_shape.Slice; ++i) {
      for (int j = 0; j < co4; ++j) {
        for (int k = 0; k < C4NUM; ++k) {
          for (int l = 0; l < C4NUM; ++l) {
            size_t src_ci = i * C4NUM + l;
            size_t src_co = j * C4NUM + k;
            if (src_ci < intensor_shape.C && static_cast<int>(src_co) < CO_) {
              int originId = (nhw * intensor_shape.C + src_ci) * CO_ + src_co;
              if (transposeB) {
                originId = src_co * intensor_shape.C * nhw_remainder + nhw * intensor_shape.C + src_ci;
              }
              padWeight[index++] = originWeight[originId];
            } else {
              index++;
            }
          }
        }
      }
    }
  }
  if (allocator->UnmapBuffer(padWeight_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  FreeStoredData(stored_weight_);
  return RET_OK;
}

int FullConnectionOpenCLKernel::InitBias() {
  // pad FC Bias
  auto allocator = ocl_runtime_->GetAllocator();
  int co4 = UP_DIV(CO_, C4NUM);
  size_t dtype_size = sizeof(float);
  size_t im_dst_x, im_dst_y;
  im_dst_x = co4;
  im_dst_y = 1;
  size_t img_dtype = CL_FLOAT;
  ImageSize img_size{im_dst_x, im_dst_y, img_dtype};
  bias_ = allocator->Malloc(img_size);
  if (bias_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  bias_ = allocator->MapBuffer(bias_, CL_MAP_WRITE, nullptr, true);
  if (bias_ == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  memset(bias_, 0x00, co4 * C4NUM * dtype_size);
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3) {
    void *src_data = stored_bias_ == nullptr ? in_tensors_.at(kBiasIndex)->data() : stored_bias_;
    MS_ASSERT(src_data);
    memcpy(bias_, src_data, CO_ * dtype_size);
  }
  if (allocator->UnmapBuffer(bias_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  FreeStoredData(stored_bias_);
  return RET_OK;
}
#endif

int FullConnectionOpenCLKernel::SetGlobalLocal() {
  local_size_ = {32, 4, 1};
  size_t CO = CO_;
  size_t N = N_;
  global_size_ = {UP_DIV(CO, C4NUM), C4NUM, N};
  AlignGlobalLocal(global_size_, local_size_);
  return RET_OK;
}

int FullConnectionOpenCLKernel::SetConstArgs() {
  if (!weight_var_) {
    if (ocl_runtime_->SetKernelArg(kernel_, 2, padWeight_, true) != CL_SUCCESS) {  // arg index 2
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  int arg_count = 3;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, bias_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, N_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  auto intensor_shape = GpuTensorInfo(in_tensors_[0]);
  int CI4 = CI_remainder_ * intensor_shape.Slice;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, CI4) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, UP_DIV(CO_, C4NUM)) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  auto in_shape_info = GpuTensorInfo(in_tensors_[0]);
  cl_int2 in_img_shape = {static_cast<int>(in_shape_info.height), static_cast<int>(in_shape_info.width)};
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_img_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count, static_cast<cl_int>(param->act_type_)) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int FullConnectionOpenCLKernel::StoreConstData() {
  if (!InferShapeDone()) {
    stored_weight_ = StoreTensorData(in_tensors_.at(kWeightIndex));
    if (stored_weight_ == nullptr) {
      MS_LOG(ERROR) << "Store weight failed.";
      return RET_ERROR;
    }
    if (in_tensors_.size() > kBiasIndex) {
      stored_bias_ = StoreTensorData(in_tensors_.at(kBiasIndex));
      if (stored_bias_ == nullptr) {
        MS_LOG(ERROR) << "Store bias failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int FullConnectionOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_count = 0;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, out_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (weight_var_) {
    if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_tensors_[1]->data()) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_FullConnection, OpenCLKernelCreator<FullConnectionOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_FullConnection, OpenCLKernelCreator<FullConnectionOpenCLKernel>)
}  // namespace mindspore::kernel
