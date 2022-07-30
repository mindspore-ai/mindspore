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
#include "src/litert/kernel/opencl/kernel/matmul.h"
#include "src/litert/kernel/opencl/kernel/strassen.h"
#include "src/common/utils.h"
#include "src/litert/kernel/opencl/cl/strassen.cl.inc"

using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;

namespace mindspore::kernel {
namespace {
const int half = 2;
}

int StrassenOpenCLKernel::Prepare() {
  const std::string kernel_name = "MatMul_Strassen_NHWC4_2d";
  std::string source = strassen_source;
  const std::string program_name = "MatMul";
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);

  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  ocl_runtime_->BuildKernel(kernel_IMG_add_sub_2, program_name, "MatMul_IMG_Add_Sub_2", build_options_ext);
  ocl_runtime_->BuildKernel(kernel_BUF_add_sub_2, program_name, "MatMul_BUF_Add_Sub_2", build_options_ext);
  ocl_runtime_->BuildKernel(kernel_back_result, program_name, "Strassen_Back_Result", build_options_ext);
  ocl_runtime_->BuildKernel(MatMul_StrassenBUFFilled, program_name, "MatMul_BUF_Filled", build_options_ext);
  ocl_runtime_->BuildKernel(MatMul_StrassenIMGFilled, program_name, "MatMul_IMG_Filled", build_options_ext);
  ret = InitWeights();
  if (ret != RET_OK) {
    return ret;
  }
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  ret = SetGlobalLocal();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetGlobalLocal failed.";
    return ret;
  }

  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int StrassenOpenCLKernel::AllocatorMemoryForStrassen(int NumA, int NumB) {
  auto allocator = ocl_runtime_->GetAllocator();
  size_t img_dtype = enable_fp16_ ? CL_HALF_FLOAT : CL_FLOAT;
  ImageSize img_size{static_cast<size_t>(UP_DIV(NumA, C4NUM)), static_cast<size_t>(NumA), img_dtype};
  size_t dtype_size = enable_fp16_ ? sizeof(cl_half) : sizeof(cl_float);
  size_t memB = NumB * NumB * dtype_size;
  for (int depth = 0; depth < MAXDEPTH; depth++) {
    B_temp[depth] = allocator->Malloc(memB, lite::opencl::MemType::BUF);
    if (B_temp[depth] == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
    A_temp[depth] = allocator->Malloc(img_size);
    if (A_temp[depth] == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
    M1[depth] = allocator->Malloc(img_size);
    if (M1[depth] == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
    M2[depth] = allocator->Malloc(img_size);
    if (M2[depth] == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
    M3[depth] = allocator->Malloc(img_size);
    if (M3[depth] == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
    M4[depth] = allocator->Malloc(img_size);
    if (M4[depth] == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
    M5[depth] = allocator->Malloc(img_size);
    if (M5[depth] == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
    M6[depth] = allocator->Malloc(img_size);
    if (M6[depth] == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
    M7[depth] = allocator->Malloc(img_size);
    if (M7[depth] == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

#ifdef ENABLE_FP16
int StrassenOpenCLKernel::InitWeights() {
  // ABMCI @ ABCICO = ABMCO
  auto allocator = ocl_runtime_->GetAllocator();
  int NumA = in_tensors_[0]->shape()[0];
  int NumB = in_tensors_[1]->shape()[0];
  size_t dtype_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  padWeight_ = allocator->Malloc(NumA * NumB * dtype_size, lite::opencl::MemType::BUF);
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
  memset(padWeight_, 0x00, NumA * NumB * dtype_size);
  auto weight_tensor_data = in_tensors_.at(kWeightIndex)->data();
  MS_ASSERT(weight_tensor_data);
  auto originWeightFp32 = reinterpret_cast<float *>(weight_tensor_data);
  auto originWeightFp16 = reinterpret_cast<float16_t *>(weight_tensor_data);
  bool isModelFp16 = in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat16;
  if (AllocatorMemoryForStrassen(NumA / half, NumB / half) != RET_OK) {
    MS_LOG(ERROR) << "AllocatorMemoryForStrassen failed.";
    return RET_ERROR;
  }
  size_t size = NumA * NumB * dtype_size;
  if (isModelFp16) {
    if (enable_fp16_) {
      memcpy(padWeightFp16, originWeightFp16, size);
    } else {
      for (int i = 0; i < NumA * NumB; ++i) {
        padWeightFp32[i] = static_cast<float>(originWeightFp16[i]);
      }
    }
  } else {
    if (enable_fp16_) {
      for (int i = 0; i < NumA * NumB; ++i) {
        padWeightFp16[i] = static_cast<float16_t>(originWeightFp32[i]);
      }
    } else {
      memcpy(padWeightFp32, originWeightFp32, size);
    }
  }
  if (allocator->UnmapBuffer(padWeight_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
#else
int StrassenOpenCLKernel::InitWeights() {
  // ABMCI @ ABCICO = ABMCO
  auto allocator = ocl_runtime_->GetAllocator();
  int NumA = in_tensors_[0]->shape()[0];
  int NumB = in_tensors_[1]->shape()[0];
  size_t dtype_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  padWeight_ = allocator->Malloc(NumA * NumB * dtype_size, lite::opencl::MemType::BUF);
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
  memset(padWeight_, 0x00, NumA * NumB * dtype_size);
  auto weight_tensor_data = in_tensors_.at(kWeightIndex)->data();
  MS_ASSERT(weight_tensor_data);
  auto originWeightFp32 = reinterpret_cast<float *>(weight_tensor_data);
  if (AllocatorMemoryForStrassen(NumA / half, NumB / half) != RET_OK) {
    MS_LOG(ERROR) << "AllocatorMemoryForStrassen failed.";
    return RET_ERROR;
  }
  size_t size = NumA * NumB * dtype_size;
  memcpy(padWeightFp32, originWeightFp32, size);
  if (allocator->UnmapBuffer(padWeight_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
#endif

void AlignStrassenGlobalLocal(const std::vector<size_t> &global, const std::vector<size_t> &local,
                              cl::NDRange *global_range, cl::NDRange *local_range) {
  *local_range = cl::NDRange(local[kNHWC_N], local[kNHWC_H], local[kNHWC_W]);
  *global_range = cl::NDRange(UP_ROUND(global[kNHWC_N], local[kNHWC_N]), UP_ROUND(global[kNHWC_H], local[kNHWC_H]),
                              UP_ROUND(global[kNHWC_W], local[kNHWC_W]));
}

// 0 : global_size_, 1: global_size_add_sub
int StrassenOpenCLKernel::StrassenSetGlobalLocal(size_t strassen_size, int type_flag) {
  size_t strassen_size_C4 = UP_DIV(strassen_size, C4NUM);
  local_size_add_sub = {16, 1, 16};
  if (type_flag == 0) {
    global_size_ = {strassen_size_C4, 1, strassen_size};
    AlignGlobalLocal(global_size_, local_size_);
  } else {
    global_size_add_sub = {strassen_size_C4, 1, strassen_size};
    AlignStrassenGlobalLocal(global_size_add_sub, local_size_add_sub, &global_add_sub_, &local_add_sub_);
  }
  return RET_OK;
}

int StrassenOpenCLKernel::SetGlobalLocal() {
  // local size should be less than MAX_GROUP_SIZE
  local_size_ = {32, 4, 1};
  global_size_ = {1, 1, 1};
  size_t strassen_size = outShape[kNHWC_C] / half;
  int ret = StrassenSetGlobalLocal(strassen_size, CLARGSINDEX0);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StrassenSetGlobalLocal 0 failed.";
    return ret;
  }
  ret = StrassenSetGlobalLocal(strassen_size, CLARGSINDEX1);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StrassenSetGlobalLocal 1 failed.";
    return ret;
  }
  ret = StrassenSetGlobalLocal(strassen_size, CLARGSINDEX2);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StrassenSetGlobalLocal 2 failed.";
    return ret;
  }
  return RET_OK;
}

int StrassenOpenCLKernel::StrassenSetConstArgs(cl::Kernel *kernel, int index, int strassen_size,
                                               bool is_matmul_kernel) {
  cl_int4 shape;
  if (is_matmul_kernel) {
    shape = {1, 1, strassen_size, strassen_size};
  } else {
    shape = {strassen_size, 1, 1, UP_DIV(strassen_size, C4NUM)};
  }
  if (ocl_runtime_->SetKernelArg(*kernel, index, shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int StrassenOpenCLKernel::SetConstArgs() {
  int strassen_size = inShape[kNHWC_C] / half;
  StrassenSetConstArgs(&kernel_IMG_add_sub_2, CLARGSINDEX3, strassen_size, false);
  StrassenSetConstArgs(&kernel_BUF_add_sub_2, CLARGSINDEX2, strassen_size, false);
  return RET_OK;
}

int StrassenOpenCLKernel::StrassenDataFilled(cl::Kernel *kernel, void *input, void *output, const int size,
                                             cl_int2 offset, lite::opencl::MemType mem_type) {
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "StrassenDataFilled input or output can not nullptr";
    return RET_ERROR;
  }
  if (mem_type == lite::opencl::MemType::IMG) {
    if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX0, input) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX1, output) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  } else {
    if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX0, input, true) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX1, output, true) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  StrassenSetConstArgs(kernel, CLARGSINDEX2, size, false);
  if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX3, offset) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(*kernel, global_add_sub_, local_add_sub_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int StrassenOpenCLKernel::StrassenAddSub(cl::Kernel *kernel, void *input, void *output, const int size, cl_int4 offset,
                                         int flag, lite::opencl::MemType mem_type) {
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "StrassenAddSub input or output can not nullptr";
    return RET_ERROR;
  }
  if (mem_type == lite::opencl::MemType::IMG) {
    if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX0, input) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX1, output) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  } else {
    if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX0, input, true) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX1, output, true) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  StrassenSetConstArgs(kernel, CLARGSINDEX2, size, false);
  if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX3, offset) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX4, flag) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(*kernel, global_add_sub_, local_add_sub_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int StrassenOpenCLKernel::StrassenBackResult(cl::Kernel *kernel, void *input1, void *input2, void *input3, void *input4,
                                             void *input5, void *input6, void *input7, void *output, const int size) {
  if (input1 == nullptr || input2 == nullptr || input3 == nullptr || input4 == nullptr || input5 == nullptr ||
      input6 == nullptr || input7 == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "StrassenBackResult input or output can not nullptr";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX0, input1) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX1, input2) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX2, input3) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX3, input4) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX4, input5) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX5, input6) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX6, input7) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(*kernel, CLARGSINDEX7, output) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  StrassenSetConstArgs(kernel, CLARGSINDEX8, size, false);
  if (ocl_runtime_->RunKernel(*kernel, global_add_sub_, local_add_sub_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int StrassenOpenCLKernel::StrassenRunMmatmul(void *input, void *weight, void *output, const int size) {
  if (input == nullptr || weight == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "StrassenRunMmatmul input ,weight or output can not nullptr";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX0, input) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX1, output) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX2, weight, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  StrassenSetConstArgs(&kernel_, CLARGSINDEX3, size, true);
  StrassenSetConstArgs(&kernel_, CLARGSINDEX4, size, true);
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int StrassenOpenCLKernel::DoStrassen(void *data, void *weight, void *result, const int size, const int depth,
                                     const int threshold) {
  const int size_2 = size / half;
  int C4 = UP_DIV(size_2, C4NUM);
  if (size <= threshold) {
    //   run matmul;
    StrassenSetGlobalLocal(size, 0);
    StrassenRunMmatmul(data, weight, result, size);
    return RET_OK;
  }
  // flag = 0 : add   otherwise flag = 1 : sub
  //   M1 = A11 * ( B12- B22)
  StrassenSetGlobalLocal(size_2, 1);
  StrassenDataFilled(&MatMul_StrassenIMGFilled, data, A_temp[depth + 1], size_2, {0, 0}, lite::opencl::MemType::IMG);
  StrassenAddSub(&kernel_BUF_add_sub_2, weight, B_temp[depth + 1], size_2, {0, C4, size_2, C4}, 1,
                 lite::opencl::MemType::BUF);
  DoStrassen(A_temp[depth + 1], B_temp[depth + 1], M1[depth + 1], size_2, depth + 1, threshold);

  // M2 = (A11 + A12) * B22
  StrassenSetGlobalLocal(size_2, 1);
  StrassenDataFilled(&MatMul_StrassenBUFFilled, weight, B_temp[depth + 1], size_2, {size_2, C4},
                     lite::opencl::MemType::BUF);
  StrassenAddSub(&kernel_IMG_add_sub_2, data, A_temp[depth + 1], size_2, {0, 0, 0, C4}, 0, lite::opencl::MemType::IMG);
  DoStrassen(A_temp[depth + 1], B_temp[depth + 1], M2[depth + 1], size_2, depth + 1, threshold);

  // M3 = (A21 + A22) * B11
  StrassenSetGlobalLocal(size_2, 1);
  StrassenDataFilled(&MatMul_StrassenBUFFilled, weight, B_temp[depth + 1], size_2, {0, 0}, lite::opencl::MemType::BUF);
  StrassenAddSub(&kernel_IMG_add_sub_2, data, A_temp[depth + 1], size_2, {size_2, 0, size_2, C4}, 0,
                 lite::opencl::MemType::IMG);
  DoStrassen(A_temp[depth + 1], B_temp[depth + 1], M3[depth + 1], size_2, depth + 1, threshold);

  // M4 = A22 * (B21 - B11)
  StrassenSetGlobalLocal(size_2, 1);
  StrassenDataFilled(&MatMul_StrassenIMGFilled, data, A_temp[depth + 1], size_2, {size_2, C4},
                     lite::opencl::MemType::IMG);
  StrassenAddSub(&kernel_BUF_add_sub_2, weight, B_temp[depth + 1], size_2, {size_2, 0, 0, 0}, 1,
                 lite::opencl::MemType::BUF);
  DoStrassen(A_temp[depth + 1], B_temp[depth + 1], M4[depth + 1], size_2, depth + 1, threshold);

  // M5 = (A11 + A22) * (B11 + B22)
  StrassenSetGlobalLocal(size_2, 1);
  StrassenAddSub(&kernel_IMG_add_sub_2, data, A_temp[depth + 1], size_2, {0, 0, size_2, C4}, 0,
                 lite::opencl::MemType::IMG);
  // (B11 + B22)
  StrassenAddSub(&kernel_BUF_add_sub_2, weight, B_temp[depth + 1], size_2, {0, 0, size_2, C4}, 0,
                 lite::opencl::MemType::BUF);
  DoStrassen(A_temp[depth + 1], B_temp[depth + 1], M5[depth + 1], size_2, depth + 1, threshold);

  // M6 = (A12 - A22) * (B21 + B22)
  StrassenSetGlobalLocal(size_2, 1);
  StrassenAddSub(&kernel_IMG_add_sub_2, data, A_temp[depth + 1], size_2, {0, C4, size_2, C4}, 1,
                 lite::opencl::MemType::IMG);
  StrassenAddSub(&kernel_BUF_add_sub_2, weight, B_temp[depth + 1], size_2, {size_2, 0, size_2, C4}, 0,
                 lite::opencl::MemType::BUF);
  DoStrassen(A_temp[depth + 1], B_temp[depth + 1], M6[depth + 1], size_2, depth + 1, threshold);

  // M7 = (A11 - A21) * (B11 + B12)
  StrassenSetGlobalLocal(size_2, 1);
  StrassenAddSub(&kernel_IMG_add_sub_2, data, A_temp[depth + 1], size_2, {0, 0, size_2, 0}, 1,
                 lite::opencl::MemType::IMG);
  StrassenAddSub(&kernel_BUF_add_sub_2, weight, B_temp[depth + 1], size_2, {0, 0, 0, C4}, 0,
                 lite::opencl::MemType::BUF);
  DoStrassen(A_temp[depth + 1], B_temp[depth + 1], M7[depth + 1], size_2, depth + 1, threshold);

  //   BackResult
  StrassenSetGlobalLocal(size_2, 1);
  StrassenBackResult(&kernel_back_result, M1[depth + 1], M2[depth + 1], M3[depth + 1], M4[depth + 1], M5[depth + 1],
                     M6[depth + 1], M7[depth + 1], result, size_2);
  return RET_OK;
}

int StrassenOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int threshold;
  const int up_bound = 1024;
  const int down_bound = 256;
  if (in_tensors_.at(0)->shape()[0] >= up_bound) {
    threshold = UP_DIV(in_tensors_.at(0)->shape()[0], C4NUM) / half;
  } else if (in_tensors_.at(0)->shape()[0] <= down_bound) {
    threshold = in_tensors_.at(0)->shape()[0];
  } else {
    threshold = UP_DIV(in_tensors_.at(0)->shape()[0], C4NUM);
  }
  DoStrassen(in_tensors_.at(0)->data(), padWeight_, out_tensors_.at(0)->data(), in_tensors_.at(0)->shape()[0], 0,
             threshold);
  return RET_OK;
}
}  // namespace mindspore::kernel
