/**
 * Copyright 2019 Huawei Technologies n., Ltd
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
#include "src/runtime/kernel/opencl/kernel/matmul.h"
#include "src/runtime/kernel/opencl/kernel/strassen.h"
#include "src/common/utils.h"
#include "src/runtime/kernel/opencl/cl/strassen.cl.inc"

using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;

namespace mindspore::kernel {

int StrassenOpenCLKernel::Prepare() {
  std::string kernel_name = "MatMul_Strassen_NHWC4_2d";
  std::string source = strassen_source;
  std::string program_name = "MatMul";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  ocl_runtime_->BuildKernel(kernel_IMG_add_sub_2, program_name, "MatMul_IMG_Add_Sub_2");
  ocl_runtime_->BuildKernel(kernel_BUF_add_sub_2, program_name, "MatMul_BUF_Add_Sub_2");
  ocl_runtime_->BuildKernel(kernel_back_result, program_name, "Strassen_Back_Result");
  ocl_runtime_->BuildKernel(MatMul_StrassenBUFFilled, program_name, "MatMul_BUF_Filled");
  ocl_runtime_->BuildKernel(MatMul_StrassenIMGFilled, program_name, "MatMul_IMG_Filled");
  auto ret = InitWeights();
  if (ret != RET_OK) {
    return ret;
  }
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

void StrassenOpenCLKernel::AllocatorMemoryForStrassen(int NumA, int NumB) {
  auto allocator = ocl_runtime_->GetAllocator();
  size_t img_dtype = enable_fp16_ ? CL_HALF_FLOAT : CL_FLOAT;
  ImageSize img_size{static_cast<size_t>(UP_DIV(NumA, C4NUM)), static_cast<size_t>(NumA), img_dtype};
  size_t dtype_size = enable_fp16_ ? sizeof(cl_half) : sizeof(cl_float);
  size_t memB = NumB * NumB * dtype_size;
  for (int depth = 0; depth < MAXDEPTH; depth++) {
    B_temp[depth] = allocator->Malloc(memB);
    A_temp[depth] = allocator->Malloc(img_size);
    M1[depth] = allocator->Malloc(img_size);
    M2[depth] = allocator->Malloc(img_size);
    M3[depth] = allocator->Malloc(img_size);
    M4[depth] = allocator->Malloc(img_size);
    M5[depth] = allocator->Malloc(img_size);
    M6[depth] = allocator->Malloc(img_size);
    M7[depth] = allocator->Malloc(img_size);
  }
}

int StrassenOpenCLKernel::InitWeights() {
  // ABMCI @ ABCICO = ABMCO
  auto ret = DequantWeight();
  if (ret != RET_OK) {
    return ret;
  }
  auto allocator = ocl_runtime_->GetAllocator();
  int NumA = in_tensors_[0]->shape()[0];
  int NumB = in_tensors_[1]->shape()[0];
  size_t dtype_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  padWeight_ = allocator->Malloc(NumA * NumB * dtype_size);
  padWeight_ = allocator->MapBuffer(padWeight_, CL_MAP_WRITE, nullptr, true);
  auto padWeightFp32 = reinterpret_cast<float *>(padWeight_);
  auto padWeightFp16 = reinterpret_cast<float16_t *>(padWeight_);
  memset(padWeight_, 0x00, NumA * NumB * dtype_size);
  auto originWeightFp32 = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->data_c());
  auto originWeightFp16 = reinterpret_cast<float16_t *>(in_tensors_.at(kWeightIndex)->data_c());
  bool isModelFp16 = in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat16;
  AllocatorMemoryForStrassen(NumA / 2, NumB / 2);
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
  allocator->UnmapBuffer(padWeight_);
  FreeDequantedWeight();
  return RET_OK;
}

void AlignStrassenGlobalLocal(const std::vector<size_t> &global, const std::vector<size_t> &local,
                              cl::NDRange *global_range, cl::NDRange *local_range) {
  *local_range = cl::NDRange(local[0], local[1], local[2]);
  *global_range =
    cl::NDRange(UP_ROUND(global[0], local[0]), UP_ROUND(global[1], local[1]), UP_ROUND(global[2], local[2]));
}

// 0 : global_size_, 1: global_size_add_sub
void StrassenOpenCLKernel::StrassenSetGlobalLocal(size_t strassen_size, int type_flag) {
  size_t strassen_size_C4 = UP_DIV(strassen_size, C4NUM);
  local_size_add_sub = {16, 1, 16};
  if (type_flag == 0) {
    global_size_ = {strassen_size_C4, 1, strassen_size};
    AlignGlobalLocal(global_size_, local_size_);
  } else {
    global_size_add_sub = {strassen_size_C4, 1, strassen_size};
    AlignStrassenGlobalLocal(global_size_add_sub, local_size_add_sub, &global_add_sub_, &local_add_sub_);
  }
}

void StrassenOpenCLKernel::SetGlobalLocal() {
  // local size should less than MAX_GROUP_SIZE
  local_size_ = {32, 4, 1};
  global_size_ = {1, 1, 1};
  size_t strassen_size = outShape[3] / 2;
  StrassenSetGlobalLocal(strassen_size, 0);  // set global_ and local
  StrassenSetGlobalLocal(strassen_size, 1);  // set global_size_add_sub
  StrassenSetGlobalLocal(strassen_size, 2);  // set global_size_weights
}

void StrassenOpenCLKernel::StrassenSetConstArgs(cl::Kernel *kernel, int index, int strassen_size,
                                                bool is_matmul_kernel) {
  cl_int4 shape;
  if (is_matmul_kernel) {
    shape = {1, 1, strassen_size, strassen_size};
  } else {
    shape = {strassen_size, 1, 1, UP_DIV(strassen_size, C4NUM)};
  }
  ocl_runtime_->SetKernelArg(*kernel, index, shape);
}

void StrassenOpenCLKernel::SetConstArgs() {
  int arg_count = 2;
  cl_int4 in_shape = {inShape[0], inShape[1], inShape[2], inShape[3]};
  cl_int4 out_shape = {outShape[0], outShape[1], outShape[2], outShape[3]};
  cl_int4 shape_offset = {0, 0, 0, 0};
  int strassen_size = inShape[3] / 2;
  out_shape.s[2] = in_shape.s[2] = in_shape.s[2] / 2;
  out_shape.s[3] = in_shape.s[3] = in_shape.s[3] / 2;
  StrassenSetConstArgs(&kernel_IMG_add_sub_2, 3, strassen_size, false);
  StrassenSetConstArgs(&kernel_BUF_add_sub_2, 2, strassen_size, false);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, out_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, shape_offset);
}

void StrassenOpenCLKernel::StrassenDataFilled(cl::Kernel *kernel, void *input, void *output, const int size,
                                              cl_int2 offset, lite::opencl::MemType mem_type) {
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "StrassenDataFilled input or output can not nullptr";
    return;
  }
  if (mem_type == lite::opencl::MemType::IMG) {
    ocl_runtime_->SetKernelArg(*kernel, 0, input);
    ocl_runtime_->SetKernelArg(*kernel, 1, output);
  } else {
    ocl_runtime_->SetKernelArg(*kernel, 0, input, lite::opencl::MemType::BUF);
    ocl_runtime_->SetKernelArg(*kernel, 1, output, lite::opencl::MemType::BUF);
  }
  StrassenSetConstArgs(kernel, 2, size, false);
  ocl_runtime_->SetKernelArg(*kernel, 3, offset);
  ocl_runtime_->RunKernel(*kernel, global_add_sub_, local_add_sub_, nullptr, &event_);
}

void StrassenOpenCLKernel::StrassenAddSub(cl::Kernel *kernel, void *input, void *output, const int size, cl_int4 offset,
                                          int flag, lite::opencl::MemType mem_type) {
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "StrassenAddSub input or output can not nullptr";
    return;
  }
  if (mem_type == lite::opencl::MemType::IMG) {
    ocl_runtime_->SetKernelArg(*kernel, 0, input, lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(*kernel, 1, output, lite::opencl::MemType::IMG);
  } else {
    ocl_runtime_->SetKernelArg(*kernel, 0, input, lite::opencl::MemType::BUF);
    ocl_runtime_->SetKernelArg(*kernel, 1, output, lite::opencl::MemType::BUF);
  }
  StrassenSetConstArgs(kernel, 2, size, false);
  ocl_runtime_->SetKernelArg(*kernel, 3, offset);
  ocl_runtime_->SetKernelArg(*kernel, 4, flag);
  ocl_runtime_->RunKernel(*kernel, global_add_sub_, local_add_sub_, nullptr, &event_);
}

void StrassenOpenCLKernel::StrassenBackResult(cl::Kernel *kernel, void *input1, void *input2, void *input3,
                                              void *input4, void *input5, void *input6, void *input7, void *output,
                                              const int size) {
  if (input1 == nullptr || input2 == nullptr || input3 == nullptr || input4 == nullptr || input5 == nullptr ||
      input6 == nullptr || input7 == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "StrassenBackResult input or output can not nullptr";
    return;
  }
  ocl_runtime_->SetKernelArg(*kernel, 0, input1);
  ocl_runtime_->SetKernelArg(*kernel, 1, input2);
  ocl_runtime_->SetKernelArg(*kernel, 2, input3);
  ocl_runtime_->SetKernelArg(*kernel, 3, input4);
  ocl_runtime_->SetKernelArg(*kernel, 4, input5);
  ocl_runtime_->SetKernelArg(*kernel, 5, input6);
  ocl_runtime_->SetKernelArg(*kernel, 6, input7);
  ocl_runtime_->SetKernelArg(*kernel, 7, output);
  StrassenSetConstArgs(kernel, 8, size, false);
  ocl_runtime_->RunKernel(*kernel, global_add_sub_, local_add_sub_, nullptr, &event_);
}

void StrassenOpenCLKernel::StrassenRunMmatmul(void *input, void *weight, void *output, const int size) {
  if (input == nullptr || weight == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "StrassenRunMmatmul input ,weight or output can not nullptr";
    return;
  }
  ocl_runtime_->SetKernelArg(kernel_, 0, input);
  ocl_runtime_->SetKernelArg(kernel_, 1, output);
  ocl_runtime_->SetKernelArg(kernel_, 2, weight, lite::opencl::MemType::BUF);
  StrassenSetConstArgs(&kernel_, 3, size, true);
  StrassenSetConstArgs(&kernel_, 4, size, true);
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
}

void StrassenOpenCLKernel::DoStrassen(void *data, void *weight, void *result, const int size, const int depth,
                                      const int threshold) {
  const int size_2 = size / 2;
  int C4 = UP_DIV(size_2, C4NUM);
  if (size <= threshold) {
    //   run matmul;
    StrassenSetGlobalLocal(size, 0);
    StrassenRunMmatmul(data, weight, result, size);
    return;
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
}

int StrassenOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int threshold;
  const int up_bound = 1024;
  const int down_bound = 256;
  if (in_tensors_.at(0)->shape()[0] >= up_bound) {
    threshold = UP_DIV(in_tensors_.at(0)->shape()[0], C4NUM) / 2;
  } else if (in_tensors_.at(0)->shape()[0] <= down_bound) {
    threshold = in_tensors_.at(0)->shape()[0];
  } else {
    threshold = UP_DIV(in_tensors_.at(0)->shape()[0], C4NUM);
  }
  DoStrassen(in_tensors_.at(0)->data_c(), padWeight_, out_tensors_.at(0)->data_c(), in_tensors_.at(0)->shape()[0], 0,
             threshold);
  return RET_OK;
}
}  // namespace mindspore::kernel
