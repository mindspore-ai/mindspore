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

#include "src/runtime/kernel/opencl/kernel/winograd.h"
#include <memory>
#include "src/runtime/kernel/opencl/cl/winograd.cl.inc"
#include "nnacl/base/minimal_filtering_generator.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

namespace {
void Align(const std::vector<int> &global, const std::vector<int> &local, cl::NDRange *global_range,
           cl::NDRange *local_range) {
  *local_range = cl::NDRange(local[0], local[1], local[2]);
  *global_range =
    cl::NDRange(UP_ROUND(global[0], local[0]), UP_ROUND(global[1], local[1]), UP_ROUND(global[2], local[2]));
}

constexpr float Gt[] = {1.0000000000, 1.0000000000, 1.0000000000,  1.0000000000, 1.0000000000,  0.0000000000,
                        0.0000000000, 0.7071067691, -0.7071067691, 1.4142135382, -1.4142135382, 0.0000000000,
                        0.0000000000, 0.4999999702, 0.4999999702,  1.9999998808, 1.9999998808,  1.0000000000};

#ifndef ENABLE_ARM64
constexpr float G[] = {1.0000000000, 0.0000000000,  0.0000000000, 1.0000000000, 0.7071067691, 0.4999999702,
                       1.0000000000, -0.7071067691, 0.4999999702, 1.0000000000, 1.4142135382, 1.9999998808,
                       1.0000000000, -1.4142135382, 1.9999998808, 0.0000000000, 0.0000000000, 1.0000000000};
std::vector<float> GenerateWinogradFilter(void *src, TypeId dtype, size_t CO, size_t CI) {
  auto src_fp32 = reinterpret_cast<float *>(src);
  auto src_fp16 = reinterpret_cast<float16_t *>(src);
  std::function<float(int)> access_func;
  if (dtype == kNumberTypeFloat32) {
    access_func = [=](int idx) { return src_fp32[idx]; };
  } else {
    access_func = [=](int idx) { return static_cast<float>(src_fp16[idx]); };
  }
  // OHWI -> O66I
  std::vector<float> dst(CO * 6 * 6 * CI);
  if (src == nullptr) {
    return dst;
  }
  for (int co = 0; co < CO; ++co) {
    for (int ci = 0; ci < CI; ++ci) {
      float in_vals[9];
      for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
          const int f_index = ((co * 3 + kh) * 3 + kw) * CI + ci;
          in_vals[kh * 3 + kw] = access_func(f_index);
        }
      }
      auto temp_vals = MatrixMultiply(G, in_vals, 6, 3, 3);
      auto out_vals = MatrixMultiply(temp_vals.data(), Gt, 6, 3, 6);
      for (int kh = 0; kh < 6; ++kh) {
        for (int kw = 0; kw < 6; ++kw) {
          const int f_index = ((co * 6 + kh) * 6 + kw) * CI + ci;
          dst[f_index] = out_vals[kh * 6 + kw];
        }
      }
    }
  }
  return dst;
}
#endif

}  // namespace

void WinogradOpenCLKernel::BuildKernel() {
  std::string program_name = "winograd";
  ocl_runtime_->LoadSource(program_name, GetActDefines() + winograd_source);
  ocl_runtime_->BuildKernel(kernel_4x4to36_, program_name, "Winograd4x4To36");
  ocl_runtime_->BuildKernel(kernel_, program_name,
                            filter_type_ == MemType::IMG ? "WinogradConv2D_Img" : "WinogradConv2D");
  ocl_runtime_->BuildKernel(kernel_36to4x4_, program_name, "Winograd36To4x4");
}

void WinogradOpenCLKernel::InitFilter() {
  auto allocator = ocl_runtime_->GetAllocator();

  auto ret = DequantWeight();
  if (ret != RET_OK) {
    return;
  }

  // allocate opencl memory: buffer or image2d
  size_t size = 0;
  int Ogroup = 2;
  if (filter_type_ == MemType::IMG) {
    size_t width = 6 * 6 * UP_ROUND(CI_, CI_TILE);
    size_t height = CO_SLICES_;
    size_t dtype = use_fp16_ ? CL_HALF_FLOAT : CL_FLOAT;
    size = width * height * CO_TILE * sizeof_FLT_;
    packed_filter_ = allocator->Malloc({width, height, dtype});
  } else {
    size = UP_DIV(CO_SLICES_, Ogroup) * 6 * 6 * CI_SLICES_ * Ogroup * CI_TILE * CO_TILE * sizeof_FLT_;
    packed_filter_ = allocator->Malloc(size);
  }

  // rearrange filter
  auto filter_tensor = in_tensors_.at(1);
#ifndef ENABLE_ARM64
  auto winograd_filter = GenerateWinogradFilter(filter_tensor->data_c(), filter_tensor->data_type(), CO_, CI_);
  void *src_data = winograd_filter.data();
#else
  std::unique_ptr<float[]> winograd_filter(new float[CO_ * 6 * 6 * CI_]);
  WinogradWeightTransform(reinterpret_cast<const float *>(filter_tensor->data_c()),
                          reinterpret_cast<float *>(winograd_filter.get()), nullptr, Gt, 1, 6, 3, CI_, CO_, false);

  void *src_data = winograd_filter.get();
#endif

  auto src_dtype = kNumberTypeFloat32;
  auto dst_dtype = use_fp16_ ? kNumberTypeFloat16 : kNumberTypeFloat32;
  std::vector<char> tmp(size, 0);
  if (filter_type_ == MemType::IMG) {
    ConvertFilter(src_data, tmp.data(), src_dtype, dst_dtype, OHWI, OHWIOgroupI4O4, CO_, 6, 6, CI_);
  } else {
    ConvertFilter(src_data, tmp.data(), src_dtype, dst_dtype, OHWI, OHWIOgroupI4O4, CO_, 6, 6, CI_, Ogroup);
  }

  // unmap
  if (filter_type_ == MemType::IMG) {
    ocl_runtime_->WriteImage(packed_filter_, tmp.data());
  } else {
    allocator->MapBuffer(packed_filter_, CL_MAP_WRITE, nullptr, true);
    memcpy(packed_filter_, tmp.data(), size);
    allocator->UnmapBuffer(packed_filter_);
  }

  FreeDequantedWeight();
}

void WinogradOpenCLKernel::AllocateMemory() {
  auto allocator = ocl_runtime_->GetAllocator();
  size_t img_dtype = use_fp16_ ? CL_HALF_FLOAT : CL_FLOAT;

  size_t width = TILE_HW_;
  size_t height = CI_SLICES_ * 36;
  winograd_mem0_ = allocator->Malloc({width, height, img_dtype});

  width = TILE_HW_;
  height = CO_SLICES_ * 36;
  winograd_mem1_ = allocator->Malloc({width, height, img_dtype});
}

void WinogradOpenCLKernel::SetConstArgs() {
  AllocateMemory();

  int arg_cn = 1;
  cl_int4 input_shape = {batch_size_, OH_, OW_, CI_SLICES_};  // maybe pad=0, so use OH/OW
  ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, winograd_mem0_);
  ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, input_shape);
  ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, TILE_HW_);
  ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn, param_->pad_u_);

  arg_cn = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, winograd_mem0_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, winograd_mem1_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, packed_filter_, filter_type_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, TILE_HW_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, CI_SLICES_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn, CO_SLICES_);

  arg_cn = 2;
  cl_int4 output_shape = {batch_size_, OH_, OW_, CO_SLICES_};
  ocl_runtime_->SetKernelArg(kernel_36to4x4_, 0, winograd_mem1_);
  ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, packed_bias_, MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, output_shape);
  ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, TILE_HW_);
  ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, param_->act_type_);
  ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn, alpha_);
}

void WinogradOpenCLKernel::SetGlobalLocal() {
  Align({TILE_HW_, 6, CI_SLICES_}, {8, 6, 4}, &global_4x4to36_, &local_4x4to36_);
  Align({UP_DIV(TILE_HW_, 2), 36, UP_DIV(CO_SLICES_, 2)}, {8, 3, 8}, &global_range_, &local_range_);
  Align({TILE_HW_, 4, CO_SLICES_}, {4, 4, 8}, &global_36to4x4_, &local_36to4x4_);
}

int WinogradOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " winograd Running!";
  MS_LOG(DEBUG) << "winograd kernel0 Running!";
  ocl_runtime_->SetKernelArg(kernel_4x4to36_, 0, in_tensors_.front()->data_c());
  ocl_runtime_->RunKernel(kernel_4x4to36_, global_4x4to36_, local_4x4to36_, nullptr, &event_);

  MS_LOG(DEBUG) << "winograd kernel1 Running!";
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &kernel2_event_);

  MS_LOG(DEBUG) << "winograd kernel2 Running!";
  ocl_runtime_->SetKernelArg(kernel_36to4x4_, 1, out_tensors_.front()->data_c());
  ocl_runtime_->RunKernel(kernel_36to4x4_, global_36to4x4_, local_36to4x4_, nullptr, &kernel3_event_);
  return RET_OK;
}

double WinogradOpenCLKernel::GetProfilingTimeMs() {
  if (!ocl_runtime_->isProfiling()) {
    return MAX_PROFILING_TIME_MILLI_SECOND;
  }
  cl_ulong time_start;
  cl_ulong time_end;
  event_.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
  event_.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
  cl_ulong time_ns = time_end - time_start;
  kernel2_event_.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
  kernel2_event_.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
  time_ns += time_end - time_start;
  kernel3_event_.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
  kernel3_event_.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
  time_ns += time_end - time_start;
  return static_cast<double>(time_ns) * 1e-6;
}
}  // namespace mindspore::kernel
