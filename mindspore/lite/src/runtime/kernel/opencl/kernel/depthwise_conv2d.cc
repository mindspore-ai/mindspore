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

#include "src/runtime/kernel/opencl/kernel/depthwise_conv2d.h"
#include <string>
#include <set>
#include <map>
#include <algorithm>
#include <utility>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "nnacl/fp32/common_func_fp32.h"
#include "nnacl/op_base.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/opencl/cl/depthwise_conv2d.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::lite::opencl::MemType;

namespace mindspore::kernel {
int DepthwiseConv2dOpenCLKernel::CheckSpecs() {
  if ((in_tensors_.size() != INPUT_TENSOR_SIZE_2 && in_tensors_.size() != INPUT_TENSOR_SIZE_3) ||
      out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_[0]->data_type() != kNumberTypeFloat32 && in_tensors_[0]->data_type() != kNumberTypeFloat16) {
    MS_LOG(WARNING) << "Unsupported data type " << in_tensors_[0]->data_type();
    return RET_ERROR;
  }
  if (!in_tensors_.at(kWeightIndex)->IsConst()) {
    MS_LOG(WARNING) << "DepthwiseConv2d don't support non-constant weight yet.";
    return RET_ERROR;
  }
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3 && !in_tensors_.at(kBiasIndex)->IsConst()) {
    MS_LOG(WARNING) << "DepthwiseConv2d don't support non-constant bias yet.";
    return RET_ERROR;
  }
  return RET_OK;
}

int DepthwiseConv2dOpenCLKernel::Prepare() {
  std::string kernel_name = "DepthwiseConv2d";
  if (out_mem_type_ == MemType::BUF) {
    kernel_name += "_BUF";
  } else {
    kernel_name += "_IMG";
  }
  kernel_name += "_NHWC4";
  auto parameter = reinterpret_cast<ConvParameter *>(op_parameter_);
  if (parameter->kernel_h_ == 1 && parameter->kernel_w_ == 1) {
    kernel_name += "_1x1";
  }
  if (filter_type_ == lite::opencl::MemType::BUF) {
    kernel_name += "_b" + std::to_string(block_size_.H) + std::to_string(block_size_.W) + std::to_string(block_size_.C);
  } else {
    block_size_.C = block_size_.H = block_size_.W = 1;
  }
  const std::string program_name = "DepthwiseConv2d";
  std::string source = depthwise_conv2d_source;
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
  ret = InitWeights();
  if (ret != RET_OK) {
    return ret;
  }
  ret = InitBias();
  if (ret != RET_OK) {
    return ret;
  }
  SetGlobalLocal();
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << kernel_name << " Init Done! mem type=" << static_cast<int>(out_mem_type_);
  return RET_OK;
}

int DepthwiseConv2dOpenCLKernel::InitWeights() {
  auto parameter = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto allocator = ocl_runtime_->GetAllocator();
  bool is_fp16 = ocl_runtime_->GetFp16Enable();

  size_t dtype_size = is_fp16 ? sizeof(int16_t) : sizeof(float);
  auto out_info = GpuTensorInfo(out_tensors_[0]);
  // weight: o, h, w, i; o == group, i == 1
  void *origin_weight = stored_weight_ == nullptr ? in_tensors_.at(kWeightIndex)->data() : stored_weight_;
  MS_ASSERT(origin_weight);
  int CO4 = UP_DIV(out_info.C, C4NUM);
  int pack_weight_size = C4NUM * CO4 * parameter->kernel_h_ * parameter->kernel_w_;

  int plane_in = parameter->kernel_h_ * parameter->kernel_w_;
  int plane_out = plane_in * C4NUM;
  if (filter_type_ == MemType::IMG) {
    int alignment = ocl_runtime_->GetImagePitchAlignment();
    plane_out = UP_ROUND(plane_out, alignment) * C4NUM;
    pack_weight_size = plane_out * CO4;
  }
  pack_weight_size = pack_weight_size * dtype_size;
  auto ConvertFilter = [](void *src, void *dst, TypeId src_type, TypeId dst_type, size_t plane_in, size_t plane_out,
                          size_t channel) {
    if (dst_type == kNumberTypeFloat16) {
      if (src_type == kNumberTypeFloat16) {
        std::function<int16_t(int16_t)> to_dtype = [](int16_t x) -> int16_t { return x; };
        PackNCHWToNC4HW4<int16_t, int16_t>(src, dst, 1, plane_in, plane_out, channel, to_dtype);
      } else if (src_type == kNumberTypeFloat32) {
        std::function<float16_t(float)> to_dtype = [](float x) -> float16_t { return static_cast<float16_t>(x); };
        PackNCHWToNC4HW4<float, float16_t>(src, dst, 1, plane_in, plane_out, channel, to_dtype);
      } else {  // int8 or int16
        std::function<int16_t(int16_t)> to_dtype = [](int16_t x) -> int16_t { return x; };
        PackNCHWToNC4HW4<int16_t, int16_t>(src, dst, 1, plane_in, plane_out, channel, to_dtype);
      }
    } else {
      if (src_type == kNumberTypeFloat32) {
        std::function<float(float)> to_dtype = [](float x) -> float { return x; };
        PackNCHWToNC4HW4<float, float>(src, dst, 1, plane_in, plane_out, channel, to_dtype);
      } else if (src_type == kNumberTypeFloat16) {
        std::function<float(float16_t)> to_dtype = [](float16_t x) -> float { return static_cast<float>(x); };
        PackNCHWToNC4HW4<float16_t, float>(src, dst, 1, plane_in, plane_out, channel, to_dtype);
      } else {  // int8 or int16
        std::function<float(float)> to_dtype = [](float x) -> float { return x; };
        PackNCHWToNC4HW4<float, float>(src, dst, 1, plane_in, plane_out, channel, to_dtype);
      }
    }
  };
  std::vector<char> temp_filter(pack_weight_size);
  auto src_type = in_tensors_.at(kWeightIndex)->data_type();
  auto dst_type = is_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32;
  ConvertFilter(origin_weight, temp_filter.data(), src_type, dst_type, plane_in, plane_out, out_info.C);
  if (filter_type_ == MemType::IMG) {
    size_t img_dtype = ocl_runtime_->GetFp16Enable() ? CL_HALF_FLOAT : CL_FLOAT;
    ImageSize img_size{(size_t)plane_out / C4NUM, (size_t)out_info.N * CO4, img_dtype};
    packed_weight_ = allocator->Malloc(img_size, temp_filter.data());

  } else {
    packed_weight_ = allocator->Malloc(pack_weight_size, temp_filter.data());
  }
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  FreeStoredData(stored_weight_);
  return RET_OK;
}

int DepthwiseConv2dOpenCLKernel::InitBias() {
  auto allocator = ocl_runtime_->GetAllocator();
  bool is_fp16 = ocl_runtime_->GetFp16Enable();

  size_t dtype_size = is_fp16 ? sizeof(int16_t) : sizeof(float);
  auto out_info = GpuTensorInfo(out_tensors_[0]);
  int CO4 = UP_DIV(out_info.C, C4NUM);
  auto src_type = in_tensors_.at(kWeightIndex)->data_type();
  auto dst_type = is_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32;

  auto ConvertBias = [](void *src, void *dst, size_t size, size_t dtype_size, TypeId src_type, TypeId dst_type) {
    if (dst_type == kNumberTypeFloat16 && src_type == kNumberTypeFloat32) {
      float16_t *bias_ptr = static_cast<float16_t *>(dst);
      for (size_t i = 0; i < size; ++i) {
        bias_ptr[i] = static_cast<float16_t>(static_cast<float *>(src)[i]);
      }
    } else if (dst_type == kNumberTypeFloat32 && src_type == kNumberTypeFloat16) {
      float32_t *bias_ptr = static_cast<float32_t *>(dst);
      for (size_t i = 0; i < size; ++i) {
        bias_ptr[i] = static_cast<float32_t>(static_cast<float16_t *>(src)[i]);
      }
    } else {
      memcpy(dst, src, size * dtype_size);
    }
  };
  size_t bias_size = C4NUM * CO4 * dtype_size;
  std::vector<char> temp_bias(bias_size, 0);
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3) {
    src_type = in_tensors_.at(kBiasIndex)->data_type();
    dst_type = is_fp16 ? kNumberTypeFloat16 : kNumberTypeFloat32;
    auto element_size = in_tensors_.at(kBiasIndex)->ElementsNum();
    void *src_data = stored_bias_ == nullptr ? in_tensors_.at(kBiasIndex)->data() : stored_bias_;
    MS_ASSERT(src_data);
    ConvertBias(src_data, temp_bias.data(), element_size, dtype_size, src_type, dst_type);
  }
  bias_data_ = allocator->Malloc(bias_size, temp_bias.data());
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }

  FreeStoredData(stored_bias_);
  return RET_OK;
}

int DepthwiseConv2dOpenCLKernel::SetConstArgs() {
  auto parameter = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto in_info = GpuTensorInfo(in_tensors_[0]);
  auto out_info = GpuTensorInfo(out_tensors_[0]);
  size_t CO4 = UP_DIV(out_info.C, C4NUM);
  size_t CI4 = UP_DIV(in_info.C, C4NUM);

  std::map<ActType, std::pair<float, float>> relu_clips{
    {ActType_No, {-FLT_MAX, FLT_MAX}}, {ActType_Relu, {0.0, FLT_MAX}}, {ActType_Relu6, {0, 6.0}}};
  cl_int2 kernel_size = {parameter->kernel_w_, parameter->kernel_h_};
  cl_int2 stride = {parameter->stride_w_, parameter->stride_h_};
  cl_int2 padding = {-parameter->pad_l_, -parameter->pad_u_};
  cl_int2 dilation = {parameter->dilation_w_, parameter->dilation_h_};
  cl_int4 src_size = {(cl_int)in_info.W, (cl_int)in_info.H, (cl_int)CI4, (cl_int)in_info.N};
  cl_int4 dst_size = {(cl_int)out_info.W, (cl_int)out_info.H, (cl_int)CO4, (cl_int)out_info.N};

  int arg_cnt = 2;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, packed_weight_, (filter_type_ == lite::opencl::MemType::BUF)) !=
      CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, bias_data_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, kernel_size) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, stride) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, padding) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, dilation) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, src_size) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, dst_size) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, relu_clips[parameter->act_type_].first) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, relu_clips[parameter->act_type_].second) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void DepthwiseConv2dOpenCLKernel::SetGlobalLocal() {
  auto out_info = GpuTensorInfo(out_tensors_[0]);
  // set global
  size_t CO4 = UP_DIV(out_info.C, C4NUM * block_size_.C);
  global_size_ = {CO4, (size_t)UP_DIV(out_info.W, block_size_.W),
                  (size_t)UP_DIV(out_info.H, block_size_.H) * out_info.N};
  // set local
  int local_max = filter_type_ == MemType::IMG ? 64 : 128;
  if (ocl_runtime_->DeviceComputeUnits() > 16) {
    local_max = 256;
  }
  const int local_c_max = 16;
  const int OH_threshold = 100;
  const int OW_threshold = 100;
  const int OC_threshold = 64;
  size_t local_c = GetMaxDivisor(global_size_[0], local_c_max);
  local_c = std::max<size_t>(local_c, 1);
  size_t local_hw = local_max / local_c;
  size_t local_h;
  size_t local_w;
  if (out_info.H >= OH_threshold && out_info.W >= OW_threshold && out_info.C <= OC_threshold) {  // c -> w -> h
    local_w = std::min(global_size_[1], local_hw);
    local_h = std::min(local_hw / local_w, global_size_[2]);
  } else {  // c -> h -> w
    local_h = std::min(global_size_[2], local_hw);
    local_w = std::min(local_hw / local_h, global_size_[1]);
  }

  local_size_ = {local_c, local_w, local_h};
  AlignGlobalLocal(global_size_, local_size_);
}

int DepthwiseConv2dOpenCLKernel::StoreConstData() {
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

int DepthwiseConv2dOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  if (ocl_runtime_->SetKernelArg(kernel_, 0, out_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, 1, in_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
