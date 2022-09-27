/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/opencl/kernel/conv2d.h"
#include <string>
#include <set>
#include <algorithm>
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "schema/ops_generated.h"
#include "src/common/utils.h"
#include "src/litert/kernel/opencl/utils.h"
#include "src/litert/kernel/opencl/kernel/depthwise_conv2d.h"
#include "src/litert/kernel/opencl/kernel/fullconnection.h"
#include "src/litert/kernel/opencl/kernel/winograd.h"
#include "src/litert/kernel/opencl/cl/conv2d.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::ActivationType_TANH;
using mindspore::schema::PrimitiveType_Conv2DFusion;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {
int Conv2DOpenCLKernel::CheckSpecs() {
  auto ret = InputOutputCheckSpecs();
  if (ret != RET_OK) {
    return ret;
  }

  ret = FilterBiasCheckSpecs();
  if (ret != RET_OK) {
    return ret;
  }

  // for fusion: ActivationType_LEAKY_RELU ActivationType_TANH
  switch (static_cast<int>(param_->act_type_)) {
    case ActType_No:
    case ActType_Relu:
    case ActType_Relu6:
    case ActivationType_LEAKY_RELU:
    case ActivationType_TANH:
      break;
    default: {
      MS_LOG(WARNING) << "Unsupported activation type " << param_->act_type_;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int Conv2DOpenCLKernel::InputOutputCheckSpecs() {
  int inputs_num = in_tensors_.size();
  if (inputs_num != INPUT_TENSOR_SIZE_2 && inputs_num != INPUT_TENSOR_SIZE_3) {
    MS_LOG(WARNING) << "Conv2D only supports 2 or 3 input Tensor but get " << inputs_num;
    return RET_ERROR;
  }
  int outputs_num = out_tensors_.size();
  if (outputs_num != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "Conv2D only supports 1 output Tensor but get " << outputs_num;
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(in_tensors_.at(kInputIndex));
  int input_ndim = in_tensors_.at(kInputIndex)->shape().size();
  if (input_ndim != DIMENSION_4D) {
    MS_LOG(WARNING) << "Conv2D only supports 4D input Tensor but get " << input_ndim << "D.";
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(out_tensors_.at(kInputIndex));
  int output_ndim = out_tensors_.at(kOutputIndex)->shape().size();
  if (output_ndim != DIMENSION_4D) {
    MS_LOG(WARNING) << "Conv2D only supports 4D output Tensor but get " << output_ndim << "D.";
    return RET_ERROR;
  }
  return RET_OK;
}

int Conv2DOpenCLKernel::FilterBiasCheckSpecs() {
  auto *filter_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(filter_tensor);
  int filter_ndim = filter_tensor->shape().size();
  if (filter_ndim != DIMENSION_4D) {
    MS_LOG(WARNING) << "Conv2D only supports 4D filter Tensor but get " << filter_ndim << "D.";
    return RET_ERROR;
  }
  if (!filter_tensor->IsConst()) {
    bool is_const = filter_tensor->category() == lite::Category::CONST_TENSOR ||
                    filter_tensor->category() == lite::Category::CONST_SCALAR;
    if (!(is_const && stored_filter_)) {
      MS_LOG(WARNING) << "Conv2D don't support non-constant filter yet.";
      return RET_ERROR;
    }
  }

  auto *bias_tensor = in_tensors_.size() >= INPUT_TENSOR_SIZE_3 ? in_tensors_.at(kBiasIndex) : nullptr;
  if (bias_tensor != nullptr && !bias_tensor->IsConst()) {
    bool is_const = bias_tensor->category() == lite::Category::CONST_TENSOR ||
                    bias_tensor->category() == lite::Category::CONST_SCALAR;
    if (!(is_const && stored_bias_)) {
      MS_LOG(WARNING) << "Conv2D don't support non-constant bias yet.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int Conv2DOpenCLKernel::Prepare() {
  auto ret = InitAttrs();
  if (ret != RET_OK) {
    return ret;
  }
  ret = BuildKernel();
  if (ret != RET_OK) {
    return ret;
  }
  ret = InitWeights();
  if (ret != RET_OK) {
    return ret;
  }
  (void)SetGlobalLocal();
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int Conv2DOpenCLKernel::InitAttrs() {
  CHECK_NULL_RETURN(ocl_runtime_);
#ifdef ENABLE_FP16
  use_fp16_ = ocl_runtime_->GetFp16Enable();
  sizeof_FLT_ = use_fp16_ ? sizeof(float16_t) : sizeof(float);
#else
  sizeof_FLT_ = sizeof(float);
#endif
  CHECK_NULL_RETURN(in_tensors_.front());
  CHECK_NULL_RETURN(out_tensors_.front());
  auto input_shape = in_tensors_.front()->shape();
  auto output_shape = out_tensors_.front()->shape();
  CHECK_LESS_RETURN(input_shape.size(), C4NUM);
  batch_size_ = input_shape[0];
  IH_ = input_shape[kNHWC_H];
  IW_ = input_shape[kNHWC_W];
  CI_ = input_shape[kNHWC_C];
  // for fusion Conv2D and Reshape(N11C->NC)
  if (output_shape.size() == kNHWC_W) {
    OH_ = 1;
    OW_ = 1;
    CO_ = output_shape[kNHWC_H];
  } else {  // output_shape.size() == C4NUM
    OH_ = output_shape[kNHWC_H];
    OW_ = output_shape[kNHWC_W];
    CO_ = output_shape[kNHWC_C];
  }
  CI_SLICES_ = UP_DIV(CI_, CI_TILE);
  CO_SLICES_ = UP_DIV(CO_, CO_TILE);
  CHECK_NULL_RETURN(param_);
  KH_ = param_->kernel_h_;
  KW_ = param_->kernel_w_;
  // note: TILE_HW_ is only used when use_winograd_=true
  TILE_HW_ = UP_DIV(OW_, C4NUM) * UP_DIV(OH_, C4NUM);
  return RET_OK;
}

int Conv2DOpenCLKernel::BuildKernel() {
  SetBlockSize();
  const std::string program_name = "conv2d";
  std::stringstream kernel_name;
  kernel_name << "Conv2D_H" << block_size_.H << "W" << block_size_.W << "C" << block_size_.C;
  if (filter_type_ == MemType::IMG) {
    kernel_name << "_Img";
  }
  if (KW_ == 1 && KH_ == 1) {
    kernel_name << "_1x1";
  }
  if (!ocl_runtime_->LoadSource(program_name, GetActDefines() + conv2d_source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);

  std::string exceed_max_image_width_option =
    (OW_ * CO_SLICES_ <= static_cast<int>(ocl_runtime_->GetMaxImage2DWidth())) ? "" : " -DEXCEDD_MAX_IMAGE2D_WIDTH";
  build_options_ext.push_back(exceed_max_image_width_option);
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name.str(), build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  return RET_OK;
}

void Conv2DOpenCLKernel::SetBlockSize() {
  if (filter_type_ == MemType::IMG) {
    block_size_ = {2, 2, 2};
    return;
  }
  auto task_size = static_cast<float>(batch_size_ * OH_ * OW_ * CO_SLICES_);
  MS_ASSERT(ocl_runtime_->DeviceComputeUnits());
  auto task_size_per_cu = task_size / ocl_runtime_->DeviceComputeUnits();
  bool w_kernel_is_1 =
    KW_ == 1 && param_->stride_w_ == 1 && param_->dilation_w_ == 1 && param_->pad_l_ == 0 && param_->pad_r_ == 0;
  bool h_kernel_is_1 =
    KH_ == 1 && param_->stride_h_ == 1 && param_->dilation_h_ == 1 && param_->pad_u_ == 0 && param_->pad_d_ == 0;
#ifdef ENABLE_FP16
  if (use_fp16_) {
    SetMaliFp16BlockSize(task_size_per_cu, w_kernel_is_1, h_kernel_is_1);
  } else {
    SetMaliFp32BlockSize(task_size_per_cu, w_kernel_is_1, h_kernel_is_1);
  }
#else
  SetMaliFp32BlockSize(task_size_per_cu, w_kernel_is_1, h_kernel_is_1);
#endif
}

void Conv2DOpenCLKernel::SetMaliFp32BlockSize(int task_size_per_cu, bool w_kernel_is_1, bool h_kernel_is_1) {
  int block_size;
  if (task_size_per_cu <= C256NUM) {
    block_size = C1NUM;
  } else if (task_size_per_cu <= C256NUM * C4NUM) {
    block_size = C2NUM;
  } else if (task_size_per_cu <= FLT_MAX) {
    block_size = C4NUM;
  } else {
    block_size = C8NUM;
  }

  if (!w_kernel_is_1 || !h_kernel_is_1) {
    block_size = std::min(block_size, C4NUM);
  }

  if (block_size == C8NUM) {
    block_size_ = {2, 2, 2};
  } else if (block_size == C4NUM) {
    block_size_ = {2, 2, 1};
  } else if (block_size == C2NUM) {
    block_size_ = {2, 1, 1};
  } else {
    block_size_ = {1, 1, 1};
  }
}
void Conv2DOpenCLKernel::SetMaliFp16BlockSize(int task_size_per_cu, bool w_kernel_is_1, bool h_kernel_is_1) {
  int block_size;
  if (task_size_per_cu <= C256NUM) {
    block_size = C1NUM;
  } else if (task_size_per_cu <= C256NUM * C4NUM) {
    block_size = C2NUM;
  } else if (task_size_per_cu <= C256NUM * C8NUM) {
    block_size = C4NUM;
  } else {
    block_size = C8NUM;
  }

  if (!w_kernel_is_1 || !h_kernel_is_1) {
    block_size = std::min(block_size, C4NUM);
  }

  if (CO_SLICES_ >= C128NUM && OH_ >= 10 && OW_ >= 10) {  // out hw > 10x10
    block_size = C8NUM;
  }

  if (block_size == C8NUM) {
    block_size_ = {2, 2, 2};
  } else if (block_size == C4NUM) {
    block_size_ = {2, 1, 2};
  } else if (block_size == C2NUM) {
    block_size_ = {2, 1, 1};
  } else {
    block_size_ = {1, 1, 1};
  }
}

int Conv2DOpenCLKernel::InitWeights() {
  if (InitFilter() != RET_OK) {
    MS_LOG(ERROR) << "init filter failed.";
    return RET_ERROR;
  }
  if (InitBias() != RET_OK) {
    MS_LOG(ERROR) << "init bias failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

#ifdef ENABLE_FP16
void ConvertFilter(void *src, void *dst, TypeId src_dtype, TypeId dst_dtype, FilterFormat src_format,
                   FilterFormat dst_format, size_t CO, size_t KH, size_t KW, size_t CI, size_t OGroup) {
  MS_ASSERT(src);
  MS_ASSERT(dst);
  MS_ASSERT(src_dtype == kNumberTypeFloat16 || src_dtype == kNumberTypeFloat32);
  MS_ASSERT(dst_dtype == kNumberTypeFloat16 || dst_dtype == kNumberTypeFloat32);
  MS_ASSERT(src_format == OHWI);
  MS_ASSERT(dst_format == HWII4OO4 || dst_format == OHWIOgroupI4O4);
  auto src_fp16 = reinterpret_cast<float16_t *>(src);
  auto src_fp32 = reinterpret_cast<float32_t *>(src);
  auto dst_fp16 = reinterpret_cast<float16_t *>(dst);
  auto dst_fp32 = reinterpret_cast<float32_t *>(dst);
  bool src_is_fp16 = src_dtype == kNumberTypeFloat16;
  bool dst_is_fp16 = dst_dtype == kNumberTypeFloat16;
  auto CI_SLICES = UP_DIV(CI, CI_TILE);
  auto CO_SLICES = UP_DIV(CO, CO_TILE);
  for (size_t co = 0, src_idx = 0; co < CO; ++co) {
    for (size_t kh = 0; kh < KH; ++kh) {
      for (size_t kw = 0; kw < KW; ++kw) {
        for (size_t ci = 0; ci < CI; ++ci, ++src_idx) {
          size_t ci_slice = ci / CI_TILE;
          size_t ci_inner = ci % CI_TILE;
          size_t dst_idx = 0;
          size_t co_inner = co % CO_TILE;
          if (dst_format == OHWIOgroupI4O4) {
            size_t co_slice = co / (CO_TILE * OGroup);
            size_t group_idx = co % (CO_TILE * OGroup) / CO_TILE;
            dst_idx =
              (((((co_slice * KH + kh) * KW + kw) * CI_SLICES + ci_slice) * OGroup + group_idx) * CI_TILE + ci_inner) *
                CO_TILE +
              co_inner;
          } else {  // if (dst_format == HWII4OO4)
            size_t co_slice = co / CO_TILE;
            dst_idx =
              ((((kh * KW + kw) * CI_SLICES + ci_slice) * CI_TILE + ci_inner) * CO_SLICES + co_slice) * CO_TILE +
              co_inner;
          }
          if (dst_is_fp16) {
            dst_fp16[dst_idx] = src_is_fp16 ? src_fp16[src_idx] : static_cast<float16_t>(src_fp32[src_idx]);
          } else {
            dst_fp32[dst_idx] = src_is_fp16 ? static_cast<float32_t>(src_fp16[src_idx]) : src_fp32[src_idx];
          }
        }
      }
    }
  }
}
#else
void ConvertFilter(void *src, void *dst, TypeId src_dtype, TypeId dst_dtype, FilterFormat src_format,
                   FilterFormat dst_format, size_t CO, size_t KH, size_t KW, size_t CI, size_t OGroup) {
  MS_ASSERT(src);
  MS_ASSERT(dst);
  MS_ASSERT(src_format == OHWI);
  MS_ASSERT(dst_format == HWII4OO4 || dst_format == OHWIOgroupI4O4);
  auto src_fp32 = reinterpret_cast<float *>(src);
  auto dst_fp32 = reinterpret_cast<float *>(dst);
  auto CI_SLICES = UP_DIV(CI, CI_TILE);
  auto CO_SLICES = UP_DIV(CO, CO_TILE);
  for (size_t co = 0, src_idx = 0; co < CO; ++co) {
    for (size_t kh = 0; kh < KH; ++kh) {
      for (size_t kw = 0; kw < KW; ++kw) {
        for (size_t ci = 0; ci < CI; ++ci, ++src_idx) {
          size_t dst_idx = 0;
          size_t co_inner = co % CO_TILE;
          size_t ci_slice = ci / CI_TILE;
          size_t ci_inner = ci % CI_TILE;
          if (dst_format == OHWIOgroupI4O4) {
            size_t co_slice = co / (CO_TILE * OGroup);
            size_t group_idx = co % (CO_TILE * OGroup) / CO_TILE;
            dst_idx =
              (((((co_slice * KH + kh) * KW + kw) * CI_SLICES + ci_slice) * OGroup + group_idx) * CI_TILE + ci_inner) *
                CO_TILE +
              co_inner;
          } else {  // if(dst_format==HWII4OO4)
            size_t co_slice = co / CO_TILE;
            dst_idx =
              ((((kh * KW + kw) * CI_SLICES + ci_slice) * CI_TILE + ci_inner) * CO_SLICES + co_slice) * CO_TILE +
              co_inner;
          }
          dst_fp32[dst_idx] = src_fp32[src_idx];
        }
      }
    }
  }
}
#endif

int Conv2DOpenCLKernel::InitFilter() {
  auto allocator = ocl_runtime_->GetAllocator();

  // allocate opencl memory: buffer or image2d
  size_t size = 0;
  int Ogroup = block_size_.C;
  if (filter_type_ == MemType::IMG) {
    size_t width = CO_SLICES_;
    size_t height = KH_ * KW_ * UP_ROUND(CI_, CI_TILE);
    size_t dtype = use_fp16_ ? CL_HALF_FLOAT : CL_FLOAT;
    size = width * height * CO_TILE * sizeof_FLT_;
    packed_filter_ = allocator->Malloc({width, height, dtype});
    if (packed_filter_ == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
  } else {
    size = UP_DIV(CO_SLICES_, Ogroup) * KH_ * KW_ * CI_SLICES_ * Ogroup * CI_TILE * CO_TILE * sizeof_FLT_;
    packed_filter_ = allocator->Malloc(size, lite::opencl::MemType::BUF);
    if (packed_filter_ == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
  }

  // rearrange filter
  auto filter_tensor = in_tensors_.at(1);
  CHECK_NULL_RETURN(filter_tensor);
  void *src_data = stored_filter_ == nullptr ? filter_tensor->data() : stored_filter_;
  CHECK_NULL_RETURN(src_data);
  auto src_dtype = filter_tensor->data_type();
  auto dst_dtype = use_fp16_ ? kNumberTypeFloat16 : kNumberTypeFloat32;
  std::vector<char> tmp(size, 0);
  if (filter_type_ == MemType::IMG) {
    ConvertFilter(src_data, tmp.data(), src_dtype, dst_dtype, OHWI, HWII4OO4, CO_, KH_, KW_, CI_);
  } else {
    ConvertFilter(src_data, tmp.data(), src_dtype, dst_dtype, OHWI, OHWIOgroupI4O4, CO_, KH_, KW_, CI_, Ogroup);
  }

  // unmap
  if (filter_type_ == MemType::IMG) {
    ocl_runtime_->WriteImage(packed_filter_, tmp.data());
  } else {
    if (allocator->MapBuffer(packed_filter_, CL_MAP_WRITE, nullptr, true) == nullptr) {
      MS_LOG(ERROR) << "Map Buffer failed.";
      return RET_ERROR;
    }
    memcpy(packed_filter_, tmp.data(), size);
    if (allocator->UnmapBuffer(packed_filter_) != RET_OK) {
      MS_LOG(ERROR) << "UnmapBuffer failed.";
      return RET_ERROR;
    }
  }

  FreeStoredData(stored_filter_);
  return RET_OK;
}

int Conv2DOpenCLKernel::InitBias() {
  // align bias from C to C4
  auto allocator = ocl_runtime_->GetAllocator();
  size_t packed_bias_size = UP_ROUND(CO_SLICES_, block_size_.C) * CO_TILE * sizeof_FLT_;
  packed_bias_ = allocator->Malloc(packed_bias_size, lite::opencl::MemType::BUF);
  if (packed_bias_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }

  if (allocator->MapBuffer(packed_bias_, CL_MAP_WRITE, nullptr, true) == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  memset(packed_bias_, 0x00, packed_bias_size);
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3) {
    auto bias_tensor = in_tensors_.at(DIMENSION_2D);
    void *src_data = stored_bias_ == nullptr ? bias_tensor->data() : stored_bias_;
    MS_ASSERT(src_data);

#ifdef ENABLE_FP16
    if (bias_tensor->data_type() == kNumberTypeFloat16) {
      if (use_fp16_) {
        memcpy(packed_bias_, src_data, CO_ * sizeof_FLT_);
      } else {
        auto packed_bias_fp32 = reinterpret_cast<float *>(packed_bias_);
        auto origin_bias_fp16 = reinterpret_cast<float16_t *>(src_data);
        MS_ASSERT(origin_bias_fp16);
        for (int i = 0; i < CO_; ++i) {
          packed_bias_fp32[i] = static_cast<float>(origin_bias_fp16[i]);
        }
      }
    } else {
      if (use_fp16_) {
        auto packed_bias_fp16 = reinterpret_cast<float16_t *>(packed_bias_);
        auto origin_bias_fp32 = reinterpret_cast<float *>(src_data);
        MS_ASSERT(origin_bias_fp32);
        for (int i = 0; i < CO_; ++i) {
          packed_bias_fp16[i] = static_cast<float16_t>(origin_bias_fp32[i]);
        }
      } else {
        memcpy(packed_bias_, src_data, CO_ * sizeof_FLT_);
      }
    }
#else
    memcpy(packed_bias_, src_data, CO_ * sizeof_FLT_);
#endif
  }
  if (allocator->UnmapBuffer(packed_bias_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  FreeStoredData(stored_bias_);
  return RET_OK;
}

int Conv2DOpenCLKernel::SetConstArgs() {
  cl_int4 input_shape = {batch_size_, IH_, IW_, CI_SLICES_};
  cl_int4 output_shape = {batch_size_, OH_, OW_, CO_SLICES_};
  cl_int4 kernel_stride = {KH_, KW_, param_->stride_h_, param_->stride_w_};
  cl_int4 pad = {param_->pad_u_, param_->pad_d_, param_->pad_l_, param_->pad_r_};
  cl_int2 dilation = {param_->dilation_h_, param_->dilation_w_};

  int arg_cn = 2;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, packed_filter_, (filter_type_ == lite::opencl::MemType::BUF)) !=
      CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, packed_bias_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, output_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, kernel_stride) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, pad) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, dilation) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, param_->act_type_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn, alpha_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int Conv2DOpenCLKernel::SetGlobalLocal() {
  size_t global_h = batch_size_ * UP_DIV(OH_, block_size_.H);
  size_t global_w = UP_DIV(OW_, block_size_.W);
  size_t global_c = UP_DIV(CO_SLICES_, block_size_.C);
  int local_max = filter_type_ == MemType::IMG ? C64NUM : C128NUM;
  if (ocl_runtime_->DeviceComputeUnits() > C16NUM) {
    local_max = C256NUM;
  }
  const int local_c_max = 16;
  const int OH_threshold = 100;
  const int OW_threshold = 100;
  const int OC_threshold = 64;
  size_t local_c = GetMaxDivisor(global_c, local_c_max);
  local_c = std::max<size_t>(local_c, 1);
  size_t local_hw = local_max / local_c;
  size_t local_h;
  size_t local_w;
  if (OH_ >= OH_threshold && OW_ >= OW_threshold && CO_ <= OC_threshold) {  // c -> w -> h
    local_w = std::min(global_w, local_hw);
    local_h = std::min(local_hw / local_w, global_h);
  } else {  // c -> h -> w
    local_h = std::min(global_h, local_hw);
    local_w = std::min(local_hw / local_h, global_w);
  }
  global_size_ = {global_h, global_w, global_c};
  local_size_ = {local_h, local_w, local_c};
  AlignGlobalLocal(global_size_, local_size_);

  return RET_OK;
}

int Conv2DOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  if (ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_.front()->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_.front()->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

std::vector<BaseTuningParameter> Conv2DOpenCLKernel::GenerateTuningParam() {
  // don't need to tune local_c
  std::vector<BaseTuningParameter> tuning_params = {};
  BaseTuningParameter default_tuning_param = BaseTuningParameter();
  default_tuning_param.local_size = local_size_;
  tuning_params.push_back(default_tuning_param);

  std::vector<size_t> max_work_items = ocl_runtime_->GetWorkItemSize();
  size_t max_workgroup_size = ocl_runtime_->GetMaxWorkGroupSize(kernel_);
  std::set<size_t> candidate_x = GenerateLocalByGlobal(global_size_[0]);
  std::set<size_t> candidate_y = GenerateLocalByGlobal(global_size_[1]);
  for (auto x : candidate_x) {
    if (x <= max_work_items[0]) {
      for (auto y : candidate_y) {
        if (y <= max_work_items[1]) {
          auto group_size = x * y * local_size_[DIMENSION_2D];
          if (group_size <= max_workgroup_size) {
            BaseTuningParameter tuning_param = BaseTuningParameter();
            tuning_param.local_size = {x, y, local_size_[DIMENSION_2D]};
            tuning_params.push_back(tuning_param);
          }
        }
      }
    }
  }
  return tuning_params;
}

int Conv2DOpenCLKernel::StoreConstData() {
  if (!InferShapeDone()) {
    stored_filter_ = StoreTensorData(in_tensors_.at(kWeightIndex));
    if (stored_filter_ == nullptr) {
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

bool UseFcReplaceConv(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                      ConvParameter *param) {
  MS_ASSERT(param);
  MS_ASSERT(!inputs.empty());
  MS_ASSERT(!outputs.empty());
  auto input_shape = inputs.front()->shape();
  auto output_shape = inputs.front()->shape();
  // IH=1 IW=1 OH=1 OW=1
  bool hw_is_1 = input_shape.size() == DIMENSION_4D && input_shape[kNHWC_H] == 1 && input_shape[kNHWC_W] == 1 &&
                 output_shape.size() == DIMENSION_4D && output_shape[kNHWC_H] == 1 && output_shape[kNHWC_W] == 1;
  bool attr_valid = param->kernel_h_ == 1 && param->kernel_w_ == 1 && param->stride_h_ == 1 && param->stride_w_ == 1 &&
                    param->pad_u_ == 0 && param->pad_d_ == 0 && param->pad_l_ == 0 && param->pad_r_ == 0 &&
                    param->dilation_h_ == 1 && param->dilation_w_ == 1;
  return hw_is_1 && attr_valid;
}

OpParameter *CreateFcParam(const ConvParameter *conv_param, const std::vector<lite::Tensor *> &inputs) {
  auto fc_param = static_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (fc_param == nullptr) {
    MS_LOG(ERROR) << "Create FullConnection kernel param failed.";
    return nullptr;
  }
  fc_param->op_parameter_.type_ = PrimitiveType_FullConnection;
  fc_param->a_transpose_ = false;
  fc_param->b_transpose_ = true;
  fc_param->act_type_ = conv_param->act_type_;
  fc_param->has_bias_ = inputs.size() == DIMENSION_3D;
  return reinterpret_cast<OpParameter *>(fc_param);
}

bool UseWinograd4x4To6x6(const ConvParameter *param, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs) {
  if (!(inputs.size() == DIMENSION_2D || inputs.size() == DIMENSION_3D) || outputs.empty()) {
    return false;
  }
  auto input_shape = inputs.front()->shape();
  auto output_shape = outputs.front()->shape();
  if (input_shape.size() != DIMENSION_4D ||
      (output_shape.size() != DIMENSION_2D && output_shape.size() != DIMENSION_4D)) {
    return false;
  }
  int batch_size = input_shape[kNHWC_N];
  int IH = input_shape[kNHWC_H];
  int IW = input_shape[kNHWC_W];
  int CI = input_shape[kNHWC_C];
  int OH = output_shape.size() == DIMENSION_2D ? 1 : output_shape[kNHWC_H];
  int OW = output_shape.size() == DIMENSION_2D ? 1 : output_shape[kNHWC_W];
  int CO = output_shape.size() == DIMENSION_2D ? output_shape[kNHWC_H] : output_shape[kNHWC_C];
  int CI_SLICES = UP_DIV(CI, CI_TILE);
  int CO_SLICES = UP_DIV(CO, CO_TILE);
  int TILE_HW_ = UP_DIV(OH, C4NUM) * UP_DIV(OW, C4NUM);

  bool pad_is_all_0 = param->pad_u_ == 0 && param->pad_d_ == 0 && param->pad_l_ == 0 && param->pad_r_ == 0;
  bool pad_is_all_1 = param->pad_u_ == 1 && param->pad_d_ == 1 && param->pad_l_ == 1 && param->pad_r_ == 1;
  bool attr_valid = param->kernel_h_ == 3 && param->kernel_w_ == 3 && param->stride_h_ == 1 &&  // kernel 3x3
                    param->stride_w_ == 1 && param->dilation_h_ == 1 && param->dilation_w_ == 1 &&
                    (pad_is_all_0 || pad_is_all_1);

  bool shape_valid = false;
  if (pad_is_all_1) {
    shape_valid = batch_size == 1 && IH == OH && IW == OW;
  } else if (pad_is_all_0) {
    shape_valid = batch_size == 1 && IH - 2 == OH && IW - 2 == OW;  // 2 : left 1 + right 1, top 1 + button 1
  }

  bool channel_good = CI_SLICES >= C8NUM && CO_SLICES >= C8NUM;
  bool hw_good = TILE_HW_ >= C16NUM;
  return attr_valid && shape_valid && channel_good && hw_good;
}

kernel::LiteKernel *OpenCLConv2DCreator(const std::vector<lite::Tensor *> &inputs,
                                        const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                        const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(!inputs.empty());
  MS_ASSERT(!outputs.empty());
  MS_ASSERT(opParameter);
  auto *conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  int input_channel = conv_param->input_channel_;
  int output_channel = conv_param->output_channel_;
  int group = conv_param->group_;

  kernel::OpenCLKernel *kernel = nullptr;
  // case 1: depthwise conv2d
  if (group == input_channel && group == output_channel) {
    kernel =
      new (std::nothrow) DepthwiseConv2dOpenCLKernel(reinterpret_cast<OpParameter *>(conv_param), inputs, outputs, ctx);
    auto ret = reinterpret_cast<DepthwiseConv2dOpenCLKernel *>(kernel)->StoreConstData();
    if (ret != mindspore::lite::RET_OK) {
      MS_LOG(ERROR) << "Store " << opParameter->name_ << " const data failed!";
      delete kernel;
      return nullptr;
    }
    return kernel;
  }

  // case 2: group conv2d
  if (group != 1) {
    MS_LOG(ERROR) << "OpenCL doesn't support group conv2d.";
    free(conv_param);
    return nullptr;
  }

  // case 3: common conv2d
  auto shape = outputs.front()->shape();
  bool infer_shape_done = std::find(shape.begin(), shape.end(), -1) == shape.end();
  if (infer_shape_done && UseWinograd4x4To6x6(conv_param, inputs, outputs)) {
    MS_LOG(DEBUG) << "use Winograd algorithm.";
    kernel = new (std::nothrow) WinogradOpenCLKernel(reinterpret_cast<OpParameter *>(conv_param), inputs, outputs, ctx);
  } else {
    kernel = new (std::nothrow) Conv2DOpenCLKernel(reinterpret_cast<OpParameter *>(conv_param), inputs, outputs, ctx);
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create Convolution kernel failed.";
    free(conv_param);
    return nullptr;
  }
  if (!infer_shape_done) {
    auto ret = reinterpret_cast<Conv2DOpenCLKernel *>(kernel)->StoreConstData();
    if (ret != mindspore::lite::RET_OK) {
      MS_LOG(ERROR) << "Store " << opParameter->name_ << " const data failed!";
      delete kernel;
      return nullptr;
    }
    return kernel;
  }
  if (kernel->CheckSpecs() != RET_OK || kernel->OpenCLKernel::CheckSpecs() != RET_OK) {
    MS_LOG(ERROR) << "Init Convolution kernel failed.";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Conv2DFusion, OpenCLConv2DCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Conv2DFusion, OpenCLConv2DCreator)
}  // namespace mindspore::kernel
