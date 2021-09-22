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
#include <deque>
#include <string>
#include <algorithm>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/strided_slice.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/strided_slice.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SliceFusion;
using mindspore::schema::PrimitiveType_StridedSlice;

namespace mindspore::kernel {
int StridedSliceOpenCLKernel::CheckSpecs() {
  if (type() == PrimitiveType_SliceFusion) {
    if (in_tensors_.size() != INPUT_TENSOR_SIZE_3) {
      MS_LOG(WARNING) << "Slice only supports 3 input Tensor.";
      return RET_ERROR;
    }
    int in_ndim = in_tensors_.front()->shape().size();
    if (CheckParamLikeTensor("Slice", "begin", in_tensors_.at(1), kNumberTypeInt32, {in_ndim}) != RET_OK) {
      return RET_ERROR;
    }
    if (CheckParamLikeTensor("Slice", "size", in_tensors_.at(2), kNumberTypeInt32, {in_ndim}) != RET_OK) {
      return RET_ERROR;
    }
  } else if (type() == PrimitiveType_StridedSlice) {
    if (in_tensors_.size() != INPUT_TENSOR_SIZE_4) {
      MS_LOG(WARNING) << "StridedSlice only supports 4 input Tensor.";
      return RET_ERROR;
    }
    int in_ndim = in_tensors_.front()->shape().size();
    if (CheckParamLikeTensor("StridedSlice", "begin", in_tensors_.at(1), kNumberTypeInt32, {in_ndim}) != RET_OK) {
      return RET_ERROR;
    }
    if (CheckParamLikeTensor("StridedSlice", "end", in_tensors_.at(2), kNumberTypeInt32, {in_ndim}) != RET_OK) {
      return RET_ERROR;
    }
    if (CheckParamLikeTensor("StridedSlice", "stride", in_tensors_.at(3), kNumberTypeInt32, {in_ndim}) != RET_OK) {
      return RET_ERROR;
    }
  } else {
    MS_LOG(WARNING) << "type error.";
    return RET_ERROR;
  }
  const std::string kernel_name = type() == PrimitiveType_SliceFusion ? "Slice" : "StridedSlice";
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << kernel_name + " only supports 1 output Tensor.";
    return RET_ERROR;
  }
  auto in_ndim = in_tensors_.front()->shape().size();
  if (in_ndim == 0 || in_ndim > DIMENSION_4D) {
    MS_LOG(WARNING) << kernel_name + " only supports 1D-4D input tensor";
    return RET_ERROR;
  }
  auto out_ndim = out_tensors_.front()->shape().size();
  if (out_ndim > DIMENSION_4D) {
    MS_LOG(WARNING) << kernel_name + " only supports 0D-4D output tensor";
    return RET_ERROR;
  }
  if (InitConstArgs() != RET_OK) {
    MS_LOG(WARNING) << "call InitConstArgs() failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int StridedSliceOpenCLKernel::Prepare() {
  const std::string program_name = "strided_slice";
  if (!ocl_runtime_->LoadSource(program_name, strided_slice_source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, "strided_slice", build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  SetGlobalLocal();
  return RET_OK;
}

int StridedSliceOpenCLKernel::InitConstArgs() {
  auto input_info = GpuTensorInfo(in_tensors_.front());
  auto output_info = GpuTensorInfo(out_tensors_.front());
  input_shape_ = {static_cast<cl_int>(input_info.N), static_cast<cl_int>(input_info.H),
                  static_cast<cl_int>(input_info.W), static_cast<cl_int>(input_info.C)};
  output_shape_ = {static_cast<cl_int>(output_info.N), static_cast<cl_int>(output_info.H),
                   static_cast<cl_int>(output_info.W), static_cast<cl_int>(output_info.C)};
  io_slices_ = {static_cast<cl_int>(input_info.Slice), static_cast<cl_int>(output_info.Slice)};

  if (type() == PrimitiveType_SliceFusion) {
    auto *begin = reinterpret_cast<int32_t *>(in_tensors_.at(1)->data());
    MS_ASSERT(begin);
    auto *size = reinterpret_cast<int32_t *>(in_tensors_.at(2)->data());
    MS_ASSERT(size);
    Broadcast2GpuShape(begin_.s, begin, input_info.NDim, 0);
    Broadcast2GpuShape(size_.s, size, input_info.NDim, -1);
    for (int i = 0; i < 4; ++i) {
      if (begin_.s[i] < 0) {
        begin_.s[i] += input_shape_.s[i];
      }
      if (begin_.s[i] < 0 || begin_.s[i] >= input_shape_.s[i]) {
        MS_LOG(ERROR) << "Slice only supports 0<=begin<input_shape but begin[i]=" << begin_.s[i]
                      << " input_shape[i]=" << input_shape_.s[i];
        return RET_ERROR;
      }
      if (size_.s[i] < -1 || size_.s[i] == 0) {
        MS_LOG(ERROR) << "Slice only supports size=-1 or size>0 but size[i]=" << size_.s[i];
        return RET_ERROR;
      }
      if (size_.s[i] == -1 || begin_.s[i] + size_.s[i] > input_shape_.s[i]) {
        size_.s[i] = input_shape_.s[i] - begin_.s[i];
      }
    }
  } else {
    auto *begin = reinterpret_cast<int32_t *>(in_tensors_.at(1)->data());
    MS_ASSERT(begin);
    auto *end = reinterpret_cast<int32_t *>(in_tensors_.at(2)->data());
    MS_ASSERT(end);
    auto *stride = reinterpret_cast<int32_t *>(in_tensors_.at(3)->data());
    MS_ASSERT(stride);
    cl_int4 end_ = input_shape_;
    Broadcast2GpuShape(begin_.s, begin, input_info.NDim, 0);
    Broadcast2GpuShape(end_.s, end, input_info.NDim);
    Broadcast2GpuShape(stride_.s, stride, input_info.NDim, 1);

    for (int i = 0; i < 4; ++i) {
      // begin is negative
      if (begin_.s[i] < 0) {
        begin_.s[i] += input_shape_.s[i];
      }
      // avoid begin is out of range
      begin_.s[i] = std::clamp(begin_.s[i], 0, input_shape_.s[i] - 1);
      // end is negative
      if (end_.s[i] <= 0) {
        end_.s[i] += input_shape_.s[i];
      }
      // avoid end is out of range
      end_.s[i] = std::clamp(end_.s[i], 0, input_shape_.s[i]);

      // check stride begin end
      if (stride_.s[i] > 0) {
        if (begin_.s[i] >= end_.s[i]) {
          MS_LOG(ERROR) << "StridedSlice kernel only supports begin_<end when stride>0";
          return RET_ERROR;
        }
      } else if (stride_.s[i] < 0) {
        if (begin_.s[i] <= end_.s[i]) {
          MS_LOG(ERROR) << "StridedSlice kernel only supports begin_>end when stride<0";
          return RET_ERROR;
        }
      } else {
        MS_LOG(ERROR) << "StridedSlice kernel only supports stride!=0";
        return RET_ERROR;
      }
      size_.s[i] = std::ceil(static_cast<float>(end_.s[i] - begin_.s[i]) / static_cast<float>(stride_.s[i]));
    }
  }

  // check size
  std::vector<int> shape_not_1;
  std::vector<int> size_not_1;
  auto output_shape = out_tensors_.front()->shape();
  std::copy_if(output_shape.begin(), output_shape.end(), std::back_inserter(shape_not_1), [](int x) { return x > 1; });
  std::copy_if(size_.s, size_.s + 4, std::back_inserter(size_not_1), [](int x) { return x > 1; });
  if (shape_not_1 != size_not_1) {
    MS_LOG(ERROR) << "Slice/StridedSlice kernel output shape infer error";
    return RET_ERROR;
  }
  return RET_OK;
}

int StridedSliceOpenCLKernel::SetConstArgs() {
  int arg_cn = 2;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_shape_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, output_shape_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, io_slices_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, begin_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, stride_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn, size_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void StridedSliceOpenCLKernel::SetGlobalLocal() {
  auto output_info = GpuTensorInfo(out_tensors_.front());
  global_size_ = {output_info.N * output_info.H, output_info.W, output_info.Slice};

  const int max_divider = 8;
  auto max_work_group_size = ocl_runtime_->DeviceMaxWorkGroupSize();
  size_t local_c = GetMaxDivisorStrategy0(global_size_[2], max_divider);
  local_c = std::max<size_t>(local_c, 1);
  size_t local_hw = max_work_group_size / local_c;
  size_t local_h = std::min(UP_DIV(global_size_[0], 2), local_hw);
  size_t local_w = std::min(local_hw / local_h, global_size_[1]);
  local_size_ = {local_h, local_w, local_c};
  AlignGlobalLocal(global_size_, local_size_);
}

int StridedSliceOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
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

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_SliceFusion, OpenCLKernelCreator<StridedSliceOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_SliceFusion, OpenCLKernelCreator<StridedSliceOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_StridedSlice, OpenCLKernelCreator<StridedSliceOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_StridedSlice, OpenCLKernelCreator<StridedSliceOpenCLKernel>);
}  // namespace mindspore::kernel
