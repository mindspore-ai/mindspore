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

#include "src/litert/kernel/opencl/kernel/concat.h"
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/opencl/utils.h"

const std::vector<std::string> cl_index_str = {".x", ".y", ".z", ".w"};

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore {
namespace kernel {
int ConcatOpenCLKernel::RunAxis0() {
  auto allocator_ = ocl_runtime_->GetAllocator();
  ImageSize img_size;
  auto dst_data = out_tensors_[0]->data();
  MS_ASSERT(dst_data);
  auto dst_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
  auto *out_image = allocator_->GetImage(dst_data);
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    auto src_data = weight_ptrs_.at(i) == nullptr ? in_tensors_[i]->data() : weight_ptrs_.at(i);
    if (allocator_->GetImageSize(src_data, &img_size) != RET_OK) {
      MS_LOG(WARNING) << "GetImageSize failed.";
      return RET_ERROR;
    }
    auto src_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
    auto region = cl::array<cl::size_type, 3U>{img_size.width, img_size.height, 1};
    auto *input_image = allocator_->GetImage(src_data);
    if (ocl_runtime_->GetDefaultCommandQueue()->enqueueCopyImage(*input_image, *out_image, src_origin, dst_origin,
                                                                 region) != CL_SUCCESS) {
      MS_LOG(WARNING) << "enqueueCopyImage failed.";
    }
    dst_origin[1] += region[1];
  }
  return RET_OK;
}

void ConcatGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
  const int max_divider = 8;
  const int max_x = 2, max_y = 8;
  int x = std::min(GetMaxDivisorStrategy1(global[0], max_divider), max_x);
  if (x == 0) {
    return;
  }
  int yz = max_size / x;
  int y = std::min(std::min(GetMaxDivisorStrategy1(global[1], max_divider), yz), max_y);
  if (y == 0) {
    return;
  }
  int z = std::min(yz / y, static_cast<int>(UP_DIV(global[2], 2)));

  local->clear();
  local->push_back(x);
  local->push_back(y);
  local->push_back(z);
}

int ConcatOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() < INPUT_TENSOR_SIZE_2 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  std::set<lite::opencl::GpuType> mali_devices = {lite::opencl::MALI, lite::opencl::MALI_T, lite::opencl::MALI_G,
                                                  lite::opencl::MALI_G78};
  auto cur_gpu_type = ocl_runtime_->GetGpuInfo().type;
  if ((mali_devices.find(cur_gpu_type) != mali_devices.end()) && (in_tensors_.size() > INPUT_TENSOR_SIZE_16)) {
    MS_LOG(WARNING) << "For MALI serial, the size of inputs should be no more than 16, but got " << in_tensors_.size()
                    << "in Concat kernel.";
    return RET_ERROR;
  }
  auto param = reinterpret_cast<ConcatParameter *>(this->op_parameter_);
  auto out_tensors_shape_size = out_tensors_[0]->shape().size();
  MS_LOG(DEBUG) << " concat at axis = " << param->axis_;
  if (out_tensors_shape_size > DIMENSION_4D) {
    MS_LOG(WARNING) << " GPU Unsupported shape.size > 4 ";
    return RET_ERROR;
  }

  auto out_tensor_info = GpuTensorInfo(out_tensors_[0]);
  auto height = out_tensor_info.N * out_tensor_info.D * out_tensor_info.H;
  auto width = out_tensor_info.W * out_tensor_info.Slice;
  if ((height > ocl_runtime_->GetMaxImage2DHeight()) || (width > ocl_runtime_->GetMaxImage2DWidth())) {
    MS_LOG(WARNING) << "Output tensor is too larger to use OpenCL in Concat kernel.";
    return RET_ERROR;
  }

  for (auto &in_tensor : in_tensors_) {
    auto in_tensors_shape_size = in_tensor->shape().size();
    if (in_tensors_shape_size > DIMENSION_4D) {
      MS_LOG(WARNING) << " GPU Unsupported in_tensor shape.size > 4 ";
      return RET_ERROR;
    }
  }
  axis_ = param->axis_;
  if (axis_ < 0) {
    axis_ += in_tensors_.front()->shape().size();
  }
  if (axis_ < 0 || axis_ > 3) {
    MS_LOG(WARNING) << " only support axis >= 0 and axis <= 3 ";
    return RET_ERROR;
  }
  if (out_tensors_shape_size < 4 && type() == PrimitiveType_Concat && axis_ != 0) {
    if (out_tensors_shape_size == DIMENSION_2D) {
      axis_ = axis_ + 2;
    } else if (out_tensors_shape_size == DIMENSION_3D) {
      axis_ = axis_ + 1;
    } else {
      MS_LOG(WARNING) << " Unsupported axis =:  " << axis_ << "  shape().size()=:  " << out_tensors_shape_size;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

std::string ConcatOpenCLKernel::GenMainCodeAxis3UnAlign() {
  std::stringstream code;
  int result_index = 0;
  int temp_index = 0;
  int output_index = 0;
  code << "DTYPE4 result = (DTYPE4)(0);\n";
  for (size_t j = 0; j < in_tensors_.size(); j++) {
    std::vector<int> in_shape(DIMENSION_4D);
    Broadcast2GpuShape(in_tensors_[j]->shape().data(), in_tensors_[j]->shape().size(), in_shape.data(), DIMENSION_4D,
                       1);
    auto align_num = UP_DIV(in_shape[CLIDX_W], C4NUM);

    for (int k = 0; k < align_num; k++) {
      code << "DTYPE4 t" << temp_index << " = READ_IMAGE(input" << j << ", smp_zero, (int2)(((Y) * (" << align_num
           << ") + (" << k << ")), (X)));\n";
      for (int m = 0; (m < C4NUM) && (m < in_shape[CLIDX_W] - k * C4NUM); m++) {
        code << "result" << cl_index_str[result_index++ % C4NUM] << " = t" << temp_index << cl_index_str[m] << ";\n";
        if (result_index % C4NUM == 0) {
          code << "WRITE_IMAGE(output, (int2)(((Y) * (" << out_shape_.s[CLIDX_W] << ") + (" << output_index++
               << ")), (X)), result);\n";
        }
      }
      temp_index++;
    }
  }
  if (out_shape_.s[CLIDX_W] > output_index) {
    code << "WRITE_IMAGE(output, (int2)(((Y) * (" << out_shape_.s[CLIDX_W] << ") + (" << output_index++
         << ")), (X)), result);\n";
  }
  return code.str();
}

std::string ConcatOpenCLKernel::GenMainCodeOthers() {
  std::stringstream code;
  code << "DTYPE4 result;\n";
  if (axis_ == kNHWC_H) {
    code << "int IN = X / " << out_shape_.s[CLIDX_Y] << ";\n"
         << "int IH = X - IN * " << out_shape_.s[CLIDX_Y] << ";\n";
  }

  for (size_t j = 0; j < in_tensors_.size(); j++) {
    std::vector<int> in_shape(DIMENSION_4D);
    Broadcast2GpuShape(in_tensors_[j]->shape().data(), in_tensors_[j]->shape().size(), in_shape.data(), DIMENSION_4D,
                       1);
    in_shape[CLIDX_W] = UP_DIV(in_shape[CLIDX_W], C4NUM);
    std::string variable_name;
    std::string function_y;
    if (axis_ == kNHWC_H) {
      variable_name = "IH";
      function_y = "IN * " + std::to_string(in_shape[CLIDX_Y]) + " + IH";
    } else if (axis_ == kNHWC_C) {
      variable_name = "Z";
      function_y = "X";
    } else {
      variable_name = "Y";
      function_y = "X";
    }
    if (j == 0) {
      code << "int boundary0 = " << in_shape[axis_] << ";\n";
      code << "if (" << variable_name << " < boundary0) {\n";
      code << "int coordinate_x = Y * " << in_shape[CLIDX_W] << " + Z;\n";
      code << "int coordinate_y = " << function_y << ";\n";
      code << "result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y));\n";
      code << "}\n";
    } else {
      code << "int boundary" << j << " = boundary" << (j - 1) << " + " << in_shape[axis_] << ";\n";
      code << "if (" << variable_name << " >= boundary" << (j - 1) << " && " << variable_name << " < boundary" << j
           << ") {\n";
      if (axis_ == kNHWC_H) {
        code << "int coordinate_x = Y * " << in_shape[CLIDX_W] << " + Z;\n";
        code << "int coordinate_y = " << function_y << " - boundary" << (j - 1) << ";\n";
      } else if (axis_ == kNHWC_W) {
        code << "int coordinate_x = (Y - boundary" << (j - 1) << ") * " << in_shape[CLIDX_W] << " + Z;\n";
        code << "int coordinate_y = X;\n";
      } else if (axis_ == kNHWC_C) {
        code << "int coordinate_x = Y * " << in_shape[CLIDX_W] << " + Z - boundary" << (j - 1) << ";\n";
        code << "int coordinate_y = X;\n";
      }

      code << "result = READ_IMAGE(input" << j << ", smp_none, (int2)(coordinate_x, coordinate_y));\n";
      code << "}\n";
    }
  }
  code << "WRITE_IMAGE(output, (int2)((Y) * " << out_shape_.s[CLIDX_W] << " + Z, (X)), result);\n";
  return code.str();
}

std::string ConcatOpenCLKernel::GenCode() {
  std::vector<int> out_shape(DIMENSION_4D);
  Broadcast2GpuShape(out_tensors_[0]->shape().data(), out_tensors_[0]->shape().size(), out_shape.data(), DIMENSION_4D,
                     1);
  for (size_t i = 0; i < out_shape.size(); i++) {
    out_shape_.s[i] = out_shape[i];
  }
  out_shape_.s[CLIDX_W] = UP_DIV(out_shape_.s[CLIDX_W], C4NUM);

  std::stringstream code;
  auto header = OpenCLKernelHeader();
  if (header.empty()) {
    MS_LOG(ERROR) << "Generate OpenCL kernel header failed.";
    return "";
  }
  code << header;
  code << "__kernel void Concat(\n";
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    code << "__read_only image2d_t input" << i << ",\n";
  }
  code << "__write_only image2d_t output\n) {\n";

  if (axis_ == 3 && !Align_) {
    code << "int X = get_global_id(0);\n"
         << "int Y = get_global_id(1);\n"
         << "if (X >= " << out_shape[CLIDX_X] * out_shape[CLIDX_Y] << " || Y >= " << out_shape[CLIDX_Z]
         << ") return;\n";

    code << GenMainCodeAxis3UnAlign();
  } else {
    code << "int X = get_global_id(0);\n"
         << "int Y = get_global_id(1);\n"
         << "int Z = get_global_id(2);\n"
         << "if (X >= " << out_shape_.s[CLIDX_X] * out_shape_.s[CLIDX_Y] << " || Y >= " << out_shape_.s[CLIDX_Z]
         << " || Z >= " << out_shape_.s[CLIDX_W] << ") return;\n";

    code << GenMainCodeOthers();
  }
  code << "}\n";
  return code.str();
}

int ConcatOpenCLKernel::SetGlobalLocal() {
  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  if (axis_ == 3 && !Align_) {
    OH = out_shape_.s[0] * out_shape_.s[1];
    OW = out_shape_.s[2];
    global_size_ = {OH, OW, 1};
    local_size_ = {1, 1, 1};
  } else {
    OH = out_shape_.s[0] * out_shape_.s[1];
    OW = out_shape_.s[2];
    OC = out_shape_.s[3];
    global_size_ = {OH, OW, OC};
    local_size_ = {1, 1, 1};
  }
  ConcatGetWorkGroup(global_size_, &local_size_, max_global[0]);
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);

  return RET_OK;
}

int ConcatOpenCLKernel::ConvertWeightToTensor() {
  auto allocator = ocl_runtime_->GetAllocator();
  bool fp16_enable = ocl_runtime_->GetFp16Enable();
  for (auto in_tensor : in_tensors_) {
    auto in_shape = GpuTensorInfo(in_tensor);
    if (in_tensor->IsConst()) {
      std::vector<char> weight(in_shape.Image2DSize, 0);
      bool src_is_fp16 = in_tensor->data_type() == kNumberTypeFloat16;
      PackNHWCToNHWC4(in_tensor->data(), weight.data(), src_is_fp16,
                      fp16_enable && in_tensor->data_type() != kNumberTypeInt32, in_shape);
      size_t dtype;
      switch (in_tensor->data_type()) {
        case kNumberTypeInt32: {
          dtype = CL_SIGNED_INT32;
          break;
        }
        case kNumberTypeFloat32: {
          dtype = CL_FLOAT;
          break;
        }
        case kNumberTypeFloat16: {
          dtype = CL_HALF_FLOAT;
          break;
        }
        default:
          MS_LOG(ERROR) << "Unsupported data type is" << in_tensor->data_type();
          return RET_ERROR;
      }
      ImageSize img_size{in_shape.width, in_shape.height, dtype};
      auto weight_ptr_ = allocator->Malloc(img_size, weight.data());
      if (weight_ptr_ == nullptr) {
        MS_LOG(ERROR) << "Malloc failed.";
        return RET_ERROR;
      }
      weight_ptrs_.push_back(weight_ptr_);
    } else {
      weight_ptrs_.push_back(nullptr);
    }
  }
  return RET_OK;
}

int ConcatOpenCLKernel::Prepare() {
  int ret = ConvertWeightToTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvertWeightToTensor failed.";
    return ret;
  }
  if (axis_ == 0) {
    if (std::any_of(in_tensors_.begin(), in_tensors_.end(), [](lite::Tensor *t) { return t->shape().size() != 1; })) {
      return RET_OK;
    }
    axis_ = 3;
  }
  for (auto const &in_tensor : in_tensors_) {
    if (in_tensor->shape().back() % C4NUM != 0) {
      Align_ = false;
    }
  }

  std::string source = GenCode();
  // For debug.
  dump_code_ = "[" + this->name() + "]\n" + source;

  if (source.empty()) {
    MS_LOG(ERROR) << "Failed to generate source code for " << this->name();
    return RET_ERROR;
  }

  std::string program_name = "Concat\n" + source;
  std::string kernel_name = "Concat";
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }

  std::vector<std::string> build_options_ext{};
  ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  (void)SetGlobalLocal();
  return RET_OK;
}

int ConcatOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  if (axis_ == 0) {
    return RunAxis0();
  }
  int arg_cn = 0;
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    auto input_ptr = weight_ptrs_.at(i) == nullptr ? in_tensors_[i]->data() : weight_ptrs_.at(i);
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_ptr) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }

  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }

  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Concat, OpenCLKernelCreator<ConcatOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Concat, OpenCLKernelCreator<ConcatOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeInt32, PrimitiveType_Concat, OpenCLKernelCreator<ConcatOpenCLKernel>)
}  // namespace kernel
}  // namespace mindspore
