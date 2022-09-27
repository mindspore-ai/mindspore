/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/int8/resize_int8.h"
#include <vector>
#include <algorithm>
#include "include/errorcode.h"
#include "nnacl/int8/resize_int8.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INVALID_OP_ATTR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;

namespace mindspore::kernel {
namespace {
constexpr unsigned int OFFSET_BASE = 10;
}  // namespace
void ResizeInt8CPUKernel::FreeResizeBiLinear() {
  free(resize_quant_arg_.x_axis_index_);
  resize_quant_arg_.x_axis_index_ = nullptr;
  free(resize_quant_arg_.x_axis_lower_);
  resize_quant_arg_.x_axis_lower_ = nullptr;
  free(resize_quant_arg_.x_axis_upper_);
  resize_quant_arg_.x_axis_upper_ = nullptr;
  free(resize_quant_arg_.y_axis_index_);
  resize_quant_arg_.y_axis_index_ = nullptr;
  free(resize_quant_arg_.y_axis_lower_);
  resize_quant_arg_.y_axis_lower_ = nullptr;
  free(resize_quant_arg_.y_axis_upper_);
  resize_quant_arg_.y_axis_upper_ = nullptr;
}

void ResizeInt8CPUKernel::FreeFloatResizeBiLinear() {
  free(resize_float_quant_arg_.x_axis_index_);
  resize_float_quant_arg_.x_axis_index_ = nullptr;
  free(resize_float_quant_arg_.x_axis_lower_);
  resize_float_quant_arg_.x_axis_lower_ = nullptr;
  free(resize_float_quant_arg_.x_axis_upper_);
  resize_float_quant_arg_.x_axis_upper_ = nullptr;
  free(resize_float_quant_arg_.y_axis_index_);
  resize_float_quant_arg_.y_axis_index_ = nullptr;
  free(resize_float_quant_arg_.y_axis_lower_);
  resize_float_quant_arg_.y_axis_lower_ = nullptr;
  free(resize_float_quant_arg_.y_axis_upper_);
  resize_float_quant_arg_.y_axis_upper_ = nullptr;
}

ResizeInt8CPUKernel::~ResizeInt8CPUKernel() {
  if (method_ == schema::ResizeMethod_LINEAR) {
    if (quant_in_ != nullptr) {
      if (quant_in_->zp_ == 0) {
        FreeResizeBiLinear();
      } else {
        FreeFloatResizeBiLinear();
      }
    }
  }
  delete quant_out_;
  quant_out_ = nullptr;
  delete quant_in_;
  quant_in_ = nullptr;
  delete multiplier_;
  multiplier_ = nullptr;
}

int ResizeInt8CPUKernel::Prepare() {
  auto ret = ResizeBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  quant_in_ = new (std::nothrow) QuantArg;
  quant_out_ = new (std::nothrow) QuantArg;
  multiplier_ = new (std::nothrow) QuantMulArg;
  if (quant_in_ == nullptr || quant_out_ == nullptr || multiplier_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  auto input = in_tensors_.at(0);
  CHECK_LESS_RETURN(input->quant_params().size(), 1);
  quant_in_->zp_ = input->quant_params().front().zeroPoint;
  quant_in_->scale_ = input->quant_params().front().scale;
  auto output = out_tensors_.at(0);
  CHECK_LESS_RETURN(output->quant_params().size(), 1);
  quant_out_->zp_ = output->quant_params().front().zeroPoint;
  quant_out_->scale_ = output->quant_params().front().scale;

  QuantizeRoundParameterWithDoublePrecision(quant_in_->scale_ / quant_out_->scale_, &multiplier_->multiplier_,
                                            &multiplier_->left_shift_, &multiplier_->right_shift_);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ResizeInt8CPUKernel::InitResizeQuantArg() {
  auto out_shape = out_tensors_.front()->shape();
  resize_quant_arg_.x_axis_index_ = reinterpret_cast<int32_t *>(malloc(out_shape.at(kNHWC_W) * sizeof(int32_t)));
  if (resize_quant_arg_.x_axis_index_ == nullptr) {
    MS_LOG(ERROR) << "malloc x axis index array failed.";
    return RET_ERROR;
  }
  resize_quant_arg_.x_axis_lower_ = reinterpret_cast<int32_t *>(malloc(out_shape.at(kNHWC_W) * sizeof(int32_t)));
  if (resize_quant_arg_.x_axis_lower_ == nullptr) {
    MS_LOG(ERROR) << "malloc x_axis_lower_ array failed.";
    return RET_ERROR;
  }
  resize_quant_arg_.x_axis_upper_ = reinterpret_cast<int32_t *>(malloc(out_shape.at(kNHWC_W) * sizeof(int32_t)));
  if (resize_quant_arg_.x_axis_upper_ == nullptr) {
    MS_LOG(ERROR) << "malloc x_axis_upper_ array failed.";
    return RET_ERROR;
  }
  resize_quant_arg_.y_axis_index_ = reinterpret_cast<int32_t *>(malloc(out_shape.at(kNHWC_H) * sizeof(int32_t)));
  if (resize_quant_arg_.y_axis_index_ == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_index_ array failed.";
    return RET_ERROR;
  }
  resize_quant_arg_.y_axis_lower_ = reinterpret_cast<int32_t *>(malloc(out_shape.at(kNHWC_H) * sizeof(int32_t)));
  if (resize_quant_arg_.y_axis_lower_ == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_lower_ array failed.";
    return RET_ERROR;
  }
  resize_quant_arg_.y_axis_upper_ = reinterpret_cast<int32_t *>(malloc(out_shape.at(kNHWC_H) * sizeof(int32_t)));
  if (resize_quant_arg_.y_axis_upper_ == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_upper_ array failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeInt8CPUKernel::CalRatio() {
  auto in_tensor = in_tensors_.front();
  auto in_width = in_tensor->Width();
  auto in_height = in_tensor->Height();
  auto out_tensor = out_tensors_.front();
  auto out_width = out_tensor->Width();
  auto out_height = out_tensor->Height();
  resize_quant_arg_.ratio_x_ = ((1 << OFFSET_BASE) * in_width + out_width / 2) / out_width;
  resize_quant_arg_.ratio_y_ = ((1 << OFFSET_BASE) * in_height + out_height / 2) / out_height;
  bool align_corners = coordinate_transform_mode_ == schema::CoordinateTransformMode_ALIGN_CORNERS;
  if (align_corners && out_width > 1) {
    resize_quant_arg_.ratio_x_ = ((1 << OFFSET_BASE) * (in_width - 1) + (out_width - 1) / 2) / (out_width - 1);
  }
  if (align_corners && out_height > 1) {
    resize_quant_arg_.ratio_y_ = ((1 << OFFSET_BASE) * (in_height - 1) + (out_height - 1) / 2) / (out_height - 1);
  }
  return RET_OK;
}

int ResizeInt8CPUKernel::CalInterpolationRange() {
  for (int i = 0; i < out_tensors_.front()->Height(); ++i) {
    int32_t scaled_index = i * resize_quant_arg_.ratio_y_;
    resize_quant_arg_.y_axis_index_[i] = scaled_index;
    resize_quant_arg_.y_axis_lower_[i] = std::max(scaled_index / (1 << OFFSET_BASE), 0);
    resize_quant_arg_.y_axis_upper_[i] =
      std::min(scaled_index / (1 << OFFSET_BASE) + 1, in_tensors_.front()->Height() - 1);
  }
  for (int i = 0; i < out_tensors_.front()->Width(); ++i) {
    int32_t scaled_index = i * resize_quant_arg_.ratio_x_;
    resize_quant_arg_.x_axis_index_[i] = scaled_index;
    resize_quant_arg_.x_axis_lower_[i] = std::max(scaled_index / (1 << OFFSET_BASE), 0);
    resize_quant_arg_.x_axis_upper_[i] =
      std::min(scaled_index / (1 << OFFSET_BASE) + 1, in_tensors_.front()->Width() - 1);
  }
  return RET_OK;
}

int ResizeInt8CPUKernel::InitResizeFloatQuantArg() {
  auto out_shape = out_tensors_.front()->shape();
  resize_float_quant_arg_.x_axis_index_ = reinterpret_cast<float *>(malloc(out_shape[kNHWC_W] * sizeof(float)));
  if (resize_float_quant_arg_.x_axis_index_ == nullptr) {
    MS_LOG(ERROR) << "malloc x axis index array failed.";
    return RET_ERROR;
  }
  resize_float_quant_arg_.x_axis_lower_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_W] * sizeof(int32_t)));
  if (resize_float_quant_arg_.x_axis_lower_ == nullptr) {
    MS_LOG(ERROR) << "malloc x_axis_lower_ array failed.";
    return RET_ERROR;
  }
  resize_float_quant_arg_.x_axis_upper_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_W] * sizeof(int32_t)));
  if (resize_float_quant_arg_.x_axis_upper_ == nullptr) {
    MS_LOG(ERROR) << "malloc x_axis_upper_ array failed.";
    return RET_ERROR;
  }
  resize_float_quant_arg_.y_axis_index_ = reinterpret_cast<float *>(malloc(out_shape[kNHWC_H] * sizeof(float)));
  if (resize_float_quant_arg_.y_axis_index_ == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_index_ array failed.";
    return RET_ERROR;
  }
  resize_float_quant_arg_.y_axis_lower_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_H] * sizeof(int32_t)));
  if (resize_float_quant_arg_.y_axis_lower_ == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_lower_ array failed.";
    return RET_ERROR;
  }
  resize_float_quant_arg_.y_axis_upper_ = reinterpret_cast<int32_t *>(malloc(out_shape[kNHWC_H] * sizeof(int32_t)));
  if (resize_float_quant_arg_.y_axis_upper_ == nullptr) {
    MS_LOG(ERROR) << "malloc y_axis_upper_ array failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeInt8CPUKernel::CalFloatRatio() {
  auto in_tensor = in_tensors_.front();
  auto in_width = in_tensor->Width();
  auto in_height = in_tensor->Height();
  auto out_tensor = out_tensors_.front();
  auto out_width = out_tensor->Width();
  auto out_height = out_tensor->Height();
  resize_float_quant_arg_.ratio_x_ = static_cast<float>(in_width) / out_width;
  resize_float_quant_arg_.ratio_y_ = static_cast<float>(in_height) / out_height;
  bool align_corners = coordinate_transform_mode_ == schema::CoordinateTransformMode_ALIGN_CORNERS;
  if (align_corners && out_width > 1) {
    resize_float_quant_arg_.ratio_x_ = static_cast<float>(in_width - 1) / (out_width - 1);
  }
  if (align_corners && out_height > 1) {
    resize_float_quant_arg_.ratio_y_ = static_cast<float>(in_height - 1) / (out_height - 1);
  }
  return RET_OK;
}

int ResizeInt8CPUKernel::CalFloatInterpolationRange() {
  for (int i = 0; i < out_tensors_.front()->Height(); ++i) {
    float scaled_index = i * resize_float_quant_arg_.ratio_y_;
    int lower_index = std::floor(scaled_index);
    resize_float_quant_arg_.y_axis_index_[i] = scaled_index;
    resize_float_quant_arg_.y_axis_lower_[i] = std::max(lower_index, 0);
    resize_float_quant_arg_.y_axis_upper_[i] = std::min(lower_index + 1, in_tensors_.front()->Height() - 1);
  }
  for (int i = 0; i < out_tensors_.front()->Width(); ++i) {
    float scaled_index = i * resize_float_quant_arg_.ratio_x_;
    int lower_index = std::floor(scaled_index);
    resize_float_quant_arg_.x_axis_index_[i] = scaled_index;
    resize_float_quant_arg_.x_axis_lower_[i] = std::max(lower_index, 0);
    resize_float_quant_arg_.x_axis_upper_[i] = std::min(lower_index + 1, in_tensors_.front()->Width() - 1);
  }
  return RET_OK;
}

int ResizeInt8CPUKernel::InitResizeBiLinear() {
  auto ret = InitResizeQuantArg();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize Int8 Op Resize Failed.";
    return ret;
  }
  ret = CalRatio();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Cal ratio Failed.";
    return ret;
  }
  ret = CalInterpolationRange();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Cal range of interpolation Failed.";
    return ret;
  }
  return RET_OK;
}

int ResizeInt8CPUKernel::InitFloatResizeBiLinear() {
  auto ret = InitResizeFloatQuantArg();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize Int8 Op Resize Failed.";
    return ret;
  }
  ret = CalFloatRatio();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Cal ratio Failed.";
    return ret;
  }
  ret = CalFloatInterpolationRange();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Cal range of interpolation Failed.";
    return ret;
  }
  return RET_OK;
}

int ResizeInt8CPUKernel::ReSize() {
  if (method_ == schema::ResizeMethod_LINEAR) {
    if (quant_in_->zp_ == 0) {
      return InitResizeBiLinear();
    } else {
      return InitFloatResizeBiLinear();
    }
  }
  if (in_tensors_.front()->quant_params().empty() || out_tensors_.front()->quant_params().empty() ||
      in_tensors_.front()->quant_params().front().scale != out_tensors_.front()->quant_params().front().scale ||
      in_tensors_.front()->quant_params().front().zeroPoint != out_tensors_.front()->quant_params().front().zeroPoint ||
      in_tensors_.front()->quant_params().front().zeroPoint > INT8_MAX ||
      in_tensors_.front()->quant_params().front().zeroPoint < INT8_MIN ||
      in_tensors_.front()->quant_params().front().scale < 0) {
    MS_LOG(ERROR) << "Resize quant param is invalid.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeInt8Impl(void *cdata, int task_id, float, float) {
  auto resize = reinterpret_cast<ResizeInt8CPUKernel *>(cdata);
  auto error_code = resize->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Resize Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeInt8CPUKernel::RunImpl(int task_id) {
  auto input = in_tensors_.at(0);
  auto input_data = reinterpret_cast<const int8_t *>(input->data());
  if (input_data == nullptr) {
    return RET_NULL_PTR;
  }
  auto output_data = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data());
  if (output_data == nullptr) {
    return RET_NULL_PTR;
  }
  auto input_shape = input->shape();

  if (ms_context_ == nullptr) {
    return RET_NULL_PTR;
  }

  int ret = 0;
  switch (method_) {
    case static_cast<int>(schema::ResizeMethod_LINEAR): {
      auto out_tensor = out_tensors_.front();
      auto out_c = out_tensor->Channel();
      int plane = out_tensor->Height() * out_tensor->Width();
      int num = UP_DIV(plane, op_parameter_->thread_num_);
      int start_index = task_id * num;
      int count = plane - start_index;
      count = count > num ? num : count;
      auto out_ptr = output_data + start_index * out_c;
      if (quant_in_->zp_ == 0) {
        ret =
          ResizeBilinearInt8(input_data, out_ptr, out_tensor->Batch(), input->Height(), input->Width(),
                             out_tensor->Height(), out_tensor->Width(), out_c, start_index, count, resize_quant_arg_);
      } else {
        ret = ResizeBilinearWithFloatScaleInt8(input_data, out_ptr, out_tensor->Batch(), input->Height(),
                                               input->Width(), out_tensor->Height(), out_tensor->Width(), out_c,
                                               start_index, count, resize_float_quant_arg_);
      }

      break;
    }
    case static_cast<int>(schema::ResizeMethod_NEAREST): {
      bool same_zp = quant_in_->zp_ == quant_out_->zp_;
      bool same_scale = abs(quant_out_->scale_ - quant_in_->scale_) < 1e-6;
      bool align_corners = coordinate_transform_mode_ == schema::CoordinateTransformMode_ALIGN_CORNERS;
      auto out_shape = out_tensors_[0]->shape();
      if (same_zp && same_scale) {
        ret = ResizeNearestNeighborInt8Simple(input_data, output_data, input_shape.data(), out_shape.data(),
                                              align_corners, task_id, op_parameter_->thread_num_);
      } else {
        ret = ResizeNearestNeighborInt8(input_data, output_data, input_shape.data(), out_shape.data(), align_corners,
                                        multiplier_, quant_in_, quant_out_, task_id, op_parameter_->thread_num_);
      }
      break;
    }
    case schema::ResizeMethod_UNKNOWN:
    default: {
      MS_LOG(ERROR) << "Resize unknown method " << method_;
      ret = RET_ERROR;
    }
  }
  return ret;
}

int ResizeInt8CPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, ResizeInt8Impl, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Resize run error, error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Resize, LiteKernelCreator<ResizeInt8CPUKernel>)
}  // namespace mindspore::kernel
