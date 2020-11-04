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

#include "src/runtime/kernel/arm/fp32/pad_fp32.h"
#include <string>
#include "src/kernel_registry.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "nnacl/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Pad;

namespace mindspore::kernel {
namespace {
constexpr size_t kMirrorPadInputSize = 2;
}
int PadCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PadCPUKernel::ReSize() {
  auto input = in_tensors_.at(0);
  auto rank = input->shape().size();
  if (rank > DEFAULT_PAD_NDIMS) {
    MS_LOG(ERROR) << "Pad input rank should <= " << DEFAULT_PAD_NDIMS << ", got " << rank;
    return RET_ERROR;
  }
  auto output = out_tensors_.at(0);
  if (pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_CONSTANT)) {
    auto ret = ExtendShape(in_, DEFAULT_PAD_NDIMS, input->shape().data(), rank);
    if (ret != RET_OK) {
      return ret;
    }
    ret = ExtendShape(out_, DEFAULT_PAD_NDIMS, output->shape().data(), rank);
    if (ret != RET_OK) {
      return ret;
    }
    if (pad_param_->padding_length < MAX_PAD_SIZE) {
      int ori_paddings[MAX_PAD_SIZE];
      for (auto i = 0; i < pad_param_->padding_length; ++i) {
        ori_paddings[i] = pad_param_->paddings_[i];
      }
      ret = ExtendPaddings(pad_param_->paddings_, MAX_PAD_SIZE, ori_paddings, pad_param_->padding_length);
      if (ret != RET_OK) {
        return ret;
      }
      pad_param_->padding_length = MAX_PAD_SIZE;
    }
  }
  return RET_OK;
}

int PadCPUKernel::ExtendShape(int *shape, int length, const int *ori_shape, int rank) {
  if (shape == nullptr || ori_shape == nullptr) {
    return RET_NULL_PTR;
  }
  for (auto i = 0; i < length - rank; ++i) {
    shape[i] = 1;
  }
  for (auto i = length - rank; i < length; ++i) {
    shape[i] = ori_shape[i - (length - rank)];
  }
  return RET_OK;
}

int PadCPUKernel::ExtendPaddings(int *paddings, int length, const int *ori_paddings, int ori_length) {
  if (paddings == nullptr || ori_paddings == nullptr) {
    return RET_NULL_PTR;
  }
  for (auto i = 0; i < length - ori_length; ++i) {
    paddings[i] = 0;
  }
  for (auto i = length - ori_length; i < length; ++i) {
    paddings[i] = ori_paddings[i - (length - ori_length)];
  }
  return RET_OK;
}

int PadImpl(void *cdata, int task_id) {
  auto padKernel = reinterpret_cast<PadCPUKernel *>(cdata);
  int error_code = padKernel->RunImpl(task_id);
  if (error_code != NNACL_OK) {
    MS_LOG(ERROR) << "Pad Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PadCPUKernel::RunImpl(int task_id) {
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);

  auto input_data = reinterpret_cast<float *>(input->MutableData());
  auto output_data = reinterpret_cast<float *>(output->MutableData());

  Pad(input_data, output_data, in_, out_, pad_param_->paddings_, task_id, context_->thread_num_);

  return RET_OK;
}

int MirrorPadImpl(void *cdata, int task_id) {
  auto padKernel = reinterpret_cast<PadCPUKernel *>(cdata);
  int error_code = padKernel->RunMirrorPadImpl(task_id);
  if (error_code != NNACL_OK) {
    MS_LOG(ERROR) << "Pad Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PadCPUKernel::RunMirrorPadImpl(int task_id) {
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  auto output_data = reinterpret_cast<float *>(output->MutableData());

  int unit = UP_DIV(output->ElementsNum(), context_->thread_num_);
  int begin = unit * task_id;
  int end = MSMIN(begin + unit, output->ElementsNum());
  MirrorPad(input_data, output_data, in_, pad_param_, begin, end);
  return RET_OK;
}

int PadCPUKernel::CheckPaddings(int *paddings, int length, int *input_shape, int mode) {
  if (paddings == nullptr || input_shape == nullptr) {
    return RET_NULL_PTR;
  }
  std::string prefix;
  int offset;
  if (mode == static_cast<int>(schema::PaddingMode_SYMMETRIC)) {
    prefix = "For Pad SYMMETRIC ";
    offset = 0;
  } else {
    prefix = "For Pad REFLECT ";
    offset = 1;
  }
  for (auto i = 0; i < length; ++i) {
    int max_valid = input_shape[i] - offset;
    if (paddings[i * 2] > max_valid) {
      MS_LOG(WARNING) << prefix << "paddings " << paddings[i * 2] << "should be less than " << max_valid + 1;
    }
    if (paddings[i * 2 + 1] > max_valid) {
      MS_LOG(WARNING) << prefix << "paddings " << paddings[i * 2 + 1] << "should be less than " << max_valid + 1;
    }
  }
  return RET_OK;
}

int PadCPUKernel::CopyPaddingFromInput() {
  if (in_tensors_.size() != kMirrorPadInputSize) {
    MS_LOG(ERROR) << "Pad Reflect or Symmetric mode need 2 inputs, got " << in_tensors_.size();
    return RET_ERROR;
  }
  auto padding_tensor = in_tensors_.at(1);
  auto paddings = reinterpret_cast<int *>(padding_tensor->MutableData());
  if (paddings == nullptr) {
    MS_LOG(ERROR) << "Pad second input data nullptr";
    return RET_ERROR;
  }
  auto input_shape = in_tensors_.at(0)->shape();
  int rank = static_cast<int>(input_shape.size());
  if (padding_tensor->ElementsNum() != rank * 2) {
    MS_LOG(ERROR) << "Pad second input elements num" << padding_tensor->ElementsNum() << ", should be " << rank * 2;
    return RET_ERROR;
  }

  auto ret = ExtendShape(in_, DEFAULT_PAD_NDIMS, input_shape.data(), rank);
  if (ret != RET_OK) {
    return ret;
  }
  ret = ExtendPaddings(pad_param_->paddings_, MAX_PAD_SIZE, paddings, padding_tensor->ElementsNum());
  if (ret != RET_OK) {
    return ret;
  }
  pad_param_->padding_length = MAX_PAD_SIZE;
  return RET_OK;
}

void PadCPUKernel::CalculateStrides() {
  pad_param_->in_strides[DEFAULT_PAD_NDIMS - 1] = 1;
  for (auto i = DEFAULT_PAD_NDIMS - 2; i >= 0; --i) {
    pad_param_->in_strides[i] = in_[i + 1] * pad_param_->in_strides[i + 1];
  }
  for (auto i = 0; i < DEFAULT_PAD_NDIMS; ++i) {
    out_[i] = in_[i] + pad_param_->paddings_[i * 2] + pad_param_->paddings_[i * 2 + 1];
  }
  pad_param_->out_strides[DEFAULT_PAD_NDIMS - 1] = 1;
  for (auto i = DEFAULT_PAD_NDIMS - 2; i >= 0; --i) {
    pad_param_->out_strides[i] = out_[i + 1] * pad_param_->out_strides[i + 1];
  }
}

int PadCPUKernel::HandleMirrorPad() {
  if (in_tensors_.size() == 1) {
    auto input_shape = in_tensors_.at(0)->shape();
    int rank = static_cast<int>(input_shape.size());
    auto ret = ExtendShape(in_, DEFAULT_PAD_NDIMS, input_shape.data(), rank);
    if (ret != RET_OK) {
      return ret;
    }
  } else {
    auto ret = CopyPaddingFromInput();
    if (ret != RET_OK) {
      return ret;
    }
  }
  auto ret = CheckPaddings(pad_param_->paddings_, DEFAULT_PAD_NDIMS, in_, pad_param_->pad_mode_);
  if (ret != RET_OK) {
    return ret;
  }
  CalculateStrides();
  pad_param_->mirror_offset_ = pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_REFLECT) ? 1 : 0;
  return RET_OK;
}

int PadCPUKernel::Run() {
  int error_code;
  if (pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_CONSTANT)) {
    auto output = out_tensors_.at(0);
    int output_size = output->ElementsNum();
    auto output_data = reinterpret_cast<float *>(output->MutableData());
    if (pad_param_->constant_value_ - 0.0f < 1e-5) {
      memset(output_data, 0, output_size * sizeof(float));
    } else {
      for (auto i = 0; i < output_size; ++i) {
        output_data[i] = pad_param_->constant_value_;
      }
    }
    error_code = ParallelLaunch(this->context_->thread_pool_, PadImpl, this, context_->thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Pad run error, error_code[" << error_code << "]";
      return RET_ERROR;
    }
  } else {
    // mirror pad case
    HandleMirrorPad();

    error_code = ParallelLaunch(this->context_->thread_pool_, MirrorPadImpl, this, context_->thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Pad Reflect or Symmetric mode run error, error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }

  return RET_OK;
}
}  // namespace mindspore::kernel
