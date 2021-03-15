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

#include "src/runtime/kernel/arm/int8/pad_int8.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_PadFusion;
namespace mindspore::kernel {

namespace {
constexpr size_t kMirrorPadInputSize = 2;
}

void PadInt8CPUKernel::FreeQuantParam() {
  if (pad_param_->pad_quant_arg_.in_quant_args_ != nullptr) {
    free(pad_param_->pad_quant_arg_.in_quant_args_);
    pad_param_->pad_quant_arg_.in_quant_args_ = nullptr;
  }
  if (pad_param_->pad_quant_arg_.out_quanr_args_ != nullptr) {
    free(pad_param_->pad_quant_arg_.out_quanr_args_);
    pad_param_->pad_quant_arg_.out_quanr_args_ = nullptr;
  }
  if (pad_param_->pad_quant_arg_.constant_value_ != nullptr) {
    free(pad_param_->pad_quant_arg_.constant_value_);
    pad_param_->pad_quant_arg_.constant_value_ = nullptr;
  }
}

int PadInt8CPUKernel::SetQuantParam() {
  PadQuantArg *pad_quant_args = &pad_param_->pad_quant_arg_;
  pad_quant_args->in_quant_args_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (pad_quant_args->in_quant_args_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  pad_quant_args->out_quanr_args_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (pad_quant_args->out_quanr_args_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  pad_quant_args->constant_value_ = reinterpret_cast<int8_t *>(malloc(sizeof(int8_t)));
  if (pad_quant_args->constant_value_ == nullptr) {
    return RET_MEMORY_FAILED;
  }

  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto in_quant_arg = input_tensor->quant_params();
  auto out_quant_arg = out_tensor->quant_params();

  pad_quant_args->in_quant_args_->zp_ = in_quant_arg.front().zeroPoint;
  pad_quant_args->in_quant_args_->scale_ = in_quant_arg.front().scale;
  pad_quant_args->out_quanr_args_->zp_ = out_quant_arg.front().zeroPoint;
  pad_quant_args->out_quanr_args_->scale_ = out_quant_arg.front().scale;

  if (pad_quant_args->in_quant_args_->scale_ != pad_quant_args->out_quanr_args_->scale_ ||
      pad_quant_args->in_quant_args_->zp_ != pad_quant_args->out_quanr_args_->zp_) {
    MS_LOG(ERROR) << "Pad int8 op : scale & zp of output and input must be equal.";
    return RET_ERROR;
  }

  pad_quant_args->constant_value_[0] = QuantizeToInt8(
    pad_param_->constant_value_, pad_quant_args->in_quant_args_->scale_, pad_quant_args->in_quant_args_->zp_);
  return RET_OK;
}

int PadInt8CPUKernel::InitPadParam() {
  auto in_dims = in_tensors_.at(0)->shape();
  auto out_dims = out_tensors_.at(0)->shape();
  int ndims = in_dims.size();

  int in[] = {1, 1, 1, 1};
  int out[] = {1, 1, 1, 1};

  for (int i = 0; i < ndims; i++) {
    in[COMM_SHAPE_SIZE - ndims + i] = in_dims[i];
    out[COMM_SHAPE_SIZE - ndims + i] = out_dims[i];
  }

  memcpy(in_dims_, in, COMM_SHAPE_SIZE * sizeof(int));
  memcpy(out_dims_, out, COMM_SHAPE_SIZE * sizeof(int));

  return RET_OK;
}

int PadInt8CPUKernel::ReSize() {
  int error_code = InitPadParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "InitPadParam failed. errorcode: " << error_code;
    return error_code;
  }
  return RET_OK;
}

int PadInt8CPUKernel::Init() {
  MS_ASSERT(pad_param_);
  auto error_code = SetQuantParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SetQuantParam failed. errorcode: " << error_code;
    return error_code;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PadInt8CPUKernel::RunImpl(int task_id) {
  MS_ASSERT(in_data_);
  MS_ASSERT(out_data_);
  MS_ASSERT(in_dims_);
  MS_ASSERT(out_dims_);
  return PadConstant4D(in_data_, out_data_, in_dims_, out_dims_, pad_param_->paddings_, task_id, context_->thread_num_);
}

int PadInt8Impl(void *cdata, int task_id) {
  auto resize = reinterpret_cast<PadInt8CPUKernel *>(cdata);
  auto error_code = resize->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Resize Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PadInt8CPUKernel::HandleMirrorPad() {
  auto ret = CopyPaddingFromInput();
  if (ret != RET_OK) {
    return ret;
  }
  ret = CheckPaddings(pad_param_->paddings_, COMM_SHAPE_SIZE, in_dims_, pad_param_->pad_mode_);
  if (ret != RET_OK) {
    return ret;
  }
  CalculateStrides();
  pad_param_->mirror_offset_ = pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_REFLECT) ? 1 : 0;
  return RET_OK;
}

void PadInt8CPUKernel::CalculateStrides() {
  pad_param_->in_strides[COMM_SHAPE_SIZE - 1] = 1;
  for (auto i = COMM_SHAPE_SIZE - 2; i >= 0; --i) {
    pad_param_->in_strides[i] = in_dims_[i + 1] * pad_param_->in_strides[i + 1];
  }
  for (auto i = 0; i < COMM_SHAPE_SIZE; ++i) {
    out_dims_[i] = in_dims_[i] + pad_param_->paddings_[i * 2] + pad_param_->paddings_[i * 2 + 1];
  }
  pad_param_->out_strides[COMM_SHAPE_SIZE - 1] = 1;
  for (auto i = COMM_SHAPE_SIZE - 2; i >= 0; --i) {
    pad_param_->out_strides[i] = out_dims_[i + 1] * pad_param_->out_strides[i + 1];
  }
}

int PadInt8CPUKernel::ExtendPaddings(int *paddings, int length, const int *ori_paddings, int ori_length) {
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

int PadInt8CPUKernel::RunMirrorPadImpl(int task_id) {
  auto input = in_tensors_.at(0);
  MS_ASSERT(input);
  auto output = out_tensors_.at(0);
  MS_ASSERT(output);
  auto input_data = reinterpret_cast<int8_t *>(input->MutableData());
  MS_ASSERT(input_data);
  auto output_data = reinterpret_cast<int8_t *>(output->MutableData());
  MS_ASSERT(output_data);

  int unit = UP_DIV(output->ElementsNum(), context_->thread_num_);
  int begin = unit * task_id;
  int end = MSMIN(begin + unit, output->ElementsNum());
  MirrorPadInt8(input_data, output_data, in_dims_, pad_param_, begin, end);
  return RET_OK;
}

int MirrorPadImplInt8(void *cdata, int task_id) {
  auto padKernel = reinterpret_cast<PadInt8CPUKernel *>(cdata);
  int error_code = padKernel->RunMirrorPadImpl(task_id);
  if (error_code != NNACL_OK) {
    MS_LOG(ERROR) << "Pad Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PadInt8CPUKernel::CheckPaddings(const int *paddings, int length, const int *input_shape, int mode) {
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
      MS_LOG(WARNING) << prefix << "paddings " << paddings[i * 2] << "should be more than " << max_valid + 1;
    }
    if (paddings[i * 2 + 1] > max_valid) {
      MS_LOG(WARNING) << prefix << "paddings " << paddings[i * 2 + 1] << "should be less than " << max_valid + 1;
    }
  }
  return RET_OK;
}

int PadInt8CPUKernel::CopyPaddingFromInput() {
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

  auto ret = ExtendPaddings(pad_param_->paddings_, MAX_SHAPE_SIZE, paddings, padding_tensor->ElementsNum());
  if (ret != RET_OK) {
    return ret;
  }
  pad_param_->padding_length = MAX_SHAPE_SIZE;
  return RET_OK;
}

int PadInt8CPUKernel::Run() {
  in_data_ = reinterpret_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  out_data_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());

  int error_code;
  if (pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_CONSTANT)) {
    memset(out_data_, pad_param_->pad_quant_arg_.constant_value_[0], out_tensors_[0]->ElementsNum() * sizeof(int8_t));
    error_code = ParallelLaunch(this->context_->thread_pool_, PadInt8Impl, this, context_->thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Resize run error, error_code[" << error_code << "]";
      return RET_ERROR;
    }
  } else {
    // mirror pad case
    error_code = HandleMirrorPad();
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Handle mirror pad failed, error_code[" << error_code << "]";
      return error_code;
    }

    error_code = ParallelLaunch(this->context_->thread_pool_, MirrorPadImplInt8, this, context_->thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Pad Reflect or Symmetric mode run error, error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_PadFusion, LiteKernelCreator<PadInt8CPUKernel>)
}  // namespace mindspore::kernel
