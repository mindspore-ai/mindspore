/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/int8/pad_int8_coder.h"
#include <cfloat>
#include "include/errorcode.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_PadFusion;

namespace mindspore::lite::micro::nnacl {
constexpr int kDouble = 2;
int PadInt8Coder::Prepare(CoderContext *context) {
  pad_param_ = reinterpret_cast<PadParameter *>(parameter_);
  if (pad_param_->pad_mode_ != static_cast<int>(schema::PaddingMode_CONSTANT)) {
    MS_LOG(ERROR) << "The int8 pad operator only supports PaddingMode_CONSTANT mode at present.";
    return RET_ERROR;
  }
  MS_CHECK_TRUE_RET(input_tensors_.size() == kInputSize1 || input_tensors_.size() == kInputSize2, RET_ERROR);
  MS_CHECK_TRUE_RET(output_tensors_.size() == 1, RET_ERROR);
  CHECK_NULL_RETURN(input_tensors_[0]);
  CHECK_NULL_RETURN(input_tensors_[1]);
  CHECK_NULL_RETURN(output_tensors_[0]);
  CHECK_NULL_RETURN(pad_param_);
  // param check, padding length must equal 2 * len(input_x)
  if (input_tensors_[kInputIndex]->shape().size() * kDouble != static_cast<size_t>(pad_param_->padding_length)) {
    MS_LOG(ERROR) << "Input shape size not match padding length.";
    return RET_ERROR;
  }
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

  auto *input_tensor = input_tensors_.at(kInputIndex);
  auto *out_tensor = output_tensors_.at(kOutputIndex);
  auto in_quant_arg = input_tensor->quant_params();
  MS_CHECK_TRUE_RET(!in_quant_arg.empty(), RET_ERROR);
  auto out_quant_arg = out_tensor->quant_params();
  MS_CHECK_TRUE_RET(!out_quant_arg.empty(), RET_ERROR);

  pad_quant_args->in_quant_args_->zp_ = in_quant_arg.front().zeroPoint;
  pad_quant_args->in_quant_args_->scale_ = in_quant_arg.front().scale;
  pad_quant_args->out_quanr_args_->zp_ = out_quant_arg.front().zeroPoint;
  pad_quant_args->out_quanr_args_->scale_ = out_quant_arg.front().scale;

  if (std::abs(pad_quant_args->in_quant_args_->scale_ - pad_quant_args->out_quanr_args_->scale_) > FLT_EPSILON ||
      pad_quant_args->in_quant_args_->zp_ != pad_quant_args->out_quanr_args_->zp_) {
    MS_LOG(ERROR) << "Pad int8 op : scale & zp of output and input must be equal.";
    return RET_ERROR;
  }

  pad_quant_args->constant_value_[0] = QuantizeToInt8(
    pad_param_->constant_value_, pad_quant_args->in_quant_args_->scale_, pad_quant_args->in_quant_args_->zp_);
  return InitPadParam();
}

int PadInt8Coder::InitPadParam() {
  auto in_dims = input_tensors_.at(0)->shape();
  auto out_dims = output_tensors_.at(0)->shape();
  int ndims = static_cast<int>(in_dims.size());

  int in[COMM_SHAPE_SIZE] = {1, 1, 1, 1};
  int out[COMM_SHAPE_SIZE] = {1, 1, 1, 1};

  for (int i = 0; i < ndims; i++) {
    in[COMM_SHAPE_SIZE - ndims + i] = in_dims[i];
    out[COMM_SHAPE_SIZE - ndims + i] = out_dims[i];
  }

  if (memcpy_s(in_dims_, COMM_SHAPE_SIZE * sizeof(int), in, COMM_SHAPE_SIZE * sizeof(int)) != EOK) {
    MS_LOG(ERROR) << "memcpy failed.";
    return RET_ERROR;
  }
  if (memcpy_s(out_dims_, COMM_SHAPE_SIZE * sizeof(int), out, COMM_SHAPE_SIZE * sizeof(int)) != EOK) {
    MS_LOG(ERROR) << "memcpy failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

int PadInt8Coder::ExtendPaddings(int *paddings, int length, const int *ori_paddings, int ori_length) const {
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

int PadInt8Coder::CopyPaddingFromInput() {
  auto padding_tensor = input_tensors_.at(1);
  auto paddings = reinterpret_cast<int *>(padding_tensor->MutableData());
  if (paddings == nullptr) {
    MS_LOG(ERROR) << "Pad second input data nullptr";
    return RET_ERROR;
  }
  auto input_shape = input_tensors_.at(0)->shape();
  int rank = static_cast<int>(input_shape.size());
  MS_CHECK_GT(padding_tensor->ElementsNum(), 0, RET_ERROR);

  if (padding_tensor->ElementsNum() != rank * kDouble) {
    MS_LOG(ERROR) << "Pad second input elements num" << padding_tensor->ElementsNum() << ", should be "
                  << rank * kDouble;
    return RET_ERROR;
  }

  auto ret = ExtendPaddings(pad_param_->paddings_, MAX_SHAPE_SIZE, paddings, padding_tensor->ElementsNum());
  if (ret != RET_OK) {
    return ret;
  }
  pad_param_->padding_length = MAX_SHAPE_SIZE;
  return RET_OK;
}

int PadInt8Coder::CheckPaddings(const int *paddings, int length, const int *input_shape, int mode) {
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
    if (paddings[i * kDouble] > max_valid) {
      MS_LOG(WARNING) << prefix << "paddings " << paddings[i * kDouble] << "should be more than " << (max_valid + 1);
    }
    if (paddings[i * kDouble + 1] > max_valid) {
      MS_LOG(WARNING) << prefix << "paddings " << paddings[i * kDouble + 1] << "should be less than "
                      << (max_valid + 1);
    }
  }
  return RET_OK;
}

int PadInt8Coder::CalculateStrides() {
  pad_param_->in_strides[COMM_SHAPE_SIZE - 1] = 1;
  for (auto i = COMM_SHAPE_SIZE - 2; i >= 0; --i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(in_dims_[i + 1], pad_param_->in_strides[i + 1]), RET_ERROR, "mul overflow");
    pad_param_->in_strides[i] = in_dims_[i + 1] * pad_param_->in_strides[i + 1];
  }
  for (auto i = 0; i < COMM_SHAPE_SIZE; ++i) {
    out_dims_[i] = in_dims_[i] + pad_param_->paddings_[i * kDouble] + pad_param_->paddings_[i * kDouble + 1];
  }
  pad_param_->out_strides[COMM_SHAPE_SIZE - 1] = 1;
  for (auto i = COMM_SHAPE_SIZE - 2; i >= 0; --i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(out_dims_[i + 1], pad_param_->out_strides[i + 1]), RET_ERROR, "mul overflow");
    pad_param_->out_strides[i] = out_dims_[i + 1] * pad_param_->out_strides[i + 1];
  }
  return RET_OK;
}

int PadInt8Coder::HandleMirrorPad() {
  auto ret = CopyPaddingFromInput();
  if (ret != RET_OK) {
    return ret;
  }
  ret = CheckPaddings(pad_param_->paddings_, COMM_SHAPE_SIZE, in_dims_, pad_param_->pad_mode_);
  if (ret != RET_OK) {
    return ret;
  }
  ret = CalculateStrides();
  if (ret != RET_OK) {
    return ret;
  }
  pad_param_->mirror_offset_ = pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_REFLECT) ? 1 : 0;
  return RET_OK;
}

int PadInt8Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/int8/pad_int8.h",
          },
          {
            "pad_int8.c",
          });
  NNaclInt8Serializer code;
  MS_CHECK_GT(output_tensors_[0]->ElementsNum(), 0, RET_ERROR);

  code << "int in_dims[" << COMM_SHAPE_SIZE << "]=" << ToString(in_dims_) << ";\n";
  if (pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_CONSTANT)) {
    code.CodeFunction("memset", output_tensor_, pad_param_->pad_quant_arg_.constant_value_[0], output_tensor_->Size());
    code << "int out_dims[" << COMM_SHAPE_SIZE << "]=" << ToString(out_dims_) << ";\n";
    code << "int paddings[" << MAX_PAD_SIZE << "]=" << ToString(pad_param_->paddings_) << ";\n";

    code.CodeFunction("PadConstant4D", input_tensor_, output_tensor_, "in_dims", "out_dims", "paddings", 0, 1);
  } else {
    if (HandleMirrorPad() != RET_OK) {
      MS_LOG(ERROR) << "Handle mirror pad failed.";
      return RET_ERROR;
    }
    code << "QuantArg in_quant_args[1]={{" << pad_param_->pad_quant_arg_.in_quant_args_->scale_ << ","
         << pad_param_->pad_quant_arg_.in_quant_args_->zp_ << "}};\n";
    code << "QuantArg out_quant_args[1]={{" << pad_param_->pad_quant_arg_.out_quanr_args_->scale_ << ","
         << pad_param_->pad_quant_arg_.out_quanr_args_->zp_ << "}};\n";
    code << "unsigned char constant_value[1]={{" << pad_param_->pad_quant_arg_.constant_value_ << "}};\n";

    pad_param_->op_parameter_.thread_num_ = 1;
    code.CodeStruct("param", *pad_param_);
    code.CodeFunction("MirrorPadInt8", input_tensor_, output_tensor_, "in_dims", "&param", 0,
                      output_tensor_->ElementsNum());
  }

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_PadFusion, CPUOpCoderCreator<PadInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
