/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/cmsis-nn/int8/conv2d_int8_coder.h"
#include <string>
#include <vector>
#include <memory>
#include "coder/opcoders/serializers/serializer.h"
#include "coder/opcoders/file_collector.h"
#include "src/common/prim_util.h"

using mindspore::schema::PrimitiveType_Conv2DFusion;

namespace mindspore::lite::micro::cmsis {
int Conv2DInt8Coder::Prepare(CoderContext *const context) {
  Conv2DBaseCoder::Init();
  MS_CHECK_RET_CODE(micro::Conv2DBaseCoder::CheckLayout(input_tensor_), "CheckLayout failed");
  MS_CHECK_RET_CODE(micro::Conv2DBaseCoder::SetQuantParam(), "SetQuantParam failed");
  MS_CHECK_RET_CODE(Conv2DBaseCoder::SetQuantArgs(), "SetQuantArgs failed");
  MS_CHECK_RET_CODE(SetParameters(), "SetParameters failed");
  CheckSupportOptimize();
  MS_CHECK_RET_CODE(InitTmpBuffer(), "InitTmpBuffer failed");
  return RET_OK;
}

int Conv2DInt8Coder::DoCode(CoderContext *const context) {
  Serializer code;
  code.precision(kPrecision);
  Collect(context,
          {
            "CMSIS/NN/Include/arm_nnfunctions.h",
          },
          {});
  if (opt_ != Convolve_1x1_fast) {
    code.CodeFunction("memset", buffer_, 0, buffer_size_);
  }
  code.CodeArray("output_shift", output_shift_, output_ch_);
  code.CodeArray("output_mult", output_mult_, output_ch_);
  switch (opt_) {
    case Basic:
      Collect(context, {},
              {
                "arm_convolve_s8.c",
                "arm_nn_mat_mult_kernel_s8_s16.c",
                "arm_q7_to_q15_with_offset.c",
              });
      code.CodeFunction("arm_convolve_s8", input_tensor_, input_x_, input_y_, input_ch_, input_batches_, filter_tensor_,
                        output_ch_, kernel_x_, kernel_y_, pad_x_, pad_y_, stride_x_, stride_y_, bias_tensor_,
                        output_tensor_, "output_shift", "output_mult", out_offset_, input_offset_, out_activation_min_,
                        out_activation_max_, output_x_, output_y_, buffer_);
      break;
    case Convolve_1_x_n:
      Collect(context, {},
              {
                "arm_convolve_1_x_n_s8.c",
                "arm_nn_mat_mul_core_1x_s8.c",
              });
      code.CodeFunction("arm_convolve_1_x_n_s8", input_tensor_, input_x_, input_ch_, input_batches_, filter_tensor_,
                        output_ch_, kernel_x_, pad_x_, stride_x_, bias_tensor_, output_tensor_, "output_shift",
                        "output_mult", out_offset_, input_offset_, out_activation_min_, out_activation_max_, output_x_,
                        buffer_);
      break;
    case Convolve_1x1_fast:
      Collect(context, {},
              {
                "arm_convolve_1x1_s8_fast.c",
                "arm_nn_mat_mult_nt_t_s8.c",
                "arm_nn_mat_mul_core_4x_s8.c",
                "arm_nn_mat_mul_core_1x_s8.c",
              });
      code.CodeFunction("arm_convolve_1x1_s8_fast", input_tensor_, input_x_, input_y_, input_ch_, input_batches_,
                        filter_tensor_, output_ch_, pad_x_, pad_y_, stride_x_, stride_y_, bias_tensor_, output_tensor_,
                        "output_shift", "output_mult", out_offset_, input_offset_, out_activation_min_,
                        out_activation_max_, output_x_, output_y_, buffer_);
      break;
    default:
      MS_LOG(ERROR) << "opt enum value is not defined";
      return RET_ERROR;
  }

  context->AppendCode(code.str());
  return RET_OK;
}

int Conv2DInt8Coder::SetParameters() {
  MS_CHECK_TRUE(input_tensor_->Channel() == filter_tensor_->DimensionSize(kNHWC_C),
                "input Channel and filter size not match!");
  MS_CHECK_TRUE(output_tensor_->Channel() == filter_tensor_->DimensionSize(kNHWC_N),
                "output Channel and filter size not match!");

  input_x_ = input_tensor_->Width();
  input_y_ = input_tensor_->Height();
  input_ch_ = input_tensor_->Channel();
  input_batches_ = input_tensor_->Batch();

  kernel_x_ = filter_tensor_->DimensionSize(kNHWC_W);
  kernel_y_ = filter_tensor_->DimensionSize(kNHWC_H);
  pad_x_ = conv_param_->pad_l_;
  pad_y_ = conv_param_->pad_u_;

  stride_x_ = conv_param_->stride_w_;
  stride_y_ = conv_param_->stride_h_;

  MS_CHECK_TRUE(!input_tensor_->quant_params().empty(), "input quant_params is empty");
  MS_CHECK_TRUE(!output_tensor_->quant_params().empty(), "output quant_params is empty");
  LiteQuantParam input_quant_arg = input_tensor_->quant_params().at(0);
  LiteQuantParam output_quant_arg = output_tensor_->quant_params().at(0);

  input_offset_ = -input_quant_arg.zeroPoint;
  out_offset_ = output_quant_arg.zeroPoint;

  output_x_ = output_tensor_->DimensionSize(kNHWC_W);
  output_y_ = output_tensor_->DimensionSize(kNHWC_H);
  output_ch_ = output_tensor_->Channel();

  CalculateActivationRangeQuantized(conv_param_->act_type_ == ActType_Relu, conv_param_->act_type_ == ActType_Relu6,
                                    output_quant_arg.zeroPoint, static_cast<float>(output_quant_arg.scale),
                                    &out_activation_min_, &out_activation_max_);
  return RET_OK;
}

void Conv2DInt8Coder::CheckSupportOptimize() {
  if ((pad_x_ == 0) && (pad_y_ == 0) && (input_ch_ % 4 == 0) && (stride_x_ == 1) && (stride_y_ == 1) &&
      (kernel_x_ == 1) && (kernel_y_ == 1)) {
    opt_ = Convolve_1x1_fast;
    return;
  }

  if ((output_x_ == 1) && (input_x_ == 1) && (kernel_y_ == 1) && (output_x_ % 4 == 0) && (input_batches_ == 1)) {
    opt_ = Convolve_1_x_n;
    return;
  }
  opt_ = Basic;
}

int Conv2DInt8Coder::InitTmpBuffer() {
  const size_t kPartial = 2;
  switch (opt_) {
    case Basic:
      buffer_size_ =
        static_cast<size_t>(kPartial * input_tensor_->Channel() * filter_tensor_->Width() * filter_tensor_->Height()) *
        sizeof(int16_t);
      break;
    case Convolve_1_x_n:
      buffer_size_ =
        static_cast<size_t>(kPartial * input_tensor_->Channel() * filter_tensor_->Width() * filter_tensor_->Height()) *
        sizeof(int16_t);
      break;
    case Convolve_1x1_fast:
      // do nothing
      buffer_size_ = 0;
      return RET_OK;
    default:
      MS_LOG(ERROR) << "opt enum value is not defined";
      return RET_ERROR;
  }
  buffer_ = static_cast<int16_t *>(allocator_->Malloc(kNumberTypeInt16, buffer_size_, kWorkspace));
  MS_CHECK_PTR(buffer_);
  return RET_OK;
}

std::unique_ptr<OperatorCoder> CmsisConv2DInt8OpCoderCreator(const std::vector<Tensor *> &in_tensors,
                                                             const std::vector<Tensor *> &out_tensors,
                                                             const Model::Node *node, size_t node_index, Target target,
                                                             int schema_version) {
  MS_CHECK_PTR_RET_NULL(node);
  std::unique_ptr<Conv2DInt8Coder> coder =
    std::make_unique<Conv2DInt8Coder>(in_tensors, out_tensors, node, node_index, target);
  return coder;
}

REG_OPERATOR_CODER(kARM32M, kNumberTypeInt8, PrimitiveType_Conv2DFusion, CPUOpCoderCreator<Conv2DInt8Coder>)
}  // namespace mindspore::lite::micro::cmsis
