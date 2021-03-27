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

#include <string>
#include <vector>
#include "coder/opcoders/cmsis-nn/int8/pooling_int8_coder.h"
#include "coder/opcoders/serializers/serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;

namespace mindspore::lite::micro::cmsis {
int PoolingInt8Coder::Prepare(CoderContext *const context) {
  this->pooling_parameter_ = reinterpret_cast<PoolingParameter *>(parameter_);
  // get tensors
  MS_CHECK_RET_CODE(SetParameters(), "SetParameters failed");

  if (pooling_parameter_->pool_mode_ == PoolMode_AvgPool) {
    buffer_size_ = input_tensor_->Channel() * sizeof(int32_t);
    buffer_ = static_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, buffer_size_, kWorkspace));
    MS_CHECK_PTR(buffer_);
  }

  return RET_OK;
}

int PoolingInt8Coder::DoCode(CoderContext *const context) {
  // init struct PoolingParameters
  std::string pooling_func;

  std::vector<std::string> cFiles;
  if (pooling_parameter_->pool_mode_ == PoolMode_AvgPool) {
    cFiles = {"arm_avgpool_s8.c"};
    pooling_func = "arm_avgpool_s8";
  } else if (pooling_parameter_->pool_mode_ == PoolMode_MaxPool) {
    cFiles = {"arm_max_pool_s8.c"};
    pooling_func = "arm_max_pool_s8";
  } else {
    MS_LOG(ERROR) << "unsupported pad mode";
    return RET_ERROR;
  }
  Collect(context, {"CMSIS/NN/Include/arm_nnfunctions.h"}, cFiles);

  Serializer code;
  code.precision(kPrecision);

  code.CodeFunction(pooling_func, dim_src_height_, dim_src_width_, dim_dst_height_, dim_dst_width_, stride_height_,
                    stride_width_, dim_kernel_height_, dim_kernel_width_, padding_height_, padding_width_, act_min_,
                    act_max_, ch_src_, input_tensor_, buffer_, output_tensor_);
  context->AppendCode(code.str());
  return RET_OK;
}

int PoolingInt8Coder::SetParameters() {
  dim_src_height_ = input_tensor_->Height();
  dim_src_width_ = input_tensor_->Width();
  dim_dst_height_ = output_tensor_->DimensionSize(1);
  dim_dst_width_ = output_tensor_->DimensionSize(2);
  ch_src_ = input_tensor_->Channel();

  stride_height_ = pooling_parameter_->stride_h_;
  stride_width_ = pooling_parameter_->stride_w_;

  dim_kernel_height_ = pooling_parameter_->window_h_;
  dim_kernel_width_ = pooling_parameter_->window_w_;

  // only use pad_u_ and pad_l_ because their value is consistent with tf
  // ref: mindspore/lite/src/ops/conv2d.cc:ConvInferShape
  padding_height_ = pooling_parameter_->pad_u_;
  padding_width_ = pooling_parameter_->pad_l_;

  MS_CHECK_TRUE(!output_tensor_->quant_params().empty(), "output quant_params is empty");
  QuantArg output_quant_arg = output_tensor_->quant_params().at(0);
  CalculateActivationRangeQuantized(pooling_parameter_->act_type_ == ActType_Relu,
                                    pooling_parameter_->act_type_ == ActType_Relu6, output_quant_arg.zeroPoint,
                                    output_quant_arg.scale, &act_min_, &act_max_);

  MS_CHECK_TRUE(input_tensor_->Channel() == output_tensor_->Channel(),
                "input Channel and output Channel size not match!");
  return RET_OK;
}

REG_OPERATOR_CODER(kARM32M, kNumberTypeInt8, PrimitiveType_AvgPoolFusion, CPUOpCoderCreator<PoolingInt8Coder>)
REG_OPERATOR_CODER(kARM32M, kNumberTypeInt8, PrimitiveType_MaxPoolFusion, CPUOpCoderCreator<PoolingInt8Coder>)

}  // namespace mindspore::lite::micro::cmsis
