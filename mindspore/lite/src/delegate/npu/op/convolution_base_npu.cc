/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/delegate/npu/op/convolution_base_npu.h"
#include "src/delegate/npu/npu_converter_utils.h"
#include "src/delegate/npu/transpose_kernel.h"

namespace mindspore {
ConvolutionBaseNPUOp::~ConvolutionBaseNPUOp() {
  if (act_ != nullptr) {
    delete act_;
    act_ = nullptr;
  }
  if (weight_ != nullptr) {
    delete weight_;
    weight_ = nullptr;
  }
  if (bias_ != nullptr) {
    delete bias_;
    bias_ = nullptr;
  }
}

void ConvolutionBaseNPUOp::FreeTmpWeight() {
  if (fp32_weight_ != nullptr) {
    free(fp32_weight_);
    fp32_weight_ = nullptr;
  }
  if (nchw_weight_ != nullptr) {
    free(nchw_weight_);
    nchw_weight_ = nullptr;
  }
}

int ConvolutionBaseNPUOp::InitWeightConst(const std::vector<mindspore::MSTensor> &inputs) {
  weight_ = new (std::nothrow) hiai::op::Const(name_ + "_w");
  if (weight_ == nullptr) {
    MS_LOG(ERROR) << "New weight const failed.";
    return RET_ERROR;
  }
  auto w_shape = inputs[1].Shape();
  auto origin_weight = inputs[1].Data().get();
  MS_ASSERT(origin_weight);

  nchw_weight_ = reinterpret_cast<float *>(malloc(inputs[1].ElementNum() * sizeof(float)));
  if (nchw_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }

  if (inputs[1].DataType() == DataType::kNumberTypeFloat16) {
#ifdef ENABLE_ARM64
    fp32_weight_ = reinterpret_cast<float *>(malloc(inputs[1].ElementNum() * sizeof(float)));
    if (fp32_weight_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      FreeTmpWeight();
      return RET_ERROR;
    }
    // weight fp16->fp32
    Float16ToFloat32(reinterpret_cast<const float16_t *>(origin_weight), reinterpret_cast<float *>(fp32_weight_),
                     inputs[1].ElementNum());
    PackNHWCToNCHWFp32(fp32_weight_, nchw_weight_, w_shape[NHWC_N], w_shape[NHWC_H] * w_shape[NHWC_W], w_shape[NHWC_C]);
#else
    MS_LOG(ERROR) << "This platform does not support fp16.";
    FreeTmpWeight();
    return RET_ERROR;
#endif
  } else if (inputs[1].DataType() == DataType::kNumberTypeFloat32) {
    PackNHWCToNCHWFp32(origin_weight, nchw_weight_, w_shape[NHWC_N], w_shape[NHWC_H] * w_shape[NHWC_W],
                       w_shape[NHWC_C]);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of weight tensor for npu convolution.";
    FreeTmpWeight();
    return RET_ERROR;
  }

  auto weight_tensor = std::make_shared<ge::Tensor>();
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "new weight_tensor failed.";
    FreeTmpWeight();
    return RET_ERROR;
  }
  ge::TensorDesc tensor_desc(ConverterToNPUShape({w_shape[NHWC_N], w_shape[NHWC_C], w_shape[NHWC_H], w_shape[NHWC_W]}),
                             ge::FORMAT_NCHW, ConverterToNPUDataType(inputs[1].DataType()));
  weight_tensor->SetTensorDesc(tensor_desc);
  weight_tensor->SetData(reinterpret_cast<const uint8_t *>(nchw_weight_), inputs[1].ElementNum() * sizeof(float));

  weight_->set_attr_value(weight_tensor);
  FreeTmpWeight();
  return RET_OK;
}

int ConvolutionBaseNPUOp::InitBiasConst(const std::vector<mindspore::MSTensor> &inputs) {
  if (inputs.size() >= 3) {
    bias_ = new (std::nothrow) hiai::op::Const(name_ + "_b");
    if (bias_ == nullptr) {
      MS_LOG(ERROR) << "New bias const failed.";
      return RET_ERROR;
    }
    std::shared_ptr<ge::Tensor> bias_tensor = ConverterToNPUTensor(inputs[BIAS_INDEX]);
    if (bias_tensor == nullptr) {
      MS_LOG(ERROR) << "Get bias_tensor failed.";
      return RET_ERROR;
    }
    bias_->set_attr_value(bias_tensor);
  }
  return RET_OK;
}

int ConvolutionBaseNPUOp::SetActivation(const ge::Operator *input, schema::ActivationType act_type) {
  act_ = new (std::nothrow) hiai::op::Activation(name_ + "_act");
  if (act_ == nullptr) {
    MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  act_->set_input_x(*input);
  auto act_mode = ConverterToNPUActivationMode(act_type);
  if (act_mode == ACTIVATION_INVALID) {
    MS_LOG(ERROR) << "Unsupported activation type for convolution op " << name_;
    return RET_ERROR;
  }
  act_->set_attr_mode(act_mode);
  return RET_OK;
}
}  // namespace mindspore
