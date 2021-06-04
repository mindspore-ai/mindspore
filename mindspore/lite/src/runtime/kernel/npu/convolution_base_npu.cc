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

#include "src/runtime/kernel/npu/convolution_base_npu.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
#include "nnacl/pack.h"

namespace mindspore::kernel {
namespace {
constexpr int BATCH_INDEX = 0;
constexpr int HEIGHT_INDEX = 1;
constexpr int WIDTH_INDEX = 2;
constexpr int CHANNEL_INDEX = 3;
constexpr size_t WITH_BIAS_SIZE = 3;
constexpr int BIAS_INDEX = 2;
constexpr int RELU_MODE = 1;
constexpr int RELU6_MODE = 14;
}  // namespace
ConvolutionBaseNPUKernel::~ConvolutionBaseNPUKernel() {
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

int ConvolutionBaseNPUKernel::InitWeightConst(const std::vector<lite::Tensor *> &inputs) {
  weight_ = new (std::nothrow) hiai::op::Const(name_ + "_w");
  if (weight_ == nullptr) {
    MS_LOG(ERROR) << "New weight const failed.";
    return RET_ERROR;
  }
  auto w_shape = inputs[1]->shape();
  auto nhwc_data = inputs[1]->data_c();
  auto nchw_data = reinterpret_cast<float *>(malloc(inputs[1]->ElementsNum() * sizeof(float)));
  if (nchw_data == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  PackNHWCToNCHWFp32(nhwc_data, nchw_data, w_shape[BATCH_INDEX], w_shape[HEIGHT_INDEX] * w_shape[WIDTH_INDEX],
                     w_shape[CHANNEL_INDEX], 0, 0);

  std::shared_ptr<ge::Tensor> weight_tensor = std::make_shared<ge::Tensor>();
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "new weight_tensor failed.";
    return RET_ERROR;
  }
  ge::TensorDesc tensor_desc(lite::ConverterToNPUShape({w_shape[BATCH_INDEX], w_shape[CHANNEL_INDEX],
                                                        w_shape[HEIGHT_INDEX], w_shape[WIDTH_INDEX]}),
                             ge::FORMAT_NCHW, lite::ConverterToNPUDataType(inputs[1]->data_type()));
  weight_tensor->SetTensorDesc(tensor_desc);
  weight_tensor->SetData(reinterpret_cast<const uint8_t *>(nchw_data), inputs[1]->Size());

  weight_->set_attr_value(weight_tensor);
  free(nchw_data);
  return RET_OK;
}

int ConvolutionBaseNPUKernel::InitBiasConst(const std::vector<lite::Tensor *> &inputs) {
  if (inputs.size() >= WITH_BIAS_SIZE) {
    bias_ = new (std::nothrow) hiai::op::Const(name_ + "_b");
    if (bias_ == nullptr) {
      MS_LOG(ERROR) << "New bias const failed.";
      return RET_ERROR;
    }
    inputs[BIAS_INDEX]->set_format(schema::Format_NCHW);
    auto bias_tensor = mindspore::lite::ConverterToNPUTensor(inputs[BIAS_INDEX]);
    bias_->set_attr_value(bias_tensor);
    inputs[BIAS_INDEX]->set_format(schema::Format_NHWC);
  }
  return RET_OK;
}

int ConvolutionBaseNPUKernel::SetActivation(const ge::Operator *input, ActType act_type) {
  act_ = new (std::nothrow) hiai::op::Activation(name_ + "_act");
  if (act_ == nullptr) {
    MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  act_->set_input_x(*input);
  if (act_type == ActType_Relu) {
    act_->set_attr_mode(RELU_MODE);
  } else if (act_type == ActType_Relu6) {
    act_->set_attr_mode(RELU6_MODE);
  } else {
    MS_LOG(ERROR) << "Unsupported activation type for convolution.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
