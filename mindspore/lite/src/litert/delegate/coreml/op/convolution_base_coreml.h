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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_CONVOLUTION_BASE_COREML_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_CONVOLUTION_BASE_COREML_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <unordered_map>
#include "src/litert/delegate/coreml/op/coreml_op.h"
namespace mindspore::lite {
class ConvolutionBaseCoreMLOp : public CoreMLOp {
 public:
  ConvolutionBaseCoreMLOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                          const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : CoreMLOp(primitive, in_tensors, out_tensors, name) {
    input_h_ = static_cast<int>(in_tensors.at(0).Shape().at(kNHWC_H));
    input_w_ = static_cast<int>(in_tensors.at(0).Shape().at(kNHWC_W));
    kernel_h_ = static_cast<int>(in_tensors.at(1).Shape().at(MS_WT_H));
    kernel_w_ = static_cast<int>(in_tensors.at(1).Shape().at(MS_WT_W));
    output_h_ = static_cast<int>(out_tensors.at(0).Shape().at(kNHWC_H));
    output_w_ = static_cast<int>(out_tensors.at(0).Shape().at(kNHWC_W));
  }

  int BuildLayer() override;

 protected:
  virtual int SetConvParam() { return RET_OK; }

  virtual int SetConvWeight();

  virtual int SetConvBias();

 protected:
  int input_h_;
  int input_w_;
  int kernel_h_;
  int kernel_w_;
  int output_h_;
  int output_w_;
  CoreML::Specification::ConvolutionLayerParams *conv_param_ = nullptr;
  schema::ActivationType act_type_ = schema::ActivationType_NO_ACTIVATION;
  std::unique_ptr<CoreML::Specification::NeuralNetworkLayer> trans_in_op_ = nullptr;
  std::unique_ptr<CoreML::Specification::NeuralNetworkLayer> trans_out_op_ = nullptr;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_CONVOLUTION_BASE_COREML_H_
