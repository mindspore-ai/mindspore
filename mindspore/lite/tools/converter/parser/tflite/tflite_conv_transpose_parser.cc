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

#include "tools/converter/parser/tflite/tflite_conv_transpose_parser.h"
#include <vector>
#include <memory>
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TfliteDeConvParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Conv2dTransposeFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  prim->set_pad({0, 0, 0, 0});
  prim->set_group(1);
  prim->set_format(mindspore::Format::NHWC);
  prim->set_activation_type(mindspore::ActivationType::NO_ACTIVATION);
  prim->set_dilation({1, 1});
  prim->set_output_paddings({0, 0});

  const auto &tflite_attr = tflite_op->builtin_options.AsTransposeConvOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get deconv attr failed";
    return nullptr;
  }
  prim->set_stride({tflite_attr->stride_h, tflite_attr->stride_w});
  auto padMode = GetPadMode(tflite_attr->padding);
  prim->set_pad_mode(padMode);

  // get weight tensor
  if (tflite_op->inputs.size() < kInputSize2) {
    MS_LOG(ERROR) << "the tflite_op shape is illegal";
    return nullptr;
  }
  if (tflite_op->inputs.size() == kInputSize2) {
    (void)prim->AddAttr(ops::kHasBias, api::MakeValue<bool>(true));
  }
  const auto &weight_tensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(SECOND_INPUT));
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "the weight tensor is null";
    return nullptr;
  }
  auto weight_shape = weight_tensor->shape;
  if (weight_shape.empty() || weight_shape.size() < DIMENSION_4D) {
    MS_LOG(ERROR) << "the weight shape is illegal";
    return nullptr;
  }
  prim->set_in_channel(weight_shape[kWeightChannelIn]);
  prim->set_out_channel(weight_shape[kWeightChannelOut]);
  prim->set_kernel_size({weight_shape[kWeightKernelH], weight_shape[kWeightKernelW]});

  // calculate pad params
  const auto &data_tensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(THIRD_INPUT));
  if (data_tensor == nullptr) {
    MS_LOG(ERROR) << "data_tensor is nullptr";
    return nullptr;
  }
  std::vector<int64_t> params;
  int status = getPaddingParam(data_tensor, padMode, tflite_attr->stride_h, tflite_attr->stride_w,
                               weight_shape[kWeightKernelH], weight_shape[kWeightKernelW], &params);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get padding params failed";
    return nullptr;
  } else if (status == RET_OK) {
    prim->set_pad_list(params);
  }

  return prim->GetPrim();
}

TfliteNodeRegister g_tfliteDeConv2DParser(tflite::BuiltinOperator_TRANSPOSE_CONV, new TfliteDeConvParser());
}  // namespace lite
}  // namespace mindspore
