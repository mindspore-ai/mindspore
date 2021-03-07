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

#include "tools/converter/parser/tflite/tflite_pooling_parser.h"
#include <vector>
#include <memory>
#include <string>
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fusion/max_pool_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteAvgPoolParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                            const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::AvgPoolFusion>();

  prim->set_format(mindspore::Format::NHWC);
  prim->set_round_mode(mindspore::RoundMode::FLOOR);
  prim->set_global(false);

  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  if (tflite_subgraph == nullptr) {
    MS_LOG(ERROR) << "tflite_subgraph is nullptr";
    return nullptr;
  }
  const auto &tflite_attr = tflite_op->builtin_options.AsPool2DOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: conv attr failed";
    return nullptr;
  }
  prim->set_kernel_size({tflite_attr->filter_height, tflite_attr->filter_width});
  prim->set_strides({tflite_attr->stride_h, tflite_attr->stride_w});
  auto padMode = GetPadMode(tflite_attr->padding);
  prim->set_pad_mode(padMode);
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  // calculate pad params
  const auto &dataTensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(0));
  std::vector<int64_t> params;
  int status = getPaddingParam(dataTensor, padMode, tflite_attr->stride_h, tflite_attr->stride_w,
                               tflite_attr->filter_height, tflite_attr->filter_width, &params);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get padding params failed";
    return nullptr;
  } else if (status == RET_OK) {
    prim->set_pad(params);
  }

  return prim.release();
}

ops::PrimitiveC *TfliteMaxPoolParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                            const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::MaxPoolFusion>();

  prim->set_format(mindspore::Format::NHWC);
  prim->set_round_mode(mindspore::RoundMode::FLOOR);
  prim->set_global(false);

  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  if (tflite_subgraph == nullptr) {
    MS_LOG(ERROR) << "tflite_subgraph is nullptr";
    return nullptr;
  }
  const auto &tflite_attr = tflite_op->builtin_options.AsPool2DOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: conv attr failed";
    return nullptr;
  }
  prim->set_kernel_size({tflite_attr->filter_height, tflite_attr->filter_width});
  prim->set_strides({tflite_attr->stride_h, tflite_attr->stride_w});
  auto padMode = GetPadMode(tflite_attr->padding);
  prim->set_pad_mode(padMode);
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  // calculate pad params
  const auto &dataTensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(0));
  std::vector<int64_t> params;
  int status = getPaddingParam(dataTensor, padMode, tflite_attr->stride_h, tflite_attr->stride_w,
                               tflite_attr->filter_height, tflite_attr->filter_width, &params);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get padding params failed";
    return nullptr;
  } else if (status == RET_OK) {
    prim->set_pad(params);
  }

  return prim.release();
}

TfliteNodeRegister g_tfliteMeanPoolingParser(tflite::BuiltinOperator_AVERAGE_POOL_2D, new TfliteAvgPoolParser());
TfliteNodeRegister g_tfliteMaxPoolingParser(tflite::BuiltinOperator_MAX_POOL_2D, new TfliteMaxPoolParser());
}  // namespace lite
}  // namespace mindspore
