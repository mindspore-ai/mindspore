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

#include "parser/caffe/caffe_scale_parser.h"
#include <memory>
#include "common/op_attr.h"
#include "common/op_enum.h"
#include "ops/fusion/scale_fusion.h"

namespace mindspore {
namespace lite {
STATUS CaffeScaleParser::GetAxisIndex(const int32_t &axis, uint32_t *axis_index) {
  if (axis < dpico::kAxisLowerBound || axis > dpico::kAxisUpperBound) {
    MS_LOG(ERROR) << "Scale axis value(" << axis << ") is not correct";
    return RET_ERROR;
  }

  if (axis == -1) {
    MS_LOG(WARNING) << "axis with -1 may lead to calculation errors when input less than 4 dims.";
  }

  if (axis_index == nullptr) {
    MS_LOG(ERROR) << "axis_index is nullptr.";
    return RET_ERROR;
  }
  *axis_index = (static_cast<uint32_t>(axis) + dpico::kDims4) % dpico::kDims4;
  return RET_OK;
}

BaseOperatorPtr CaffeScaleParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::ScaleFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  if (weight.blobs_size() + weight.bottom_size() < kNums2) {
    MS_LOG(ERROR) << "Scale bottom size:" << weight.bottom_size() << ", blobs size:" << weight.blobs_size()
                  << " invalid in layer " << weight.name().c_str();
    return nullptr;
  }

  const caffe::ScaleParameter &scaleParam = weight.scale_param();
  prim->set_axis(1);
  if (scaleParam.has_axis()) {
    uint32_t axis_index = 1;
    if (GetAxisIndex(scaleParam.axis(), &axis_index)) {
      MS_LOG(ERROR) << "scale get axis failed for layer " << weight.name().c_str();
      return nullptr;
    }
    prim->set_axis(axis_index);
  }

  if (scaleParam.has_bias_term()) {
    (void)prim->AddAttr(dpico::kBiasTerm, api::MakeValue<bool>(scaleParam.bias_term()));
  }

  if (scaleParam.has_num_axes()) {
    (void)prim->AddAttr(dpico::kNumAxes, api::MakeValue<int64_t>(scaleParam.num_axes()));
  }
  return prim;
}

CaffeNodeRegistrar g_caffeScaleParser("Scale", new CaffeScaleParser());
}  // namespace lite
}  // namespace mindspore
