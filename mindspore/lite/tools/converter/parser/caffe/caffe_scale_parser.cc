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

#include <memory>
#include "mindspore/lite/tools/converter/parser/caffe/caffe_scale_parser.h"

const int32_t NCHW_DIM_C = 1;
const int32_t DIM_DEFAULT_SIZE = 4;

namespace mindspore {
namespace lite {
STATUS CaffeScaleParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight,
                               schema::CNodeT *op, std::vector<schema::TensorT *> *weightVec) {
  std::unique_ptr<schema::ScaleT> attr(new schema::ScaleT());

  if (weight.blobs_size() + weight.bottom_size() < 2) {
    // MS_LOGE("Scale bottom size:%d, blobs size:%d invalid in layer %s", weight.bottom_size(), weight.blobs_size(),
    //        weight.name().c_str());
    return RET_ERROR;
  }

  const caffe::ScaleParameter scaleParam = weight.scale_param();
  int axis = NCHW_DIM_C;
  if (scaleParam.has_axis()) {
    uint32_t axis_index = NCHW_DIM_C;
    if (GetAxisIndex(scaleParam.axis(), &axis_index)) {
      // MS_LOGE("scale get axis failed for layer %s.", weight.name().c_str());
    }
  }
  attr->axis = axis;

  // parse scale
  // todo expect only weight as scale not bias
  if (weight.blobs().size() == 1) {
    auto scale = ConvertWeight(weight.blobs(0));
    if (scale == nullptr) {
      // MS_LOGE("Scale Convert blobs(0) for layer %s failed.", weight.name().c_str());
      return RET_ERROR;
    }
    weightVec->push_back(scale);
  } else if (weight.blobs().size() >= 2) {
    auto scale = ConvertWeight(weight.blobs(0));
    if (scale == nullptr) {
      // MS_LOGE("Scale Convert blobs(0) for layer %s failed.", weight.name().c_str());
      return RET_ERROR;
    }
    weightVec->push_back(scale);

    // parse bias
    bool scaleBias = scaleParam.bias_term();
    if (scaleBias) {
      auto bias = ConvertWeight(weight.blobs_size() > 1 ? weight.blobs(1) : weight.blobs(0));
      if (bias == nullptr) {
        // MS_LOGE("Scale Convert blobs(1) for layer %s failed.", weight.name().c_str());
        return RET_ERROR;
      }
      weightVec->push_back(bias);
    }
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.value = attr.release();
  op->primitive->value.type = schema::PrimitiveType_Scale;
  return RET_OK;
}

STATUS CaffeScaleParser::GetAxisIndex(const int32_t &axis, uint32_t *axis_index) {
  if (axis < -DIM_DEFAULT_SIZE || axis >= DIM_DEFAULT_SIZE) {
    // MS_LOGE("Scale axis value(%d) is not correct, ", axis);
    return RET_PARAM_INVALID;
  }

  if (axis == -1) {
    // MS_LOGW("axis with -1 may lead to calculation errors when input less than 4 dims.");
  }

  *axis_index = (axis + DIM_DEFAULT_SIZE) % DIM_DEFAULT_SIZE;
  return RET_OK;
}

CaffeNodeRegistrar g_caffeScaleParser("Scale", new CaffeScaleParser());
}  // namespace lite
}  // namespace mindspore
