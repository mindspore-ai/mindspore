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

#include "mindspore/lite/tools/converter/parser/caffe/caffe_scale_parser.h"
#include <memory>

const int32_t NCHW_DIM_C = 1;
const int32_t DIM_DEFAULT_SIZE = 4;

namespace mindspore {
namespace lite {
STATUS CaffeScaleParser::Parse(const caffe::LayerParameter &proto,
                               const caffe::LayerParameter &weight,
                               schema::CNodeT *op,
                               std::vector<schema::TensorT *> *weightVec) {
  MS_LOG(DEBUG) << "parse CaffeScaleParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::ScaleT> attr = std::make_unique<schema::ScaleT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  if (weight.blobs_size() + weight.bottom_size() < 2) {
    MS_LOG(ERROR) << "Scale bottom size:" << weight.bottom_size() << ", blobs size:" << weight.blobs_size()
                  << " invalid in layer " << weight.name().c_str();
    return RET_ERROR;
  }

  const caffe::ScaleParameter scaleParam = weight.scale_param();
  int axis = NCHW_DIM_C;
  if (scaleParam.has_axis()) {
    uint32_t axis_index = NCHW_DIM_C;
    if (GetAxisIndex(scaleParam.axis(), &axis_index)) {
      MS_LOG(ERROR) << "scale get axis failed for layer " << weight.name().c_str();
      return RET_ERROR;
    }
  }
  attr->axis = axis;

  // parse scale
  if (weight.blobs().size() == 1) {
    auto scale = ConvertWeight(weight.blobs(0));
    if (scale == nullptr) {
      MS_LOG(ERROR) << "Scale Convert blobs(0) for layer " << weight.name().c_str() << " failed.";
      return RET_ERROR;
    }
    weightVec->push_back(scale);
  } else if (weight.blobs().size() >= 2) {
    auto scale = ConvertWeight(weight.blobs(0));
    if (scale == nullptr) {
      MS_LOG(ERROR) << "Scale Convert blobs(0) for layer " << weight.name().c_str() << " failed.";
      return RET_ERROR;
    }
    weightVec->push_back(scale);

    // parse bias
    bool scaleBias = scaleParam.bias_term();
    if (scaleBias) {
      auto bias = ConvertWeight(weight.blobs_size() > 1 ? weight.blobs(1) : weight.blobs(0));
      if (bias == nullptr) {
        MS_LOG(ERROR) << "Scale Convert blobs(1) for layer " << weight.name().c_str() << " failed.";
        return RET_ERROR;
      }
      weightVec->push_back(bias);
    }
  }

  op->name = proto.name();
  op->primitive->value.type = schema::PrimitiveType_Scale;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

STATUS CaffeScaleParser::GetAxisIndex(const int32_t &axis, uint32_t *axis_index) {
  if (axis < -DIM_DEFAULT_SIZE || axis >= DIM_DEFAULT_SIZE) {
    MS_LOG(ERROR) << "Scale axis value(" << axis << ") is not correct";
    return RET_ERROR;
  }

  if (axis == -1) {
    MS_LOG(WARNING) << "axis with -1 may lead to calculation errors when input less than 4 dims.";
  }

  *axis_index = (axis + DIM_DEFAULT_SIZE) % DIM_DEFAULT_SIZE;
  return RET_OK;
}

CaffeNodeRegistrar g_caffeScaleParser("Scale", new CaffeScaleParser());
}  // namespace lite
}  // namespace mindspore
