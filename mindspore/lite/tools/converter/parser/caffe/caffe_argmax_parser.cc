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

#include "mindspore/lite/tools/converter/parser/caffe/caffe_argmax_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS CaffeArgMaxParser::Parse(const caffe::LayerParameter &proto,
                                const caffe::LayerParameter &weight,
                                schema::CNodeT *op,
                                std::vector<schema::TensorT *> *weightVec) {
  op->name = proto.name();
  std::unique_ptr<schema::ArgMaxT> attr(new schema::ArgMaxT());
  const caffe::ArgMaxParameter argmaxParam = proto.argmax_param();

  int32_t axisType = 0;
  int32_t axis = 0;
  if (!argmaxParam.has_axis()) {
    axisType = 2;
  } else {
    axisType = 1;
    axis = (int64_t)argmaxParam.axis();
    if (axis == -1) {
      // MS_LOGE("axis with -1 may lead to calculation errors when input less than 4 dims.");
      return RET_ERROR;
    }
  }

  attr->axis = axis;
  attr->axisType = axisType;
  attr->outMaxValue = argmaxParam.out_max_val();
  attr->topK = argmaxParam.top_k();
  attr->keepDims = true;

  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.value = attr.release();
  op->primitive->value.type = schema::PrimitiveType_ArgMax;
  return RET_OK;
}

CaffeNodeRegistrar g_caffeArgMaxParser("ArgMax", new CaffeArgMaxParser());
}  // namespace lite
}  // namespace mindspore

