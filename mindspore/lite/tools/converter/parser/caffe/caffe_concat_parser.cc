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

#include "mindspore/lite/tools/converter/parser/caffe/caffe_concat_parser.h"
#include <memory>

const int32_t CONCAT_DEFAULT_AXIS = 1;

namespace mindspore {
namespace lite {
STATUS CaffeConcatParser::Parse(const caffe::LayerParameter &proto,
                                const caffe::LayerParameter &weight,
                                schema::CNodeT *op,
                                std::vector<schema::TensorT *> *weightVec) {
  MS_LOG(DEBUG) << "parse CaffeConcatParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::ConcatT> attr = std::make_unique<schema::ConcatT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const caffe::ConcatParameter concatParam = proto.concat_param();
  if (concatParam.has_axis() && concatParam.has_concat_dim()) {
    MS_LOG(ERROR) << "Concat param in caffe have concat_dim and axis simultaneously, return fail";
    return RET_ERROR;
  }

  if (concatParam.has_concat_dim()) {
    MS_LOG(DEBUG) << "Concat dim , set axis: " << concatParam.concat_dim();
    int32_t concat_dim_value = (int32_t)concatParam.concat_dim();
    if (concat_dim_value < 0) {
      MS_LOG(ERROR) << "concat_dim value in model is smaller than 0:" << concat_dim_value;
      return RET_ERROR;
    }
    attr->axis = concat_dim_value;
  } else if (concatParam.has_axis()) {
    MS_LOG(DEBUG) << "axis , set axis: " << concatParam.axis();
    int32_t tmpInt = (int32_t)concatParam.axis();
    attr->axis = tmpInt;
  } else {
    MS_LOG(DEBUG) << "default , set axis: " << CONCAT_DEFAULT_AXIS;
    attr->axis = CONCAT_DEFAULT_AXIS;
  }
  attr->n = proto.bottom_size();

  op->name = proto.name();
  op->primitive->value.type = schema::PrimitiveType_Concat;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_caffeConcatParser("Concat", new CaffeConcatParser());
}  // namespace lite
}  // namespace mindspore

