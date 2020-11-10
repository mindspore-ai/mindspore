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

#include "tools/converter/parser/caffe/caffe_concat_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
PrimitiveC *CaffeConcatParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                                  const caffe::LayerParameter &weight) {
  std::unique_ptr<schema::ConcatT> attr = std::make_unique<schema::ConcatT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const caffe::ConcatParameter &concatParam = proto.concat_param();
  if (concatParam.has_axis() && concatParam.has_concat_dim()) {
    MS_LOG(ERROR) << "Concat param in caffe have concat_dim and axis simultaneously, return fail";
    return nullptr;
  }

  if (concatParam.has_concat_dim()) {
    MS_LOG(DEBUG) << "Concat dim , set axis: " << concatParam.concat_dim();
    auto concat_dim_value = (int32_t)concatParam.concat_dim();
    if (concat_dim_value < 0) {
      MS_LOG(ERROR) << "concat_dim value in model is smaller than 0:" << concat_dim_value;
      return nullptr;
    }
    attr->axis = concat_dim_value;
  } else if (concatParam.has_axis()) {
    MS_LOG(DEBUG) << "set axis: " << concatParam.axis();
    auto tmpInt = (int32_t)concatParam.axis();
    attr->axis = tmpInt;
  } else {
    MS_LOG(DEBUG) << "by default, set axis = 1";
    attr->axis = 1;
  }
  attr->n = proto.bottom_size();

  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Concat;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

CaffeNodeRegistrar g_caffeConcatParser("Concat", new CaffeConcatParser());
}  // namespace lite
}  // namespace mindspore
