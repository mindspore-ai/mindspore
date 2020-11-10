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

#include "tools/converter/parser/caffe/caffe_argmax_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
lite::PrimitiveC *CaffeArgMaxParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                                        const caffe::LayerParameter &weight) {
  std::unique_ptr<schema::ArgMaxT> attr = std::make_unique<schema::ArgMaxT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  attr->outMaxValue = false;
  attr->topK = 1;
  const caffe::ArgMaxParameter &argmaxParam = proto.argmax_param();
  if (argmaxParam.has_out_max_val()) {
    attr->outMaxValue = argmaxParam.out_max_val();
  }
  if (argmaxParam.has_top_k()) {
    attr->topK = argmaxParam.top_k();
  }
  int32_t axisType = 0;
  int32_t axis = 0;
  if (!argmaxParam.has_axis()) {
    axisType = 2;
  } else {
    axisType = 1;
    axis = (int64_t)argmaxParam.axis();
  }
  attr->axis = axis;
  attr->axisType = axisType;
  attr->keepDims = true;
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_ArgMax;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

CaffeNodeRegistrar g_caffeArgMaxParser("ArgMax", new CaffeArgMaxParser());
}  // namespace lite
}  // namespace mindspore
