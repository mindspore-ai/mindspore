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
#include "mindspore/lite/tools/converter/parser/caffe/caffe_reshape_parser.h"

namespace mindspore {
namespace lite {
STATUS CaffeReshapeParser::Parse(const caffe::LayerParameter &proto,
                                 const caffe::LayerParameter &weight,
                                 schema::CNodeT *op,
                                 std::vector<schema::TensorT *> *weightVec) {
  std::unique_ptr<schema::ReshapeT> attr(new schema::ReshapeT());
  attr->format = schema::Format_NCHW;

  const caffe::ReshapeParameter reshapeParam = proto.reshape_param();

  if (!reshapeParam.has_shape()) {
    // MS_LOGE("Reshape has no shape info, ret fail");
    return RET_ERROR;
  }

  const caffe::BlobShape &blob_shape = reshapeParam.shape();
  for (int i = 0; i < blob_shape.dim_size(); i++) {
    attr->shape.push_back(blob_shape.dim(i));
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.type = schema::PrimitiveType_Reshape;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_caffeReshapeParser("Reshape", new CaffeReshapeParser());
}  // namespace lite
}  // namespace mindspore

