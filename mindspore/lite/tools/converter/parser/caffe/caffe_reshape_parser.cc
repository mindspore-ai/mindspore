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

#include "tools/converter/parser/caffe/caffe_reshape_parser.h"
#include <memory>
#include "ops/reshape.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeReshapeParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Reshape>();

  const caffe::ReshapeParameter &reshapeParam = proto.reshape_param();
  if (!reshapeParam.has_shape()) {
    MS_LOG(ERROR) << "Reshape has no shape info, ret fail";
    return nullptr;
  }
  std::vector<int32_t> shape;
  const caffe::BlobShape &blob_shape = reshapeParam.shape();
  for (int i = 0; i < blob_shape.dim_size(); i++) {
    shape.push_back(blob_shape.dim(i));
  }
  prim->AddAttr("shape", MakeValue(shape));

  return prim.release();
}

CaffeNodeRegistrar g_caffeReshapeParser("Reshape", new CaffeReshapeParser());
}  // namespace lite
}  // namespace mindspore
