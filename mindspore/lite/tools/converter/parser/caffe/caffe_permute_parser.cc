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

#include "tools/converter/parser/caffe/caffe_permute_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
PrimitiveC *CaffePermuteParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                                   const caffe::LayerParameter &weight) {
  std::unique_ptr<schema::TransposeT> attr = std::make_unique<schema::TransposeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const caffe::PermuteParameter &permuteParam = proto.permute_param();
  const int num_order_dims = permuteParam.order_size();
  attr->perm.resize(num_order_dims);
  for (int i = 0; i < num_order_dims; ++i) {
    attr->perm[i] = (int32_t)permuteParam.order()[i];
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Transpose;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

CaffeNodeRegistrar g_caffePermuteParser("Permute", new CaffePermuteParser());
}  // namespace lite
}  // namespace mindspore
