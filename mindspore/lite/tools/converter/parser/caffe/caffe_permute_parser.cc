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

#include "mindspore/lite/tools/converter/parser/caffe/caffe_permute_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS CaffePermuteParser::Parse(const caffe::LayerParameter &proto,
                                const caffe::LayerParameter &weight,
                                schema::CNodeT *op,
                                std::vector<schema::TensorT *> *weightVec) {
  op->name = proto.name();
  std::unique_ptr<schema::TransposeT> attr(new schema::TransposeT());
  const caffe::PermuteParameter permuteParam = proto.permute_param();

  const int num_order_dims = permuteParam.order_size();
  attr->perm.resize(num_order_dims);
  for (int i = 0; i < num_order_dims; ++i) {
    attr->perm[i] = (int32_t)permuteParam.order()[i];
  }
  attr->conjugate = false;

  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.value = attr.release();
  op->primitive->value.type = schema::PrimitiveType_Transpose;
  return RET_OK;
}

CaffeNodeRegistrar g_caffePermuteParser("Permute", new CaffePermuteParser());
}  // namespace lite
}  // namespace mindspore

