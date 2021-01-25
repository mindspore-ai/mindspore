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
#include "ops/transpose.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffePermuteParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto primitive_c = new (std::nothrow) ops::Transpose();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Transpose failed";
    return nullptr;
  }

  std::vector<int32_t> perm;
  const caffe::PermuteParameter &permuteParam = proto.permute_param();
  const int num_order_dims = permuteParam.order_size();
  perm.resize(num_order_dims);
  for (int i = 0; i < num_order_dims; ++i) {
    perm[i] = permuteParam.order()[i];
  }
  primitive_c->AddAttr("perm", MakeValue(perm));

  return primitive_c;
}

CaffeNodeRegistrar g_caffePermuteParser("Permute", new CaffePermuteParser());
}  // namespace lite
}  // namespace mindspore
