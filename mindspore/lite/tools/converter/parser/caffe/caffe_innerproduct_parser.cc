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

#include "tools/converter/parser/caffe/caffe_innerproduct_parser.h"
#include <memory>
#include "ops/fusion/full_connection.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeInnerProductParser::Parse(const caffe::LayerParameter &proto,
                                                const caffe::LayerParameter &weight) {
  auto primitive_c = new (std::nothrow) ops::FullConnection();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new FullConnection failed";
    return nullptr;
  }

  const caffe::InnerProductParameter &innerProductParam = proto.inner_product_param();
  if (!innerProductParam.has_num_output()) {
    MS_LOG(ERROR) << "InnerProduct Parse num_output for " << proto.name().c_str() << " failed.";
    return nullptr;
  }

  if (innerProductParam.axis() == 1) {
    primitive_c->set_axis(1);
    primitive_c->set_use_axis(true);
  } else {
    MS_LOG(ERROR) << "InnerProduct Parse axis only support default 1, but actually " << innerProductParam.axis();
    return nullptr;
  }
  if (innerProductParam.bias_term()) {
    primitive_c->set_has_bias(true);
  }

  primitive_c->set_activation_type(mindspore::ActivationType::NO_ACTIVATION);

  return primitive_c;
}

CaffeNodeRegistrar g_caffeInnerProductParser("InnerProduct", new CaffeInnerProductParser());
}  // namespace lite
}  // namespace mindspore
