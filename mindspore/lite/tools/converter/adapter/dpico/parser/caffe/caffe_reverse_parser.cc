/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "parser/caffe/caffe_reverse_parser.h"
#include <memory>
#include "ops/reverse_v2.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeReverseParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::ReverseV2>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }

  if (proto.has_reverse_param() && proto.reverse_param().has_axis()) {
    prim->set_axis({proto.reverse_param().axis()});
  } else {
    prim->set_axis({0});
  }

  return prim;
}

CaffeNodeRegistrar g_caffeReverseParser("Reverse", new CaffeReverseParser());
}  // namespace lite
}  // namespace mindspore
