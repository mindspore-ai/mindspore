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
#include "ops/concat.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeConcatParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Concat>();

  const caffe::ConcatParameter &concatParam = proto.concat_param();
  if (concatParam.has_axis() && concatParam.has_concat_dim()) {
    MS_LOG(ERROR) << "Concat param in caffe have concat_dim and axis simultaneously, return fail";
    return nullptr;
  }

  int64_t axis = 1;
  if (concatParam.has_concat_dim()) {
    MS_LOG(DEBUG) << "Concat dim , set axis: " << concatParam.concat_dim();
    axis = concatParam.concat_dim();
    if (axis < 0) {
      MS_LOG(ERROR) << "concat_dim value in model is smaller than 0:" << axis;
      return nullptr;
    }
  } else if (concatParam.has_axis()) {
    MS_LOG(DEBUG) << "set axis: " << concatParam.axis();
    axis = concatParam.axis();
  }
  prim->set_axis(axis);

  return prim.release();
}

CaffeNodeRegistrar g_caffeConcatParser("Concat", new CaffeConcatParser());
}  // namespace lite
}  // namespace mindspore
