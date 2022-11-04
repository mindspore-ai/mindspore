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

#include "parser/caffe/caffe_reshape_parser.h"
#include <memory>
#include "common/op_attr.h"
#include "ops/reshape.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeReshapeParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Reshape>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  const caffe::ReshapeParameter &reshapeParam = proto.reshape_param();
  if (!reshapeParam.has_shape()) {
    MS_LOG(ERROR) << "Reshape has no shape info, ret fail";
    return nullptr;
  }
  std::vector<int64_t> shape;
  const caffe::BlobShape &blob_shape = reshapeParam.shape();
  for (int i = 0; i < blob_shape.dim_size(); i++) {
    shape.push_back(blob_shape.dim(i));
  }
  (void)prim->AddAttr(ops::kShape, api::MakeValue(shape));

  if (reshapeParam.has_axis()) {
    (void)prim->AddAttr(ops::kAxis, api::MakeValue(reshapeParam.axis()));
  }

  if (reshapeParam.has_num_axes()) {
    (void)prim->AddAttr(dpico::kNumAxes, api::MakeValue(reshapeParam.num_axes()));
  }

  return prim;
}

CaffeNodeRegistrar g_caffeReshapeParser("Reshape", new CaffeReshapeParser());
}  // namespace lite
}  // namespace mindspore
