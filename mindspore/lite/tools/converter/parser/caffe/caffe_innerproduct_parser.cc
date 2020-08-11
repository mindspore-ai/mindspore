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
#include "mindspore/lite/tools/converter/parser/caffe/caffe_innerproduct_parser.h"

namespace mindspore {
namespace lite {
STATUS CaffeInnerProductParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight,
                                      schema::CNodeT *op, std::vector<schema::TensorT *> *weightVec) {
  const caffe::InnerProductParameter innerProductParam = proto.inner_product_param();
  std::unique_ptr<schema::FullConnectionT> attr(new schema::FullConnectionT());

  if (!innerProductParam.has_num_output()) {
    // MS_LOGE("InnerProduct Parse num_output for %s failed.", proto.name().c_str());
    return RET_ERROR;
  }

  if (innerProductParam.axis() == 1) {
    attr->axis = 1;
    attr->useAxis = true;
  } else {
    // MS_LOG(ERROR) << "InnerProduct Parse axis only support default 1, but actually " << innerProductParam.axis();
    return RET_ERROR;
  }

  if (innerProductParam.bias_term()) {
    attr->hasBias = true;
  }

  // parse weight
  if (weight.blobs_size() == 0) {
    // MS_LOGE("InnerProduct No filter data in layer %s", weight.name().c_str());
    return RET_ERROR;
  }

  // parse filter
  auto filter = ConvertWeight(weight.blobs(0));
  if (filter == nullptr) {
    // MS_LOGE("InnerProduct parse weight for layer %s failed", weight.name().c_str());
    return RET_ERROR;
  }
  weightVec->push_back(filter);

  // parse bias
  if (innerProductParam.bias_term() && weight.blobs_size() > 1) {
    auto bias = ConvertWeight(weight.blobs(1));
    if (bias == nullptr) {
      // MS_LOGE("InnerProduct parse bias for layer %s failed", weight.name().c_str());
      return RET_ERROR;
    }
    weightVec->push_back(bias);
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.value = attr.release();
  op->primitive->value.type = schema::PrimitiveType_FullConnection;
  return RET_OK;
}

CaffeNodeRegistrar g_caffeInnerProductParser("InnerProduct", new CaffeInnerProductParser());
}  // namespace lite
}  // namespace mindspore

