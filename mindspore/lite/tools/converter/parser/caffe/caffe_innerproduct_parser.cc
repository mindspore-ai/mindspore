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

#include "mindspore/lite/tools/converter/parser/caffe/caffe_innerproduct_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS CaffeInnerProductParser::Parse(const caffe::LayerParameter &proto,
                                      const caffe::LayerParameter &weight,
                                      schema::CNodeT *op,
                                      std::vector<schema::TensorT *> *weightVec) {
  MS_LOG(DEBUG) << "parse CaffeInnerProductParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::FullConnectionT> attr = std::make_unique<schema::FullConnectionT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const caffe::InnerProductParameter innerProductParam = proto.inner_product_param();
  if (!innerProductParam.has_num_output()) {
    MS_LOG(ERROR) << "InnerProduct Parse num_output for " << proto.name().c_str() << " failed.";
    return RET_ERROR;
  }

  if (innerProductParam.axis() == 1) {
    attr->axis = 1;
    attr->useAxis = true;
  } else {
    MS_LOG(ERROR) << "InnerProduct Parse axis only support default 1, but actually " << innerProductParam.axis();
    return RET_ERROR;
  }

  if (innerProductParam.bias_term()) {
    attr->hasBias = true;
  }
  attr->activationType = schema::ActivationType_NO_ACTIVATION;

  // parse weight
  if (weight.blobs_size() == 0) {
    MS_LOG(ERROR) << "InnerProduct No filter data in layer " << weight.name().c_str();
    return RET_ERROR;
  }

  // parse filter
  auto filter = ConvertWeight(weight.blobs(0));
  if (filter == nullptr) {
    MS_LOG(ERROR) << "InnerProduct parse weight for layer " << weight.name().c_str() << " failed";
    return RET_ERROR;
  }
  weightVec->push_back(filter);

  // parse bias
  if (innerProductParam.bias_term() && weight.blobs_size() > 1) {
    auto bias = ConvertWeight(weight.blobs(1));
    if (bias == nullptr) {
      MS_LOG(ERROR) << "InnerProduct parse bias for layer " << weight.name().c_str() << " failed";
      return RET_ERROR;
    }
    weightVec->push_back(bias);
  }

  op->name = proto.name();
  op->primitive->value.type = schema::PrimitiveType_FullConnection;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_caffeInnerProductParser("InnerProduct", new CaffeInnerProductParser());
}  // namespace lite
}  // namespace mindspore

