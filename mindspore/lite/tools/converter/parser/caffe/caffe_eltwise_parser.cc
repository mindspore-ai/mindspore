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

#include "tools/converter/parser/caffe/caffe_eltwise_parser.h"
#include <cmath>
#include <memory>

const int ELTWISE_MIN_INPUT_SIZE = 2;
const float ELTWISE_SUM_COEFF_EPSILON = 1e-5;

namespace mindspore {
namespace lite {
STATUS CaffeEltwiseParser::Parse(const caffe::LayerParameter &proto,
                                 const caffe::LayerParameter &weight,
                                 schema::CNodeT *op,
                                 std::vector<schema::TensorT *> *weightVec) {
  MS_LOG(DEBUG) << "parse CaffeEltwiseParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::EltwiseT> attr = std::make_unique<schema::EltwiseT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  if (proto.bottom_size() < ELTWISE_MIN_INPUT_SIZE) {
    MS_LOG(ERROR) << "Eltwise Op " << proto.name() << " need at least 2 inputs,but input size is "
                  << proto.bottom_size();
    return RET_ERROR;
  }

  const caffe::EltwiseParameter eltwiseParam = proto.eltwise_param();
  if (eltwiseParam.coeff_size() != 0 && eltwiseParam.coeff_size() != proto.bottom_size()) {
    MS_LOG(ERROR) << "Coeff size(" << eltwiseParam.coeff_size()
                  << ") check fail, Eltwise Layer takes one coefficient per bottom blob.";
    return RET_ERROR;
  }

  if (eltwiseParam.operation() == caffe::EltwiseParameter::PROD && eltwiseParam.coeff_size() != 0) {
    MS_LOG(ERROR) << "Eltwise layer only takes coefficients for summation.";
    return RET_ERROR;
  }

  if (eltwiseParam.coeff_size() != 0 && (fabs(eltwiseParam.coeff(0) - 1) > ELTWISE_SUM_COEFF_EPSILON ||
                                         fabs(eltwiseParam.coeff(1) - 1) > ELTWISE_SUM_COEFF_EPSILON)) {
    MS_LOG(ERROR) << "Eltwise only support coefficient 1 for summation now.";
    return RET_ERROR;
  }

  if (proto.has_eltwise_param() && eltwiseParam.has_operation()) {
    switch (eltwiseParam.operation()) {
      case caffe::EltwiseParameter::PROD:
        attr->mode = schema::EltwiseMode_PROD;
        break;
      case caffe::EltwiseParameter::SUM:
        attr->mode = schema::EltwiseMode_SUM;
        break;
      case caffe::EltwiseParameter::MAX:
        attr->mode = schema::EltwiseMode_MAXIMUM;
        break;
      default:
        MS_LOG(ERROR) << "Eltwise parse params fail, unsupported opration: " << eltwiseParam.operation();
        return RET_ERROR;
    }
  } else {
    attr->mode = schema::EltwiseMode_SUM;
  }

  op->name = proto.name();
  op->primitive->value.type = schema::PrimitiveType_Eltwise;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_caffeEltwiseParser("Eltwise", new CaffeEltwiseParser());
}  // namespace lite
}  // namespace mindspore
