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
#include "ops/eltwise.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeEltwiseParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Eltwise>();

  if (proto.bottom_size() < 2) {
    MS_LOG(ERROR) << "Eltwise Op " << proto.name() << " need at least 2 inputs,but input size is "
                  << proto.bottom_size();
    return nullptr;
  }

  const caffe::EltwiseParameter &eltwiseParam = proto.eltwise_param();
  if (eltwiseParam.coeff_size() != 0 && eltwiseParam.coeff_size() != proto.bottom_size()) {
    MS_LOG(ERROR) << "Coeff size(" << eltwiseParam.coeff_size()
                  << ") check fail, Eltwise Layer takes one coefficient per bottom blob.";
    return nullptr;
  }

  if (eltwiseParam.operation() == caffe::EltwiseParameter::PROD && eltwiseParam.coeff_size() != 0) {
    MS_LOG(ERROR) << "Eltwise layer only takes coefficients for summation.";
    return nullptr;
  }

  if (eltwiseParam.coeff_size() != 0 &&
      (std::fabs(eltwiseParam.coeff(0) - 1) > 1e-5 || std::fabs(eltwiseParam.coeff(1) - 1) > 1e-5)) {
    MS_LOG(ERROR) << "Eltwise only support coefficient 1 for summation now.";
    return nullptr;
  }

  if (proto.has_eltwise_param() && eltwiseParam.has_operation()) {
    switch (eltwiseParam.operation()) {
      case caffe::EltwiseParameter::PROD:
        prim->set_mode(mindspore::EltwiseMode::PROD);
        break;
      case caffe::EltwiseParameter::SUM:
        prim->set_mode(mindspore::EltwiseMode::SUM);
        break;
      case caffe::EltwiseParameter::MAX:
        prim->set_mode(mindspore::EltwiseMode::MAXIMUM);
        break;
      default:
        MS_LOG(ERROR) << "Eltwise parse params fail, unsupported operation: " << eltwiseParam.operation();
        return nullptr;
    }
  } else {
    prim->set_mode(mindspore::EltwiseMode::SUM);
  }

  return prim.release();
}

CaffeNodeRegistrar g_caffeEltwiseParser("Eltwise", new CaffeEltwiseParser());
}  // namespace lite
}  // namespace mindspore
