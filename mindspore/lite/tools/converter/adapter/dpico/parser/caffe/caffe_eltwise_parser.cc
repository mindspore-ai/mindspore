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

#include "parser/caffe/caffe_eltwise_parser.h"
#include <cmath>
#include <limits>
#include <memory>
#include "ops/fusion/sub_fusion.h"
#include "ops/eltwise.h"
#include "ops/custom.h"
#include "common/op_attr.h"
#include "common/check_base.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
namespace {
bool IsSubOp(const caffe::EltwiseParameter &eltwiseParam) {
  return eltwiseParam.operation() == caffe::EltwiseParameter::SUM && eltwiseParam.coeff_size() != 0 &&
         (std::fabs(eltwiseParam.coeff(0) - 1) <= std::numeric_limits<float>::epsilon() &&
          std::fabs(eltwiseParam.coeff(1) + 1) <= std::numeric_limits<float>::epsilon());
}
bool IsEltwiseOp(const caffe::EltwiseParameter &eltwiseParam) {
  return eltwiseParam.coeff_size() == 0 ||
         (eltwiseParam.operation() != caffe::EltwiseParameter::PROD &&
          std::fabs(eltwiseParam.coeff(0) - 1) <= std::numeric_limits<float>::epsilon() &&
          std::fabs(eltwiseParam.coeff(1) - 1) <= std::numeric_limits<float>::epsilon());
}
int SetEltwiseMode(const caffe::EltwiseParameter &eltwiseParam, const BaseOperatorPtr &prim) {
  MS_CHECK_TRUE_MSG(prim != nullptr, RET_ERROR, "prim is nullptr.");
  if (eltwiseParam.has_operation()) {
    switch (eltwiseParam.operation()) {
      case caffe::EltwiseParameter::PROD:
        (void)prim->AddAttr(ops::kMode, api::MakeValue(static_cast<int64_t>(mindspore::EltwiseMode::PROD)));
        break;
      case caffe::EltwiseParameter::SUM:
        (void)prim->AddAttr(ops::kMode, api::MakeValue(static_cast<int64_t>(mindspore::EltwiseMode::SUM)));
        break;
      case caffe::EltwiseParameter::MAX:
        (void)prim->AddAttr(ops::kMode, api::MakeValue(static_cast<int64_t>(mindspore::EltwiseMode::MAXIMUM)));
        break;
      default:
        MS_LOG(ERROR) << "Eltwise parse params fail, unsupported operation: " << eltwiseParam.operation();
        return RET_ERROR;
    }
  } else {
    (void)prim->AddAttr(ops::kMode, api::MakeValue(static_cast<int64_t>(mindspore::EltwiseMode::SUM)));
  }
  return RET_OK;
}
BaseOperatorPtr ParseToCustomOp(const caffe::EltwiseParameter &eltwiseParam) {
  auto prim = std::make_shared<ops::Custom>();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is nullptr.");
  prim->set_type("Eltwise");
  if (eltwiseParam.coeff_size() != 0) {
    auto coeff_vals = std::vector<float>(eltwiseParam.coeff().begin(), eltwiseParam.coeff().end());
    (void)prim->AddAttr(dpico::kCoeffs, api::MakeValue(coeff_vals));
  }
  if (SetEltwiseMode(eltwiseParam, prim) != RET_OK) {
    MS_LOG(ERROR) << "set eltwise mode failed.";
    return nullptr;
  }
  return prim;
}
}  // namespace
BaseOperatorPtr CaffeEltwiseParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  const caffe::EltwiseParameter &eltwiseParam = proto.eltwise_param();
  if (eltwiseParam.coeff_size() != 0 && eltwiseParam.coeff_size() != proto.bottom_size()) {
    MS_LOG(ERROR) << "Coeff size(" << eltwiseParam.coeff_size()
                  << ") check fail, Eltwise Layer takes one coefficient per bottom blob.";
    return nullptr;
  }
  if (proto.bottom_size() < kNums2) {
    MS_LOG(ERROR) << "Eltwise Op " << proto.name() << " need at least 2 inputs,but input size is "
                  << proto.bottom_size();
    return nullptr;
  } else if (proto.bottom_size() == kNums2) {
    if (IsSubOp(eltwiseParam)) {
      auto prim = std::make_shared<ops::SubFusion>();
      MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is nullptr.");
      return prim;
    } else if (IsEltwiseOp(eltwiseParam)) {
      auto prim = std::make_shared<ops::Eltwise>();
      MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is nullptr.");
      if (SetEltwiseMode(eltwiseParam, prim) != RET_OK) {
        MS_LOG(ERROR) << "set eltwise mode failed. " << proto.name();
        return nullptr;
      }
      return prim;
    } else {
      return ParseToCustomOp(eltwiseParam);
    }
  } else {
    return ParseToCustomOp(eltwiseParam);
  }
}

CaffeNodeRegistrar g_caffeEltwiseParser("Eltwise", new CaffeEltwiseParser());
}  // namespace lite
}  // namespace mindspore
