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

#include "checker/pow_checker.h"
#include <vector>
#include <limits>
#include <cmath>
#include "common/fetch_content.h"
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
bool PowFusionChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, kInputIndex1, format, kMaxInputWOf4Dims)) {
    MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }
  float power;
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  DataInfo data_info;
  if (op->inputs().size() > kInputIndex2 && FetchDataFromParameterNode(op, kInputIndex2, &data_info) == lite::RET_OK) {
    if (data_info.data_type_ != static_cast<int>(kNumberTypeFloat32)) {
      MS_LOG(ERROR) << "data_type not correct";
      return false;
    }
    if (data_info.data_.data() == nullptr) {
      MS_LOG(ERROR) << "data is nullptr. " << op->fullname_with_scope();
      return false;
    }
    power = *(reinterpret_cast<float *>(data_info.data_.data()));
  } else if (primitive->GetAttr(ops::kPower) != nullptr) {
    power = api::GetValue<float>(primitive->GetAttr(ops::kPower));
  } else {
    MS_LOG(ERROR) << "null param";
    return false;
  }
  if (!(std::fmod(std::fabs(power), 1.0) >
          std::numeric_limits<float>::epsilon() &&  // support power: -0.5, 0.5, integers
        std::fabs(std::fabs(power) - 0.5) > std::numeric_limits<float>::epsilon())) {
    return true;
  } else {
    MS_LOG(WARNING) << "power val only supports -0.5, 0.5, integers " << op->fullname_with_scope();
    return false;
  }
}

OpCheckerRegistrar g_PowFusionChecker("PowFusion", new PowFusionChecker());
}  // namespace dpico
}  // namespace mindspore
