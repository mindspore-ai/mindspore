/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "common/infer_util.h"
#include <iostream>
#include <vector>
#include "mindapi/base/logging.h"
#include "include/errorcode.h"
#include "common/check_base.h"
#include "common/string_util.h"
#include "common/op_attr.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore {
namespace dpico {
int CheckCustomInputOutput(const std::vector<mindspore::MSTensor> *inputs,
                           const std::vector<mindspore::MSTensor> *outputs, const schema::Primitive *primitive) {
  if (inputs == nullptr) {
    MS_LOG(ERROR) << "inputs is nullptr.";
    return RET_ERROR;
  }
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "outputs is nullptr.";
    return RET_ERROR;
  }
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr.";
    return RET_ERROR;
  }
  if (inputs->empty()) {
    MS_LOG(ERROR) << "Inputs size 0.";
    return RET_ERROR;
  }
  if (outputs->empty()) {
    MS_LOG(ERROR) << "Outputs size 0.";
    return RET_ERROR;
  }
  if (primitive->value_type() != schema::PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Primitive type is not PrimitiveType_Custom.";
    return RET_ERROR;
  }
  return RET_OK;
}

int CheckCustomParam(const schema::Custom *param, const std::string &param_name) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "param is nullptr";
    return RET_ERROR;
  }
  if (param->type() == nullptr) {
    MS_LOG(ERROR) << "param->type() is nullptr";
    return RET_ERROR;
  }
  if (param->type()->str() != param_name) {
    MS_LOG(ERROR) << "current custom node should be " << param_name << ", but in fact it's " << param->type()->str();
    return RET_ERROR;
  }
  return RET_OK;
}

int GetOmNetType(const schema::Primitive *primitive, OmNetType *om_net_type) {
  MS_CHECK_TRUE_MSG(primitive != nullptr && om_net_type != nullptr, RET_ERROR, "input params contain nullptr.");
  auto op = primitive->value_as_Custom();
  MS_CHECK_TRUE_MSG(op != nullptr, RET_ERROR, "custom op is nullptr.");
  auto attrs = op->attr();
  MS_CHECK_TRUE_MSG(attrs != nullptr && attrs->size() >= 1, RET_ERROR, "custom op attr is invalid.");
  std::string om_net_type_str;
  for (size_t i = 0; i < attrs->size(); i++) {
    auto attr = attrs->Get(i);
    MS_CHECK_TRUE_MSG(attr != nullptr && attr->name() != nullptr, RET_ERROR, "invalid attr.");
    if (attr->name()->str() != kNetType) {
      continue;
    }
    auto data_info = attr->data();
    MS_CHECK_TRUE_MSG(data_info != nullptr, RET_ERROR, "attr data is nullptr");
    int data_size = static_cast<int>(data_info->size());
    for (int j = 0; j < data_size; j++) {
      om_net_type_str.push_back(static_cast<char>(data_info->Get(j)));
    }
    break;
  }
  if (om_net_type_str.empty()) {
    *om_net_type = OmNetType::kCnn;
    return RET_OK;
  }
  if (!IsValidUnsignedNum(om_net_type_str)) {
    MS_LOG(ERROR) << "net_type attr data is invalid num.";
    return RET_ERROR;
  }
  int om_net_type_val = stoi(om_net_type_str);
  if (om_net_type_val < static_cast<int>(OmNetType::kCnn) ||
      om_net_type_val > static_cast<int>(OmNetType::kRecurrent)) {
    MS_LOG(ERROR) << "net_type val is invalid. " << om_net_type_val;
    return RET_ERROR;
  }
  *om_net_type = static_cast<OmNetType>(om_net_type_val);
  return RET_OK;
}
}  // namespace dpico
}  // namespace mindspore
