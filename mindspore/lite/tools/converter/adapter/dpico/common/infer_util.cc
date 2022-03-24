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

#include "common/infer_util.h"
#include <iostream>
#include <vector>
#include "mindapi/base/logging.h"
#include "include/errorcode.h"

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
}  // namespace dpico
}  // namespace mindspore
