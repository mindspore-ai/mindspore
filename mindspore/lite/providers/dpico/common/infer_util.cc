/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "common/log_util.h"
#include "include/errorcode.h"
#include "include/svp_acl_rt.h"
#include "include/svp_acl.h"
#include "include/svp_acl_ext.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore {
namespace lite {
static bool kThreadRunning = true;

Status FetchAttrs(const schema::Primitive &primitive, std::map<std::string, std::string> *attrs) {
  if (attrs == nullptr) {
    MS_LOG(ERROR) << "function input parameter is nullptr.";
    return kLiteError;
  }
  auto param = primitive.value_as_Custom();
  if (lite::CheckCustomParam(param, "DPICO") != RET_OK) {
    MS_LOG(ERROR) << "custom param is invalid.";
    return kLiteError;
  }
  if (param->attr() == nullptr) {
    MS_LOG(ERROR) << "param attr is nullptr.";
    return kLiteError;
  }
  if (param->attr()->size() < 1) {
    MS_LOG(ERROR) << "There are at least 1 attribute of Custom";
    return kLiteError;
  }
  for (size_t i = 0; i < param->attr()->size(); i++) {
    if (param->attr()->Get(i) == nullptr || param->attr()->Get(i)->name() == nullptr) {
      MS_LOG(ERROR) << "param->attr()->Get(i) is nullptr or param->attr()->Get(i)->name() is nullptr";
      return kLiteError;
    }
    auto output_info = param->attr()->Get(i)->data();
    if (output_info == nullptr) {
      MS_LOG(ERROR) << "output_info is nullptr";
      return kLiteError;
    }
    int buf_size = static_cast<int>(output_info->size());
    std::string attr;
    for (int j = 0; j < buf_size; j++) {
      attr.push_back(static_cast<char>(output_info->Get(j)));
    }
    auto attr_name = param->attr()->Get(i)->name()->str();
    attrs->emplace(attr_name, attr);
  }
  return kSuccess;
}

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

void AicpuThread() {
  MS_LOG(INFO) << "create aicpu thread success";
  while (kThreadRunning) {
    svp_acl_error ret = svp_acl_ext_process_aicpu_task(1000);  // 1000 ms
    if (ret != SVP_ACL_SUCCESS && ret != SVP_ACL_ERROR_RT_REPORT_TIMEOUT) {
      MS_LOG(ERROR) << "create aicpu thread failed!";
      break;
    }
  }
  MS_LOG(INFO) << "end to destroy aicpu thread";
}

int DpicoAicpuThreadManager::CreateAicpuThread(uint32_t model_id) {
  uint32_t aicpu_task_num = 0;
  svp_acl_ext_get_mdl_aicpu_task_num(model_id, &aicpu_task_num);
  all_aicpu_task_num_ += aicpu_task_num;
  if (all_aicpu_task_num_ > 0 && !is_aicpu_thread_activity_) {
    g_threadExitFlag_ = true;
    kThreadRunning = g_threadExitFlag_;
    aicpu_thread_ = std::thread(AicpuThread);
    is_aicpu_thread_activity_ = true;
  }
  return RET_OK;
}

int DpicoAicpuThreadManager::DestroyAicpuThread() {
  if (all_aicpu_task_num_ > 0 && is_aicpu_thread_activity_) {
    g_threadExitFlag_ = false;
    kThreadRunning = g_threadExitFlag_;
    if (aicpu_thread_.joinable()) {
      aicpu_thread_.join();
    }
    all_aicpu_task_num_ = 0;
    is_aicpu_thread_activity_ = false;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
