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

#ifndef MINDSPORE_CCSRC_C_API_SRC_HELPER_H_
#define MINDSPORE_CCSRC_C_API_SRC_HELPER_H_

#include <memory>
#include "base/base.h"
#include "c_api/src/resource_manager.h"
#include "c_api/include/context.h"
#include "c_api/src/common.h"

Handle GetRawPtr(ResMgrHandle res_mgr, const BasePtr &src_ptr);

template <typename T>
T GetSrcPtr(ResMgrHandle res_mgr, ConstHandle raw_ptr) {
  auto res_mgr_ptr = reinterpret_cast<ResourceManager *>(res_mgr);
  BasePtr base_ptr = res_mgr_ptr->GetSrcPtr(raw_ptr);
  if (base_ptr == nullptr) {
    MS_LOG(ERROR) << "Get source shared pointer failed.";
    return nullptr;
  }
  auto res_ptr = base_ptr->cast<T>();
  return res_ptr;
}
#endif  // MINDSPORE_CCSRC_C_API_SRC_HELPER_H_
