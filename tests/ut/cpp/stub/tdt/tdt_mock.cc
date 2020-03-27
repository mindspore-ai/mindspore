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

#ifndef TDT_MOCK_H
#define TDT_MOCK_H

#include "tdt/tsd_client.h"

namespace tdt {
StatusFactory* StatusFactory::GetInstance() {
  static StatusFactory instance;
  return &instance;
}

void StatusFactory::RegisterErrorNo(const uint32_t err, const std::string& desc) { return; }

std::string StatusFactory::GetErrDesc(const uint32_t err) { return "Error"; }

std::string StatusFactory::GetErrCodeDesc(uint32_t errCode) { return "Error"; }

StatusFactory::StatusFactory() {}

std::mutex& StatusFactory::GetMutex() { return GetInstance()->rwMutex_; }

TsdClient* TsdClient::GetInstance() {
  static TsdClient instance;
  return &instance;
}

/**
 * @ingroup TsdClient
 * @brief 构造函数
 */
TsdClient::TsdClient() { rankSize_ = 1; }

/**
 * @ingroup TsdClient
 * @brief 析构函数
 */
TsdClient::~TsdClient() = default;

/**
 * @ingroup TsdClient
 * @brief framework发送拉起hccp和computer process的命令
 * @param [in] phyDeviceId : FMK传入物理ID
 * @param [in] phyDeviceId : FMK传入rankSize
 * @return TDT_OK:成功 或者其他错误码
 */
TDT_StatusT TsdClient::Open(const uint32_t deviceId, const uint32_t rankSize) { return TDT_OK; }

/**
 * @ingroup TsdClient
 * @brief 通知TsdClient关闭相关资源
 * @param 无
 * @return TDT_OK:成功 或者其他错误码
 */
TDT_StatusT TsdClient::Close() { return TDT_OK; }

}  // namespace tdt
#endif  // TDT_MOCK_H
