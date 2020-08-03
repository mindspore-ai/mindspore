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

#include "backend/session/kernel_build_client.h"

#include <iostream>
#include <memory>

namespace mindspore {
namespace kernel {
void ReplaceStr(std::string *dest, const std::string &replace, char new_char) {
  std::string::size_type start = 0;
  while ((start = (*dest).find(replace, start)) != std::string::npos) {
    (*dest).replace(start, replace.size(), 1, new_char);
    start++;  // Replaced 1 charactor.
  }
}

int KernelBuildClient::Start(const std::string &json) {
  // Start compiling..
  std::string res = SendRequest(kSTART);
  if (res != kACK) {
    MS_LOG(ERROR) << "START failed, res: " << res;
    return -1;
  }
  // Send the json data.
  res = SendRequest(json);
  if (res == kFAILED) {
    MS_LOG(ERROR) << "START send data failed, res: " << res;
    return -1;
  }
  // Return task id.
  return std::stoi(res);
}

bool KernelBuildClient::Wait(int *task_id, std::string *task_result, std::string *pre_build_result) {
  // Start waiting..
  std::string res = SendRequest(kWAIT);
  if (res != kACK) {
    MS_LOG(ERROR) << "WAIT failed, res: " << res;
    return false;
  }
  // Request task id.
  *task_id = std::stoi(SendRequest(kCONT));
  // Requst task result.
  *task_result = SendRequest(kCONT);
  // Request prebuild result.
  *pre_build_result = SendRequest(kCONT);
  return true;
}

void KernelBuildClient::Reset() {
  // Start compiling..
  std::string res = SendRequest(kRESET);
  if (res != kACK) {
    MS_LOG(EXCEPTION) << "RESET response is: " << res;
  }
}

std::string KernelBuildClient::SelectFormat(const std::string &json) {
  // Start compiling..
  std::string res = SendRequest(kFORMAT);
  if (res != kACK) {
    MS_LOG(ERROR) << "FORMAT failed, res: " << res;
    return "";
  }
  // Send the json data.
  res = SendRequest(json);
  if (res == kERR) {
    MS_LOG(ERROR) << "FORMAT send data failed, res: " << res;
    return "";
  }
  return res;
}

bool KernelBuildClient::CheckSupported(const std::string &json) {
  // Checking support..
  std::string res = SendRequest(kSUPPORT);
  if (res != kACK) {
    MS_LOG(ERROR) << "SUPPORT failed, res: " << res;
    return false;
  }
  // Send the json data.
  res = SendRequest(json);
  if (res != kTRUE) {
    MS_LOG(ERROR) << "SUPPORT send data failed, res: " << res;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
