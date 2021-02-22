/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <memory>

namespace mindspore {
namespace kernel {
inline static bool init_flag = false;
void ReplaceStr(std::string *dest, const std::string &replace, char new_char) {
  std::string::size_type start = 0;
  while ((start = (*dest).find(replace, start)) != std::string::npos) {
    (*dest).replace(start, replace.size(), 1, new_char);
    start++;  // Replaced 1 character.
  }
}

bool KernelBuildClient::AkgStart(int process_num, int wait_time) {
  // Start compiling..
  auto res = SendRequest(kAkgStart);
  if (res != kAck) {
    MS_LOG(ERROR) << "AKG/START failed, res: " << res;
    return false;
  }
  std::string process_num_str = std::to_string(process_num);
  res = SendRequest(process_num_str);
  if (res != kAck) {
    MS_LOG(ERROR) << "AKG/START(process_num) responds failed, res: " << res;
    return false;
  }
  std::string wait_time_str = std::to_string(wait_time);
  res = SendRequest(wait_time_str);
  if (res != kAck) {
    MS_LOG(ERROR) << "AKG/START(wait_time) responds failed, res: " << res;
    return false;
  }
  return true;
}

bool KernelBuildClient::AkgSendData(const std::vector<std::string> &jsons) {
  auto res = SendRequest(kAkgData);
  if (res != kAck) {
    MS_LOG(ERROR) << "AKG/DATA failed, res: " << res;
    return false;
  }
  for (auto &json : jsons) {
    res = SendRequest(json);
    if (res != kAck) {
      MS_LOG(ERROR) << "AKG/DATA.. responds failed, res: " << res << ", when sending [" << json << "]";
      return false;
    }
  }
  return true;
}

// Fetch the result of AKG compiling.
bool KernelBuildClient::AkgWait() {
  auto res = SendRequest(kAkgWait);
  if (res != kTrue) {
    MS_LOG(ERROR) << "AKG/WAIT failed, res: " << res;
    return false;
  }
  return true;
}

bool AscendKernelBuildClient::TbePre(const std::string &mode) {
  auto res = SendRequest(kTbePre);
  if (res.find(kSuccess) == res.npos) {
    MS_LOG(EXCEPTION) << "PRE failed, res: " << res;
  }
  MS_LOG(INFO) << "Pre " << res;
  // init env for auto tune
  res = SendRequest(kTbeTune);
  if (res != kAck) {
    MS_LOG(EXCEPTION) << "Send tune single failed, res: " << res;
  }
  res = SendRequest(mode);
  if (res != kSuccess) {
    MS_LOG(EXCEPTION) << "PRE failed, res: " << res;
  }

  return true;
}

int AscendKernelBuildClient::TbeStart(const std::string &json, const std::string &mode) {
  if (!init_flag) {
    if (!TbePre(mode)) {
      MS_LOG(EXCEPTION) << "START failed";
    }
    init_flag = true;
  }
  // Start compiling..
  auto res = SendRequest(kTbeStart);
  if (res != kAck) {
    MS_LOG(ERROR) << "START failed, res: " << res;
    return -1;
  }
  // Send the json data.
  res = SendRequest(json);
  if (res == kFailed) {
    MS_LOG(ERROR) << "TBE/START responds failed, res: " << res;
    return -1;
  }
  // Return task id.
  return std::stoi(res);
}

bool AscendKernelBuildClient::TbeWait(int *task_id, std::string *task_result, std::string *pre_build_result) {
  // Start waiting..
  auto res = SendRequest(kTbeWait);
  if (res != kAck) {
    MS_LOG(ERROR) << "TBE/WAIT failed, res: " << res;
    return false;
  }
  // Request task id.
  *task_id = std::stoi(SendRequest(kContinue));
  // Request task result.
  *task_result = SendRequest(kContinue);
  // Request prebuild result.
  *pre_build_result = SendRequest(kContinue);
  return true;
}

void AscendKernelBuildClient::TbeReset() {
  // Start compiling..
  init_flag = false;
  auto res = SendRequest(kTbeReset);
  if (res != kAck) {
    MS_LOG(EXCEPTION) << "TBE/RESET response is: " << res;
  }
}

std::string AscendKernelBuildClient::SelectFormat(const std::string &json) {
  // Start compiling..
  auto res = SendRequest(kFormat);
  if (res != kAck) {
    MS_LOG(ERROR) << "FORMAT failed, res: " << res;
    return "";
  }
  // Send the json data.
  res = SendRequest(json);
  if (res == kErr) {
    MS_LOG(ERROR) << "FORMAT responds failed, res: " << res;
    return "";
  }
  return res;
}

bool AscendKernelBuildClient::CheckSupported(const std::string &json) {
  // Checking support..
  auto res = SendRequest(kSupport);
  if (res != kAck) {
    MS_LOG(ERROR) << "SUPPORT failed, res: " << res;
    return false;
  }
  // Send the json data.
  res = SendRequest(json);
  if (res != kTrue) {
    MS_LOG(INFO) << "SUPPORT responds failed, res: " << res;
    return false;
  }
  return true;
}

int GpuKernelBuildClient::AkgGetPid() {
  auto res = SendRequest(kAkgPid);
  if (res == kErr) {
    MS_LOG(ERROR) << "AKG/PID failed, res: " << res;
    return -1;
  }
  return std::stoi(res);
}

bool GpuKernelBuildClient::AkgCompileSingle(const std::string json) {
  auto res = SendRequest(kAkgCompileOp);
  if (res != kAck) {
    MS_LOG(ERROR) << "AKG/COMPILE failed, res: " << res;
    return false;
  }
  // Send single json data.
  res = SendRequest(json);
  if (res != kAck) {
    MS_LOG(ERROR) << "AKG/COMPILE responds failed, res: " << res;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
