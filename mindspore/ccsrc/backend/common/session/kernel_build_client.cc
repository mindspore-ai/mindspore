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

#include "backend/common/session/kernel_build_client.h"
#include <memory>

namespace mindspore {
namespace kernel {
void ReplaceStr(std::string *dest, const std::string &replace, char new_char) {
  std::string::size_type start = 0;
  while ((start = (*dest).find(replace, start)) != std::string::npos) {
    (*dest).replace(start, replace.size(), 1, new_char);
    start++;  // Replaced 1 character.
  }
}

std::string GetPyExe() {
  // get real python executable path
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    return kEnv;
  }
  auto env = ms_context->get_param<std::string>(MS_CTX_PYTHON_EXE_PATH);
  if (env.empty()) {
    return kEnv;
  }
  return env;
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

bool KernelBuildClient::AkgSendAttr(const std::string &attr) {
  auto res = SendRequest(kAkgAttr);
  if (res != kAck) {
    MS_LOG(ERROR) << "AKG/ATTR failed, res: " << res;
    return false;
  }
  res = SendRequest(attr);
  if (res != kAck) {
    MS_LOG(ERROR) << "AKG/ATTR.. responds failed, res: " << res << ", when sending [" << attr << "]";
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

std::string AscendKernelBuildClient::DispatchToServer(const std::string &job_json_str) {
  auto res = SendRequest(job_json_str);
  if (res == kFailed) {
    MS_LOG(ERROR) << "Send TBE job json failed, res: " << res;
    return "";
  }
  return res;
}

AscendKernelBuildClient &AscendKernelBuildClient::Instance() {
  static AscendKernelBuildClient instance{};
  return instance;
}

AkgKernelBuildClient &AkgKernelBuildClient::Instance() {
  static AkgKernelBuildClient instance{};
  return instance;
}
}  // namespace kernel
}  // namespace mindspore
