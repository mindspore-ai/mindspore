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

#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_KERNEL_BUILD_CLIENT_H_
#define MINDSPORE_CCSRC_BACKEND_SESSION_KERNEL_BUILD_CLIENT_H_

#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <mutex>

#include "include/common/duplex_pipe.h"
#include "include/backend/visible.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
void ReplaceStr(std::string *dest, const std::string &replace, char new_char);

constexpr inline static int kBufferSize = 4096;
constexpr inline static auto kEnv = "python";
// The TAG as prefix of real command from remote.
constexpr inline static auto kTag = "[~]";
BACKEND_EXPORT std::string GetPyExe();
BACKEND_EXPORT std::string GetCmdResult();

class BACKEND_EXPORT KernelBuildClient {
 public:
  // Send Finish request to server
  constexpr inline static auto kFinish = "FINISH";
  constexpr inline static auto kAkgStart = "AKG/START";
  constexpr inline static auto kAkgData = "AKG/DATA";
  constexpr inline static auto kAkgAttr = "AKG/ATTR";
  constexpr inline static auto kAkgWait = "AKG/WAIT";
  // Receive the response from server
  constexpr inline static auto kAck = "ACK";
  constexpr inline static auto kErr = "ERR";
  constexpr inline static auto kTrue = "True";
  constexpr inline static auto kSuccess = "Success";

  // Revert \n, \r, [space].
  constexpr inline static auto kLF = "[LF]";
  constexpr inline static auto kCR = "[CR]";
  constexpr inline static auto kSP = "[SP]";

  virtual std::string GetEnv() = 0;
  virtual std::string GetScript() = 0;

  void Open() {
    if (!init_) {
      // Exception's thrown if open failed
      if (dp_->Open({GetEnv(), GetScript()}, true) != -1) {
        dp_->SetFinalizeCallback(std::make_shared<std::function<void()>>([this]() { Close(); }));
        init_ = true;
      }
    }
  }
  void Close() noexcept {
    if (init_) {
      dp_->Close();
      init_ = false;
    }
  }

  // Send a request and fetch its response
  std::string SendRequest(const std::string &data) {
    std::lock_guard<std::mutex> locker(mutex_);
    Request(data);
    return Response();
  }
  void Request(const std::string &req) {
    if (!init_) {
      MS_LOG(EXCEPTION) << "Try to send request before Open()";
    }
    *dp_ << req;
  }
  std::string Response() {
    if (!init_) {
      MS_LOG(EXCEPTION) << "Try to get response before Open()";
    }
    std::string res;
    *dp_ >> res;
    // Filter out the interference
    if (res.empty()) {
      MS_LOG(EXCEPTION) << "Response is empty";
    }
    auto start = res.find(kTag);
    if (start == std::string::npos) {
      MS_LOG(EXCEPTION) << "Response seems incorrect, res: " << res;
    }
    auto pos = start + std::strlen(kTag);
    if (pos > res.size()) {  // Safe check for codedex
      MS_LOG(EXCEPTION) << "Response seems incorrect, res(" << res.size() << "): {" << res << "}, start: " << start;
    }
    res = res.substr(pos);
    // Revert the line feed and space
    if (res != kSuccess && res != kAck && res != kErr && res != kTrue) {
      ReplaceStr(&res, kLF, '\n');
      ReplaceStr(&res, kSP, ' ');
    }
    return res;
  }

  // Run AKG building.
  bool AkgStart(int process_num, int wait_time);
  bool AkgSendAttr(const std::string &attr);
  bool AkgSendData(const std::vector<std::string> &jsons);
  bool AkgWait();

 protected:
  KernelBuildClient() : init_(false), dp_(std::make_shared<DuplexPipe>()) {}
  virtual ~KernelBuildClient() = default;

 private:
  // Support multi-thread.
  std::mutex mutex_;
  bool init_;
  std::shared_ptr<DuplexPipe> dp_;
};

static std::string GetCmdResult(const std::string &cmd) {
#ifdef _MSC_VER
  FILE *fpipe = _popen(cmd.c_str(), "r");
#else
  FILE *fpipe = popen(cmd.c_str(), "r");
#endif
  if (fpipe == nullptr) {
    MS_LOG(EXCEPTION) << "popen failed, errno: " << errno;
  }
  bool start = false;
  std::string result;
  char buf[kBufferSize];
  while (std::fgets(buf, sizeof(buf), fpipe) != nullptr) {
    auto len = std::strlen(buf);
    if (len == 0 || len >= kBufferSize) {
      // Safe check for codedex
      // Should never reach here
      MS_LOG(EXCEPTION) << "fgets() failed, len: " << len << ", errno: " << errno;
    }
    if (std::strncmp(buf, kTag, std::strlen(kTag)) == 0) {
      start = true;
    }
    // Filter with 'kTAG' and '\n'
    if (start) {
      bool line_end = buf[len - 1] == '\n';
      (void)result.append(buf, line_end ? len - 1 : len);
      if (line_end) {
        break;
      }
    }
  }
#ifdef _MSC_VER
  (void)_pclose(fpipe);
#else
  (void)pclose(fpipe);
#endif
  return result;
}

static std::string GetScriptFilePath(const std::string &cmd_env, const std::string &cmd_script,
                                     const std::string &server_script) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto server_dir = ms_context->get_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR);
  if (!server_dir.empty()) {
    return server_dir + server_script;
  }

  std::string cmd = cmd_env;
  (void)cmd.append(1, ' ').append(cmd_script);
  auto result = GetCmdResult(cmd);
  const std::string py_suffix = ".py";
  if (result.empty() || result.rfind(py_suffix) != (result.length() - py_suffix.length())) {
    MS_LOG(EXCEPTION) << "py file seems incorrect, result: {" << result << "}";
  }
  if (strlen(kTag) > result.size()) {  // Safe check for codedex
    MS_LOG(EXCEPTION) << "result size seems incorrect, result(" << result.size() << "): {" << result << "}";
  }
  result = result.substr(strlen(kTag));
  MS_LOG(DEBUG) << "result: " << result;
  return result;
}

class BACKEND_EXPORT AscendKernelBuildClient : public KernelBuildClient {
 public:
  // Server configure
  constexpr inline static auto kGetPathScript =
    "-c "
    "\""
    "import pkgutil;"
    "path = pkgutil"
    ".get_loader(\\\"mindspore._extends.remote.kernel_build_server_ascend\\\")"  // Server module name
    ".get_filename();"
    "print('[~]' + path)"
    "\"";

  constexpr inline static auto kServerScript = "kernel_build_server_ascend.py";

  // Receive the response from server
  constexpr inline static auto kFailed = "-1";

  // Send server info. query to server
  constexpr inline static auto kFormat = "FORMAT";
  constexpr inline static auto kSupport = "SUPPORT";

  static AscendKernelBuildClient &Instance();

  std::string GetEnv() override { return GetPyExe(); }

  std::string GetScript() override {
    auto env = GetPyExe();
    return GetScriptFilePath(env, kGetPathScript, kServerScript);
  }
  // Run TBE building.
  std::string DispatchToServer(const std::string &job_json_str);

  AscendKernelBuildClient(const AscendKernelBuildClient &) = delete;
  AscendKernelBuildClient &operator=(const AscendKernelBuildClient &) = delete;

  AscendKernelBuildClient(AscendKernelBuildClient &&) = delete;
  AscendKernelBuildClient &operator=(AscendKernelBuildClient &&) = delete;

 protected:
  ~AscendKernelBuildClient() override { Close(); }

 private:
  AscendKernelBuildClient() { Open(); }
};

class BACKEND_EXPORT AkgKernelBuildClient : public KernelBuildClient {
 public:
  // Server configure
  constexpr inline static auto kGetPathScript =
    "-c "
    "\""
    "import pkgutil;"
    "path = pkgutil"
    ".get_loader(\\\"mindspore._extends.remote.kernel_build_server_akg\\\")"  // Server module name
    ".get_filename();"
    "print('[~]' + path)"
    "\"";

  constexpr inline static auto kServerScript = "kernel_build_server_akg.py";

  static AkgKernelBuildClient &Instance();

  std::string GetEnv() override { return GetPyExe(); }

  std::string GetScript() override {
    auto env = GetPyExe();
    return GetScriptFilePath(env, kGetPathScript, kServerScript);
  }

  AkgKernelBuildClient(const AkgKernelBuildClient &) = delete;
  AkgKernelBuildClient &operator=(const AkgKernelBuildClient &) = delete;

  AkgKernelBuildClient(AkgKernelBuildClient &&) = delete;
  AkgKernelBuildClient &operator=(AkgKernelBuildClient &&) = delete;

 protected:
  ~AkgKernelBuildClient() override { Close(); }

 private:
  AkgKernelBuildClient() { Open(); }
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_KERNEL_BUILD_CLIENT_H_
