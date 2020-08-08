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

#ifndef MINDSPORE_CCSRC_COMMON_DUPLEX_PIPE_H_
#define MINDSPORE_CCSRC_COMMON_DUPLEX_PIPE_H_

#include <unistd.h>
#include <string>
#include <memory>
#include <initializer_list>
#include <functional>

#include "utils/log_adapter.h"
#define DP_DEBUG MS_LOG(DEBUG) << "[DuplexPipe] "
#define DP_INFO MS_LOG(INFO) << "[DuplexPipe] "
#define DP_ERROR MS_LOG(ERROR) << "[DuplexPipe] "
#define DP_EXCEPTION MS_LOG(EXCEPTION) << "[DuplexPipe] "

namespace mindspore {
// A tool to run a command as child process and build a duplex pipe between them.
// Similar to 'popen()', but use duplex not simplex pipe, more like 'socketpair'.
class DuplexPipe : public std::enable_shared_from_this<mindspore::DuplexPipe> {
 public:
  constexpr inline static int kBufferSize = 4096;
  constexpr inline static unsigned int kTimeOutSeconds = 5;

  DuplexPipe() = default;
  ~DuplexPipe() = default;

  // Create a subprocess and open a duplex pipe between local and remote
  int Open(std::initializer_list<std::string> arg_list, bool append_fds = false);
  void Close();
  void SetTimeOutSeconds(unsigned int secs) { time_out_secs_ = secs; }
  void SetTimeOutCallback(const std::function<void()> &cb) {
    has_time_out_callback_ = true;
    time_out_callback_ = cb;
  }

  // Write the 'buf' to remote stdin
  void Write(const std::string &buf, bool flush = true);
  // Read from remote stdout/stderr into 'c_buf_'
  std::string Read();

  void WriteWithStdout(const std::string &buf, bool flush);
  std::string ReadWithStdin();

  DuplexPipe &operator<<(const std::string &buf);
  DuplexPipe &operator>>(std::string &buf);

 private:
  void SetTimeOut() { alarm_.Set(shared_from_this(), time_out_secs_); }
  void CancelTimeOut() { alarm_.Cancel(); }
  void TimeOut() {
    if (has_time_out_callback_) {
      time_out_callback_();
    }
    Close();
    DP_EXCEPTION << "Time out when read from pipe";
  }

  // Subprocess id in parent process,
  // otherwise zero in child process.
  pid_t pid_;

  // Pipe: { Local:fd1_[1] --> Remote:fd1_[0] }
  // Remote:fd1_[0] would be redirected by subprocess's stdin.
  // Local:fd1_[1] would be used by 'Write()' as output.
  int fd1_[2];

  // Pipe: { Remote:fd2_[1] --> Local:fd2_[0] }
  // Remote:fd2_[1] would be redirected by subprocess's stdout.
  // Local:fd2_[0] would be used by 'Read()' as input.
  int fd2_[2];

  // // Used and returned by 'Read()'.
  // std::string buf_;
  char c_buf_[kBufferSize];

  int local_stdin_;
  int local_stdout_;
  int local_stderr_;
  int remote_stdin_;
  int remote_stdout_;
  int remote_stderr_;

  class Alarm {
   public:
    Alarm() = default;
    ~Alarm() = default;

    void Set(std::shared_ptr<DuplexPipe> dp, unsigned int interval_secs);
    void Cancel();

   private:
    static void SigHandler(int sig) {
      DP_INFO << "Signal: " << sig;
      dp_->TimeOut();
    }

    inline static std::shared_ptr<DuplexPipe> dp_;
  };

  unsigned int time_out_secs_ = kTimeOutSeconds;
  bool has_time_out_callback_ = false;
  std::function<void()> time_out_callback_;
  Alarm alarm_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_COMMON_DUPLEX_PIPE_H_
