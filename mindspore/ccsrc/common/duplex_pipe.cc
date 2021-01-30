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

#include "common/duplex_pipe.h"

#include <sys/wait.h>
#include <iostream>
#include <vector>
#include <algorithm>

namespace mindspore {
DuplexPipe::~DuplexPipe() {
  // pid_ < 0 means the child process is invalid or closed, pid_ == 0 means this process is child
  if (pid_ > 0) {
    (void)kill(pid_, SIGKILL);
  }
}

int DuplexPipe::Open(std::initializer_list<std::string> arg_list, bool append_fds) {
  if (pipe(fd1_) == -1) {
    DP_EXCEPTION << "pipe 1 failed, errno: " << errno;
  }
  if (pipe(fd2_) == -1) {
    close(fd1_[0]);
    close(fd1_[1]);
    DP_EXCEPTION << "pipe 2 failed, errno: " << errno;
  }

  pid_ = fork();
  if (pid_ < 0) {
    close(fd1_[0]);
    close(fd1_[1]);
    close(fd2_[0]);
    close(fd2_[1]);
    DP_EXCEPTION << "fork failed, errno: " << errno;
  } else if (pid_ == 0) {  // Remote process
    remote_stdout_ = dup(STDOUT_FILENO);
    remote_stdin_ = dup(STDIN_FILENO);
    close(fd1_[1]);
    close(fd2_[0]);
    if (!append_fds) {
      dup2(fd1_[0], STDIN_FILENO);
      dup2(fd2_[1], STDOUT_FILENO);
    }
    std::vector<const char *> args;
    std::transform(arg_list.begin(), arg_list.end(), std::back_inserter(args),
                   [](const std::string &arg) -> const char * { return arg.c_str(); });
    if (append_fds) {
      std::string fd10 = std::to_string(fd1_[0]).c_str();
      args.emplace_back(fd10.c_str());
      std::string fd21 = std::to_string(fd2_[1]).c_str();
      args.emplace_back(fd21.c_str());
    }
    args.emplace_back(nullptr);
    if (execvp(args[0], const_cast<char *const *>(&args[0])) == -1) {
      DP_EXCEPTION << "execute " << args[0] << " failed, errno: " << errno;
    }
  } else {  // Local process
    DP_INFO << "Local process, id: " << getpid() << ", " << fd2_[0] << "/" << fd1_[1];
    local_stdout_ = dup(STDOUT_FILENO);
    local_stdin_ = dup(STDIN_FILENO);
    close(fd1_[0]);
    close(fd2_[1]);

    signal_handler_ = std::make_shared<SignalHandler>(weak_from_this(), &pid_);
  }
  return 0;
}

void DuplexPipe::Write(const std::string &buf, bool flush) {
  // Write the string into pipe
  if (write(fd1_[1], buf.data(), buf.size()) == -1) {
    DP_ERROR << "write failed, errno: " << errno;
    return;
  }
  if (flush) {
    // Flush into the pipe
    if (write(fd1_[1], "\n", 1) == -1) {
      DP_ERROR << "write failed, errno: " << errno;
      return;
    }
  }
  DP_DEBUG << "<< [" << buf << "]";
}

std::string DuplexPipe::Read() {
  // Read the string from pipe
  std::string buf;
  // Read one line or multiple lines
  while (1) {
    SetTimeOut();
    ssize_t size = read(fd2_[0], c_buf_, kBufferSize);  // MAYBE BLOCKED, Till reading something
    if (size <= 0) {
      break;
    }
    CancelTimeOut();
    bool line_end = c_buf_[size - 1] == '\n';
    buf.append(c_buf_, line_end ? size - 1 : size);  // Copy without the last '\n'
    if (line_end) {
      break;
    }
  }
  DP_DEBUG << ">> [" << buf << "]";
  return buf;
}

void DuplexPipe::WriteWithStdout(const std::string &buf, bool flush) {
  dup2(fd1_[1], STDOUT_FILENO);
  // Write the string into pipe
  std::cout << buf;
  if (flush) {
    // Flush into the pipe
    std::cout << std::endl;
  }
  dup2(local_stdout_, STDOUT_FILENO);
}

std::string DuplexPipe::ReadWithStdin() {
  std::string buf;
  dup2(fd2_[0], STDIN_FILENO);
  // Maybe blocked
  SetTimeOut();
  std::getline(std::cin, buf);  // Not use 'std::cin >>' to include space
  CancelTimeOut();
  dup2(local_stdin_, STDIN_FILENO);
  return buf;
}

DuplexPipe &DuplexPipe::operator<<(const std::string &buf) {
  Write(buf);
  return *this;
}

DuplexPipe &DuplexPipe::operator>>(std::string &buf) {
  buf = Read();
  return *this;
}

void DuplexPipe::Close() {
  close(fd1_[0]);
  close(fd1_[1]);
  close(fd2_[0]);
  close(fd2_[1]);
  pid_ = -1;
}

DuplexPipe::SignalHandler::SignalHandler(const std::weak_ptr<DuplexPipe> &dp, pid_t *pid) {
  dp_ = dp;
  child_pid_ = pid;
  signal(SIGCHLD, SigChildHandler);
  signal(SIGPIPE, SigPipeHandler);
}

DuplexPipe::SignalHandler::~SignalHandler() {}

void DuplexPipe::SignalHandler::SetAlarm(unsigned int interval_secs) {
  signal(SIGALRM, SigAlarmHandler);
  alarm(interval_secs);
}

void DuplexPipe::SignalHandler::CancelAlarm() { alarm(0); }

void DuplexPipe::SignalHandler::SigAlarmHandler(int sig) {
  DP_INFO << "Signal: " << sig << ", child_pid_: " << child_pid_;
  auto shared_dp = dp_.lock();
  if (shared_dp != nullptr) {
    shared_dp->NotifyTimeOut();
  }
  if (child_pid_ != nullptr) {
    *child_pid_ = -1;
  }
}

void DuplexPipe::SignalHandler::SigPipeHandler(int sig) {
  DP_INFO << "Signal: " << sig << ", child_pid_: " << child_pid_;
  auto shared_dp = dp_.lock();
  if (shared_dp != nullptr) {
    shared_dp->NotifyFinalize();
  }
  if (child_pid_ != nullptr) {
    *child_pid_ = -1;
  }
}

void DuplexPipe::SignalHandler::SigChildHandler(int sig) {
  int status;
  if (child_pid_ != nullptr) {
    (void)waitpid(*child_pid_, &status, WNOHANG | WUNTRACED);
    *child_pid_ = -1;
  }
}
}  // namespace mindspore
