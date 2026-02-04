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

#include "runtime/hardware_abstract/kernel_base/duplex_pipe.h"

namespace mindspore {
DuplexPipe::~DuplexPipe() {}

int DuplexPipe::Open(const std::initializer_list<std::string> &arg_list, bool append_fds) {
  DP_EXCEPTION << "Not support for Windows by now.";
}

void DuplexPipe::Write(const std::string &buf, bool flush) const { DP_EXCEPTION << "Not support for Windows by now."; }

std::string DuplexPipe::Read() { DP_EXCEPTION << "Not support for Windows by now."; }

void DuplexPipe::WriteWithStdout(const std::string &buf, bool flush) {
  DP_EXCEPTION << "Not support for Windows by now.";
}

std::string DuplexPipe::ReadWithStdin() { DP_EXCEPTION << "Not support for Windows by now."; }

const DuplexPipe &DuplexPipe::operator<<(const std::string &buf) const {
  DP_EXCEPTION << "Not support for Windows by now.";
}

DuplexPipe &DuplexPipe::operator>>(std::string &buf) { DP_EXCEPTION << "Not support for Windows by now."; }

void DuplexPipe::Close() noexcept { DP_EXCEPTION << "Not support for Windows by now."; }

void DuplexPipe::SignalHandler::SetAlarm(unsigned int interval_secs) const {
  DP_EXCEPTION << "Not support for Windows by now.";
}

void DuplexPipe::SignalHandler::CancelAlarm() const { DP_EXCEPTION << "Not support for Windows by now."; }
}  // namespace mindspore
