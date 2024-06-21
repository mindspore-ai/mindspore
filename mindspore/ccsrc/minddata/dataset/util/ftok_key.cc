/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/ftok_key.h"

#include <fstream>
#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#endif

#include "minddata/dataset/callback/callback_param.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/core/message_queue.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
std::atomic<uint32_t> inc_id = 0;

Status GetKey(key_t *key) {
  RETURN_UNEXPECTED_IF_NULL(key);
  // ipc file name
  auto pid = getpid();
  auto tid = std::this_thread::get_id();
  std::ostringstream oss;
  oss << tid;
  std::string ipc_key_file =
    "/tmp/dataset_ipc_" + std::to_string(pid) + "_" + oss.str() + "_" + std::to_string(inc_id++);
  MS_LOG(INFO) << "ipc file: " << ipc_key_file;

  // create the ipc file
  std::fstream fs1;
  fs1.open(ipc_key_file, std::fstream::out);
  fs1.close();

  platform::ChangeFileMode(ipc_key_file, S_IRUSR | S_IWUSR);

  // get key by ftok
  *key = ftok(ipc_key_file.c_str(), 0x600);
  if (*key < 0) {
    RETURN_STATUS_UNEXPECTED("ftok a new key error, ipc file: " + ipc_key_file);
  }

  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
