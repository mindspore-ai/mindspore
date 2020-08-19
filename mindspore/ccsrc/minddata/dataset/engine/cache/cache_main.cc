/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "minddata/dataset/engine/cache/cache_server.h"
#include <sys/types.h>
#include <unistd.h>
#ifdef USE_GLOG
#include <glog/logging.h>
#endif
#include <cstdlib>

namespace ds = mindspore::dataset;

int main(int argc, char **argv) {
  ds::Status rc;
  ds::CacheServer::Builder builder;

  // This executable is not to be called directly, and should be invoked by cache_admin executable.
  if (argc != 7) {
    rc = ds::Status(ds::StatusCode::kSyntaxError);
    std::cerr << rc.ToString() << std::endl;
    return static_cast<int>(rc.get_code());
  }

  builder.SetRootDirectory(argv[1])
    .SetNumWorkers(strtol(argv[2], nullptr, 10))
    .SetPort(strtol(argv[3], nullptr, 10))
    .SetSharedMemorySizeInGB(strtol(argv[4], nullptr, 10));

#ifdef USE_GLOG
  FLAGS_minloglevel = strtol(argv[5], nullptr, 10);
#endif

  auto daemonize_string = argv[6];
  bool daemonize = strcmp(daemonize_string, "true") == 0 || strcmp(daemonize_string, "TRUE") == 0 ||
                   strcmp(daemonize_string, "t") == 0 || strcmp(daemonize_string, "T") == 0;

  // We always change directory to / on unix rather than using the directory where the cache_server
  // is called. This is a standard procedure for daemonize a process on unix.
  if (chdir("/") == -1) {
    std::string errMsg = "Unable to change directory to /. Errno = " + std::to_string(errno);
    std::cerr << errMsg << std::endl;
    return -1;
  }

  // Simple check of the parameters before we move on.
  rc = builder.SanityCheck();
  if (rc.IsError()) {
    std::cerr << rc.ToString() << std::endl;
    return static_cast<int>(rc.get_code());
  }

#ifdef USE_GLOG
  FLAGS_log_dir = "/tmp";
  google::InitGoogleLogging(argv[0]);
#endif

  if (daemonize) {
    // fork the child process to become the daemon
    pid_t pid = fork();
    // failed to fork
    if (pid < 0) {
      std::string err_msg = "Failed to fork process for cache server: " + std::to_string(errno);
      std::cerr << err_msg << std::endl;
      return errno;
    } else if (pid > 0) {
      // Parent
      std::cerr << "cache server daemon process has been created as process id: " << pid
                << "\nCheck log file for any start up error" << std::endl;
      signal(SIGCHLD, SIG_IGN);  // ignore sig child signal.
      return 0;
    } else {
      // Child process will continue from here if daemonize and parent has already exited.
      // If we are running in the foreground, none of the code in block below will be run.
      pid_t sid;
      umask(0);
      sid = setsid();
      if (sid < 0) {
        MS_LOG(ERROR) << "Failed to setsid(). Errno = " << std::to_string(errno);
        return errno;
      }
      close(0);
      close(1);
      close(2);
    }
  }

  // Dump the summary
  MS_LOG(INFO) << builder << std::endl;
  rc = builder.Build();
  if (rc.IsOk()) {
    ds::CacheServer &cs = ds::CacheServer::GetInstance();
    // Kick off the threads. Loop forever and never return unless error.
    rc = cs.Run();
    if (rc.get_code() == ds::StatusCode::kDuplicateKey) {
      std::string errMsg = "Server is already started";
      MS_LOG(ERROR) << errMsg;
      std::cerr << errMsg << std::endl;
      return 0;
    }
  }
  if (rc.IsError()) {
    MS_LOG(ERROR) << rc.ToString();
    std::cerr << rc.ToString() << std::endl;
    return static_cast<int>(rc.get_code());
  }
  return 0;
}
