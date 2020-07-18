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
#include <unistd.h>
#include <iostream>
#ifdef USE_GLOG
#include <glog/logging.h>
#endif
#include "minddata/dataset/engine/cache/cache_admin_arg.h"

namespace ds = mindspore::dataset;

int main(int argc, char **argv) {
  ds::Status rc;
  ds::CacheAdminArgHandler args;
  std::stringstream arg_stream;

#ifdef USE_GLOG
  FLAGS_log_dir = "/tmp";
  google::InitGoogleLogging(argv[0]);
#endif

  std::string warningMsg;
  warningMsg.reserve(512);
  warningMsg += "WARNING:\n";
  warningMsg += "cache_admin and the cache server that it controls are currently only used for experimental research";
  warningMsg += " purposes at this time.\n";
  warningMsg += "It is not intended for general availability yet as it may not be stable.  Use it at your own risk.\n";

  // A warning message until the code is mature enough.
  std::cerr << warningMsg << std::endl;

  if (argc == 1) {
    args.Help();
    return 0;
  }

  // ingest all the args into a string stream for parsing
  for (int i = 1; i < argc; ++i) {
    arg_stream << " " << std::string(argv[i]);
  }

  // Parse the args
  rc = args.ParseArgStream(&arg_stream);
  if (!rc.IsOk()) {
    std::cerr << rc.ToString() << std::endl;
    return 1;
  }

  // Execute the command
  rc = args.RunCommand();
  if (!rc.IsOk()) {
    std::cerr << rc.ToString() << std::endl;
    return 1;
  }

  return 0;
}
