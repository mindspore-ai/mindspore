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

#include "minddata/dataset/engine/cache/perf/cache_pipeline_run.h"
#include <string.h>
#include "mindspore/core/utils/log_adapter.h"

namespace ms = mindspore;
namespace ds = mindspore::dataset;

int main(int argc, char **argv) {
#ifdef USE_GLOG
#define google mindspore_private
  FLAGS_logtostderr = false;
  FLAGS_log_dir = "/tmp";
  google::InitGoogleLogging(argv[0]);
#undef google
#endif
  ds::CachePipelineRun cachePipelineRun;
  if (cachePipelineRun.ProcessArgs(argc, argv) == 0) {
    ms::Status rc = cachePipelineRun.Run();
    // If we hit any error, send the rc back to the parent.
    if (rc.IsError()) {
      ds::ErrorMsg proto;
      proto.set_rc(static_cast<int32_t>(rc.StatusCode()));
      proto.set_msg(rc.ToString());
      ds::CachePerfMsg msg;
      (void)cachePipelineRun.SendMessage(&msg, ds::CachePerfMsg::MessageType::kError, &proto);
    }
    return static_cast<int>(rc.StatusCode());
  }
  return 0;
}
