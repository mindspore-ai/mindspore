/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "profiler/device/cpu/cpu_data_saver.h"
#include <fstream>
#include <numeric>
#include "sys/stat.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace profiler {
namespace cpu {
void CpuDataSaver::WriteFile(std::string out_path_dir) {
  if (op_detail_infos_.empty() || op_type_infos_.empty()) {
    MS_LOG(INFO) << "No cpu operation detail infos to write.";
    return;
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device_id_ = std::to_string(device_id);
  op_side_ = "cpu";
  WriteOpDetail(out_path_dir);
  WriteOpType(out_path_dir);
  WriteOpTimestamp(out_path_dir);
}
}  // namespace cpu
}  // namespace profiler
}  // namespace mindspore
