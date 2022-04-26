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
std::shared_ptr<CpuDataSaver> CpuDataSaver::cpu_data_saver_inst_ = std::make_shared<CpuDataSaver>();

void CpuDataSaver::WriteFile(const std::string out_path_dir) {
  if (op_detail_infos_.empty() || op_type_infos_.empty()) {
    MS_LOG(INFO) << "No cpu operation detail infos to write.";
    return;
  }
#if ENABLE_GPU
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device_id_ = std::to_string(device_id);
#else
  auto rank_id = common::GetEnv("RANK_ID");
  // If RANK_ID is not set, default value is 0.
  if (rank_id.empty()) {
    rank_id = "0";
  }
  rank_id = std::string(rank_id);
  // When the value of RANK_ID is not a number, set its value to 0.
  for (int i = 0; i < static_cast<int>(rank_id.size()); i++) {
    if (std::isdigit(rank_id[i]) == 0) {
      rank_id = "0";
      break;
    }
  }
  device_id_ = rank_id;
#endif
  op_side_ = "cpu";
  WriteOpDetail(out_path_dir);
  WriteOpType(out_path_dir);
  WriteOpTimestamp(out_path_dir);
}

void CpuDataSaver::WriteFrameWork(const std::string &base_dir, const std::vector<CurKernelInfo> &all_kernel_info) {
  std::string file_path = base_dir + "/cpu_framework_" + device_id_ + ".txt";
  std::ofstream ofs(file_path);
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << file_path << "' failed!";
    return;
  }
  for (auto kernel_info : all_kernel_info) {
    auto op_name = kernel_info.op_name;
    auto op_type = kernel_info.op_type;
    auto cur_kernel_all_inputs_info = kernel_info.cur_kernel_all_inputs_info;
    try {
      ofs << op_type << ";" << op_name << ";";
      for (auto cur_kernel_input_info : cur_kernel_all_inputs_info) {
        ofs << "input_" << cur_kernel_input_info.input_id << ":" << cur_kernel_input_info.shape << ";";
      }
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Write " << file_path << "failed:" << e.what();
      ofs.close();
      return;
    }
    ofs << std::endl;
  }
  ofs.close();
  ChangeFileMode(file_path);
  MS_LOG(INFO) << "Write framework infos into file: " << file_path;
}
OpTimestampInfo &CpuDataSaver::GetOpTimeStampInfo() { return op_timestamps_map_; }

std::shared_ptr<CpuDataSaver> &CpuDataSaver::GetInstance() { return cpu_data_saver_inst_; }
}  // namespace cpu
}  // namespace profiler
}  // namespace mindspore
