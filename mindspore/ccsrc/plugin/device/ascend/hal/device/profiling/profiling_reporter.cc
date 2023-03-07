/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/profiling/profiling_reporter.h"
#include <map>
#include <algorithm>
#include <fstream>
#include "kernel/kernel.h"
#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"
#include "include/common/utils/utils.h"
#include "include/backend/kernel_graph.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/profiler/ascend_profiling.h"
#include "plugin/device/ascend/hal/profiler/parallel_strategy_profiling.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/profiler/options.h"

namespace mindspore {
namespace device {
namespace ascend {
constexpr uint32_t kProfilingModelStartLogId = 0;
constexpr uint32_t kProfilingModelEndLogId = 1;

// GE task info task_type
enum class TaskInfoTaskType {
  MSPROF_AI_CORE = 0,
  MSPROF_AI_CPU = 1,
  MSPROF_AIV = 2,
  MSPROF_HCCL = 10,
  MSPROF_RTS = 11,
  MSPROF_UNKNOWN_TYPE = 1000,
};

// MS kernel to GE task info task_type
static std::map<KernelType, TaskInfoTaskType> KernelType2TaskTypeEnum{
  {KernelType::TBE_KERNEL, TaskInfoTaskType::MSPROF_AI_CORE},
  {KernelType::AKG_KERNEL, TaskInfoTaskType::MSPROF_AI_CORE},
  {KernelType::AICPU_KERNEL, TaskInfoTaskType::MSPROF_AI_CPU},
  {KernelType::RT_KERNEL, TaskInfoTaskType::MSPROF_RTS},
  {KernelType::HCCL_KERNEL, TaskInfoTaskType::MSPROF_HCCL},
  {KernelType::HOST_KERNEL, TaskInfoTaskType::MSPROF_UNKNOWN_TYPE},
  {KernelType::CPU_KERNEL, TaskInfoTaskType::MSPROF_UNKNOWN_TYPE},
  {KernelType::GPU_KERNEL, TaskInfoTaskType::MSPROF_UNKNOWN_TYPE},
  {KernelType::BISHENG_KERNEL, TaskInfoTaskType::MSPROF_UNKNOWN_TYPE},
  {KernelType::ACL_KERNEL, TaskInfoTaskType::MSPROF_UNKNOWN_TYPE},
  {KernelType::UNKNOWN_KERNEL_TYPE, TaskInfoTaskType::MSPROF_UNKNOWN_TYPE}};

// 0 means unknown format
static std::map<string, uint32_t> OpFormat2Index{{kOpFormat_DEFAULT, 1},
                                                 {kOpFormat_NC1KHKWHWC0, 2},
                                                 {kOpFormat_ND, 3},
                                                 {kOpFormat_NCHW, 4},
                                                 {kOpFormat_NHWC, 5},
                                                 {kOpFormat_HWCN, 6},
                                                 {kOpFormat_NC1HWC0, 7},
                                                 {kOpFormat_FRAC_Z, 8},
                                                 {kOpFormat_C1HWNCoC0, 9},
                                                 {kOpFormat_FRAC_NZ, 10},
                                                 {kOpFormat_NC1HWC0_C04, 11},
                                                 {kOpFormat_FRACTAL_Z_C04, 12},
                                                 {kOpFormat_NDHWC, 13},
                                                 {kOpFormat_FRACTAL_ZN_LSTM, 14},
                                                 {kOpFormat_FRACTAL_ZN_RNN, 15},
                                                 {kOpFormat_ND_RNN_BIAS, 16},
                                                 {kOpFormat_NDC1HWC0, 17},
                                                 {kOpFormat_NCDHW, 18},
                                                 {kOpFormat_FRACTAL_Z_3D, 19},
                                                 {kOpFormat_DHWNC, 20},
                                                 {kOpFormat_DHWCN, 21}};

bool ProfilingReporter::CheckStreamTaskValid() const {
  if (cnode_list_.size() != stream_ids_.size() || cnode_list_.size() != task_ids_.size()) {
    MS_LOG(ERROR) << "CNode size is not equal stream size or not equal task size, "
                     "can not support to report profiling data. CNode size is "
                  << cnode_list_.size() << ", stream size is " << stream_ids_.size() << ", task size is "
                  << task_ids_.size();
    return false;
  }
  return true;
}

void ProfilingReporter::ReportTasks() const {
  MS_LOG(INFO) << "Profiling start to report tasks.";
  if (!CheckStreamTaskValid()) {
    return;
  }
  size_t task_index = 0;
  for (const auto &node : cnode_list_) {
    MS_EXCEPTION_IF_NULL(node);
    KernelType kernel_type = AnfAlgo::GetKernelType(node);
    auto stream_id = stream_ids_[task_index];
    auto task_id = task_ids_[task_index];
    ReportTask(node, stream_id, task_id, kernel_type);
    ReportNode(node, stream_id, task_id, MSPROF_GE_TENSOR_TYPE_INPUT);
    ReportNode(node, stream_id, task_id, MSPROF_GE_TENSOR_TYPE_OUTPUT);

    ++task_index;
  }
  MS_LOG(INFO) << "Profiling report task data finish. cnode_size: " << task_index;
}

// This function only report model start and model end.
void ProfilingReporter::ReportStepPoint(const std::vector<std::shared_ptr<StepPointDesc>> &points) {
  MS_LOG(INFO) << "Profiling start to report model start and end point data.";
  if (!CheckStreamTaskValid()) {
    return;
  }
  ConstructNodeNameIndexMap();
  for (const auto &point : points) {
    if (point->tag() != kProfilingModelStartLogId && point->tag() != kProfilingModelEndLogId) {
      continue;
    }
    auto op_name = point->op_name();
    MsprofGeProfStepData step_point{};
    step_point.modelId = rt_model_id_;
    step_point.streamId = GetStreamId(op_name);
    step_point.taskId = GetTaskId(op_name);
    step_point.timeStamp = 0;
    step_point.curIterNum = 0;
    step_point.threadId = 0;
    step_point.tag = point->tag();
    ReportData(device_id_, reinterpret_cast<unsigned char *>(&step_point), sizeof(step_point), "step_info");

    auto cnode = GetCNode(op_name);
    MS_EXCEPTION_IF_NULL(cnode);
    const auto stream_id = AnfAlgo::GetStreamId(cnode);
    const auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
    MS_EXCEPTION_IF_NULL(stream);
    // The tag of this function should report all tags, it will be saved to ts_track.data.<device_id>.slice_<index>
    // The first step index set to 1, here keep same with ge
    (void)rtProfilerTraceEx(1, rt_model_id_, point->tag(), stream);

    MS_LOG(INFO) << "Report step point, rt_model_id_: " << rt_model_id_ << ", op name: " << point->op_name()
                 << ", stream id: " << GetStreamId(op_name) << ", task id: " << GetTaskId(op_name)
                 << ", tag: " << point->tag();
  }
}

void ProfilingReporter::DynamicNodeReport(const CNodePtr &node, uint32_t stream_id, uint32_t task_id,
                                          const KernelType kernel_type) const {
  ReportTask(node, stream_id, task_id, kernel_type);
  ReportNode(node, stream_id, task_id, MSPROF_GE_TENSOR_TYPE_INPUT);
  ReportNode(node, stream_id, task_id, MSPROF_GE_TENSOR_TYPE_OUTPUT);
  MS_LOG(INFO) << "Profiling report one dynamic node <" << node->fullname_with_scope() << "> data finish.";
}

const CNodePtr ProfilingReporter::GetCNode(const std::string &name) const {
  for (const CNodePtr &cnode : cnode_list_) {
    MS_EXCEPTION_IF_NULL(cnode);
    std::string fullname = cnode->fullname_with_scope();
    if (fullname.find(name) != std::string::npos && fullname.find("Push") != std::string::npos) {
      return cnode;
    }
  }
  return nullptr;
}

uint32_t ProfilingReporter::GetStreamId(const string &node_name) {
  auto index = node_name_index_map_[node_name];
  return stream_ids_[(uint32_t)index];
}

uint32_t ProfilingReporter::GetTaskId(const string &node_name) {
  auto index = node_name_index_map_[node_name];
  return task_ids_[(uint32_t)index];
}

void ProfilingReporter::ReportParallelStrategy() const {
  std::string parallel_data = profiler::ascend::ParallelStrategy::GetInstance()->GetParallelStrategyForReport();
  if (parallel_data.empty()) {
    return;
  }
  MS_LOG(INFO) << "Start to report parallel strategy data to Ascend Profiler.";
  std::string tag_name = "parallel_strategy";
  constexpr int report_max_len = 256;
  while (!parallel_data.empty()) {
    std::string data_str = parallel_data.substr(0, report_max_len);
    ReportData(device_id_, reinterpret_cast<unsigned char *>(data_str.data()), data_str.size(), tag_name);
    parallel_data = parallel_data.substr(data_str.size());
  }
  MS_LOG(INFO) << "Stop to report parallel strategy data to Ascend Profiler.";
}

std::tuple<std::string, std::string> ProfilingReporter::GetTraceDataFilePath() const {
  const std::string dir = mindspore::profiler::ascend::GetOutputPath();
  std::map<std::string, std::string> trace_data_paths = {
    {"device_queue", dir + "/" + "device_queue_profiling_" + common::GetEnv("RANK_ID") + ".txt"},
    {"dataset_iterator", dir + "/" + "dataset_iterator_profiling_" + common::GetEnv("RANK_ID") + ".txt"}};
  for (auto trace_data_path : trace_data_paths) {
    if (TraceDataPathValid(trace_data_path.second)) {
      return {trace_data_path.second, trace_data_path.first};
    }
  }
  return {"", ""};
}

bool ProfilingReporter::TraceDataPathValid(const std::string &path) const {
  int ret = access(path.c_str(), F_OK);
  if (ret == -1) {
    return false;
  }

  ret = access(path.c_str(), R_OK);
  if (ret == -1) {
    MS_LOG(WARNING) << "file: " << path << " is not readable.";
    return false;
  }

  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    MS_LOG(WARNING) << "file: " << path << " is not open.";
    return false;
  }
  std::string str;
  ifs >> str;
  ifs.close();
  if (str.empty()) {
    MS_LOG(WARNING) << "file: " << path << " is empty.";
    return false;
  }
  return true;
}

void ProfilingReporter::ReportMDTraceData() const {
  auto [trace_data_path, tag_name] = GetTraceDataFilePath();
  if (trace_data_path.empty() || tag_name.empty()) {
    return;
  }
  std::ifstream ifs(trace_data_path, std::ios::binary);
  if (!ifs.is_open()) {
    return;
  }
  std::string line_data;
  MS_LOG(INFO) << "Start to report trace data to Ascend Profiler.";
  while (std::getline(ifs, line_data)) {
    line_data += "\n";
    ReportData(device_id_, reinterpret_cast<unsigned char *>(line_data.data()), line_data.size(), tag_name);
  }
  MS_LOG(INFO) << "Stop to report trace data to Ascend Profiler.";
}

void ProfilingReporter::ReportData(uint32_t device_id, unsigned char *data, size_t data_size,
                                   const string &tag_name) const {
  ReporterData report_data{};
  report_data.deviceId = static_cast<int32_t>(device_id);
  report_data.data = data;
  report_data.dataLen = data_size;
  auto ret = memcpy_s(report_data.tag, MSPROF_ENGINE_MAX_TAG_LEN + 1, tag_name.c_str(), tag_name.length());
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Report data failed, tag is " << tag_name << ", ret: " << ret;
  }

  auto report_ret = ProfilingManager::GetInstance().CallMsprofReport(NOT_NULL(&report_data));
  if (report_ret != 0) {
    MS_LOG(EXCEPTION) << "Report data failed, tag is " << tag_name << ", ret: " << report_ret << ".";
  }
}

void ProfilingReporter::ConstructNodeNameIndexMap() {
  if (!node_name_index_map_.empty()) {
    return;
  }

  size_t task_index = 0;
  for (const auto &node : cnode_list_) {
    MS_EXCEPTION_IF_NULL(node);
    (void)node_name_index_map_.insert(pair<string, size_t>(node->fullname_with_scope(), task_index));
    ++task_index;
  }
}

uint32_t ProfilingReporter::GetBlockDim(const CNodePtr &node) {
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  auto ascend_kernel_mod = dynamic_cast<kernel::AscendKernelMod *>(kernel_mod);
  MS_EXCEPTION_IF_NULL(ascend_kernel_mod);
  return ascend_kernel_mod->block_dim();
}

void ProfilingReporter::ReportTask(const CNodePtr &node, const uint32_t stream_id, uint32_t task_id,
                                   KernelType kernel_type) const {
  MsprofGeProfTaskData task_info{};
  task_info.taskType = static_cast<uint32_t>(KernelType2TaskTypeEnum[kernel_type]);
  std::string opType = common::AnfAlgo::GetCNodeName(node);
  SetAlternativeValue(&task_info.opName, MSPROF_MIX_DATA_STRING_LEN, node->fullname_with_scope(), device_id_);
  SetAlternativeValue(&task_info.opType, MSPROF_GE_OP_TYPE_LEN, opType, device_id_);
  // Note: Currently, the profiler supports only static shapes.
  task_info.shapeType = static_cast<uint32_t>(MSPROF_GE_SHAPE_TYPE_STATIC);
  task_info.blockDims = GetBlockDim(node);
  // Note: Currently, all steps are hardcoded to 0.
  task_info.curIterNum = 0;
  task_info.modelId = rt_model_id_;
  task_info.streamId = stream_id;
  task_info.taskId = task_id;
  task_info.timeStamp = 0;
  task_info.threadId = 0;

  ReportData(device_id_, reinterpret_cast<unsigned char *>(&task_info), sizeof(task_info), "task_desc_info");
}

void ProfilingReporter::ReportNode(const CNodePtr &node, uint32_t stream_id, uint32_t task_id,
                                   uint32_t tensor_type) const {
  const std::string tag_name = "tensor_data_info";

  size_t total_size = 0;
  if (tensor_type == MSPROF_GE_TENSOR_TYPE_INPUT) {
    total_size = common::AnfAlgo::GetInputTensorNum(node);
  } else {
    total_size = AnfAlgo::GetOutputTensorNum(node);
  }

  const size_t batch_size = total_size / MSPROF_GE_TENSOR_DATA_NUM;
  for (size_t i = 0U; i < batch_size; i++) {
    MsprofGeProfTensorData tensor_info{};
    BuildProfTensorDataCommon(&tensor_info, stream_id, task_id);
    tensor_info.tensorNum = MSPROF_GE_TENSOR_DATA_NUM;
    for (size_t j = 0U; j < MSPROF_GE_TENSOR_DATA_NUM; j++) {
      size_t cur_index = i * MSPROF_GE_TENSOR_DATA_NUM + j;
      MsprofGeTensorData tensor_data{};
      BuildTensorData(&tensor_data, node, cur_index, tensor_type);
      tensor_info.tensorData[j] = tensor_data;
    }
    ReportData(device_id_, reinterpret_cast<unsigned char *>(&tensor_info), sizeof(tensor_info), tag_name);
  }

  size_t remain_size = total_size % MSPROF_GE_TENSOR_DATA_NUM;
  if (remain_size == 0) {
    return;
  }

  MsprofGeProfTensorData tensor_info{};
  BuildProfTensorDataCommon(&tensor_info, stream_id, task_id);
  tensor_info.tensorNum = remain_size;
  for (size_t i = 0U; i < remain_size; ++i) {
    MsprofGeTensorData tensor_data{};
    size_t cur_index = batch_size * MSPROF_GE_TENSOR_DATA_NUM + i;
    BuildTensorData(&tensor_data, node, cur_index, tensor_type);
    tensor_info.tensorData[i] = tensor_data;
  }
  ReportData(device_id_, reinterpret_cast<unsigned char *>(&tensor_info), sizeof(tensor_info), tag_name);
}

void ProfilingReporter::BuildProfTensorDataCommon(MsprofGeProfTensorData *tensor_info, uint32_t stream_id,
                                                  uint32_t task_id) const {
  MS_EXCEPTION_IF_NULL(tensor_info);
  tensor_info->modelId = rt_model_id_;
  tensor_info->streamId = stream_id;
  tensor_info->taskId = task_id;
  // Note: Currently, all steps are hardcoded to 0.
  tensor_info->curIterNum = 0;
}

void ProfilingReporter::BuildTensorData(MsprofGeTensorData *tensor_data, const CNodePtr &node, size_t index,
                                        uint32_t tensor_type) const {
  MS_EXCEPTION_IF_NULL(tensor_data);
  tensor_data->tensorType = tensor_type;
  std::vector<int64_t> shape;
  string data_format;
  uint32_t vm_data_type;
  if (tensor_type == MSPROF_GE_TENSOR_TYPE_INPUT) {
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(node, index);
    auto input_node = input_node_with_index.first;
    auto input_index = input_node_with_index.second;
    shape = AnfAlgo::GetOutputDeviceShape(input_node, input_index);
    data_format = AnfAlgo::GetOutputFormat(input_node, input_index);
    vm_data_type = static_cast<uint32_t>(AnfAlgo::GetOutputDeviceDataType(input_node, input_index));
  } else {
    shape = AnfAlgo::GetOutputDeviceShape(node, index);
    data_format = AnfAlgo::GetOutputFormat(node, index);
    vm_data_type = static_cast<uint32_t>(AnfAlgo::GetOutputDeviceDataType(node, index));
  }
  tensor_data->dataType = vm_data_type + MSPROF_DIFFERENCE;
  tensor_data->format = OpFormat2Index[data_format] + MSPROF_DIFFERENCE;
  auto shape_size = std::min(static_cast<uint64_t>(MSPROF_GE_TENSOR_DATA_SHAPE_LEN), shape.size());
  (void)std::copy(shape.begin(), shape.begin() + shape_size, tensor_data->shape);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
