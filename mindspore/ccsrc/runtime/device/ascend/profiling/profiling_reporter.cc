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
#include <map>
#include <algorithm>
#include "runtime/device/ascend/profiling/profiling_reporter.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/ascend_kernel_mod.h"
#include "utils/utils.h"

namespace mindspore {
namespace device {
namespace ascend {
static std::map<enum KernelType, MsprofGeTaskType> KernelType2TaskTypeEnum{{TBE_KERNEL, MSPROF_GE_TASK_TYPE_AI_CORE},
                                                                           {AKG_KERNEL, MSPROF_GE_TASK_TYPE_AI_CORE},
                                                                           {AICPU_KERNEL, MSPROF_GE_TASK_TYPE_AI_CPU}};

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

bool ProfilingReporter::CheckStreamTaskValid() {
  if (cnode_list_.size() != stream_ids_.size() || cnode_list_.size() != task_ids_.size()) {
    MS_LOG(ERROR) << "CNode size is not equal stream size or not equal task size, "
                     "can not support to report profiling data. CNode size is "
                  << cnode_list_.size() << ", stream size is " << stream_ids_.size() << ", task size is "
                  << task_ids_.size();
    return false;
  }
  return true;
}

void ProfilingReporter::ReportTasks() {
  MS_LOG(INFO) << "Profiling start to report tasks.";
  if (!CheckStreamTaskValid()) {
    return;
  }
  size_t task_index = 0;
  for (const auto &node : cnode_list_) {
    MS_EXCEPTION_IF_NULL(node);
    KernelType kernel_type = AnfAlgo::GetKernelType(node);
    // Note: some kernel stream id or task id will conflict, such as RT_KERNEL,
    // and CPU_KERNEL does not have stream id, task id.
    if (kernel_type != TBE_KERNEL && kernel_type != AKG_KERNEL && kernel_type != AICPU_KERNEL &&
        kernel_type != HCCL_KERNEL) {
      MS_LOG(INFO) << "This node is not TBE_KERNEL, AKG_KERNEL, AICPU_KERNEL, HCCL_KERNEL, will skip, node name:"
                   << node->fullname_with_scope();
      ++task_index;
      continue;
    }
    auto stream_id = stream_ids_[task_index];
    auto task_id = task_ids_[task_index];
    (void)ReportTask(node, stream_id, task_id, kernel_type);
    (void)ReportNode(node, stream_id, task_id, MSPROF_GE_TENSOR_TYPE_INPUT);
    (void)ReportNode(node, stream_id, task_id, MSPROF_GE_TENSOR_TYPE_OUTPUT);

    ++task_index;
  }
  MS_LOG(INFO) << "Profiling report task data finish.";
}

void ProfilingReporter::ReportStepPoint(const std::vector<std::shared_ptr<StepPointDesc>> &points) {
  MS_LOG(INFO) << "Profiling start to report step point data.";
  if (!CheckStreamTaskValid()) {
    return;
  }

  ConstructNodeNameIndexMap();
  for (const auto &point : points) {
    MsprofGeProfStepData step_point{};
    step_point.modelId = graph_id_;
    auto op_name = point->op_name();
    step_point.streamId = GetStreamId(op_name);
    step_point.taskId = GetTaskId(op_name);
    step_point.timeStamp = 0;
    step_point.curIterNum = 0;
    step_point.threadId = 0;
    step_point.tag = point->tag();
    (void)ReportData(device_id_, reinterpret_cast<unsigned char *>(&step_point), sizeof(step_point), "step_info");
  }
}

uint32_t ProfilingReporter::GetStreamId(const string &node_name) {
  auto index = node_name_index_map_[node_name];
  return stream_ids_[index];
}

uint32_t ProfilingReporter::GetTaskId(const string &node_name) {
  auto index = node_name_index_map_[node_name];
  return task_ids_[index];
}

void ProfilingReporter::ReportData(int32_t device_id, unsigned char *data, size_t data_size, const string &tag_name) {
  ReporterData report_data{};
  report_data.deviceId = device_id;
  report_data.data = data;
  report_data.dataLen = data_size;
  auto ret = memcpy_s(report_data.tag, MSPROF_ENGINE_MAX_TAG_LEN + 1, tag_name.c_str(), tag_name.length());
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Report data failed, tag is " << tag_name.c_str() << ", ret: " << ret;
  }

  auto report_ret = ProfilingManager::GetInstance().CallMsprofReport(NOT_NULL(&report_data));
  if (report_ret != 0) {
    MS_LOG(EXCEPTION) << "Report data failed, tag is " << tag_name.c_str() << ", ret: " << ret;
  }
}

void ProfilingReporter::ConstructNodeNameIndexMap() {
  if (!node_name_index_map_.empty()) {
    return;
  }

  size_t task_index = 0;
  for (const auto &node : cnode_list_) {
    MS_EXCEPTION_IF_NULL(node);
    node_name_index_map_.insert(pair<string, uint32_t>(node->fullname_with_scope(), task_index));
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
                                   KernelType kernel_type) {
  MsprofGeProfTaskData task_info{};
  task_info.taskType = static_cast<uint32_t>(KernelType2TaskTypeEnum[kernel_type]);
  (void)SetAlternativeValue(task_info.opName, MSPROF_MIX_DATA_STRING_LEN, node->fullname_with_scope(), device_id_);
  (void)SetAlternativeValue(task_info.opType, MSPROF_GE_OP_TYPE_LEN, AnfAlgo::GetCNodeName(node), device_id_);
  // Note: Currently, the profiler supports only static shapes.
  task_info.shapeType = static_cast<uint32_t>(MSPROF_GE_SHAPE_TYPE_STATIC);
  task_info.blockDims = GetBlockDim(node);
  // Note: Currently, all steps are hardcoded to 0.
  task_info.curIterNum = 0;
  task_info.modelId = graph_id_;
  task_info.streamId = stream_id;
  task_info.taskId = task_id;
  task_info.timeStamp = 0;
  task_info.threadId = 0;

  (void)ReportData(device_id_, reinterpret_cast<unsigned char *>(&task_info), sizeof(task_info), "task_desc_info");
}

void ProfilingReporter::ReportNode(const CNodePtr &node, uint32_t stream_id, uint32_t task_id, uint32_t tensor_type) {
  const std::string tag_name = "tensor_data_info";

  size_t total_size = 0;
  if (tensor_type == MSPROF_GE_TENSOR_TYPE_INPUT) {
    total_size = AnfAlgo::GetInputTensorNum(node);
  } else {
    total_size = AnfAlgo::GetOutputTensorNum(node);
  }

  const size_t batch_size = total_size / MSPROF_GE_TENSOR_DATA_NUM;
  for (size_t i = 0U; i < batch_size; i++) {
    MsprofGeProfTensorData tensor_info{};
    (void)BuildProfTensorDataCommon(tensor_info, stream_id, task_id);
    tensor_info.tensorNum = MSPROF_GE_TENSOR_DATA_NUM;
    for (size_t j = 0U; j < MSPROF_GE_TENSOR_DATA_NUM; j++) {
      size_t cur_index = i * MSPROF_GE_TENSOR_DATA_NUM + j;
      MsprofGeTensorData tensor_data{};
      (void)BuildTensorData(tensor_data, node, cur_index, tensor_type);
      tensor_info.tensorData[j] = tensor_data;
    }
    (void)ReportData(device_id_, reinterpret_cast<unsigned char *>(&tensor_info), sizeof(tensor_info), tag_name);
  }

  size_t remain_size = total_size % MSPROF_GE_TENSOR_DATA_NUM;
  if (remain_size == 0) {
    return;
  }

  MsprofGeProfTensorData tensor_info{};
  (void)BuildProfTensorDataCommon(tensor_info, stream_id, task_id);
  tensor_info.tensorNum = remain_size;
  for (size_t i = 0U; i < remain_size; ++i) {
    MsprofGeTensorData tensor_data{};
    size_t cur_index = batch_size * MSPROF_GE_TENSOR_DATA_NUM + i;
    (void)BuildTensorData(tensor_data, node, cur_index, tensor_type);
    tensor_info.tensorData[i] = tensor_data;
  }
  (void)ReportData(device_id_, reinterpret_cast<unsigned char *>(&tensor_info), sizeof(tensor_info), tag_name);
}

void ProfilingReporter::BuildProfTensorDataCommon(MsprofGeProfTensorData &tensor_info, uint32_t stream_id,
                                                  uint32_t task_id) {
  tensor_info.modelId = graph_id_;
  tensor_info.streamId = stream_id;
  tensor_info.taskId = task_id;
  // Note: Currently, all steps are hardcoded to 0.
  tensor_info.curIterNum = 0;
}

void ProfilingReporter::BuildTensorData(MsprofGeTensorData &tensor_data, const CNodePtr &node, size_t index,
                                        uint32_t tensor_type) {
  tensor_data.tensorType = tensor_type;
  std::vector<size_t> shape;
  string data_format;
  if (tensor_type == MSPROF_GE_TENSOR_TYPE_INPUT) {
    auto input_node_with_index = AnfAlgo::GetPrevNodeOutput(node, index);
    auto input_node = input_node_with_index.first;
    auto input_index = input_node_with_index.second;
    shape = AnfAlgo::GetOutputDeviceShape(input_node, input_index);
    data_format = AnfAlgo::GetOutputFormat(input_node, input_index);
    tensor_data.dataType = static_cast<uint32_t>(AnfAlgo::GetOutputDeviceDataType(input_node, input_index));
  } else {
    shape = AnfAlgo::GetOutputDeviceShape(node, index);
    data_format = AnfAlgo::GetOutputFormat(node, index);
    tensor_data.dataType = static_cast<uint32_t>(AnfAlgo::GetOutputDeviceDataType(node, index));
  }

  tensor_data.format = OpFormat2Index[data_format];
  auto shape_size = std::min(static_cast<uint64_t>(MSPROF_GE_TENSOR_DATA_SHAPE_LEN), shape.size());
  std::copy(shape.begin(), shape.begin() + shape_size, tensor_data.shape);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
