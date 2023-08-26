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

#ifndef MINDSPORE_PROFILING_REPORTER_H
#define MINDSPORE_PROFILING_REPORTER_H
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <map>
#include <tuple>

#include "utils/log_adapter.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "toolchain/prof_common.h"
#include "toolchain/prof_reporter.h"
#include "include/backend/kernel_graph.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace device {
namespace ascend {
using std::pair;
using std::string;
using std::vector;

const uint32_t MSPROF_DIFFERENCE = 200;

// GE task info task_type
enum class TaskInfoTaskType {
  TASK_TYPE_AI_CORE = 0,
  TASK_TYPE_AI_CPU = 1,
  TASK_TYPE_AIV = 2,
  TASK_TYPE_WRITE_BACK = 3,
  TASK_TYPE_MIX_AIC = 4,
  TASK_TYPE_MIX_AIV = 5,
  TASK_TYPE_FFTS_PLUS = 6,
  TASK_TYPE_DSA = 7,
  TASK_TYPE_DVPP = 8,
  TASK_TYPE_HCCL = 9,
  MSPROF_RTS = 11,
  MSPROF_UNKNOWN_TYPE = 1000,
};

// MS kernel to GE task info task_type
static std::map<KernelType, TaskInfoTaskType> KernelType2TaskTypeEnum{
  {KernelType::TBE_KERNEL, TaskInfoTaskType::TASK_TYPE_AI_CORE},
  {KernelType::AKG_KERNEL, TaskInfoTaskType::TASK_TYPE_AI_CORE},
  {KernelType::AICPU_KERNEL, TaskInfoTaskType::TASK_TYPE_AI_CPU},
  {KernelType::RT_KERNEL, TaskInfoTaskType::MSPROF_RTS},
  {KernelType::HCCL_KERNEL, TaskInfoTaskType::TASK_TYPE_HCCL},
  {KernelType::HOST_KERNEL, TaskInfoTaskType::MSPROF_UNKNOWN_TYPE},
  {KernelType::CPU_KERNEL, TaskInfoTaskType::MSPROF_UNKNOWN_TYPE},
  {KernelType::GPU_KERNEL, TaskInfoTaskType::MSPROF_UNKNOWN_TYPE},
  {KernelType::BISHENG_KERNEL, TaskInfoTaskType::MSPROF_UNKNOWN_TYPE},
  {KernelType::ACL_KERNEL, TaskInfoTaskType::MSPROF_UNKNOWN_TYPE},
  {KernelType::OPAPI_KERNEL, TaskInfoTaskType::MSPROF_UNKNOWN_TYPE},
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

class StepPointDesc {
 public:
  StepPointDesc(string op_name, uint32_t tag) : op_name_(std::move(op_name)), tag_(tag) {}
  ~StepPointDesc() = default;

  string op_name() const { return op_name_; }
  uint32_t tag() const { return tag_; }

 private:
  string op_name_;
  uint32_t tag_;
};

class ProfilingReporter {
 public:
  ProfilingReporter(int device_id, uint32_t graph_id, uint32_t rt_model_id, vector<CNodePtr> cnode_list,
                    const vector<uint32_t> &stream_ids, const vector<uint32_t> &task_ids)
      : device_id_(device_id),
        graph_id_(graph_id),
        rt_model_id_(rt_model_id),
        cnode_list_(std::move(cnode_list)),
        stream_ids_(stream_ids),
        task_ids_(task_ids) {}
  ~ProfilingReporter() = default;

  void ReportTasks() const;
  void DynamicNodeReport(const CNodePtr &node, uint32_t stream_id, uint32_t task_id,
                         const KernelType kernel_type) const;
  void ReportStepPoint(const vector<std::shared_ptr<StepPointDesc>> &points);
  void ReportParallelStrategy() const;
  void ReportMDTraceData() const;

 private:
  uint32_t device_id_;
  uint32_t graph_id_;
  uint32_t rt_model_id_;
  vector<CNodePtr> cnode_list_;
  vector<uint32_t> stream_ids_;
  vector<uint32_t> task_ids_;
  std::map<string, int> node_name_index_map_;
  const uint32_t DEFAULT_CONTEXT_ID = 4294967295;

  bool CheckStreamTaskValid() const;
  static uint32_t GetBlockDim(const CNodePtr &node);
  void ConstructNodeNameIndexMap();
  uint32_t GetStreamId(const string &node_name);
  uint32_t GetTaskId(const string &node_name);
  const CNodePtr GetCNode(const std::string &name) const;
  std::tuple<std::string, std::string> GetTraceDataFilePath() const;
  bool TraceDataPathValid(const std::string &path) const;

  void ReportData(uint32_t device_id, unsigned char *data, size_t data_size, const std::string &tag_name) const;
  void ReportTask(const CNodePtr &node, uint32_t stream_id, uint32_t task_id, KernelType kernel_type) const;
  void ReportNode(const CNodePtr &node, uint32_t stream_id, uint32_t task_id, uint32_t tensor_type) const;
  void BuildProfTensorDataCommon(MsprofGeProfTensorData *tensor_info, uint32_t stream_id, uint32_t task_id) const;
  void BuildTensorData(MsprofGeTensorData *tensor_data, const CNodePtr &node, size_t index, uint32_t tensor_type) const;

  template <typename T>
  void SetAlternativeValue(T *property, const size_t property_size, const string &value,
                           const int32_t &device_id) const {
    MS_EXCEPTION_IF_NULL(property);
    if (value.size() < property_size) {
      property->type = static_cast<uint8_t>(MSPROF_MIX_DATA_STRING);
      const auto ret = strncpy_s(property->data.dataStr, property_size, value.c_str(), value.size());
      if (ret != EOK) {
        MS_LOG(ERROR) << "[Profiling] strncpy_s value " << value.c_str() << " error!";
        return;
      }
    } else {
      property->type = static_cast<uint8_t>(MSPROF_MIX_DATA_HASH_ID);
      uint64_t hash_id;
      ProfilingManager::GetInstance().QueryHashId(device_id, value, &hash_id);
      property->data.hashId = hash_id;
    }
  }
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_PROFILING_REPORTER_H
