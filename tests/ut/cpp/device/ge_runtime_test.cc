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
#include <memory>
#include "common/common_test.h"
#define private public
#include "runtime/device/ascend/ge_runtime/model_runner.h"
#include "runtime/device/ascend/ge_runtime/runtime_model.h"
#include "runtime/device/ascend/ge_runtime/task/task_factory.h"
#include "runtime/device/ascend/ge_runtime/task/aicpu_task.h"
#include "runtime/device/ascend/ge_runtime/task/event_record_task.h"
#include "runtime/device/ascend/ge_runtime/task/event_wait_task.h"
#include "runtime/device/ascend/ge_runtime/task/hccl_task.h"
#include "runtime/device/ascend/ge_runtime/task/label_goto_task.h"
#include "runtime/device/ascend/ge_runtime/task/label_manager.h"
#include "runtime/device/ascend/ge_runtime/task/label_set_task.h"
#include "runtime/device/ascend/ge_runtime/task/label_switch_task.h"
#include "runtime/device/ascend/ge_runtime/task/memcpy_async_task.h"
#include "runtime/device/ascend/ge_runtime/task/profiler_task.h"
#include "runtime/device/ascend/ge_runtime/task/stream_active_task.h"
#include "runtime/device/ascend/ge_runtime/task/stream_switch_task.h"
#include "runtime/device/ascend/ge_runtime/task/tbe_task.h"
#undef private
#include "common/opskernel/ops_kernel_info_store.h"

using namespace mindspore::ge::model_runner;
using namespace testing;

class MockOpsKernelInfoStore : public ge::OpsKernelInfoStore {
 public:
  ge::Status Initialize(const map<string, string> &) override { return ge::SUCCESS; }
  ge::Status Finalize() override { return ge::SUCCESS; }
  void GetAllOpsKernelInfo(std::map<string, ge::OpInfo> &infos) const override {}
  bool CheckSupported(const ge::OpDescPtr &opDescPtr, std::string &un_supported_reason) const override { return true; }
  ge::Status LoadTask(ge::GETaskInfo &task) override { return ge::SUCCESS; }
};

namespace mindspore {
class TestAscendGeRuntime : public UT::Common {
 public:
  TestAscendGeRuntime() {}

 private:
  void TearDown() override {
    {
      std::lock_guard<std::mutex> lock(HcclTask::model_stream_mapping_mutex_);
      HcclTask::model_stream_mapping_.clear();
    }
  }
};

TEST_F(TestAscendGeRuntime, test_task_create_null_task_info_failed) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1), reinterpret_cast<rtStream_t>(1)},
                             {reinterpret_cast<rtLabel_t>(1)}, {reinterpret_cast<rtEvent_t>(1)});
  ASSERT_TRUE(TaskFactory::GetInstance().Create(model_context, nullptr) == nullptr);
}

TEST_F(TestAscendGeRuntime, test_aicpu_task_create_one_stream_success) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1)}, {reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> aicpu_task_info = std::make_shared<AicpuTaskInfo>(
    "op_name", 0, "so_name", "kernel_name", "node_def", "ext_info", std::vector<void *>{reinterpret_cast<void *>(1)},
    std::vector<void *>{reinterpret_cast<void *>(1)}, true);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, aicpu_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<AicpuTask>(task) != nullptr);
  ASSERT_NO_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_aicpu_task_create_multi_stream_success) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1), reinterpret_cast<rtStream_t>(1)},
                             {reinterpret_cast<rtLabel_t>(1)}, {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> aicpu_task_info = std::make_shared<AicpuTaskInfo>(
    "op_name", 0, "so_name", "kernel_name", "node_def", "", std::vector<void *>{reinterpret_cast<void *>(1)},
    std::vector<void *>{reinterpret_cast<void *>(1)}, true);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, aicpu_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<AicpuTask>(task) != nullptr);
  ASSERT_NO_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_aicpu_task_create_invalid_stream_id_failed) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1), reinterpret_cast<rtStream_t>(1)},
                             {reinterpret_cast<rtLabel_t>(1)}, {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> aicpu_task_info = std::make_shared<AicpuTaskInfo>(
    "op_name", 5, "so_name", "kernel_name", "node_def", "", std::vector<void *>{reinterpret_cast<void *>(1)},
    std::vector<void *>{reinterpret_cast<void *>(1)}, true);
  ASSERT_ANY_THROW(TaskFactory::GetInstance().Create(model_context, aicpu_task_info));
}

TEST_F(TestAscendGeRuntime, test_event_record_task_create_success) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1)}, {reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> event_record_task_info = std::make_shared<EventRecordTaskInfo>("op_name", 0, 0);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, event_record_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<EventRecordTask>(task) != nullptr);
  ASSERT_NO_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_event_record_task_create_invalid_event_id_failed) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1)}, {reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> event_record_task_info = std::make_shared<EventRecordTaskInfo>("op_name", 0, 10);
  ASSERT_ANY_THROW(TaskFactory::GetInstance().Create(model_context, event_record_task_info));
}

TEST_F(TestAscendGeRuntime, test_event_wait_task_create_success) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1)}, {reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> event_record_task_info = std::make_shared<EventWaitTaskInfo>("op_name", 0, 0);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, event_record_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<EventWaitTask>(task) != nullptr);
  ASSERT_NO_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_event_wait_task_create_invalid_event_id_failed) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1)}, {reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> event_record_task_info = std::make_shared<EventWaitTaskInfo>("op_name", 0, 10);
  ASSERT_ANY_THROW(TaskFactory::GetInstance().Create(model_context, event_record_task_info));
}

TEST_F(TestAscendGeRuntime, test_hccl_task_create_success) {
  MockOpsKernelInfoStore ops_kernel_info_store;
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1)}, {reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> hccl_task_info = std::make_shared<HcclTaskInfo>(
    "op_name", 0, "hccl_type", reinterpret_cast<void *>(1), reinterpret_cast<void *>(2), reinterpret_cast<void *>(3), 4,
    5, std::vector<uint8_t>(6, 7), reinterpret_cast<void *>(&ops_kernel_info_store), 9, 10, 11, 12, "group", true);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, hccl_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<HcclTask>(task) != nullptr);
  ASSERT_NO_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_hccl_task_create_stream_reuse_success) {
  const rtModel_t model = reinterpret_cast<rtModel_t>(0x12345678);
  const rtStream_t stream = reinterpret_cast<rtStream_t>(0x87654321);
  constexpr uint32_t stream_id = 0;
  constexpr int64_t task1_stream_num = 3;
  constexpr int64_t task2_stream_num = 5;
  constexpr int64_t task3_stream_num = 4;
  MockOpsKernelInfoStore ops_kernel_info_store;
  ModelContext model_context(0, 0, 0, model, reinterpret_cast<rtStream_t>(2), {stream},
                             {reinterpret_cast<rtLabel_t>(1)}, {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> hccl_task_info_1 = std::make_shared<HcclTaskInfo>(
    "op_name", stream_id, "hccl_type", reinterpret_cast<void *>(1), reinterpret_cast<void *>(2),
    reinterpret_cast<void *>(3), 4, task1_stream_num, std::vector<uint8_t>(6, 7),
    reinterpret_cast<void *>(&ops_kernel_info_store), 9, 10, 11, 12, "group", true);
  std::shared_ptr<TaskInfo> hccl_task_info_2 = std::make_shared<HcclTaskInfo>(
    "op_name", stream_id, "hccl_type", reinterpret_cast<void *>(1), reinterpret_cast<void *>(2),
    reinterpret_cast<void *>(3), 4, task2_stream_num, std::vector<uint8_t>(6, 7),
    reinterpret_cast<void *>(&ops_kernel_info_store), 9, 10, 11, 12, "group", true);
  std::shared_ptr<TaskInfo> hccl_task_info_3 = std::make_shared<HcclTaskInfo>(
    "op_name", stream_id, "hccl_type", reinterpret_cast<void *>(1), reinterpret_cast<void *>(2),
    reinterpret_cast<void *>(3), 4, task3_stream_num, std::vector<uint8_t>(6, 7),
    reinterpret_cast<void *>(&ops_kernel_info_store), 9, 10, 11, 12, "group", true);
  std::shared_ptr<Task> task_1 = TaskFactory::GetInstance().Create(model_context, hccl_task_info_1);
  std::shared_ptr<Task> task_2 = TaskFactory::GetInstance().Create(model_context, hccl_task_info_2);
  std::shared_ptr<Task> task_3 = TaskFactory::GetInstance().Create(model_context, hccl_task_info_3);
  ASSERT_TRUE(std::dynamic_pointer_cast<HcclTask>(task_1) != nullptr);
  ASSERT_TRUE(std::dynamic_pointer_cast<HcclTask>(task_2) != nullptr);
  ASSERT_TRUE(std::dynamic_pointer_cast<HcclTask>(task_3) != nullptr);
  ASSERT_NO_THROW(task_1->Distribute());
  ASSERT_NO_THROW(task_2->Distribute());
  ASSERT_NO_THROW(task_3->Distribute());
  {
    std::lock_guard<std::mutex> lock(HcclTask::model_stream_mapping_mutex_);
    auto model_iter = HcclTask::model_stream_mapping_.find(model);
    ASSERT_NE(model_iter, HcclTask::model_stream_mapping_.end());
    auto stream_iter = model_iter->second.find(stream_id);
    ASSERT_NE(stream_iter, model_iter->second.end());
    const auto &stream_vec = stream_iter->second;
    ASSERT_EQ(stream_vec.size(), std::max(task1_stream_num, std::max(task2_stream_num, task3_stream_num)));
    for (const auto &s : stream_vec) {
      auto shared = s.lock();
      ASSERT_TRUE(shared != nullptr);
    }
  }
}

TEST_F(TestAscendGeRuntime, test_label_goto_task_create_success) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1)}, {reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> label_goto_task_info = std::make_shared<LabelGotoTaskInfo>("op_name", 0, 0);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, label_goto_task_info);
  auto label_goto_task = std::dynamic_pointer_cast<LabelGotoTask>(task);
  ASSERT_TRUE(label_goto_task != nullptr);
  ASSERT_NO_THROW(task->Distribute());
  label_goto_task->index_value_ = new uint8_t[5];
}

TEST_F(TestAscendGeRuntime, test_label_goto_task_create_invalid_label_id_failed) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1)}, {reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> label_goto_task_info = std::make_shared<LabelGotoTaskInfo>("op_name", 0, 1);
  ASSERT_ANY_THROW(TaskFactory::GetInstance().Create(model_context, label_goto_task_info));
}

TEST_F(TestAscendGeRuntime, test_label_goto_task_reuse_success) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1)}, {reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> label_goto_task_info = std::make_shared<LabelGotoTaskInfo>("op_name", 0, 0);
  std::shared_ptr<Task> task1 = TaskFactory::GetInstance().Create(model_context, label_goto_task_info);
  std::shared_ptr<Task> task2 = TaskFactory::GetInstance().Create(model_context, label_goto_task_info);
  auto label_goto_task_1 = std::dynamic_pointer_cast<LabelGotoTask>(task1);
  auto label_goto_task_2 = std::dynamic_pointer_cast<LabelGotoTask>(task2);
  ASSERT_TRUE(label_goto_task_1 != nullptr);
  ASSERT_NO_THROW(task1->Distribute());
  ASSERT_TRUE(label_goto_task_2 != nullptr);
  ASSERT_NO_THROW(task2->Distribute());
  ASSERT_EQ(label_goto_task_1->label_info_, label_goto_task_2->label_info_);
}

TEST_F(TestAscendGeRuntime, test_label_set_task_create_success) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1)}, {reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> label_set_task_info = std::make_shared<LabelSetTaskInfo>("op_name", 0, 0);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, label_set_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<LabelSetTask>(task) != nullptr);
  ASSERT_NO_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_label_set_task_create_invalid_label_id_failed) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1)}, {reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> label_set_task_info = std::make_shared<LabelGotoTaskInfo>("op_name", 0, 1);
  ASSERT_ANY_THROW(TaskFactory::GetInstance().Create(model_context, label_set_task_info));
}

TEST_F(TestAscendGeRuntime, test_label_switch_task_create_success) {
  ModelContext model_context(
    0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2), {reinterpret_cast<rtStream_t>(1)},
    {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)}, {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> label_switch_task_info =
    std::make_shared<LabelSwitchTaskInfo>("op_name", 0, 2, std::vector<uint32_t>{0, 1}, reinterpret_cast<void *>(1));
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, label_switch_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<LabelSwitchTask>(task) != nullptr);
  ASSERT_NO_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_label_switch_task_create_invalid_stream_id_failed) {
  ModelContext model_context(
    0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2), {reinterpret_cast<rtStream_t>(1)},
    {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)}, {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> label_switch_task_info =
    std::make_shared<LabelSwitchTaskInfo>("op_name", 1, 2, std::vector<uint32_t>{0, 1}, reinterpret_cast<void *>(1));
  ASSERT_ANY_THROW(TaskFactory::GetInstance().Create(model_context, label_switch_task_info));
}

TEST_F(TestAscendGeRuntime, test_label_switch_task_create_invalid_label_id_failed) {
  ModelContext model_context(
    0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2), {reinterpret_cast<rtStream_t>(1)},
    {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)}, {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> label_switch_task_info =
    std::make_shared<LabelSwitchTaskInfo>("op_name", 0, 3, std::vector<uint32_t>{0, 1, 2}, reinterpret_cast<void *>(1));
  ASSERT_ANY_THROW(TaskFactory::GetInstance().Create(model_context, label_switch_task_info));
}

TEST_F(TestAscendGeRuntime, test_label_switch_task_reuse_success) {
  ModelContext model_context(
    0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2), {reinterpret_cast<rtStream_t>(1)},
    {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)}, {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> label_switch_task_info =
    std::make_shared<LabelSwitchTaskInfo>("op_name", 0, 2, std::vector<uint32_t>{0, 1}, reinterpret_cast<void *>(1));
  std::shared_ptr<Task> task1 = TaskFactory::GetInstance().Create(model_context, label_switch_task_info);
  std::shared_ptr<Task> task2 = TaskFactory::GetInstance().Create(model_context, label_switch_task_info);
  auto label_switch_task_1 = std::dynamic_pointer_cast<LabelSwitchTask>(task1);
  auto label_switch_task_2 = std::dynamic_pointer_cast<LabelSwitchTask>(task2);
  ASSERT_TRUE(label_switch_task_1 != nullptr);
  ASSERT_TRUE(label_switch_task_2 != nullptr);
  ASSERT_NO_THROW(task1->Distribute());
  ASSERT_NO_THROW(task2->Distribute());
  ASSERT_EQ(label_switch_task_1->label_info_, label_switch_task_2->label_info_);
}

TEST_F(TestAscendGeRuntime, test_memcpy_async_task_create_success) {
  ModelContext model_context(
    0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2), {reinterpret_cast<rtStream_t>(1)},
    {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)}, {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> memcpy_task_info = std::make_shared<MemcpyAsyncTaskInfo>(
    "op_name", 0, reinterpret_cast<void *>(1), 2, reinterpret_cast<void *>(3), 4, 5, true);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, memcpy_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<MemcpyAsyncTask>(task) != nullptr);
  ASSERT_NO_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_memcpy_async_task_create_invalid_stream_id_failed) {
  ModelContext model_context(
    0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2), {reinterpret_cast<rtStream_t>(1)},
    {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)}, {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> memcpy_task_info = std::make_shared<MemcpyAsyncTaskInfo>(
    "op_name", 1, reinterpret_cast<void *>(1), 2, reinterpret_cast<void *>(3), 4, 5, true);
  ASSERT_ANY_THROW(TaskFactory::GetInstance().Create(model_context, memcpy_task_info));
}

TEST_F(TestAscendGeRuntime, test_profiler_task_create_success) {
  ModelContext model_context(
    0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2), {reinterpret_cast<rtStream_t>(1)},
    {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)}, {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> profiler_task_info = std::make_shared<ProfilerTraceTaskInfo>("op_name", 0, 1, true, 2);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, profiler_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<ProfilerTask>(task) != nullptr);
  ASSERT_NO_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_profiler_task_create_invalid_stream_id_failed) {
  ModelContext model_context(
    0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2), {reinterpret_cast<rtStream_t>(1)},
    {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)}, {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> profiler_task_info = std::make_shared<ProfilerTraceTaskInfo>("op_name", 1, 1, true, 2);
  ASSERT_ANY_THROW(TaskFactory::GetInstance().Create(model_context, profiler_task_info));
}

TEST_F(TestAscendGeRuntime, test_stream_active_task_create_success) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1), reinterpret_cast<rtStream_t>(2)},
                             {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> stream_active_task_info = std::make_shared<StreamActiveTaskInfo>("op_name", 0, 1);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, stream_active_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<StreamActiveTask>(task) != nullptr);
  ASSERT_NO_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_stream_active_task_create_invalid_active_stream_id_failed) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1), reinterpret_cast<rtStream_t>(2)},
                             {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> stream_active_task_info = std::make_shared<StreamActiveTaskInfo>("op_name", 0, 2);
  ASSERT_ANY_THROW(TaskFactory::GetInstance().Create(model_context, stream_active_task_info));
}

TEST_F(TestAscendGeRuntime, test_stream_switch_task_create_success) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1), reinterpret_cast<rtStream_t>(2)},
                             {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> stream_switch_task_info = std::make_shared<StreamSwitchTaskInfo>(
    "op_name", 0, 1, reinterpret_cast<void *>(2), reinterpret_cast<void *>(3), 4, 5);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, stream_switch_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<StreamSwitchTask>(task) != nullptr);
  ASSERT_NO_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_stream_switch_task_create_invalid_true_stream_id_failed) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1), reinterpret_cast<rtStream_t>(2)},
                             {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> stream_switch_task_info = std::make_shared<StreamSwitchTaskInfo>(
    "op_name", 0, 2, reinterpret_cast<void *>(2), reinterpret_cast<void *>(3), 4, 5);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, stream_switch_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<StreamSwitchTask>(task) != nullptr);
  ASSERT_ANY_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_stream_switch_task_create_invalid_stream_id_failed) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1), reinterpret_cast<rtStream_t>(2)},
                             {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> stream_switch_task_info = std::make_shared<StreamSwitchTaskInfo>(
    "op_name", 2, 1, reinterpret_cast<void *>(2), reinterpret_cast<void *>(3), 4, 5);
  ASSERT_ANY_THROW(TaskFactory::GetInstance().Create(model_context, stream_switch_task_info));
}

TEST_F(TestAscendGeRuntime, test_tbe_task_create_success) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1), reinterpret_cast<rtStream_t>(2)},
                             {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> tbe_task_info = std::make_shared<TbeTaskInfo>(
    "op_name", 0, "stub_func", 1, std::vector<uint8_t>(100, 2), 100, std::vector<uint8_t>{5, 6},
    reinterpret_cast<void *>(7), 8, std::vector<uint8_t>{9, 10},
    std::vector<void *>{reinterpret_cast<void *>(11), reinterpret_cast<void *>(12)},
    std::vector<void *>{reinterpret_cast<void *>(13), reinterpret_cast<void *>(14)},
    std::vector<void *>{reinterpret_cast<void *>(15), reinterpret_cast<void *>(16)}, true);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, tbe_task_info);
  auto tbe_task = std::dynamic_pointer_cast<TbeTask>(task);
  ASSERT_TRUE(tbe_task != nullptr);
  ASSERT_NO_THROW(task->Distribute());
  tbe_task->args_ = new uint8_t[5];
}

TEST_F(TestAscendGeRuntime, test_tbe_task_create_invalid_stream_id_failed) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1), reinterpret_cast<rtStream_t>(2)},
                             {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> tbe_task_info = std::make_shared<TbeTaskInfo>(
    "op_name", 3, "stub_func", 1, std::vector<uint8_t>(100, 2), 100, std::vector<uint8_t>{5, 6},
    reinterpret_cast<void *>(7), 8, std::vector<uint8_t>{9, 10},
    std::vector<void *>{reinterpret_cast<void *>(11), reinterpret_cast<void *>(12)},
    std::vector<void *>{reinterpret_cast<void *>(13), reinterpret_cast<void *>(14)},
    std::vector<void *>{reinterpret_cast<void *>(15), reinterpret_cast<void *>(16)}, true);
  ASSERT_ANY_THROW(TaskFactory::GetInstance().Create(model_context, tbe_task_info));
}

TEST_F(TestAscendGeRuntime, test_tbe_task_create_empty_stub_func_failed) {
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1), reinterpret_cast<rtStream_t>(2)},
                             {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> tbe_task_info = std::make_shared<TbeTaskInfo>(
    "op_name", 0, "", 1, std::vector<uint8_t>(100, 2), 100, std::vector<uint8_t>{5, 6}, reinterpret_cast<void *>(7), 8,
    std::vector<uint8_t>{9, 10}, std::vector<void *>{reinterpret_cast<void *>(11), reinterpret_cast<void *>(12)},
    std::vector<void *>{reinterpret_cast<void *>(13), reinterpret_cast<void *>(14)},
    std::vector<void *>{reinterpret_cast<void *>(15), reinterpret_cast<void *>(16)}, true);
  std::shared_ptr<Task> task = TaskFactory::GetInstance().Create(model_context, tbe_task_info);
  ASSERT_TRUE(std::dynamic_pointer_cast<TbeTask>(task) != nullptr);
  ASSERT_ANY_THROW(task->Distribute());
}

TEST_F(TestAscendGeRuntime, test_model_runner_success) {
  constexpr uint32_t model_id = 0;
  ModelContext model_context(0, 0, 0, reinterpret_cast<rtModel_t>(1), reinterpret_cast<rtStream_t>(2),
                             {reinterpret_cast<rtStream_t>(1), reinterpret_cast<rtStream_t>(2)},
                             {reinterpret_cast<rtLabel_t>(1), reinterpret_cast<rtLabel_t>(1)},
                             {reinterpret_cast<rtEvent_t>(1)});
  std::shared_ptr<TaskInfo> tbe_task_info = std::make_shared<TbeTaskInfo>(
    "op_name", 0, "stub_func", 1, std::vector<uint8_t>(100, 2), 100, std::vector<uint8_t>{5, 6},
    reinterpret_cast<void *>(7), 8, std::vector<uint8_t>{9, 10},
    std::vector<void *>{reinterpret_cast<void *>(11), reinterpret_cast<void *>(12)},
    std::vector<void *>{reinterpret_cast<void *>(13), reinterpret_cast<void *>(14)},
    std::vector<void *>{reinterpret_cast<void *>(15), reinterpret_cast<void *>(16)}, true);
  std::shared_ptr<TaskInfo> aicpu_task_info = std::make_shared<AicpuTaskInfo>(
    "op_name", 0, "so_name", "kernel_name", "node_def", "ext_info", std::vector<void *>{reinterpret_cast<void *>(1)},
    std::vector<void *>{reinterpret_cast<void *>(1)}, true);
  auto davice_model =
    std::make_shared<DavinciModel>(std::vector<std::shared_ptr<TaskInfo>>{tbe_task_info, aicpu_task_info},
                                   std::vector<uint32_t>{}, std::vector<uint32_t>{}, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0);
  ASSERT_NO_THROW(ModelRunner::Instance().LoadDavinciModel(0, 0, model_id, davice_model));
  auto iter = ModelRunner::Instance().runtime_models_.find(model_id);
  ASSERT_TRUE(iter != ModelRunner::Instance().runtime_models_.end());
  auto &task_list = iter->second->task_list_;
  task_list.clear();
  ASSERT_NO_THROW(task_list.emplace_back(TaskFactory::GetInstance().Create(model_context, tbe_task_info)));
  ASSERT_NO_THROW(task_list.emplace_back(TaskFactory::GetInstance().Create(model_context, aicpu_task_info)));
  ASSERT_NO_THROW(ModelRunner::Instance().DistributeTask(model_id));
  ASSERT_NO_THROW(ModelRunner::Instance().LoadModelComplete(model_id));
  ASSERT_NO_THROW(ModelRunner::Instance().RunModel(model_id));
  ASSERT_FALSE(ModelRunner::Instance().GetTaskIdList(model_id).empty());
  ASSERT_FALSE(ModelRunner::Instance().GetStreamIdList(model_id).empty());
  ASSERT_FALSE(ModelRunner::Instance().GetRuntimeInfoMap(model_id).empty());
  ASSERT_NO_THROW(ModelRunner::Instance().GetModelHandle(model_id));
  ASSERT_NO_THROW(ModelRunner::Instance().UnloadModel(model_id));
}
}  // namespace mindspore
