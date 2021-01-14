/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "runtime/hccl_adapter/hccl_adapter.h"
#include <map>
#include <algorithm>
#define google ascend_private
#include "register/ops_kernel_builder_registry.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "external/ge/ge_api_types.h"
#undef google
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "runtime/hccl_adapter/converter.h"
#include "runtime/hccl_adapter/hcom_graph_adaptor.h"

static constexpr const char *kHcclOpsKernelInfoStore = "ops_kernel_info_hccl";
static constexpr const char *kHcclDeployModeEnv = "DEPLOY_MODE";
// following global var, thread safety is not guaranteed
static std::shared_ptr<ge::OpsKernelInfoStore> ops_kernel_info_store = nullptr;
static ge::OpsKernelBuilderPtr ops_kernel_builder = nullptr;

namespace mindspore::hccl {
static std::map<std::string, std::string> GenHcclOptions(uint32_t device_id, std::string_view rank_id,
                                                         std::string_view rank_file) {
  auto env_deploy_mode = common::GetEnv(kHcclDeployModeEnv);
  if (env_deploy_mode.empty()) {
    MS_LOG(WARNING) << kHcclDeployModeEnv << " is not set in ENV. Now set to default value 0";
    env_deploy_mode = "0";
  }

  return std::map<std::string, std::string>({{ge::OPTION_EXEC_IS_USEHCOM, "1"},
                                             {ge::OPTION_EXEC_IS_USEHVD, "0"},
                                             {ge::OPTION_EXEC_HCCL_FLAG, "1"},
                                             {ge::OPTION_EXEC_DEVICE_ID, std::to_string(device_id)},
                                             {ge::OPTION_EXEC_RANK_ID, rank_id.data()},
                                             {ge::OPTION_EXEC_POD_NAME, rank_id.data()},
                                             {ge::OPTION_EXEC_RANK_TABLE_FILE, rank_file.data()},
                                             {ge::OPTION_GRAPH_RUN_MODE, "1"},
                                             {ge::OPTION_EXEC_HCCL_FLAG, "1"},
                                             {ge::OPTION_EXEC_DEPLOY_MODE, env_deploy_mode}});
}

bool InitHccl(uint32_t device_id, std::string_view rank_id, std::string_view rank_file) {
  MS_LOG(INFO) << "Start init hccl adapter.";
  // get ops_kernel_builder
  std::map<std::string, ge::OpsKernelBuilderPtr> all_builders = ge::OpsKernelBuilderRegistry::GetInstance().GetAll();
  if (all_builders.size() != 1) {
    MS_LOG(EXCEPTION) << "Builders size should be 1 (hccl builder), but is " << all_builders.size();
  }

  MS_LOG(INFO) << "Get builder " << all_builders.begin()->first;
  ops_kernel_builder = all_builders.begin()->second;
  MS_EXCEPTION_IF_NULL(ops_kernel_builder);
  // init ops_kernel_builder
  auto options = GenHcclOptions(device_id, rank_id, rank_file);
  auto ret = ops_kernel_builder->Initialize(options);
  if (ret != ge::SUCCESS) {
    MS_LOG(EXCEPTION) << "Init builder failed, ret = " << ret;
  }

  // get ops_kernel_info_store
  ret = ::Initialize(options);
  if (ret != ge::SUCCESS) {
    MS_LOG(EXCEPTION) << "Init plugin so failed, ret = " << ret;
  }

  std::map<std::string, std::shared_ptr<ge::OpsKernelInfoStore>> all_ops_kernel_info_stores;
  ::GetOpsKernelInfoStores(all_ops_kernel_info_stores);
  for (auto &[name, ptr] : all_ops_kernel_info_stores) {
    if (name == kHcclOpsKernelInfoStore) {
      ops_kernel_info_store = ptr;
      break;
    }
  }
  MS_EXCEPTION_IF_NULL(ops_kernel_info_store);
  ret = ops_kernel_info_store->Initialize(options);
  if (ret != ge::SUCCESS) {
    MS_LOG(EXCEPTION) << "Init info store failed, ret = " << ret;
  }
  MS_LOG(INFO) << "Init hccl adapter success.";
  return true;
}

bool FinalizeHccl() {
  MS_LOG(INFO) << "Start destroy hccl adapter.";
  if (ops_kernel_info_store != nullptr) {
    auto ret = ops_kernel_info_store->Finalize();
    if (ret != ge::SUCCESS) {
      MS_LOG(ERROR) << "Destroy info store failed, ret = " << ret;
      return false;
    }
  }

  if (ops_kernel_builder != nullptr) {
    auto ret = ops_kernel_builder->Finalize();
    if (ret != ge::SUCCESS) {
      MS_LOG(ERROR) << "Destroy builder failed, ret = " << ret;
      return false;
    }
  }

  ::Finalize();
  ops_kernel_info_store.reset();
  ops_kernel_builder.reset();
  MS_LOG(INFO) << "Destroy hccl adapter success.";
  return true;
}

bool GenTask(const AnfNodePtr &node, HcclDataType datatype, std::vector<HcclTaskInfo> *task_info_lists) {
  MS_EXCEPTION_IF_NULL(ops_kernel_builder);
  MS_EXCEPTION_IF_NULL(task_info_lists);
  MS_LOG(INFO) << "Start generate task for hccl node " << node->DebugString();
  auto [ge_node, ge_graph] = GenerateStubGeNode(node, datatype);
  MS_EXCEPTION_IF_NULL(ge_node);
  auto op = ge_node->GetOpDesc();
  MS_EXCEPTION_IF_NULL(op);

  MS_LOG(INFO) << "Start to call CalcOpRunningParam";
  ge::Status ret = ops_kernel_builder->CalcOpRunningParam(*ge_node);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "OpsKernelBuilder CalcOpRunningParam failed, ret = " << ret;
    return false;
  }
  MS_LOG(INFO) << "Start to call GenerateTask";
  ge::RunContext unused_ctx;
  std::vector<domi::TaskDef> domi_tasks;
  ret = ops_kernel_builder->GenerateTask(*ge_node, unused_ctx, domi_tasks);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "OpsKernelBuilder GenerateTask failed, ret = " << ret;
    return false;
  }

  task_info_lists->clear();
  std::transform(domi_tasks.begin(), domi_tasks.end(), std::back_inserter(*task_info_lists),
                 [&op](const domi::TaskDef &task_def) -> HcclTaskInfo { return ParseDomiTask(op, task_def); });
  MS_LOG(INFO) << "Generate task for node " << node->DebugString() << " success.";
  ge_graph.reset();
  return true;
}

int64_t CalcWorkspaceSize(const AnfNodePtr &node, HcclDataType datatype) {
  MS_EXCEPTION_IF_NULL(ops_kernel_builder);
  MS_LOG(INFO) << "Start calc workspace size for hccl node " << node->DebugString() << " ,dtype is " << datatype;
  auto [ge_node, ge_graph] = GenerateStubGeNode(node, datatype);
  MS_EXCEPTION_IF_NULL(ge_node);
  auto op = ge_node->GetOpDesc();
  MS_EXCEPTION_IF_NULL(op);

  MS_LOG(INFO) << "Start to call CalcOpRunningParam";
  ge::Status ret = ops_kernel_builder->CalcOpRunningParam(*ge_node);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "OpsKernelBuilder CalcOpRunningParam failed, ret = " << ret;
    return false;
  }

  auto workspace_sizes = op->GetWorkspaceBytes();
  if (workspace_sizes.size() != 1) {
    MS_LOG(EXCEPTION) << "Unexpected workspace size " << workspace_sizes.size();
  }
  int64_t workspace_size = workspace_sizes[0];
  MS_LOG(INFO) << "Node " << node->DebugString() << " workspace size is " << workspace_size;
  ge_graph.reset();
  return workspace_size;
}

void *GetHcclOpsKernelInfoStore() { return ops_kernel_info_store.get(); }

std::string GetHcclType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return GetGeNodeName(cnode);
}
}  // namespace mindspore::hccl
