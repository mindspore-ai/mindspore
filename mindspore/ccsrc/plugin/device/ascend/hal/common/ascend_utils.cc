/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include <vector>
#include <string>
#include "common/util/error_manager/error_manager.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "utils/ms_context.h"
#include "runtime/dev.h"

namespace mindspore {
namespace device {
namespace ascend {
constexpr auto kUnknowErrorString = "Unknown error occurred";
constexpr auto kSOC_VERSION = "SOC_VERSION";

std::string GetErrorMessage(bool add_title) {
  const string &error_message = ErrorManager::GetInstance().GetErrorMessage();
  if (!error_message.empty() && error_message.find(kUnknowErrorString) == string::npos) {
    return add_title ? "#umsg#Ascend Error Message:#umsg#" + error_message : error_message;
  }
  return "";
}

void SetErrorManagerContext() { ErrorManager::GetInstance().GenWorkStreamIdDefault(); }

std::string GetWarningMessage() {
  const string &warning_message = ErrorManager::GetInstance().GetWarningMessage();
  if (!warning_message.empty()) {
    return warning_message;
  }
  return "";
}

bool IsGraphMode() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode;
}

bool IsDynamicShapeGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  return std::any_of(node_list.begin(), node_list.end(),
                     [](const AnfNodePtr &node) { return common::AnfAlgo::IsDynamicShape(node); });
}

std::string GetSocVersion() {
  // Get default soc version.
  static std::string version{};
  if (version.empty()) {
    const int kSocVersionLen = 50;
    char soc_version[kSocVersionLen] = {0};
    auto ret = rtGetSocVersion(soc_version, kSocVersionLen);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "GetSocVersion failed.";
    }
    // Get soc version from env value.
    const char *soc_version_env = nullptr;
    std::string str_soc_version_env = common::GetEnv(kSOC_VERSION);
    if (!str_soc_version_env.empty()) {
      soc_version_env = common::SafeCStr(str_soc_version_env);
    }
    if (soc_version_env != nullptr) {
      if (std::strcmp(soc_version, soc_version_env) != 0) {
        MS_LOG(DEBUG) << "Detected the env SOC_VERSION, so the SocVersion will be changed to " << str_soc_version_env
                      << ".";
        ret = rtSetSocVersion(soc_version_env);
        if (ret != RT_ERROR_NONE) {
          MS_LOG(EXCEPTION) << "SetSocVersion failed, errorno: " << ret;
        }
        version = soc_version_env;
        return soc_version_env;
      }
    }
    version = soc_version;
  }
  return version;
}

void AssignOutputNopNodeDeviceAddress(const KernelGraphPtr &graph, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  auto outputs = common::AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
  for (auto output : outputs) {
    if (!output->isa<CNode>() || !AnfUtils::IsRealKernel(output)) {
      continue;
    }

    if (!common::AnfAlgo::IsNopNode(output)) {
      continue;
    }

    if (!common::AnfAlgo::IsNeedSkipNopOpAddr(output)) {
      continue;
    }

    size_t input_num = common::AnfAlgo::GetInputTensorNum(output);
    if (input_num != 1) {
      MS_LOG(WARNING) << "The input number of nop node :" << output->fullname_with_scope() << " is " << input_num
                      << ", not equal 1";
      continue;
    }

    auto real_input_index = AnfAlgo::GetInputIndexInGraph(output, 0);
    auto pre_node_out_device_address = AnfAlgo::GetPrevNodeOutputAddr(output, real_input_index);
    MS_EXCEPTION_IF_NULL(pre_node_out_device_address);
    auto ptr = pre_node_out_device_address->GetPtr();
    auto size = pre_node_out_device_address->GetSize();
    std::string output_format = AnfAlgo::GetOutputFormat(output, 0);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(output, 0);
    auto device_address = device_context->device_res_manager_->CreateDeviceAddress(
      const_cast<void *>(ptr), size, output_format, output_type, trans::GetRuntimePaddingShape(output, 0));
    device_address->set_is_ptr_persisted(true);
    device_address->set_host_shape(trans::GetRuntimePaddingShape(output, 0));
    AnfAlgo::SetOutputAddr(device_address, 0, output.get());
    common::AnfAlgo::SetNodeAttr(kAttrSkipNopOpAddr, MakeValue(false), output);
    MS_LOG(INFO) << "Assign device address to output nop node " << output->fullname_with_scope();
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
