/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_build.h"

#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <climits>

#include "ops/structure_op_name.h"
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_mod.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_load.h"
#include "kernel/common_utils.h"
#include "cce/fwk_adpt_struct.h"
#include "external/graph/types.h"
#include "cce/aicpu_engine_struct.h"

namespace mindspore {
namespace kernel {
namespace {
static uint64_t g_aicpu_kernel_id = 0;
static uint64_t g_aicpu_session_id = 0;
}  // namespace

uint64_t SetExtInfoShapeType(char *ext_info_buf, uint64_t ext_info_offset, ::ge::UnknowShapeOpType type) {
  // deal1: unknown shape type
  auto *info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  info->infoType = static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE);
  info->infoLen = sizeof(int32_t);
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;
  auto *shape_type = reinterpret_cast<int32_t *>(ext_info_buf + ext_info_offset);
  *shape_type = static_cast<int32_t>(type);
  ext_info_offset += info->infoLen;
  return ext_info_offset;
}

uint64_t SetExtInfoInputShapeType(char *ext_info_buf, uint64_t ext_info_offset,
                                  const std::shared_ptr<AnfNode> &anf_node, size_t input_num) {
  // deal2:input ShapeAndType
  auto *info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  info->infoType = static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE);
  info->infoLen = SizeToUint(input_num * sizeof(aicpu::FWKAdapter::ShapeAndType));
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;

  auto *inputs = reinterpret_cast<aicpu::FWKAdapter::ShapeAndType *>(ext_info_buf + ext_info_offset);
  for (size_t input_index = 0; input_index < input_num; input_index++) {
    TypeId input_type = AnfAlgo::GetInputDeviceDataType(anf_node, input_index);
    std::vector<int64_t> input_shape;
    int32_t input_data_type;
    if (input_type == kObjectTypeString) {
      auto cnode = anf_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto input_node = cnode->inputs()[input_index + 1];
      auto value_ptr = GetValueNode(input_node);
      auto value = GetValue<std::string>(value_ptr);
      input_shape.push_back(1);
      input_shape.push_back(static_cast<int64_t>(value.size()));
      input_data_type = AicpuOpUtil::MsTypeToProtoType(kObjectTypeString);
    } else {
      input_shape = AnfAlgo::GetInputDeviceShape(anf_node, input_index);
      input_data_type = AicpuOpUtil::MsTypeToProtoType(input_type);
    }
    inputs[input_index].type = input_data_type;

    size_t input_shape_index = 0;
    for (; input_shape_index < input_shape.size(); input_shape_index++) {
      inputs[input_index].dims[input_shape_index] = input_shape[input_shape_index];
    }
    if (input_shape.size() < aicpu::FWKAdapter::kMaxShapeDims) {
      inputs[input_index].dims[input_shape_index] = LLONG_MIN;
    }
  }
  ext_info_offset += info->infoLen;
  return ext_info_offset;
}

uint64_t SetExtInfoOutputShapeType(char *ext_info_buf, uint64_t ext_info_offset,
                                   const std::shared_ptr<AnfNode> &anf_node, size_t output_num) {
  // deal3:output ShapeAndType
  auto *info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  info->infoType = static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE);
  info->infoLen = SizeToUint(output_num * sizeof(aicpu::FWKAdapter::ShapeAndType));
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;

  auto *outputs = reinterpret_cast<aicpu::FWKAdapter::ShapeAndType *>(ext_info_buf + ext_info_offset);
  for (size_t output_index = 0; output_index < output_num; output_index++) {
    auto output_shape = AnfAlgo::GetOutputDeviceShape(anf_node, output_index);
    TypeId output_type = AnfAlgo::GetOutputDeviceDataType(anf_node, output_index);
    int32_t output_data_type = AicpuOpUtil::MsTypeToProtoType(output_type);
    outputs[output_index].type = output_data_type;

    size_t output_shape_index = 0;
    for (; output_shape_index < output_shape.size(); output_shape_index++) {
      outputs[output_index].dims[output_shape_index] = output_shape[output_shape_index];
    }
    for (size_t index = output_shape_index; index < aicpu::FWKAdapter::kMaxShapeDims; index++) {
      outputs[output_index].dims[index] = LLONG_MIN;
    }
  }

  ext_info_offset += info->infoLen;
  return ext_info_offset;
}

uint64_t SetExtInfoAsyncWait(char *ext_info_buf, uint64_t ext_info_offset) {
  // deal5: async wait
  auto *info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  info->infoType = static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT);
  info->infoLen = sizeof(aicpu::FWKAdapter::AsyncWait);
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;
  aicpu::FWKAdapter::AsyncWait *wait_info =
    reinterpret_cast<aicpu::FWKAdapter::AsyncWait *>(ext_info_buf + ext_info_offset);
  wait_info->waitType = aicpu::FWKAdapter::FWK_ADPT_WAIT_TYPE_NULL;
  wait_info->waitId = 0;
  wait_info->timeOut = 0;
  wait_info->reserved = 0;
  ext_info_offset += info->infoLen;
  return ext_info_offset;
}

uint64_t SetExtInfoBitMap(char *ext_info_buf, uint64_t ext_info_offset, uint64_t bitmap) {
  // deal2: bit map
  auto *info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  info->infoType = static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP);
  info->infoLen = sizeof(uint64_t);
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;
  uint64_t *bit_map = reinterpret_cast<uint64_t *>(ext_info_buf + ext_info_offset);
  *bit_map = bitmap;
  ext_info_offset += info->infoLen;
  return ext_info_offset;
}

uint64_t GenerateUniqueKernelId() {
  if (g_aicpu_kernel_id == ULLONG_MAX) {
    g_aicpu_kernel_id = 0;
  }
  return g_aicpu_kernel_id++;
}

uint64_t GenerateUniqueSessionId() {
  if (g_aicpu_session_id == ULLONG_MAX) {
    g_aicpu_session_id = 0;
  }
  return g_aicpu_session_id++;
}

uint64_t SetExtInfoSessionInfo(char *ext_info_buf, uint64_t ext_info_offset) {
  // deal5: async wait
  auto *info = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(ext_info_buf + ext_info_offset);
  info->infoType = static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO);
  info->infoLen = sizeof(SessionInfo);
  ext_info_offset += aicpu::FWKAdapter::kExtInfoHeadSize;
  SessionInfo *session_info = reinterpret_cast<SessionInfo *>(ext_info_buf + ext_info_offset);
  session_info->sessionId = GenerateUniqueSessionId();
  session_info->kernelId = GenerateUniqueKernelId();
  session_info->sessFlag = false;
  ext_info_offset += info->infoLen;
  return ext_info_offset;
}

void CreateExtInfo(const std::shared_ptr<AnfNode> &anf_node, const std::shared_ptr<AicpuOpKernelMod> &kernel_mod_ptr) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return;
  }

  auto op_name = common::AnfAlgo::GetCNodeName(anf_node);
  uint64_t ext_info_head_len = aicpu::FWKAdapter::kExtInfoHeadSize;
  std::string ext_info;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(anf_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node);

  // 1.addr:unknown shape type
  uint64_t ext_info_len = ext_info.size();
  ext_info_len += ext_info_head_len + sizeof(int32_t);

  // 2.addr:bitmap, value: uint64_t
  ext_info_len += ext_info_head_len + sizeof(uint64_t);

  // 3.addr:input ShapeAndType
  ext_info_len += ext_info_head_len + input_num * sizeof(aicpu::FWKAdapter::ShapeAndType);

  // 4.addr:output ShapeAndType
  ext_info_len += ext_info_head_len + output_num * sizeof(aicpu::FWKAdapter::ShapeAndType);

  // 5.addr:session info
  ext_info_len += ext_info_head_len + sizeof(SessionInfo);

  // 5.addr:getnext async wait
  if (op_name == kGetNextOpName) {
    ext_info_len += (ext_info_head_len + sizeof(aicpu::FWKAdapter::AsyncWait));
  }

  uint64_t ext_info_offset = ext_info.size();
  ext_info.resize(ext_info_len, 0);
  char *ext_info_buf = ext_info.data();

  ::ge::UnknowShapeOpType shape_type = ::ge::UnknowShapeOpType::DEPEND_IN_SHAPE;
  if (IsOneOfComputeDepend(op_name)) {
    shape_type = ::ge::UnknowShapeOpType::DEPEND_COMPUTE;
  }
  ext_info_offset = SetExtInfoShapeType(ext_info_buf, ext_info_offset, shape_type);
  // if bitmap = 1, means static_shape, if bitmap = 0, means dynamic_shape
  uint64_t bitmap = 1;
  if (common::AnfAlgo::IsDynamicShape(anf_node)) {
    bitmap = 0;
  }
  ext_info_offset = SetExtInfoBitMap(ext_info_buf, ext_info_offset, bitmap);
  ext_info_offset = SetExtInfoInputShapeType(ext_info_buf, ext_info_offset, anf_node, input_num);
  ext_info_offset = SetExtInfoOutputShapeType(ext_info_buf, ext_info_offset, anf_node, output_num);
  ext_info_offset = SetExtInfoSessionInfo(ext_info_buf, ext_info_offset);
  if (op_name == kGetNextOpName) {
    ext_info_offset = SetExtInfoAsyncWait(ext_info_buf, ext_info_offset);
  }

  MS_LOG(INFO) << "Check ext_info_len:" << ext_info_len << " ext_info_offset:" << ext_info_offset;
  // set ext info
  kernel_mod_ptr->SetExtInfo(ext_info, input_num, output_num);
}

KernelModPtr AicpuOpBuild(const std::shared_ptr<AnfNode> &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string op_name = common::AnfAlgo::GetCNodeName(anf_node);
  if (op_name == kInitDataSetQueue) {
    op_name = kInitData;
  }

  std::shared_ptr<AicpuOpKernelMod> kernel_mod_ptr = std::make_shared<AicpuOpKernelMod>();
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);

  if (!std::static_pointer_cast<KernelMod>(kernel_mod_ptr)
         ->Init(common::AnfAlgo::GetCNodePrimitive(anf_node), input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#Initialize aicpu kernel op["
                      << anf_node->fullname_with_scope() << "] failed.";
  }
  kernel_mod_ptr->SetIsDynamicShape(common::AnfAlgo::IsDynamicShape(anf_node));
  kernel_mod_ptr->CloseTdtWingManQueue();
  kernel_mod_ptr->SetNodeScopeName(anf_node->fullname_with_scope());

  kernel_mod_ptr->SetNodeName(op_name);
  CreateExtInfo(anf_node, kernel_mod_ptr);

  if (!AicpuOpKernelLoad::GetInstance().LoadAicpuKernelSo(anf_node, kernel_mod_ptr)) {
    MS_LOG(EXCEPTION) << "Aicpu kernel so load failed. task is " << anf_node->fullname_with_scope();
  }

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (kernel::CheckResizeCondition(cnode)) {
    kernel_mod_ptr->Resize(input_kernel_tensors, output_kernel_tensors);
  }

  return kernel_mod_ptr;
}
}  // namespace kernel
}  // namespace mindspore
