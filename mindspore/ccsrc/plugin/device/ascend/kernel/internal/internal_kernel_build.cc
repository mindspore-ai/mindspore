/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef ENABLE_INTERNAL_KERNELS
#include "plugin/device/ascend/kernel/internal/internal_kernel_build.h"

namespace mindspore {
namespace kernel {
KernelModPtr InternalKernelBuild(const AnfNodePtr &anf_node) { return nullptr; }

bool IsRegisteredInternalKernel(const AnfNodePtr &anf_node) { return false; }
}  // namespace kernel
}  // namespace mindspore

#else
#include "plugin/device/ascend/kernel/internal/internal_kernel_build.h"

#include <string>
#include <utility>
#include <vector>

#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/framework_utils.h"

namespace mindspore {
namespace kernel {
KernelModPtr InternalKernelBuild(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);

  std::string op_fullname = anf_node->fullname_with_scope();
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  // Easy to compare accuracy and performance, later changed to debug
  MS_LOG(INFO) << "internal op [" << opname << "]";
  auto kernel_ptr = Factory<InternalKernelMod>::Instance().Create(opname);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "internal can't find Kernel[" << opname << "]";
    return nullptr;
  }
  kernel_ptr->set_fullname(op_fullname);
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);

  if (!std::static_pointer_cast<KernelMod>(kernel_ptr)
         ->Init(common::AnfAlgo::GetCNodePrimitive(anf_node), input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG_WITH_NODE(EXCEPTION, anf_node) << "#dmsg#Kernel build failed:#dmsg#Initialize aclnn kernel op["
                                          << anf_node->fullname_with_scope() << "] failed.";
  }

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckResizeCondition(cnode)) {
    if (kernel_ptr->Resize(input_kernel_tensors, output_kernel_tensors) == KRET_RESIZE_FAILED) {
      MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#hostapi kernel op[" << cnode->fullname_with_scope()
                        << "] Resize failed.";
    }
  }

  return kernel_ptr;
}

void GetMsTypesList(const CNodePtr &kernel, std::vector<TypeId> *ms_in_dtypes, std::vector<TypeId> *ms_out_dtypes) {
  auto input_num = common::AnfAlgo::GetInputTensorNum(kernel);
  auto output_num = AnfUtils::GetOutputTensorNum(kernel);

  for (size_t i = 0; i < input_num; i++) {
    auto cur_input_type = mindspore::device::ascend::GetInputDeviceType(kernel, i);
    if (mindspore::device::ascend::IsEmptyTupleInput(kernel, i, cur_input_type)) {
      cur_input_type = TypeId::kNumberTypeInt64;
    }
    (void)ms_in_dtypes->push_back(cur_input_type);
  }

  for (size_t i = 0; i < output_num; i++) {
    (void)ms_out_dtypes->push_back(common::AnfAlgo::GetOutputInferDataType(kernel, i));
  }
  return;
}

bool IsRegisteredInternalKernel(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  if (Factory<InternalKernelMod>::Instance().IsRegistered(opname)) {
    internal::DtypesParamPtr check_param = std::make_shared<internal::DtypesParam>();
    check_param->op_id_ = InternalKernelUtils::ToInternalOpId(opname);
    if (check_param->op_id_ == -1) {
      MS_LOG(WARNING) << "internal can't find Kernel[" << opname << "]";
      return false;
    }
    std::vector<TypeId> ms_in_dtypes;
    std::vector<TypeId> ms_out_dtypes;
    auto cnode = anf_node->cast<CNodePtr>();
    GetMsTypesList(cnode, &ms_in_dtypes, &ms_out_dtypes);
    check_param->in_dtypes_ = InternalKernelModInOutMap::GetInstance()->MapInternelInputDtypes(opname, ms_in_dtypes);
    check_param->out_dtypes_ = InternalKernelModInOutMap::GetInstance()->MapInternelOutputDtypes(opname, ms_out_dtypes);
    return internal::IsInternalKernelDtypesSupported(check_param);
  }
  return false;
}
}  // namespace kernel
}  // namespace mindspore
#endif
