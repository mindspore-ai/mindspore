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
#include "ops/array_op_name.h"

namespace mindspore {
namespace kernel {
KernelModPtr InternalKernelBuild(const AnfNodePtr &anf_node) { return nullptr; }

bool IsRegisteredInternalKernel(const AnfNodePtr &anf_node) { return false; }

void GetValidKernelBuildInfoWithInternalFormat(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                               std::vector<std::string> *output_formats) {
  return;
}
}  // namespace kernel
}  // namespace mindspore

#else
#include "plugin/device/ascend/kernel/internal/internal_kernel_build.h"

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include "plugin/device/ascend/kernel/internal/acme_kernel_mod.h"
#include "plugin/device/ascend/kernel/internal/acme/acme_helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/framework_utils.h"
#include "ops/math_op_name.h"
#include "ops/nn_op_name.h"
#include "acl/acl_base.h"
#include "transform/acl_ir/acl_helper.h"

namespace mindspore {
namespace kernel {
namespace {
static const char k310PKey[] = "Ascend310P";

// unordered_map value vector<vector<size_t>> mean:
// the first vector is input_idx, the second is output_idx
static const std::unordered_map<std::string, std::vector<std::vector<size_t> > > kNzFormatOpsList = {
  {kMatMulOpName, {{0, 1}, {0}}},
  {"QuantBatchMatmul", {{0, 1}, {0}}},
  {kPagedAttentionOpName, {{0, 1, 2}, {0}}},
  {kFlashAttentionScoreOpName, {{0, 1, 2, 6}, {3}}},
  {kReshapeAndCacheOpName, {{2, 3}, {}}}};

// unordered_map mean:
// key is input_idx, value is special_format value
// ATTENTION_INPUT_QKV: ms_nd_shape{b, s, h} need convert to {b * s, h}, then transform nz format
// ATTENTION_INPUT_MASK: ms_nd_shape{b, 1, s, s} need convert to {b, 1, s, s}, then transform nz format
static const std::unordered_map<std::string, std::unordered_map<size_t, int64_t> > kSpecialNzFormatOpsList = {
  {kPagedAttentionOpName, {{0, internal::TransDataParam::ATTENTION_INPUT_QKV}}},
  {kFlashAttentionScoreOpName,
   {{0, internal::TransDataParam::ATTENTION_INPUT_QKV},
    {1, internal::TransDataParam::ATTENTION_INPUT_QKV},
    {2, internal::TransDataParam::ATTENTION_INPUT_QKV},
    {6, internal::TransDataParam::ATTENTION_INPUT_MASK}}}};

bool IsAscend310PSoc() {
  const char *soc_name_c = aclrtGetSocName();
  if (soc_name_c == nullptr) {
    return false;
  }
  std::string soc_name(soc_name_c);
  if (soc_name.find(k310PKey) != std::string::npos) {
    return true;
  }
  return false;
}

int64_t GetSpecialFormat(const AnfNodePtr &cur_node, const AnfNodePtr &input_node, const size_t input_idx) {
  MS_EXCEPTION_IF_NULL(cur_node);
  MS_EXCEPTION_IF_NULL(input_node);
  int64_t special_format_input = internal::TransDataParam::NORMAL;

  // cur cnode has special format input
  auto special_format_iter = kSpecialNzFormatOpsList.find(AnfUtils::GetCNodeName(cur_node));
  if (special_format_iter != kSpecialNzFormatOpsList.end()) {
    auto iter = special_format_iter->second.find(input_idx);
    if (iter != special_format_iter->second.end()) {
      special_format_input = iter->second;
    } else {
      special_format_input = internal::TransDataParam::NORMAL;
    }
  } else if (input_node->isa<CNode>()) {
    // input cnode has special format output: pa & fa output format is nz
    auto special_iter = kSpecialNzFormatOpsList.find(AnfUtils::GetCNodeName(input_node));
    if (special_iter != kSpecialNzFormatOpsList.end()) {
      special_format_input = internal::TransDataParam::ATTENTION_INPUT_QKV;
    }
  }
  return special_format_input;
}

bool IsKernelGraphOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &outputs = common::AnfAlgo::GetAllOutputIndexByReturnTypes(func_graph->output());
  return std::find_if(outputs.begin(), outputs.end(), [&node](const auto &output) {
           const auto &real_pair = common::AnfAlgo::VisitKernelWithReturnType(node, 0);
           return output.first == node || (real_pair.first == output.first && real_pair.second == output.second);
         }) != outputs.end();
}

bool IsNeedInsertTransDataForGraphOut(const AnfNodePtr &node, const std::vector<std::string> &output_formats) {
  // output is graph output & format is nz
  if (IsKernelGraphOutput(node) &&
      std::any_of(output_formats.begin(), output_formats.end(),
                  [](const std::string &format) { return !transform::AclHelper::CheckDefaultSupportFormat(format); })) {
    return true;
  }
  return false;
}

bool NeedSetParameterFormat(const AnfNodePtr &input_node, const std::string &new_format,
                            const std::string &input_format) {
  std::string old_format = input_format;
  if (transform::AclHelper::CheckDefaultSupportFormat(old_format) &&
      !transform::AclHelper::CheckDefaultSupportFormat(new_format)) {
    transform::SetParameterFormat(input_node, new_format, &old_format);
    if (old_format != input_format) {
      return true;
    }
  }
  return false;
}
}  // namespace

void GetValidKernelBuildInfoWithInternalFormat(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                               std::vector<std::string> *output_formats) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input_formats);
  MS_EXCEPTION_IF_NULL(output_formats);

  bool is_310p = IsAscend310PSoc();
  if (!is_310p) {
    return;
  }

  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);

  auto format_idx_iter = kNzFormatOpsList.find(AnfUtils::GetCNodeName(node));
  if (format_idx_iter != kNzFormatOpsList.end()) {
    auto input_nz_format_idx = format_idx_iter->second[0];
    auto output_nz_format_idx = format_idx_iter->second[1];
    for (const auto &input_idx : input_nz_format_idx) {
      input_formats->at(input_idx) = kOpFormat_FRAC_NZ;
    }
    for (const auto &output_idx : output_nz_format_idx) {
      output_formats->at(output_idx) = kOpFormat_FRAC_NZ;
    }
  }

  std::vector<size_t> special_inputs;
  std::vector<int64_t> special_format_inputs;
  for (size_t i = 0; i < input_num; ++i) {
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
    std::string input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    input_format = NeedSetParameterFormat(kernel_with_index.first, input_formats->at(i), input_format)
                     ? input_formats->at(i)
                     : input_format;
    // for reshapeext input_idx == 1, do not insert transdata
    if (AnfUtils::GetCNodeName(node) == kReshapeExtOpName && i == 1) {
      continue;
    }
    if ((!transform::AclHelper::CheckDefaultSupportFormat(input_format) ||
         !transform::AclHelper::CheckDefaultSupportFormat(input_formats->at(i))) &&
        input_format != input_formats->at(i)) {
      (void)special_inputs.emplace_back(i);
      (void)special_format_inputs.emplace_back(GetSpecialFormat(node, kernel_with_index.first, i));
    }
  }
  if (!special_inputs.empty()) {
    common::AnfAlgo::SetNodeAttr(kAttrAclSpecialInputFormat, MakeValue(special_inputs), node);
    if (std::any_of(special_format_inputs.begin(), special_format_inputs.end(),
                    [](const int64_t format_type) { return format_type != internal::TransDataParam::NORMAL; })) {
      common::AnfAlgo::SetNodeAttr(kAttrInternalSepcialFormat, MakeValue(special_format_inputs), node);
    }
  }
  // if graph output is nz format need insert transdata
  if (IsNeedInsertTransDataForGraphOut(node, *output_formats)) {
    common::AnfAlgo::SetNodeAttr(kAttrAclSpecialFormat, MakeValue(true), node);
  }
}

KernelModPtr InternalKernelBuild(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);

  std::string op_fullname = anf_node->fullname_with_scope();
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  // Easy to compare accuracy and performance, later changed to debug
  MS_LOG(INFO) << "internal op [" << opname << "]";
  KernelModPtr kernel_ptr;
  if (Factory<AcmeKernelMod>::Instance().IsRegistered(opname)) {
    MS_LOG(DEBUG) << "Supported by AcmeKernel: " << opname;
    kernel_ptr = std::static_pointer_cast<KernelMod>(Factory<AcmeKernelMod>::Instance().Create(opname));
  } else {
    kernel_ptr = std::static_pointer_cast<KernelMod>(Factory<InternalKernelMod>::Instance().Create(opname));
  }
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "internal can't find Kernel[" << opname << "]";
    return nullptr;
  }
  kernel_ptr->set_fullname(op_fullname);
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);
  if (!kernel_ptr->Init(common::AnfAlgo::GetCNodePrimitive(anf_node), input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG_WITH_NODE(EXCEPTION, anf_node) << "#dmsg#Kernel build failed:#dmsg#Initialize internal kernel op["
                                          << anf_node->fullname_with_scope() << "] failed.";
  }

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckResizeCondition(cnode)) {
    if (kernel_ptr->Resize(input_kernel_tensors, output_kernel_tensors) == KRET_RESIZE_FAILED) {
      MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#internal kernel op[" << cnode->fullname_with_scope()
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
  auto cnode = anf_node->cast<CNodePtr>();
  std::vector<TypeId> ms_in_dtypes;
  std::vector<TypeId> ms_out_dtypes;
  GetMsTypesList(cnode, &ms_in_dtypes, &ms_out_dtypes);
  if (Factory<AcmeKernelMod>::Instance().IsRegistered(opname)) {
    auto acme_op_name = TransAcmeOpName(opname);
    auto acme_in_dtypes = InternalKernelModInOutMap::GetInstance()->MapAcmeInputDtypes(opname, ms_in_dtypes);
    auto acme_out_dtypes = InternalKernelModInOutMap::GetInstance()->MapAcmeOutputDtypes(opname, ms_out_dtypes);
    return acme::IsAcmeKernelDtypesSupported(acme_op_name, acme_in_dtypes, acme_out_dtypes);
  }
  if (Factory<InternalKernelMod>::Instance().IsRegistered(opname)) {
    if (opname == kReshapeOpName) {
      return true;
    }
    internal::DtypesParamPtr check_param = std::make_shared<internal::DtypesParam>();
    check_param->op_id_ = InternalKernelUtils::ToInternalOpId(opname);
    if (check_param->op_id_ == -1) {
      MS_LOG(INFO) << "internal can't find Kernel[" << opname << "]";
      return false;
    }
    check_param->in_dtypes_ = InternalKernelModInOutMap::GetInstance()->MapInternelInputDtypes(opname, ms_in_dtypes);
    check_param->out_dtypes_ = InternalKernelModInOutMap::GetInstance()->MapInternelOutputDtypes(opname, ms_out_dtypes);
    return internal::IsInternalKernelDtypesSupported(check_param);
  }
  return false;
}
}  // namespace kernel
}  // namespace mindspore
#endif
