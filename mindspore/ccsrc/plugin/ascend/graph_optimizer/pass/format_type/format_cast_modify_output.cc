/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/format_type/format_cast_modify_output.h"
#include <map>
#include <memory>
#include "acl/acl_base.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"

namespace mindspore {
namespace opt {
namespace {
std::string GetMsFormatWithAclFormat(const int64_t &acl_format) {
  static const std::map<aclFormat, std::string> kAclFormatToMsFormat = {{ACL_FORMAT_NCHW, kOpFormat_NCHW},
                                                                        {ACL_FORMAT_NHWC, kOpFormat_NHWC},
                                                                        {ACL_FORMAT_ND, kOpFormat_ND},
                                                                        {ACL_FORMAT_NC1HWC0, kOpFormat_NC1HWC0},
                                                                        {ACL_FORMAT_FRACTAL_Z, kOpFormat_FRAC_Z},
                                                                        {ACL_FORMAT_NDHWC, kOpFormat_NDHWC},
                                                                        {ACL_FORMAT_FRACTAL_NZ, kOpFormat_FRAC_NZ},
                                                                        {ACL_FORMAT_NCDHW, kOpFormat_NCDHW},
                                                                        {ACL_FORMAT_NDC1HWC0, kOpFormat_NDC1HWC0},
                                                                        {ACL_FRACTAL_Z_3D, kOpFormat_FRACTAL_Z_3D}};
  auto iter = kAclFormatToMsFormat.find(static_cast<aclFormat>(acl_format));
  if (iter == kAclFormatToMsFormat.end()) {
    MS_LOG(EXCEPTION) << "Invalid AclFormat " << acl_format << " for FormatCast.";
  }
  return iter->second;
}
}  // namespace

const BaseRef FormatCastModifyOutput::DefinePattern() const {
  VarPtr x = std::make_shared<Var>();
  VarPtr acl_format = std::make_shared<Var>();
  return VectorRef({prim::kPrimFormatCast, x, acl_format});
}

const AnfNodePtr FormatCastModifyOutput::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto acl_format_v = cnode->input(kIndex2)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(acl_format_v);
  auto acl_format_opt = GetScalarValue<int64_t>(acl_format_v->value());
  MS_EXCEPTION_IF_CHECK_FAIL(acl_format_opt.has_value(), "Can't get acl_format value from " + cnode->DebugString());

  auto builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(cnode));
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetOutputFormat(GetMsFormatWithAclFormat(acl_format_opt.value()), kIndex0);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), cnode.get());

  return cnode;
}
}  // namespace opt
}  // namespace mindspore
