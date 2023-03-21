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
#include "plugin/device/ascend/optimizer/mindir/quant_dtype_cast_adjust.h"

#include <vector>
#include <memory>
#include <string>
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"
#include "runtime/device/ms_device_shape_transfer.h"

namespace mindspore {
namespace opt {
std::vector<std::string> QuantDTypeCastAdjust::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(std::make_shared<Primitive>(kQuantDTypeCastOpName)->name());
  return ret;
}

const BaseRef QuantDTypeCastAdjust::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kQuantDTypeCastOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr QuantDTypeCastAdjust::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << cnode->fullname_with_scope() << " run QuantDTypeCastAdjust pass.";
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive->DelAttr("format");
  primitive->DelAttr("infer_done");
  std::vector<std::string> cnode_output_format = {AnfAlgo::GetOutputFormat(cnode, 0)};
  if (!cnode_output_format.empty() && cnode_output_format.at(0) == "FRACTAL_NZ") {
    auto param_node = cnode->input(1)->cast<ParameterPtr>();
    auto tensor_info = param_node->default_param()->cast<tensor::TensorPtr>();
    auto host_shape = tensor_info->shape_c();
    auto size = tensor_info->Size();
    auto host_ptr = tensor_info->data_c();
    auto device_shape = trans::TransShapeToDevice(host_shape, kOpFormat_FRAC_NZ, kNumberTypeFloat16);
    const trans::FormatArgs format_args{host_ptr,   size,         kOpFormat_NCHW, kOpFormat_FRAC_NZ,
                                        host_shape, device_shape, kNumberTypeInt8};
    auto host_tmp = std::vector<uint8_t>(size);
    MS_LOG(DEBUG) << "TransFormat host_shape:" << host_shape << " device_shape:" << device_shape;
    auto ret = trans::TransFormat(format_args, host_tmp.data(), cnode, 1);
    if (!ret) {
      MS_LOG(ERROR) << "Trans format failed.";
      return nullptr;
    }
    if (memcpy_s(tensor_info->data_c(), tensor_info->Size(), host_tmp.data(), host_tmp.size()) != EOK) {
      MS_LOG(ERROR) << "memcpy failed.";
      return nullptr;
    }
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
