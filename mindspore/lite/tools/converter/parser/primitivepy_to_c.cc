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

#define USE_DEPRECATED_API
#include "tools/converter/parser/primitivepy_to_c.h"
#include <cstring>
#include <memory>
#include <unordered_map>
#include "pybind_api/ir/primitive_py.h"
#include "ops/primitive_c.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
namespace {
std::unordered_map<string, mindspore::Format> kStr2FormatMap{{"DEFAULT_FORMAT", mindspore::Format::DEFAULT_FORMAT},
                                                             {"NCHW", mindspore::Format::NCHW},
                                                             {"NHWC", mindspore::Format::NHWC},
                                                             {"NHWC4", mindspore::Format::NHWC4},
                                                             {"HWKC", mindspore::Format::HWKC},
                                                             {"HWCK", mindspore::Format::HWCK},
                                                             {"KCHW", mindspore::Format::KCHW},
                                                             {"CKHW", mindspore::Format::CKHW},
                                                             {"KHWC", mindspore::Format::KHWC},
                                                             {"CHWK", mindspore::Format::CHWK},
                                                             {"HW", mindspore::Format::HW},
                                                             {"HW4", mindspore::Format::HW4},
                                                             {"NC", mindspore::Format::NC},
                                                             {"NC4", mindspore::Format::NC4},
                                                             {"NC4HW4", mindspore::Format::NC4HW4},
                                                             {"NUM_OF_FORMAT", mindspore::Format::NUM_OF_FORMAT},
                                                             {"NCDHW", mindspore::Format::NCDHW},
                                                             {"NWC", mindspore::Format::NWC},
                                                             {"NCW", mindspore::Format::NCW},
                                                             {"NDHWC", mindspore::Format::NDHWC},
                                                             {"NC8HW8", mindspore::Format::NC8HW8}};
}  // namespace

bool PrimitivePyToCFusion::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_ASSERT(node != nullptr);
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_ASSERT(cnode != nullptr);

    // judge if primitive is PrimitivePy
    auto primpy_ptr = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(primpy_ptr);
    if (!primpy_ptr) {
      MS_LOG(ERROR) << "Primitive of cnode " << cnode->fullname_with_scope() << " cannot be nullptr";
      return RET_ERROR;
    }
    if (!utils::isa<PrimitivePy>(primpy_ptr)) {
      continue;
    }
    MS_LOG(INFO) << "Transform a primitivePy to primitiveC for node " << cnode->fullname_with_scope();

    auto kernel_name = primpy_ptr->name();
    ops::PrimitiveCPtr primc_ptr = nullptr;
    static auto &primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
    auto primc_it = primc_fns.find(kernel_name);
    if (primc_it != primc_fns.end() && primc_it->second) {
      primc_ptr = primc_it->second();
    }
    if (primc_ptr == nullptr) {
      MS_LOG(ERROR) << "OpPrimCRegister can not find " << kernel_name;
      return RET_ERROR;
    }
    (void)primc_ptr->SetAttrs(primpy_ptr->attrs());

    if (primpy_ptr->HasAttr(ops::kFormat)) {
      MS_LOG(INFO) << "Add attr Original format to " << cnode->fullname_with_scope();
      auto format_str = GetValue<string>(primpy_ptr->GetAttr(ops::kFormat));
      auto format_it = kStr2FormatMap.find(format_str.c_str());
      if (format_it != kStr2FormatMap.end()) {
        MS_LOG(INFO) << "Add attr Original format" << format_it->second << " to " << cnode->fullname_with_scope();
        (void)primc_ptr->AddAttr(mindspore::ops::kOriginalFormat,
                                 std::dynamic_pointer_cast<mindspore::Value>(
                                   api::MakeValue<int64_t>(static_cast<int64_t>(format_it->second))->impl()));
      } else {
        MS_LOG(ERROR) << "Fail to find format " << format_str.c_str() << "in kStr2FormatMap";
        return RET_ERROR;
      }
    }

    auto new_prim = MakeValue(primc_ptr);
    auto new_value_node = NewValueNode(new_prim);
    new_value_node->set_abstract(new_prim->ToAbstract());
    cnode->set_input(0, new_value_node);
  }
  return RET_OK;
}
}  // namespace mindspore::opt
