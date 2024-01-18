/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/resize_bilinear_mapper.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kInputNum = 2;
}

STATUS ResizeBilinearMapper::Mapper(const CNodePtr &cnode) {
  if (cnode->inputs().size() != kInputNum) {
    MS_LOG(ERROR) << "Input of resize_bilinear must be " << kInputNum << ", real size: " << cnode->inputs().size()
                  << ", cnode " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed, cnode: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto size_ptr = src_prim->GetAttr(ops::kSize);
  if (size_ptr == nullptr) {
    MS_LOG(ERROR) << "Get size failed, cnode: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto size_value_int32 = opt::CastToInt(size_ptr);
  if (size_value_int32.empty()) {
    MS_LOG(ERROR) << "Cast Size Value to Int32 failed in cnode: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  src_prim->AddAttr(ops::kSize, MakeValue(size_value_int32));

  auto inputs = cnode->inputs();
  // convert "size" attr to input, and add it to inputs list
  ParameterPtr value_param =
    opt::BuildIntVecParameterNode(cnode->func_graph(), size_value_int32, cnode->fullname_with_scope() + "_size_value");
  inputs.push_back(value_param);
  cnode->set_inputs(inputs);

  // replace ResizeBilinear with ResizeBilinearV2
  PrimitivePtr dst_prim = std::make_shared<acl::ResizeBilinearV2>();
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);

  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameResizeBilinear, ResizeBilinearMapper)
}  // namespace lite
}  // namespace mindspore
