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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_PRIMITIVE_MAPPER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_PRIMITIVE_MAPPER_H_

#include <string>
#include <memory>
#include "base/base.h"
#include "include/errorcode.h"
#include "ir/anf.h"

namespace mindspore {
namespace lite {
class PrimitiveMapper {
 public:
  explicit PrimitiveMapper(const std::string &name) : name_(name) {}

  virtual ~PrimitiveMapper() = default;

  virtual STATUS Mapper(const CNodePtr &cnode);

 protected:
  STATUS AttrAdjust(const PrimitivePtr &prim, const std::string &name) const;

  STATUS MoveAttrMap(const CNodePtr &cnode, const PrimitivePtr &dst_prim) const;

  STATUS GetValueNodeAndPrimFromCnode(const CNodePtr &cnode, ValueNodePtr *value_node, PrimitivePtr *prim_ptr) const;

  STATUS AdjustPoolAttr(int fmk_type, const std::string &src_prim_name, const PrimitivePtr &dst_prim) const;

  STATUS AddFloatAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const PrimitivePtr &dst_prim,
                             const std::string &attr_name, bool empty_shape) const;

  STATUS AddIntVecAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const PrimitivePtr &dst_prim,
                              const std::string &attr_name) const;

  STATUS AddIntAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const PrimitivePtr &dst_prim,
                           const std::string &attr_name, bool empty_shape) const;

  STATUS AddAttrForDynInputPrimitive(const CNodePtr &cnode, const std::string &attr_name) const;

  STATUS AdjustAttrFormat(const PrimitivePtr &prim, const std::string &name) const;

 private:
  void AdjustCaffePoolAttr(const std::string &src_prim_name, const PrimitivePtr &dst_prim) const;

  void AdjustOnnxPoolAttr(const std::string &src_prim_name, const PrimitivePtr &dst_prim) const;

  std::string name_;
};

using PrimitiveMapperPtr = std::shared_ptr<PrimitiveMapper>;
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_PRIMITIVE_MAPPER_H_
