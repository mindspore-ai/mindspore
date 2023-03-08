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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_DYNAMINC_SHAPE_UTIL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_DYNAMINC_SHAPE_UTIL_H

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "runtime/device/ms_device_shape_transfer.h"
#include "mindspore/core/ir/anf.h"
#include "kernel/oplib/oplib.h"
namespace mindspore {
namespace kernel {
namespace tbe {
using RangePair = std::vector<std::pair<int64_t, int64_t>>;
class TbeDynamicShapeUtil {
 public:
  TbeDynamicShapeUtil() = default;
  ~TbeDynamicShapeUtil() = default;
  static ShapeVector UpdateShape(const AnfNodePtr &node, const std::string &format, const ShapeVector &shape,
                                 size_t index, bool is_input, bool *is_change_nd = nullptr);
  static bool GetDynamicShapeAttr(const CNodePtr &cnode);
  static bool GetDynamicShapeAttr(const AnfNodePtr &anf_node);
  static std::shared_ptr<OpInfo> FindOp(const std::string &op_name, const AnfNodePtr &anf_node);
  static std::shared_ptr<OpInfo> FindOp(const std::string &op_name, const CNodePtr &cnode);
  static std::shared_ptr<OpInfo> FindOp(const CNodePtr &cnode);
  static RangePair GetInputDynamicRange(const AnfNodePtr &anf_node, size_t index, const std::string &def_format,
                                        const std::string &ori_format, const TypeId &type);
  static RangePair GetOutputDynamicRange(const AnfNodePtr &anf_node, size_t index, const std::string &def_format,
                                         const std::string &ori_format, const TypeId &type);
};
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_DYNAMINC_SHAPE_UTIL_H
