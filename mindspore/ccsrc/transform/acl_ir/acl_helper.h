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

#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_HELPER_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_HELPER_H_

#include <vector>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/tensor.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace transform {
class GeAdapterInfo;
typedef enum ErrorAclType { kNormalOp, kUnknownOp, kInValidType, kSpecialOp, kInvalidBuildInfo } ErrorAclType;
void SetParameterFormat(const AnfNodePtr &node, const std::string &format, std::string *old_foramt);

class AclHelper {
 public:
  // Check is data layout unchanged format.
  static bool CheckDefaultSupportFormat(const string &format);
  static bool IsPrintDebugString();

  // Kernel select by ge_ir.
  static KernelType GetKernelInfoByInputs(const CNodePtr &cnode, const std::shared_ptr<GeAdapterInfo> &info);
  static KernelType GetKernelInfoByOutputs(const AnfNodePtr &node, const std::shared_ptr<GeAdapterInfo> &info);
  static KernelType GetKernelInfoFromGe(const AnfNodePtr &node, ErrorAclType *err_type);

  // Select kernel's device format.
  static void GetValidKernelBuildInfo(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                      std::vector<std::string> *output_formats,
                                      std::vector<std::string> *input_reshape_types,
                                      std::vector<std::string> *output_reshape_types);

  // Convert mindspore's origin information to acl's origin information.
  static void PaddingOriShape(const std::string &name, size_t idx, const std::string &format, ShapeVector *shape);
  static std::string ConvertOriginShapeAndFormat(const std::string &name, size_t idx, const std::string &dev_format,
                                                 ShapeVector *shape);

  // Get attribute to input information.
  static bool NeedCheckAttrToInput(const CNodePtr &node, const mindspore::HashMap<size_t, std::string> &attr_input_map,
                                   size_t index);
  // Get special information from kernel's attribute.
  static std::string GetFormatFromAttr(const PrimitivePtr &primitive);
  static int64_t GetFracZGroupFromAttr(const PrimitivePtr &primitive);
  static bool GetDefaultFormatFlagFromAttr(const PrimitivePtr &primitive, bool is_input);

  // Get kernel's precision mode is FORCE_FP32.
  static bool GetMoreDataTypeSupported(TypeId data_type, const std::string &op_type);

  // Check whether is nop op.
  static bool IsNopNode(const CNodePtr &node);
  static bool IsInputDtypeSupport(const std::string &kernel_name, TypeId base_type, size_t idx);

  // Set identity flag.
  static bool NeedIdentityFlag(const std::vector<std::string> &formats);
};
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_HELPER_H_
