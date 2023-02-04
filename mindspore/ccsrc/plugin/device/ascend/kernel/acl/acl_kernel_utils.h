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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_ACL_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_ACL_UTILS_H_
#include <memory>
#include <string>
#include <set>
#include <vector>
#include <map>
#include "kernel/kernel.h"
#include "mindapi/base/type_id.h"
#include "acl/acl_op_compiler.h"
#include "acl/acl_base.h"
#include "transform/graph_ir/convert.h"
#include "kernel/oplib/oplib.h"
#include "kernel/oplib/super_bar.h"

namespace mindspore {
namespace kernel {
using GeTensorDesc = transform::GeTensorDesc;
using GeTensorDescPtr = std::shared_ptr<GeTensorDesc>;
using GeOpConvertor = transform::GeOpConvertor;
using GeDataType = transform::GeDataType;
using GeFormat = transform::GeFormat;
using GeShape = transform::GeShape;
constexpr auto kSizeMax = SIZE_MAX;

template <typename T>
inline constexpr bool is_vector = false;
template <typename T, typename A>
inline constexpr bool is_vector<std::vector<T, A>> = true;

typedef enum { SET_ACL_ATTR, SET_ACL_INPUT } ProcessAttrMode;

class AclOpDesc {
 public:
  AclOpDesc(const std::string &op_type, const AnfNodePtr &anf_node_ptr);
  ~AclOpDesc();

  void AddTensorDesc(const std::vector<GeTensorDescPtr> &inputs, const std::vector<GeTensorDescPtr> &outputs);
  void AddDataBuf(const std::vector<AddressPtr> &inputs, const std::vector<size_t> &input_size_list,
                  const std::vector<AddressPtr> &outputs, const std::vector<size_t> &output_size_list);
  void ProcessAclAttrs(const std::string &attr_name, const ValuePtr &value, const ProcessAttrMode &mode);
  void ClearNullTensor();

  std::vector<aclTensorDesc *> input_tensor_desc() const { return input_tensor_desc_; }
  std::vector<aclTensorDesc *> output_tensor_desc() const { return output_tensor_desc_; }
  std::vector<aclDataBuffer *> input_tensor_data() const { return input_tensor_data_; }
  std::vector<aclDataBuffer *> output_tensor_data() const { return output_tensor_data_; }
  aclopAttr *acl_attr() const { return acl_attr_; }

 protected:
  aclTensorDesc *CreateTensorDesc(const GeTensorDescPtr &tensor_desc);
  aclDataBuffer *CreateDataBuf(const AddressPtr &address, const size_t op_size);

  void GetListAttr(const std::string &attr_name, const ValuePtr &value, const ProcessAttrMode &mode);
  void GetListListAttr(const std::string &attr_name, const ValuePtr &value, const ProcessAttrMode &mode);
  std::vector<std::vector<int64_t>> GetListListAttrBool(const std::string &attr_name,
                                                        const ValuePtrList &value_sequence);
  std::vector<std::vector<int64_t>> GetListListAttrInt(const std::string &attr_name,
                                                       const ValuePtrList &value_sequence);
  std::vector<std::vector<int64_t>> GetListListAttrFloat(const std::string &attr_name,
                                                         const ValuePtrList &value_sequence);

  void ListListToListInt(const std::vector<std::vector<int64_t>> &value_list, std::vector<int64_t> *array_list);
  aclError AclSetAttrListListInt(const std::string &attr_name, const std::vector<std::vector<int64_t>> &value_list);
  aclError AclSetAttrListString(const std::string &attr_name, const std::vector<std::string> &value_list);

  template <typename T>
  void CallFunc(const T &val, const TypeId type, const std::string &attr_name, const ProcessAttrMode &mode);
  template <typename T>
  void AddConstDescAndBuf(const T &val, const TypeId type, const std::string &attr_name);
  template <typename T>
  void CallAclAttrFunc(const T &val, const TypeId type, const std::string &attr_name);

  void CreateNullAclTensor(const size_t idx, const bool is_input);

 private:
  AnfNodeWeakPtr anf_node_;
  void *attr_to_input_{nullptr};
  size_t attr_data_offset_{0};
  std::string op_type_;
  aclopAttr *acl_attr_{nullptr};
  std::vector<aclTensorDesc *> input_tensor_desc_{};
  std::vector<aclTensorDesc *> output_tensor_desc_{};
  std::vector<aclDataBuffer *> input_tensor_data_{};
  std::vector<aclDataBuffer *> output_tensor_data_{};
};

class AclUtils {
 public:
  static aclDataType ConvertTypeIdToAclType(const ::ge::DataType &type);

  static aclFormat ConvertFormatToAclFormat(const ::ge::Format &format);

  static bool UpdateTensorDesc(const AnfNodePtr &anf_node, std::vector<GeTensorDescPtr> *inputs,
                               std::vector<GeTensorDescPtr> *outputs);

  static std::vector<GeTensorDescPtr> GetInputTensorDesc(const AnfNodePtr &anf_node);

  static ShapeVector UpdateShape(const ShapeVector &shape, const std::string &format, const AnfNodePtr &node);

  static int GetOutputKernelIdxByGraphIdx(const AnfNodePtr &node, size_t ori_idx);

  static int GetInputKernelIdxByGraphIdx(const AnfNodePtr &node, size_t ori_idx);

  static int GetInputGraphIdxByKernelIdx(const AnfNodePtr &node, size_t ori_idx);

  static std::vector<GeTensorDescPtr> GetOutputTensorDesc(const AnfNodePtr &anf_node);

  static std::vector<std::string> GetOpInputAnchorNames(const AnfNodePtr &node);

  static std::vector<std::string> GetOpOutputAnchorNames(const AnfNodePtr &node);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_ACL_UTILS_H_
