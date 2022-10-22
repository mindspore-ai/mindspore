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
#include <vector>
#include "kernel/kernel.h"
#include "mindapi/base/type_id.h"
#include "acl/acl_op_compiler.h"
#include "acl/acl_base.h"
#include "transform/graph_ir/convert.h"

namespace mindspore {
namespace kernel {
using GeTensorDesc = transform::GeTensorDesc;
using GeTensorDescPtr = std::shared_ptr<GeTensorDesc>;
using GeOpConvertor = transform::GeOpConvertor;

template <typename T>
inline constexpr bool is_vector = false;
template <typename T, typename A>
inline constexpr bool is_vector<std::vector<T, A>> = true;

class AclOpDesc {
 public:
  explicit AclOpDesc(const std::string &op_type);
  ~AclOpDesc();

  void AddTensorDesc(const std::vector<GeTensorDescPtr> &inputs, const std::vector<GeTensorDescPtr> &outputs);
  void AddDataBuf(const std::vector<AddressPtr> &inputs, const std::vector<size_t> &input_size_list,
                  const std::vector<AddressPtr> &outputs, const std::vector<size_t> &output_size_list);
  void AddTensorAttr(const std::string &attr_name, const ValuePtr &value);
  void AddConstInputTensor(const AnfNodePtr &anf_node);

  std::vector<aclTensorDesc *> input_tensor_desc() const { return input_tensor_desc_; }
  std::vector<aclTensorDesc *> output_tensor_desc() const { return output_tensor_desc_; }
  std::vector<aclDataBuffer *> input_tensor_data() const { return input_tensor_data_; }
  std::vector<aclDataBuffer *> output_tensor_data() const { return output_tensor_data_; }
  aclopAttr *acl_attr() const { return acl_attr_; }

 protected:
  aclTensorDesc *CreateTensorDesc(const GeTensorDescPtr &tensor_desc);
  aclDataBuffer *CreateDataBuf(const AddressPtr &address, const size_t op_size);
  void SetListAttr(const std::string &attr_name, const ValuePtr &value);
  bool SelectConversionDataType(const ValuePtr &value, const unsigned int index);

  template <typename T>
  void AddConstDescAndBuf(const T &val, const TypeId type, const size_t index);

 private:
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

  static std::vector<GeTensorDescPtr> GetOutputTensorDesc(const AnfNodePtr &anf_node);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_ACL_UTILS_H_
