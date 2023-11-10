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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_PROTO_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_PROTO_H_

#include <functional>
#include <string>
#include <vector>
#include "graph/types.h"
#include "utils/hash_map.h"

namespace mindspore::transform {
class OpProto {
 public:
  explicit OpProto(const std::string &name);
  ~OpProto() = default;
  OpProto &SetInput(const std::string &name, const std::string &tensor_type, bool is_optional);
  OpProto &SetOutput(const std::string &name, const std::string &tensor_type);
  OpProto &SetAttr(const std::string &name, bool is_optional);
  OpProto &DoNothing();
  OpProto &DefineDataType(const std::string &name, const std::string &tensor_type);
  OpProto &FinishRegOperator();

  size_t GetInputIndexByName(const std::string &name) const;
  size_t GetOutputIndexByName(const std::string &name) const;

  bool IsInputOptionalTypeByName(const std::string &name) const;
  bool IsAttrOptionalTypeByName(const std::string &name) const;
  std::vector<enum ge::DataType> GetInputTypesByName(const std::string &name) const;
  std::vector<enum ge::DataType> GetOutputTypesByName(const std::string &name) const;
  void ProcessPromoteTypes();

 private:
  std::string name_;
  std::vector<std::string> input_names_;
  std::vector<bool> input_optional_flags_;
  HashMap<std::string, std::vector<enum ge::DataType>> input_types_;
  std::vector<std::string> output_names_;
  HashMap<std::string, std::vector<enum ge::DataType>> output_types_;
  HashMap<std::string, bool> attr_optional_flags_;

  // temporary fields used for building operator info
  HashMap<std::string, std::string> input_types_org_;
  HashMap<std::string, std::string> output_types_org_;
  HashMap<std::string, std::vector<enum ge::DataType>> alias_types_;
  HashMap<std::string, std::vector<std::string>> promote_types_;
};

class OpProtoStorage {
 public:
  static OpProtoStorage &GetInstance();
  OpProto &GetOpProto(const std::string &name);

 private:
  HashMap<std::string, OpProto> op_proto_map_;
};
}  //  namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_PROTO_H_
