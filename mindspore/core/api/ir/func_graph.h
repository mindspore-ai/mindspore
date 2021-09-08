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

#ifndef MINDSPORE_CORE_API_IR_FUNC_GRAPH_H_
#define MINDSPORE_CORE_API_IR_FUNC_GRAPH_H_

#include <vector>
#include <memory>
#include <string>

#include "utils/visible.h"
#include "api/ir/func_graph_manager.h"

namespace mindspore::api {

class MS_CORE_API FuncGraph {
 public:
  FuncGraph() = default;
  virtual ~FuncGraph() = default;

  virtual const std::vector<AnfNodePtr> get_inputs() const = 0;
  virtual const std::vector<AnfNodePtr> &parameters() const = 0;
  virtual void add_parameter(const ParameterPtr &p) = 0;
  virtual ParameterPtr add_parameter() = 0;

  virtual AnfNodePtr output() const = 0;
  virtual CNodePtr get_return() const = 0;
  virtual void set_output(const AnfNodePtr &value, bool force_new_ret = false) = 0;
  virtual void set_return(const CNodePtr &cnode) = 0;

  virtual CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs = std::vector<AnfNodePtr>()) = 0;
  virtual CNodePtr NewCNode(const PrimitivePtr &primitive, const std::vector<AnfNodePtr> &prim_inputs) = 0;

  virtual const AnfNodeSet &nodes() const = 0;

  virtual bool has_attr(const std::string &key) const = 0;
  virtual ValuePtr get_attr(const std::string &key) const = 0;
  virtual void set_attr(const std::string &key, const ValuePtr &value) = 0;

  virtual FuncGraphManagerPtr get_manager() const = 0;

  static std::vector<AnfNodePtr> TopoSort(const AnfNodePtr &node);

  static FuncGraphPtr Create();

  static AnfNodePtr MakeValueNode(const FuncGraphPtr &func_graph);

  static FuncGraphPtr GetFuncGraphFromAnfNode(const AnfNodePtr &input);
};
}  // namespace mindspore::api
#endif  // MINDSPORE_CORE_API_IR_FUNC_GRAPH_H_
