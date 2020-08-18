/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_BASE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_BASE_H_

#include <unordered_map>
#include <string>
#include <memory>
#include <utility>
#include <vector>
#include <sstream>

#include "transform/graph_ir/util.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "transform/graph_ir/types.h"
#ifdef ENABLE_GE
#ifdef OPEN_SOURCE
#include "graph/types.h"
#endif
#endif

#include "graph/operator_reg.h"
#ifdef OPEN_SOURCE
#include "ge/client/ge_api.h"
#else
#include "external/ge/ge_api.h"
#endif
#include "graph/tensor.h"

namespace ge {
class CustomOperator : public Operator {
 public:
  CustomOperator(const string &name, const string &type) : Operator(name, type) {}

  ~CustomOperator() override{};

  void CustomInputRegister(const string &name) { Operator::InputRegister(name); }

  void CustomOutputRegister(const string &name) { Operator::OutputRegister(name); }

  void CustomInferFuncRegister(const std::function<graphStatus(Operator &)> &func) {
    Operator::InferFuncRegister(func);
  }
};
}  // namespace ge

namespace mindspore {
namespace transform {
using CusOperatorPtr = std::shared_ptr<ge::CustomOperator>;
using CustomOperator = ge::CustomOperator;

struct OutHandler {
  OperatorPtr op;
  std::string out;
  OutHandler() : op(nullptr), out("") {}
  OutHandler(const OperatorPtr &op, const std::string out) : op(op), out(out) {}
};

struct ControlEdge {
  OperatorPtr src_op;
  OperatorPtr dest_op;
};

using AttrFunc = std::function<void(OperatorPtr, ValuePtr)>;
using OutputFunc = std::function<OutHandler(OperatorPtr)>;
using InputOpFunc = std::function<void(OperatorPtr, OperatorPtr)>;
using InputHandleFunc = std::function<void(OperatorPtr, OutHandler)>;
using CreateDynInputOpFunc = std::function<void(OperatorPtr, unsigned int)>;
using DynInputOpFunc = std::function<void(OperatorPtr, unsigned int, OperatorPtr)>;
using DynInputHandleFunc = std::function<void(OperatorPtr, unsigned int, OutHandler)>;
using UpdateOutputDescFunc = std::function<void(OperatorPtr, GeTensorDesc)>;
using CreateDynOutputOpFunc = std::function<void(OperatorPtr, unsigned int)>;
using CreateDynSubGraphFunc = std::function<void(OperatorPtr, unsigned int)>;
using DynSubGraphFunc = std::function<void(OperatorPtr, unsigned int, DfGraphPtr)>;

struct AttrDesc {
  std::string name;
  AttrFunc set_attr;
};

struct InputDesc {
  std::string name;
  InputOpFunc set_op;
  InputHandleFunc set_handle;
  UpdateOutputDescFunc update_input_desc;
};

struct DynInputDesc {
  std::string name;
  CreateDynInputOpFunc create_dyn_input;
  DynInputOpFunc set_op;
  DynInputHandleFunc set_handle;
};

struct DynSubGraphDesc {
  std::string name;
  CreateDynSubGraphFunc create_dyn_subgraph;
  DynSubGraphFunc set_subgraph;
};

struct OutputDesc {
  std::string name;
  UpdateOutputDescFunc update_out_desc;
};

struct DynOutputDesc {
  std::string name;
  CreateDynOutputOpFunc create_dyn_output;
};

class BaseOpAdapter {
 public:
  virtual ~BaseOpAdapter() {}
  virtual OperatorPtr generate(const AnfNodePtr &anf) = 0;
  virtual OperatorPtr generate(const std::string &type) { return std::make_shared<ge::Operator>(type); }
  virtual int setSubgraph(const OperatorPtr &op, int index, std::shared_ptr<std::vector<DfGraph>> branches) = 0;
  virtual int setInput(const OperatorPtr &op, int index, const OperatorPtr &input) = 0;
  virtual int setInput(const OperatorPtr &op, int index, const OutHandler &handle) = 0;
  virtual int setInput(const OperatorPtr &op, int index,
                       const std::shared_ptr<std::vector<OutHandler>> &handler_vec) = 0;
  virtual int setAttr(const OperatorPtr &op, const std::string &attrKey, const ValuePtr &attrValue) = 0;
  virtual int setAttr(const OperatorPtr &op, const PrimitivePtr &prim) = 0;
  virtual int setAttr(const OperatorPtr &op, const AnfNodePtr &node) = 0;
  virtual std::unordered_map<std::string, ValuePtr> GetExtraAttr() = 0;
  template <typename T, typename _ = typename std::enable_if<!std::is_base_of<Value, T>::value>::type>
  int setAttr(const OperatorPtr &op, const std::string &attrKey, const std::shared_ptr<T> &attrValue) {
    return setAttr(op, attrKey, MakeValue(attrValue));
  }
  template <typename T, typename _ = typename std::enable_if<!is_shared_ptr<T>::value>::type>
  int setAttr(const OperatorPtr &op, const std::string &attrKey, const T &attrValue) {
    return setAttr(op, attrKey, MakeValue(attrValue));
  }
  virtual OutHandler getOutput(const OperatorPtr &op, int index) = 0;
  virtual void updateOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type,
                                const AnfNodePtr &node) = 0;
  virtual const std::unordered_map<int, InputDesc> &getInputMap() = 0;
  virtual const std::unordered_map<unsigned int, AttrDesc> &getInputAttrMap() = 0;
  virtual const std::unordered_map<int, DynInputDesc> &getDynInputMap() = 0;
  virtual const std::unordered_map<int, OutputDesc> &getOutputMap() = 0;
  virtual const std::unordered_map<int, DynSubGraphDesc> &getDynSubgraphMap() = 0;
  void AddAttrToDrawGraph(const std::string &attr_str) { attrs_vec_.push_back(attr_str); }
  const std::vector<std::string> &GetAttrsFromDrawGraph() const { return attrs_vec_; }
  void clearAttrVect() { attrs_vec_.clear(); }

 private:
  std::vector<std::string> attrs_vec_;
};

using OpAdapterPtr = std::shared_ptr<BaseOpAdapter>;

enum AttrType {
  ATTR_INT = 0,
  ATTR_FLOAT,
  ATTR_DOUBLE,
  ATTR_STRING,
  ATTR_TENSOR,
  ATTR_BOOL,
  ATTR_LIST_INT,
  ATTR_LIST_ANY_INT,
  ATTR_ENUM
};

struct GeEnum {};
struct TFType {};
struct GEType {};

// declare Any type
template <typename T>
struct AnyTraits {
  using type = T;
};

template <>
struct AnyTraits<int> {
  using type = int64_t;
};

using ExtraAttr = std::unordered_map<std::string, ValuePtr>;
}  // namespace transform
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_BASE_H_
