/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include <string>
#include <memory>
#include <utility>
#include <vector>
#include <sstream>
#include <map>

#include "utils/hash_map.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "graph/operator_reg.h"
#include "ge/ge_api.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "mindapi/base/format.h"
#include "plugin/ascend/res_manager/visible.h"

namespace ge {
class CustomOperator : public Operator {
 public:
  CustomOperator(const string &name, const string &type) : Operator(name, type) {}

  ~CustomOperator() override{};

  void CustomInputRegister(const string &name) { Operator::InputRegister(name); }

  void CustomOptionalInputRegister(const string &name) { Operator::OptionalInputRegister(name); }

  void CustomOutputRegister(const string &name) { Operator::OutputRegister(name); }

  void CustomRequiredAttrRegister(const string &name) { Operator::RequiredAttrRegister(name); }

  void CustomInferFuncRegister(const std::function<graphStatus(Operator &)> &func) {
    Operator::InferFuncRegister(func);
  }
};
}  // namespace ge
using GeTensor = ::ge::Tensor;
namespace mindspore::device::ascend {
enum Status : int { SUCCESS = 0, FAILED, INVALID_ARGUMENT, ALREADY_EXISTS, NOT_FOUND };

using MeTensor = mindspore::tensor::Tensor;
using MeTensorPtr = std::shared_ptr<MeTensor>;
using MeDataType = mindspore::TypeId;
using GeDataType = ::ge::DataType;
using GeFormat = ::ge::Format;
using GeShape = ::ge::Shape;
using GeTensorPtr = std::shared_ptr<GeTensor>;
using GeTensorDesc = ::ge::TensorDesc;
using AnfGraph = FuncGraph;
using AnfGraphPtr = FuncGraphPtr;
using Operator = ::ge::Operator;
using OperatorPtr = std::shared_ptr<::ge::Operator>;
using DfGraph = ::ge::Graph;
using DfGraphPtr = std::shared_ptr<DfGraph>;
using TensorMap = mindspore::HashMap<std::string, std::shared_ptr<MeTensor>>;
using OptionMap = std::map<std::string, std::string>;
using TensorOrderMap = std::map<std::string, std::shared_ptr<tensor::Tensor>>;
using GeAllocatorPtr = ::ge::AllocatorPtr;

struct OutHandler {
  OperatorPtr op;
  std::string out;
  AnfNodePtr node;
  OutHandler() : op(nullptr), out(""), node(nullptr) {}
  OutHandler(const OperatorPtr &op, const std::string out, const AnfNodePtr &node = nullptr)
      : op(op), out(out), node(node) {}
};

using CusOperatorPtr = std::shared_ptr<::ge::CustomOperator>;
using CustomOperator = ::ge::CustomOperator;
using AttrFunc = std::function<void(OperatorPtr, ValuePtr)>;
using GetAttrFunc = std::function<void(OperatorPtr, ValuePtr *)>;
using OutputFunc = std::function<OutHandler(OperatorPtr)>;
using InputOpFunc = std::function<void(OperatorPtr, OperatorPtr)>;
using InputHandleFunc = std::function<void(OperatorPtr, OutHandler)>;
using CreateDynInputOpFunc = std::function<void(OperatorPtr, unsigned int)>;
using CreateDynInputOpByIndexFunc = std::function<void(OperatorPtr, unsigned int, size_t)>;
using DynInputOpFunc = std::function<void(OperatorPtr, unsigned int, OperatorPtr)>;
using DynInputHandleFunc = std::function<void(OperatorPtr, unsigned int, OutHandler)>;
using UpdateOutputDescFunc = std::function<void(OperatorPtr, GeTensorDesc)>;
using CreateDynOutputOpFunc = std::function<void(OperatorPtr, unsigned int)>;
using UpdateDynOutputDescFunc = std::function<void(OperatorPtr, unsigned int, GeTensorDesc)>;
using SubGraphFunc = std::function<void(OperatorPtr, DfGraphPtr)>;
using CreateDynSubGraphFunc = std::function<void(OperatorPtr, unsigned int)>;

using DynSubGraphFunc = std::function<void(OperatorPtr, unsigned int, DfGraphPtr)>;

static HashMap<GeDataType, std::string> ge_dtype_str_map = {
  {GeDataType::DT_FLOAT, "float"},
  {GeDataType::DT_FLOAT16, "float16"},
  {GeDataType::DT_INT8, "int8"},
  {GeDataType::DT_INT16, "int16"},
  {GeDataType::DT_UINT16, "uint16"},
  {GeDataType::DT_UINT8, "uint8"},
  {GeDataType::DT_INT32, "int32"},
  {GeDataType::DT_INT64, "int64"},
  {GeDataType::DT_UINT32, "uint32"},
  {GeDataType::DT_UINT64, "uint64"},
  {GeDataType::DT_BOOL, "bool"},
  {GeDataType::DT_DOUBLE, "double"},
  {GeDataType::DT_STRING, "string"},
  {GeDataType::DT_DUAL_SUB_INT8, "dual_sub_int8"},
  {GeDataType::DT_DUAL_SUB_UINT8, "dual_sub_uint8"},
  {GeDataType::DT_COMPLEX64, "complex64"},
  {GeDataType::DT_COMPLEX128, "complex128"},
  {GeDataType::DT_DUAL, "dual"},
  {GeDataType::DT_QINT8, "qint8"},
  {GeDataType::DT_QINT16, "qint16"},
  {GeDataType::DT_QINT32, "qint32"},
  {GeDataType::DT_QUINT8, "quint8"},
  {GeDataType::DT_QUINT16, "quint16"},
  {GeDataType::DT_RESOURCE, "resource"},
  {GeDataType::DT_STRING_REF, "string ref"},
  {GeDataType::DT_VARIANT, "dt_variant"},
  {GeDataType::DT_UNDEFINED, "undefined"},
  {GeDataType::DT_INT4, "int4"},
  {GeDataType::DT_UINT1, "uint1"},
  {GeDataType::DT_INT2, "int2"},
  {GeDataType::DT_UINT2, "uint2"},
  {GeDataType::DT_COMPLEX32, "complex32"},
  {GeDataType::DT_BF16, "bf16"},
  {GeDataType::DT_HIFLOAT8, "hifloat8"},
  {GeDataType::DT_FLOAT8_E5M2, "float8_e5m2"},
  {GeDataType::DT_FLOAT8_E4M3FN, "float8_e4m3fn"},
};

struct AttrDesc {
  std::string name;
  AttrFunc set_attr;
  AttrFunc set_attr_with_ref{nullptr};
  GetAttrFunc get_attr;
  enum {
    REQUIRED = 0,
    OPTIONAL = 1,
    DEFAULT = OPTIONAL,
  } type = DEFAULT;
};

struct InputDesc {
  std::string name;
  size_t index;
  InputOpFunc set_op;
  InputHandleFunc set_handle;
  UpdateOutputDescFunc update_input_desc;
  enum {
    REQUIRED = 0,
    OPTIONAL = 1,
    DEFAULT = REQUIRED,
  } type = DEFAULT;
  std::vector<enum ::ge::DataType> supported_dtypes;
};

struct DynInputDesc {
  std::string name;
  size_t index;
  CreateDynInputOpFunc create_dyn_input;
  CreateDynInputOpByIndexFunc create_dyn_input_by_index;
  DynInputOpFunc set_op;
  DynInputHandleFunc set_handle;
  std::vector<enum ::ge::DataType> supported_dtypes;
};

struct SubGraphDesc {
  std::string name;
  SubGraphFunc set_subgraph;
};

struct DynSubGraphDesc {
  std::string name;
  CreateDynSubGraphFunc create_dyn_subgraph;
  DynSubGraphFunc set_subgraph;
};

struct OutputDesc {
  std::string name;
  size_t index;
  UpdateOutputDescFunc update_out_desc;
  std::vector<enum ::ge::DataType> supported_dtypes;
};

struct DynOutputDesc {
  std::string name;
  size_t index;
  CreateDynOutputOpFunc create_dyn_output;
  UpdateDynOutputDescFunc update_dyn_output_desc;
  std::vector<enum ::ge::DataType> supported_dtypes;
};

class ASCEND_RES_MANAGER_EXPORT BaseOpAdapter {
 public:
  virtual ~BaseOpAdapter() {}
  virtual OperatorPtr generate(const AnfNodePtr &anf) = 0;
  virtual OperatorPtr generate(const std::string &type) const { return std::make_shared<::ge::Operator>(type); }
  virtual OperatorPtr generateDynOutputOp(const AnfNodePtr &anf) { return nullptr; }
  virtual void setDynamicOutputNum(const OperatorPtr &op, size_t dyn_output_size) { return; }
  virtual void setSubgraph(const OperatorPtr &op, std::shared_ptr<std::vector<DfGraph>> subgraphs) = 0;
  virtual void setSubgraph(const OperatorPtr &op, int index, const std::shared_ptr<std::vector<DfGraph>> &branches) = 0;
  virtual int setInput(const OperatorPtr &op, int index, const OperatorPtr &input) = 0;
  virtual int setInput(const OperatorPtr &op, int index, const OutHandler &handle) = 0;
  virtual int setInput(const OperatorPtr &op, int index, const std::shared_ptr<std::vector<OutHandler>> &handler_vec,
                       bool use_create_byindex_func = false, size_t dyn_index = 0) = 0;
  virtual int setAttr(const OperatorPtr &op, const std::string &attrKey, const ValuePtr &attrValue,
                      bool is_ref = false) = 0;
  virtual int setAttr(const OperatorPtr &op, const PrimitivePtr &prim) = 0;
  virtual int setAttr(const OperatorPtr &op, const AnfNodePtr &node) = 0;
  virtual int setAttr(const std::string &attrKey, const ValuePtr &attrValue) = 0;
  virtual int setAttr(const uint32_t &input_idx, const ValuePtr &attrValue) = 0;
  virtual int getAttr(const std::string &attrKey, ValuePtr *attrValue) = 0;
  virtual int getAttr(const uint32_t &input_idx, ValuePtr *attrValue) = 0;
  virtual mindspore::HashMap<std::string, ValuePtr> GetExtraAttr() = 0;
  template <typename T, typename _ = typename std::enable_if<!std::is_base_of<Value, T>::value>::type>
  int setAttr(const OperatorPtr &op, const std::string &attrKey, const std::shared_ptr<T> &attrValue) {
    return setAttr(op, attrKey, MakeValue(attrValue));
  }
  template <typename T, typename _ = typename std::enable_if<!is_shared_ptr<T>::value>::type>
  int setAttr(const OperatorPtr &op, const std::string &attrKey, const T &attrValue) {
    return setAttr(op, attrKey, MakeValue(attrValue));
  }
  virtual std::string getOpType() = 0;
  virtual OutHandler getOutput(const OperatorPtr &op, int index) = 0;
  virtual std::vector<OutHandler> getOutputs(const OperatorPtr &op) = 0;
  virtual void updateOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type,
                                const AnfNodePtr &node) = 0;
  virtual const mindspore::HashMap<int, InputDesc> &getInputMap() = 0;
  virtual const mindspore::HashMap<unsigned int, AttrDesc> &getInputAttrMap() = 0;
  virtual const mindspore::HashMap<std::string, AttrDesc> &getAttrMap() = 0;
  virtual const mindspore::HashMap<std::string, std::string> &getAttrInputMap() = 0;
  virtual const mindspore::HashMap<int, DynInputDesc> &getDynInputMap() = 0;
  virtual const std::map<int, OutputDesc> &getOutputMap() = 0;
  virtual const mindspore::HashMap<int, DynOutputDesc> &getDynOutputMap() = 0;
  virtual const mindspore::HashMap<int, SubGraphDesc> &getSubgraphMap() = 0;
  virtual const mindspore::HashMap<int, DynSubGraphDesc> &getDynSubgraphMap() = 0;
  virtual std::map<std::string, ValuePtr> GetNormalOpAttrList(const AnfNodePtr &node) = 0;
  virtual std::map<std::string, ValuePtr> GetOpAttrList() = 0;
  virtual bool IsDynInputOp(uint64_t index) = 0;
  virtual bool IsDyOutputOp(uint64_t index) = 0;
  virtual bool IsMultipleOutputOp(const AnfNodePtr &anf) = 0;
  virtual bool GetDynamicShapeSupport() = 0;
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
struct GEEnumToStr {};

class GEDataFormat {
 public:
  static std::string ConvertEnumToString(int64_t id) {
    const auto &enum_string = FormatEnumToString(static_cast<mindspore::Format>(id));
    if (enum_string.empty()) {
      MS_LOG(EXCEPTION) << "Invalid data format " << id;
    }
    return enum_string;
  }
};

class AscendQuantRoundMode {
 public:
  static std::string ConvertEnumToString(int64_t id) {
    static const std::vector<std::string> round_mode = {"round", "trunc", "floor", "ceil"};
    if (id < 0 || id >= static_cast<int64_t>(round_mode.size())) {
      MS_LOG(EXCEPTION) << "Invalid AscendQuant round mode " << id;
      return "";
    }
    return round_mode[id];
  }
};

class ASCEND_RES_MANAGER_EXPORT FASInputLayoutMode {
 public:
  static std::string ConvertEnumToString(int64_t id) {
    static const std::vector<std::string> input_layout_modes = {"BSH", "BNSD", "SBH", "BSND",     "TND",
                                                                "TH",  "NSD",  "SH",  "BNSD_BSND"};
    if (id < 0 || id >= static_cast<int64_t>(input_layout_modes.size())) {
      MS_LOG(EXCEPTION) << "Invalid input layout mode " << id;
      return "";
    }
    return input_layout_modes[id];
  }
};

class ASCEND_RES_MANAGER_EXPORT DequantSwigluQuantInputQuantMode {
 public:
  static std::string ConvertEnumToString(int64_t id) {
    static const std::vector<std::string> input_layout_modes = {"static", "dynamic"};
    if (id < 0 || id >= static_cast<int64_t>(input_layout_modes.size())) {
      MS_LOG(EXCEPTION) << "Invalid input layout mode " << id;
      return "";
    }
    return input_layout_modes[id];
  }
};

class ASCEND_RES_MANAGER_EXPORT FFNActivationMode {
 public:
  static std::string ConvertEnumToString(int64_t id) {
    static const std::vector<std::string> activation_mode = {
      "no_activation", "relu", "sigmoid", "relu6",   "elu",      "leaky_relu",    "abs",    "relu1",     "softsign",
      "softplus",      "tanh", "selu",    "hswish",  "hsigmoid", "thresholdrelu", "linear", "hard_tanh", "sign",
      "swish",         "gelu", "glu",     "unknown", "fastgelu", "silu",          "geglu",  "swiglu",    "reglu"};
    if (id < 0 || id >= static_cast<int64_t>(activation_mode.size())) {
      MS_LOG(EXCEPTION) << "Invalid moe ffn activation " << id;
      return "";
    }
    return activation_mode[id];
  }
};

class ASCEND_RES_MANAGER_EXPORT ScatterReduceMode {
 public:
  static std::string ConvertEnumToString(int64_t id) {
    static const std::vector<std::string> reduce_mode = {"none", "add", "multiply", "update"};
    if (id < 0 || id >= static_cast<int64_t>(reduce_mode.size())) {
      MS_LOG(EXCEPTION) << "Invalid reduce mode " << id;
      return "";
    }
    return reduce_mode[id];
  }
};

class GEPadMod {
 public:
  static std::string ConvertEnumToString(int64_t id) {
    static const std::vector<std::string> pad_mods = {"PAD", "SAME", "VALID"};
    if (id < 0 || id >= static_cast<int64_t>(pad_mods.size())) {
      MS_LOG(EXCEPTION) << "Invalid pad mod " << id;
      return "";
    }
    return pad_mods[id];
  }
};

class ASCEND_RES_MANAGER_EXPORT GEReduction {
 public:
  static std::string ConvertEnumToString(int64_t id) {
    static const std::vector<std::string> reductions = {"sum", "mean", "none"};
    if (id < 0 || id >= static_cast<int64_t>(reductions.size())) {
      MS_LOG(EXCEPTION) << "Invalid reduction " << id;
      return "";
    }
    return reductions[id];
  }
};

class GECoordinateTransformMode {
 public:
  static std::string ConvertEnumToString(int64_t id) {
    static const std::vector<std::string> modes = {"asymmetric", "align_corners", "half_pixel", "crop_and_resize"};
    if (id < 0 || id >= static_cast<int64_t>(modes.size())) {
      MS_LOG(EXCEPTION) << "Invalid CoordinateTransformMode " << id;
      return "";
    }
    return modes[id];
  }
};

class GERotatedIouMode {
 public:
  static std::string ConvertEnumToString(int64_t id) {
    static const std::vector<std::string> modes = {"iou", "iof"};
    if (id < 0 || id >= static_cast<int64_t>(modes.size())) {
      MS_LOG(EXCEPTION) << "Invalid RotatedIouMode " << id;
      return "";
    }
    return modes[id];
  }
};

// declare Any type
template <typename T>
struct AnyTraits {
  using type = T;
};

template <>
struct AnyTraits<int> {
  using type = int64_t;
};

using ExtraAttr = mindspore::HashMap<std::string, ValuePtr>;
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_BASE_H_
