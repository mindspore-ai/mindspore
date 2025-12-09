/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_PYNATIVE_FORWARD_PYBOOST_CONVERTER_H
#define MINDSPORE_CCSRC_INCLUDE_PYNATIVE_FORWARD_PYBOOST_CONVERTER_H
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <Python.h>
#include "pynative/utils/base.h"
#include "include/pynative/utils/pynative_execute.h"
#include "include/utils/tensor_py.h"
#include "ops/op_def.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace pynative {
using ConvertPair = std::pair<ops::OP_DTYPE, ops::OP_DTYPE>;
struct ParserArgs;

class CPythonTuple {
 public:
  using pybind_type = py::tuple;
  static bool TypeCheck(PyObject *obj) { return PyTuple_Check(obj); }
  static PyObject *GetItem(PyObject *obj, Py_ssize_t i) { return PyTuple_GetItem(obj, i); }
  static Py_ssize_t GetSize(PyObject *obj) { return PyTuple_Size(obj); }
};

class CPythonList {
 public:
  using pybind_type = py::list;
  static bool TypeCheck(PyObject *obj) { return PyList_Check(obj); }
  static PyObject *GetItem(PyObject *obj, Py_ssize_t i) { return PyList_GetItem(obj, i); }
  static Py_ssize_t GetSize(PyObject *obj) { return PyList_Size(obj); }
};

static std::unordered_map<std::string, ops::OP_DTYPE> type_str_map = {
  {"int", ops::OP_DTYPE::DT_INT},
  {"float", ops::OP_DTYPE::DT_FLOAT},
  {"bool", ops::OP_DTYPE::DT_BOOL},
  {"number", ops::OP_DTYPE::DT_NUMBER},
  {"tuple[int]", ops::OP_DTYPE::DT_TUPLE_INT},
  {"tuple[float]", ops::OP_DTYPE::DT_TUPLE_FLOAT},
  {"tuple[bool]", ops::OP_DTYPE::DT_TUPLE_BOOL},
  {"tuple[tensor]", ops::OP_DTYPE::DT_TUPLE_TENSOR},
  {"tuple[number]", ops::OP_DTYPE::DT_TUPLE_NUMBER},
  {"tuple[str]", ops::OP_DTYPE::DT_STR},
  {"list[int]", ops::OP_DTYPE::DT_LIST_INT},
  {"list[float]", ops::OP_DTYPE::DT_LIST_FLOAT},
  {"list[bool]", ops::OP_DTYPE::DT_LIST_BOOL},
  {"list[tensor]", ops::OP_DTYPE::DT_LIST_TENSOR},
  {"list[number]", ops::OP_DTYPE::DT_LIST_NUMBER},
  {"list[str]", ops::OP_DTYPE::DT_LIST_STR},
  {"tensor", ops::OP_DTYPE::DT_TENSOR},
  {"storage", ops::OP_DTYPE::DT_STORAGE},
  {"str", ops::OP_DTYPE::DT_STR},
  {"type", ops::OP_DTYPE::DT_TYPE},
};

static std::unordered_map<std::string, ops::OP_DTYPE> type_not_in_yaml_str_map = {
  {"tuple[any]", ops::OP_DTYPE::DT_TUPLE_ANY},
  {"list[any]", ops::OP_DTYPE::DT_LIST_ANY},
  {"any", ops::OP_DTYPE::DT_ANY},
};

class PYNATIVE_EXPORT ParserDefaultObjects {
 public:
  static ParserDefaultObjects &GetInstance();

  PyObject *Get(const std::string &default_str) {
    auto iter = objects_.find(default_str);
    if (iter != objects_.end()) {
      return *(iter->second);
    }
    MS_LOG(EXCEPTION) << "The default value should be initialized before being fetched.";
  }

  void Set(const ops::OP_DTYPE &type, const std::string &value, const std::string &kw_str) {
    objects_.try_emplace(kw_str, std::make_unique<PyObject *>(StrToPyObj(type, value)));
  }

  PyObject *StrToPyObj(const ops::OP_DTYPE &type, const std::string &str);

  void ClearRes() {
    for (const auto &pair : objects_) {
      PyObject *value = *(pair.second);
      if (value == nullptr) {
        continue;
      }
      Py_XDECREF(value);
    }
    objects_.clear();
  }

 private:
  ParserDefaultObjects() {}
  ~ParserDefaultObjects() = default;
  DISABLE_COPY_AND_ASSIGN(ParserDefaultObjects);
  std::unordered_map<std::string, std::unique_ptr<PyObject *>> objects_;
};

// information of single parameter
struct FunctionParameter {
  explicit FunctionParameter(const std::string &fmt, bool is_kw_only);
  bool Check(PyObject *obj, ConvertPair &convert_type, int &error_idx) const;
  bool TypeCheck(PyObject *obj, const ops::OP_DTYPE &type, int &idx, ConvertPair &convert_type) const;
  void SetDefaultObj(const std::string &str);
  PyObject *GetDefaultValue() { return ParserDefaultObjects::GetInstance().Get(default_str_); }

  ops::OP_DTYPE type_{ops::OP_DTYPE::DT_END};
  std::vector<ops::OP_DTYPE> cast_types_{};
  std::string default_str_{""};
  bool optional_{false};
  bool allow_none_{false};
  bool kw_only_{false};
  std::string name_;
  bool is_any_{false};
  bool allow_vararg_{false};
  bool allow_scalar_tensor_{true};
};

// single overload
struct PYNATIVE_EXPORT FunctionSignature {
  explicit FunctionSignature(const std::string &fmt, int index, const std::string &name);
  bool CheckParamValid(PyObject *obj, const FunctionParameter &param, bool raise_error, std::string *out_error_msg,
                       ConvertPair &convert_type, int &error_idx);
  bool Parse(PyObject *args, PyObject *kwargs, ParserArgs &parser_args, bool raise_error = false,
             std::string *out_error_msg = nullptr);
  bool RaiseParseKeywordArgsError(size_t nkwargs, bool raise_error, std::string *out_error_msg, size_t nargs,
                                  PyObject *kwargs);
  std::string ToString();

  std::string name_;
  std::vector<FunctionParameter> params_;
  size_t max_pos_args_;
  size_t max_args_;
  size_t min_args_;
  // e.g. allow input.reshape(1, 2, 3) parse as input.reshape((1, 2, 3))
  bool allow_int_as_list_;
  int index_;
};
using FunctionSignaturePtr = std::shared_ptr<FunctionSignature>;

struct PYNATIVE_EXPORT ParserArgs {
 public:
  explicit ParserArgs(const FunctionSignaturePtr &signature) : signature_(signature) {
    arg_list_.resize(signature->params_.size());
    src_types_.resize(signature->params_.size());
    dst_types_.resize(signature->params_.size());
  }
  ValuePtr ConvertByParseDtype(size_t index);
  void InsertInputTensor(size_t index, PyObject *input);
  void SetArg(PyObject *arg, const ConvertPair &convert_type, size_t index);
  void ClearArgs();
  const int &GetOvertLoadIndex() { return signature_->index_; }
  void PrintConvertError(size_t index);
  // convert to basic type
  std::vector<int64_t> ToBasicIntVector(size_t index);
  std::optional<std::vector<int64_t>> ToBasicIntVectorOptional(size_t index);
  int64_t ToBasicInt(size_t index);
  std::optional<int64_t> ToBasicIntOptional(size_t index);

  template <typename T>
  std::shared_ptr<T> Convert(size_t index) {
    if (index >= arg_list_.size()) {
      MS_LOG(EXCEPTION) << "Invalid index" << index << "for argument convert.";
    }
    auto convert = ConvertByParseDtype(index);
    if (convert != nullptr && convert->isa<T>()) {
      return convert->cast<std::shared_ptr<T>>();
    }
    PrintConvertError(index);
    return nullptr;
  }

  template <typename T>
  std::optional<std::shared_ptr<T>> ConvertOptional(size_t index) {
    if (index >= arg_list_.size()) {
      MS_LOG(EXCEPTION) << "Invalid index" << index << "for argument convert.";
    }
    PyObject *obj = arg_list_[index];
    if (obj == Py_None) {
      return std::nullopt;
    }
    return std::make_optional(Convert<T>(index));
  }

  void CheckHasFallback();
  void CheckHasFallback(PyObject *obj);

  bool has_fallback() const { return has_fallback_; }

  FunctionSignaturePtr signature_;
  std::vector<PyObject *> arg_list_;
  // {src_type , dst_type} for convert
  std::vector<ops::OP_DTYPE> src_types_;
  std::vector<ops::OP_DTYPE> dst_types_;
  bool has_fallback_{false};
};

// parser util
struct PYNATIVE_EXPORT PythonArgParser {
  explicit PythonArgParser(std::vector<std::string> fmts, const std::string &function_name);
  inline const ParserArgs Parse(PyObject *args, PyObject *kwargs, const bool &is_method);
  const std::vector<std::string> GetParseTypeListString(PyObject *args, PyObject *kwargs);
  std::string PrintParseError(PyObject *args, PyObject *kwargs, const bool &is_method);

 private:
  std::vector<FunctionSignaturePtr> signatures_;
  std::string function_name_;
  size_t max_args_;
};

inline const ParserArgs PythonArgParser::Parse(PyObject *args, PyObject *kwargs, const bool &is_method) {
  if (signatures_.size() == 1) {
    ParserArgs parser_args(signatures_[0]);
    signatures_[0]->Parse(args, kwargs, parser_args, true);
    parser_args.CheckHasFallback();
    return parser_args;
  }

  for (auto &signature : signatures_) {
    ParserArgs parser_args(signature);
    if (signature->Parse(args, kwargs, parser_args, false)) {
      parser_args.CheckHasFallback();
      return parser_args;
    }
  }
  MS_EXCEPTION(TypeError) << PrintParseError(args, kwargs, is_method);
}

PYNATIVE_EXPORT ValuePtr UnpackTensor(PyObject *input, const std::string &func_name);

class PYNATIVE_EXPORT Converter {
 public:
  explicit Converter(ops::OpDef *op_def);
  void Parse(PyObject *python_args);
  ValuePtr ToTensor(PyObject *python_args, size_t i);
  std::optional<ValuePtr> ToTensorOptional(PyObject *python_args, size_t i);
  template <typename T>
  ValueTuplePtr ToTensorList(PyObject *python_args, size_t i);
  template <typename T>
  std::optional<ValueTuplePtr> ToTensorListOptional(PyObject *python_args, size_t i);
  Int64ImmPtr ToInt(PyObject *python_args, size_t i);
  std::optional<Int64ImmPtr> ToIntOptional(PyObject *python_args, size_t i);
  template <typename T>
  ValueTuplePtr ToIntList(PyObject *python_args, size_t i);
  template <typename T>
  std::optional<ValueTuplePtr> ToIntListOptional(PyObject *python_args, size_t i);
  BoolImmPtr ToBool(PyObject *python_args, size_t i);
  std::optional<BoolImmPtr> ToBoolOptional(PyObject *python_args, size_t i);
  template <typename T>
  ValueTuplePtr ToBoolList(PyObject *python_args, size_t i);
  template <typename T>
  std::optional<ValueTuplePtr> ToBoolListOptional(PyObject *python_args, size_t i);
  FP32ImmPtr ToFloat(PyObject *python_args, size_t i);
  std::optional<FP32ImmPtr> ToFloatOptional(PyObject *python_args, size_t i);
  template <typename T>
  ValueTuplePtr ToFloatList(PyObject *python_args, size_t i);
  template <typename T>
  std::optional<ValueTuplePtr> ToFloatListOptional(PyObject *python_args, size_t i);
  ScalarPtr ToScalar(PyObject *python_args, size_t i);
  std::optional<ScalarPtr> ToScalarOptional(PyObject *python_args, size_t i);
  StringImmPtr ToString(PyObject *python_args, size_t i);
  std::optional<StringImmPtr> ToStringOptional(PyObject *python_args, size_t i);
  Int64ImmPtr ToDtype(PyObject *python_args, size_t i);
  std::optional<Int64ImmPtr> ToDtypeOptional(PyObject *python_args, size_t i);
  ValuePtr ConvertByCastDtype(PyObject *input, const ops::OpInputArg &op_arg, size_t i);
  ValueTuplePtr ConvertValueTupleByCastDtype(PyObject *python_args, const ops::OpInputArg &op_arg, size_t index);
  std::vector<int64_t> ConvertIntVectorByCastDtype(PyObject *python_args, const ops::OpInputArg &op_arg, size_t index);
  int64_t ConvertIntByCastDtype(PyObject *python_args, const ops::OpInputArg &op_arg, size_t index);
  const std::vector<ops::OP_DTYPE> &source_type() const { return source_type_; }
  // basic type
  int64_t ToBasicInt(PyObject *python_args, size_t i);
  std::optional<int64_t> ToBasicIntOptional(PyObject *python_args, size_t i);
  template <typename T>
  std::vector<int64_t> ToBasicIntVector(PyObject *python_args, size_t i);
  template <typename T>
  std::optional<std::vector<int64_t>> ToBasicIntVectorOptional(PyObject *python_args, size_t i);
  bool has_fallback() const { return has_fallback_; }

 private:
  ops::OpDefPtr op_def_;
  // If op not type cast, source_type is default type: DT_BEGIN, if op type cast, source_type is origin type.
  std::vector<ops::OP_DTYPE> source_type_;
  bool has_fallback_{false};
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_PYNATIVE_FORWARD_PYBOOST_CONVERTER_H
