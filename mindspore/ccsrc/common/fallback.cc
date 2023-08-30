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

#include "include/common/fallback.h"

#include <queue>

#include "include/common/utils/python_adapter.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/phase.h"

namespace mindspore {
namespace fallback {
static std::queue<py::object> py_execute_output_queue = std::queue<py::object>();

bool HasPyExecuteOutput() { return !py_execute_output_queue.empty(); }

py::object PopPyExecuteOutput() {
  auto output = py_execute_output_queue.front();
  MS_LOG(DEBUG) << "output: " << output;
  py_execute_output_queue.pop();
  return output;
}

void PushPyExecuteOutput(const py::object &output) {
  MS_LOG(DEBUG) << "output: " << output;
  py_execute_output_queue.push(output);
}

int GetJitSyntaxLevel() {
  // Get jit_syntax_level from environment variable 'MS_DEV_JIT_SYNTAX_LEVEL'.
  std::string env_level_str = common::GetEnv("MS_DEV_JIT_SYNTAX_LEVEL");
  if (env_level_str.size() == 1) {
    int env_level = -1;
    try {
      env_level = std::stoi(env_level_str);
    } catch (const std::invalid_argument &ia) {
      MS_LOG(EXCEPTION) << "Invalid argument: " << ia.what() << " when parse " << env_level_str;
    }
    if (env_level >= kStrict && env_level <= kLax) {
      return env_level;
    }
  }
  if (!env_level_str.empty()) {
    MS_LOG(EXCEPTION) << "JIT syntax level should be a number and from 0 to 2, but got " << env_level_str;
  }

  // Get jit_syntax_level from jit_config, default to an empty string.
  const auto &jit_config = PhaseManager::GetInstance().jit_config();
  auto iter = jit_config.find("jit_syntax_level");
  if (iter != jit_config.end()) {
    auto level = iter->second;
    if (level == "STRICT") {
      return kStrict;
    } else if (level == "COMPATIBLE") {
      return kCompatible;
    } else if (level == "LAX") {
      return kLax;
    }
  }
  // Get jit_syntax_level from context.
  return MsContext::GetInstance()->get_param<int>(MS_CTX_JIT_SYNTAX_LEVEL);
}

template <typename T>
bool CheckSequenceElementSame(const py::sequence &obj) {
  // Check from second element, the type of first element is determined by T.
  for (size_t i = 1; i < py::len(obj); ++i) {
    if (!py::isinstance<T>(obj[i])) {
      return false;
    }
  }
  return true;
}

bool CheckSequenceToMemory(const py::sequence &obj) {
  // A sequence object can be passed to raw memory and used by other operator if:
  //   1. The length of sequence is not empty.
  //   2. The sequence is not nested.
  //   3. The sequence only contains Scalar or Tensor elements.
  //   4. All the elements in sequence should be the same.
  if (py::len(obj) == 0) {
    return false;
  }
  auto first_obj = obj[0];
  if (py::isinstance<py::bool_>(first_obj)) {
    return CheckSequenceElementSame<py::bool_>(obj);
  } else if (py::isinstance<py::int_>(first_obj)) {
    return CheckSequenceElementSame<py::int_>(obj);
  } else if (py::isinstance<py::float_>(first_obj)) {
    return CheckSequenceElementSame<py::float_>(obj);
  } else if (py::isinstance<tensor::Tensor>(first_obj)) {
    return CheckSequenceElementSame<tensor::Tensor>(obj);
  }
  return false;
}

TypePtrList GetTypeElements(const TypePtr &type) {
  if (type->isa<List>()) {
    auto type_list = type->cast_ptr<List>();
    return type_list->elements();
  }
  auto type_tuple = type->cast_ptr<Tuple>();
  MS_EXCEPTION_IF_NULL(type_tuple);
  return type_tuple->elements();
}

abstract::AbstractSequencePtr GenerateAbstractSequence(const BaseShapePtr &base_shape, const TypePtr &type,
                                                       bool is_frontend) {
  // Generate AbstractSequence for PyExecute node.
  MS_EXCEPTION_IF_NULL(base_shape);
  MS_EXCEPTION_IF_NULL(type);
  bool is_list = base_shape->isa<abstract::ListShape>() && type->isa<List>();
  bool is_tuple = base_shape->isa<abstract::TupleShape>() && type->isa<Tuple>();
  if (!is_list && !is_tuple) {
    MS_INTERNAL_EXCEPTION(TypeError) << "For GenerateAbstractSequence, the input shape and type should be both "
                                     << "list or tuple, but got shape: " << base_shape->ToString()
                                     << " and type: " << type->ToString();
  }
  auto shape_seq = base_shape->cast_ptr<abstract::SequenceShape>();
  MS_EXCEPTION_IF_NULL(shape_seq);
  const auto &type_elements = GetTypeElements(type);
  if (shape_seq->size() != type_elements.size()) {
    MS_INTERNAL_EXCEPTION(ValueError) << "For GenerateAbstractSequence, the shape and type size should be the same, "
                                      << "but got shape size: " << shape_seq->size()
                                      << " and type size: " << type_elements.size();
  }
  AbstractBasePtrList ptr_list;
  for (size_t it = 0; it < shape_seq->size(); ++it) {
    auto element_shape = (*shape_seq)[it];
    auto element_type = type_elements[it];
    bool is_external = element_type->isa<External>();
    bool is_tensor_or_scalar = element_type->isa<Number>() || element_type->isa<TensorType>();
    if (!is_external && is_tensor_or_scalar) {
      (void)ptr_list.emplace_back(abstract::MakeAbstract(element_shape, element_type));
    } else {
      if (is_frontend) {
        (void)ptr_list.emplace_back(std::make_shared<abstract::AbstractAny>());
      } else {
        // In backend, the type is correctly fixed and the shape should be fixed.
        const auto &infer_shape = std::make_shared<abstract::Shape>(ShapeVector({1}));
        (void)ptr_list.emplace_back(abstract::MakeAbstract(infer_shape, kFloat64));
      }
    }
  }
  if (!is_frontend || is_tuple) {
    return std::make_shared<abstract::AbstractTuple>(ptr_list);
  }
  return std::make_shared<abstract::AbstractList>(ptr_list);
}
}  // namespace fallback
}  // namespace mindspore
