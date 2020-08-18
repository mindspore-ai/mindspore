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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MACRO_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MACRO_H_

#include <string>
#include <unordered_map>
#include <memory>
#include "transform/graph_ir/op_adapter.h"
#include "transform/graph_ir/op_adapter_map.h"
#include "mindspore/core/base/core_ops.h"

namespace mindspore::transform {
#define DECLARE_OP_ADAPTER(T)                                        \
  using T = ge::op::T;                                               \
  template <>                                                        \
  const std::unordered_map<int, InputDesc> OpAdapter<T>::input_map_; \
  template <>                                                        \
  const std::unordered_map<std::string, AttrDesc> OpAdapter<T>::attr_map_;

#define DECLARE_OP_USE_OUTPUT(T) \
  template <>                    \
  const std::unordered_map<int, OutputDesc> OpAdapter<T>::output_map_;

#define DECLARE_OP_USE_ENUM(T) \
  template <>                  \
  const std::unordered_map<std::string, int> OpAdapter<T>::enum_map_{};

#define DECLARE_OP_USE_INPUT_ATTR(T) \
  template <>                        \
  const std::unordered_map<unsigned int, AttrDesc> OpAdapter<T>::input_attr_map_;

#define DECLARE_OP_USE_DYN_INPUT(T) \
  template <>                       \
  const std::unordered_map<int, DynInputDesc> OpAdapter<T>::dyn_input_map_;

#define DECLARE_OP_USE_DYN_SUBGRAPH(T) \
  template <>                          \
  const std::unordered_map<int, DynSubGraphDesc> OpAdapter<T>::dyn_subgraph_map_;

#define DECLARE_OP_USE_DYN_OUTPUT(T) \
  template <>                        \
  const std::unordered_map<int, DynOutputDesc> OpAdapter<T>::dyn_output_map_;

#define INPUT_MAP(T) \
  template <>        \
  const std::unordered_map<int, InputDesc> OpAdapter<T>::input_map_
#define EMPTY_INPUT_MAP std::unordered_map<int, InputDesc>()
#define INPUT_DESC(name) \
  {                      \
#name, \
    [](const OperatorPtr op, const OperatorPtr input) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_input_##name(*input); \
    }, \
    [](const OperatorPtr op, const OutHandler& handle) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_input_##name(*(handle.op), handle.out); \
    }, \
    [](const OperatorPtr op, const GeTensorDesc desc) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->update_input_desc_##name(desc); \
    }                 \
  }

#define DYN_INPUT_MAP(T) \
  template <>            \
  const std::unordered_map<int, DynInputDesc> OpAdapter<T>::dyn_input_map_
#define DYN_INPUT_DESC(name) \
  {                          \
#name, \
    [](const OperatorPtr op, unsigned int num) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->create_dynamic_input_##name(num); \
    }, \
    [](const OperatorPtr op, unsigned int index, const OperatorPtr input) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_dynamic_input_##name(index, *input); \
    }, \
    [](const OperatorPtr op, unsigned int index, const OutHandler& handle) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_dynamic_input_##name(index, *(handle.op), handle.out); \
    }                     \
  }

#define DYN_SUBGRAPH_MAP(T) \
  template <>               \
  const std::unordered_map<int, DynSubGraphDesc> OpAdapter<T>::dyn_subgraph_map_
#define DYN_SUBGRAPH_DESC(name) \
  {                             \
#name, \
    [](const OperatorPtr op, unsigned int num) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->create_dynamic_subgraph_##name(num); \
    }, \
    [](const OperatorPtr op, unsigned int index, const DfGraphPtr graph) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_dynamic_subgraph_builder_##name(index, [graph](){return *graph;}); \
    }                        \
  }

#define ATTR_MAP(T) \
  template <>       \
  const std::unordered_map<std::string, AttrDesc> OpAdapter<T>::attr_map_
#define EMPTY_ATTR_MAP std::unordered_map<std::string, AttrDesc>()
#define ATTR_DESC(name, ...) \
  {                          \
#name, \
    [](const OperatorPtr op, const ValuePtr& value) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_attr_##name(ConvertAny(value, __VA_ARGS__)); \
    }                     \
  }

#define INPUT_ATTR_MAP(T) \
  template <>             \
  const std::unordered_map<unsigned int, AttrDesc> OpAdapter<T>::input_attr_map_

#define OUTPUT_MAP(T) \
  template <>         \
  const std::unordered_map<int, OutputDesc> OpAdapter<T>::output_map_
#define OUTPUT_DESC(name) \
  {                       \
#name, \
    [](const OperatorPtr op, const GeTensorDesc desc) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->update_output_desc_##name(desc); \
    }                  \
  }

#define DYN_OUTPUT_MAP(T) \
  template <>             \
  const std::unordered_map<int, DynOutputDesc> OpAdapter<T>::dyn_output_map_

#define DYN_OUTPUT_DESC(name) \
  {                           \
#name, \
    [](const OperatorPtr op, unsigned int num) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->create_dynamic_output_##name(num); \
    }                      \
  }

#define ADPT_DESC_ONE(T) std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<T>>())
#define ADPT_DESC_TWO(T, I) \
  std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<T>>(), std::make_shared<OpAdapter<I>>())
#define GET_MACRO(_1, _2, DESC, ...) DESC
#define ADPT_DESC(...) GET_MACRO(__VA_ARGS__, ADPT_DESC_TWO, ADPT_DESC_ONE, ...)(__VA_ARGS__)
#define REG_ADPT_DESC(name, name_str, adpt_desc)                       \
  static struct RegAdptDesc##name {                                    \
   public:                                                             \
    RegAdptDesc##name() { OpAdapterMap::get()[name_str] = adpt_desc; } \
  } g_reg_adpt_desc_##name;
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MACRO_H_
