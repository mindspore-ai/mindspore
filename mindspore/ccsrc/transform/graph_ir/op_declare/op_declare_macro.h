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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MACRO_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MACRO_H_

#include <string>
#include <memory>
#include <map>
#include "ir/value.h"
#include "utils/hash_map.h"
#include "transform/graph_ir/op_adapter.h"
#include "transform/graph_ir/op_adapter_desc.h"
#include "transform/graph_ir/op_adapter_map.h"
#include "transform/graph_ir/op_declare/op_proto.h"

#define DECLARE_CANN_OP_PROTO(T)                               \
  namespace ge {                                               \
  mindspore::transform::OpProto &Get##T##OpProto();            \
  }                                                            \
  namespace mindspore::transform {                             \
  inline const auto &k##T##op_proto = ::ge::Get##T##OpProto(); \
  }

#define DECLARE_OP_ADAPTER(T)                                              \
  DECLARE_CANN_OP_PROTO(T)                                                 \
  namespace mindspore::transform {                                         \
  class cz##T {};                                                          \
  using T = cz##T;                                                         \
  template <>                                                              \
  const mindspore::HashMap<int, InputDesc> OpAdapter<T>::input_map_;       \
  template <>                                                              \
  const mindspore::HashMap<std::string, AttrDesc> OpAdapter<T>::attr_map_; \
  template <>                                                              \
  const char OpAdapter<T>::op_type_[] = #T;                                \
  }

#define DECLARE_OP_USE_OUTPUT(T)                             \
  namespace mindspore::transform {                           \
  template <>                                                \
  const std::map<int, OutputDesc> OpAdapter<T>::output_map_; \
  }

#define DECLARE_OP_USE_SUBGRAPH(T)                                         \
  namespace mindspore::transform {                                         \
  template <>                                                              \
  const mindspore::HashMap<int, SubGraphDesc> OpAdapter<T>::subgraph_map_; \
  }

#define DECLARE_OP_USE_INPUT_ATTR(T)                                              \
  namespace mindspore::transform {                                                \
  template <>                                                                     \
  const mindspore::HashMap<unsigned int, AttrDesc> OpAdapter<T>::input_attr_map_; \
  }

#define DECLARE_OP_USE_DYN_INPUT(T)                                         \
  namespace mindspore::transform {                                          \
  template <>                                                               \
  const mindspore::HashMap<int, DynInputDesc> OpAdapter<T>::dyn_input_map_; \
  }

#define DECLARE_OP_USE_DYN_SUBGRAPH(T)                                            \
  namespace mindspore::transform {                                                \
  template <>                                                                     \
  const mindspore::HashMap<int, DynSubGraphDesc> OpAdapter<T>::dyn_subgraph_map_; \
  }

#define DECLARE_OP_USE_DYN_OUTPUT(T)                                          \
  namespace mindspore::transform {                                            \
  template <>                                                                 \
  const mindspore::HashMap<int, DynOutputDesc> OpAdapter<T>::dyn_output_map_; \
  }

#define SUBGRAPH_MAP(T) \
  template <>           \
  const mindspore::HashMap<int, SubGraphDesc> OpAdapter<T>::subgraph_map_
#define SUBGRAPH_DESC(name)                                                                                       \
  {                                                                                                               \
#name, [](const OperatorPtr op,                                                                               \
              const DfGraphPtr graph) { (void)op->SetSubgraphBuilder(#name, 0, ([graph]() { return *graph; })); } \
  }

#define INPUT_INDEX(x) \
  OpProtoStorage::GetInstance().GetOpProto(OpAdapter<OpAdapter::OpType>::GetStaticOpType()).GetInputIndexByName(#x)
#define OUTPUT_INDEX(x) \
  OpProtoStorage::GetInstance().GetOpProto(OpAdapter<OpAdapter::OpType>::GetStaticOpType()).GetOutputIndexByName(#x)
#define INPUT_OPTIONAL_TYPE(x)                                   \
  OpProtoStorage::GetInstance()                                  \
    .GetOpProto(OpAdapter<OpAdapter::OpType>::GetStaticOpType()) \
    .IsInputOptionalTypeByName(#x)
#define INPUT_DTYPES(x) \
  OpProtoStorage::GetInstance().GetOpProto(OpAdapter<OpAdapter::OpType>::GetStaticOpType()).GetInputTypesByName(#x)
#define OUTPUT_DTYPES(x) \
  OpProtoStorage::GetInstance().GetOpProto(OpAdapter<OpAdapter::OpType>::GetStaticOpType()).GetOutputTypesByName(#x)
#define ATTR_OPTIONAL_TYPE(x) \
  OpProtoStorage::GetInstance().GetOpProto(OpAdapter<OpAdapter::OpType>::GetStaticOpType()).IsAttrOptionalTypeByName(#x)

#define INPUT_MAP(T) \
  template <>        \
  const mindspore::HashMap<int, InputDesc> OpAdapter<T>::input_map_
#define EMPTY_INPUT_MAP mindspore::HashMap<int, InputDesc>()
#define INPUT_DESC(name)                                                                                         \
  {                                                                                                              \
#name, INPUT_INDEX(name),                                                                                      \
      [](const OperatorPtr op, const OperatorPtr input) { (void)op->SetInput(#name, (*input)); },                  \
      [](const OperatorPtr op, const OutHandler &handle) { (void)op->SetInput(#name, *(handle.op), handle.out); }, \
      [](const OperatorPtr op, const GeTensorDesc desc) { (void)op->UpdateInputDesc(#name, (desc)); },             \
      (INPUT_OPTIONAL_TYPE(name) ? InputDesc::OPTIONAL : InputDesc::DEFAULT), INPUT_DTYPES(name), \
  }

#define DYN_INPUT_MAP(T) \
  template <>            \
  const mindspore::HashMap<int, DynInputDesc> OpAdapter<T>::dyn_input_map_
#define DYN_INPUT_DESC(name)                                                                            \
  {                                                                                                     \
#name, INPUT_INDEX(name),                                                                             \
      [](const OperatorPtr op, unsigned int num) {                                                        \
        (void)op->DynamicInputRegister(#name, (num),                                                      \
                                       "TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, " \
                                       "DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, "  \
                                       "DT_BOOL})",                                                       \
                                       true);                                                             \
      },                                                                                                  \
      [](const OperatorPtr op, unsigned int num, size_t index) {                                          \
        (void)op->DynamicInputRegisterByIndex(#name, num, index);                                         \
      },                                                                                                  \
      [](const OperatorPtr op, unsigned int index, const OperatorPtr input) {                             \
        (void) op->SetInput(#name, index, *input);                                                        \
      },                                                                                                  \
      [](const OperatorPtr op, unsigned int index, const OutHandler &handle) {                            \
        const std::string input_name = #name;                                                             \
        (void)op->SetInput(input_name.c_str(), index, *(handle.op), handle.out.c_str());                  \
      },                                                                                                  \
      INPUT_DTYPES(name), \
  }

#define DYN_SUBGRAPH_MAP(T) \
  template <>               \
  const mindspore::HashMap<int, DynSubGraphDesc> OpAdapter<T>::dyn_subgraph_map_
#define DYN_SUBGRAPH_DESC(name)                                                                         \
  {                                                                                                     \
#name, [](const OperatorPtr op, unsigned int num) { (void)op->SubgraphCountRegister(#name, (num)); }, \
      [](const OperatorPtr op, unsigned int index, const DfGraphPtr graph) {                              \
        (void)op->SetSubgraphBuilder(#name, index, [graph]() { return *graph; });                         \
      } \
  }

#define ATTR_MAP(T) \
  template <>       \
  const mindspore::HashMap<std::string, AttrDesc> OpAdapter<T>::attr_map_
#define EMPTY_ATTR_MAP mindspore::HashMap<std::string, AttrDesc>()
#define ATTR_DESC(name, ...)                                                                                         \
  {                                                                                                                  \
#name,                                                                                                             \
      [](const OperatorPtr op, const ValuePtr &value) { (void)op->SetAttr(#name, (ConvertAny(value, __VA_ARGS__))); }, \
      [](const OperatorPtr op, ValuePtr *ms_value) {                                                                   \
        if (ms_value == nullptr || *ms_value == nullptr) {                                                             \
          auto ret = GetAttrType(__VA_ARGS__);                                                                         \
          decltype(ret) attr_value;                                                                                    \
          (void)op->GetAttr(#name, attr_value);                                                                        \
          *ms_value = GetRealValue<decltype(attr_value)>(attr_value);                                                  \
        } else {                                                                                                       \
          auto real_value = ConvertAny(*ms_value, __VA_ARGS__);                                                        \
          *ms_value = GetRealValue<decltype(real_value)>(real_value);                                                  \
        }                                                                                                              \
      },                                                                                                               \
      (ATTR_OPTIONAL_TYPE(name) ? AttrDesc::OPTIONAL : AttrDesc::REQUIRED) \
  }

#define INPUT_ATTR_MAP(T) \
  template <>             \
  const mindspore::HashMap<unsigned int, AttrDesc> OpAdapter<T>::input_attr_map_

#define ATTR_INPUT_MAP(T) \
  template <>             \
  const mindspore::HashMap<std::string, std::string> OpAdapter<T>::attr_input_map_

#define OUTPUT_MAP(T) \
  template <>         \
  const std::map<int, OutputDesc> OpAdapter<T>::output_map_
#define EMPTY_OUTPUT_MAP std::map<int, OutputDesc>()
#define OUTPUT_DESC(name)                                                                            \
  {                                                                                                  \
#name, OUTPUT_INDEX(name),                                                                       \
      [](const OperatorPtr op,                                                                       \
         const GeTensorDesc desc) { (void)op->UpdateOutputDesc(#name, desc); }, OUTPUT_DTYPES(name), \
  }

#define DYN_OUTPUT_MAP(T) \
  template <>             \
  const mindspore::HashMap<int, DynOutputDesc> OpAdapter<T>::dyn_output_map_

#define DYN_OUTPUT_DESC(name)                                                                                    \
  {                                                                                                              \
#name, OUTPUT_INDEX(name),                                                                                     \
      [](const OperatorPtr op, unsigned int num) {                                                                 \
        (void)op->DynamicOutputRegister(#name, num,                                                                \
                                        "TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, "         \
                                        "DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, " \
                                        "DT_STRING})",                                                             \
                                        true);                                                                     \
      },                                                                                                           \
      [](const OperatorPtr op, uint32_t index, const GeTensorDesc tensor_desc) {                                   \
        (void)op->UpdateDynamicOutputDesc(#name, index, tensor_desc);                                              \
      },                                                                                                           \
      OUTPUT_DTYPES(name), \
  }

#define ADPT_DESC_ONE(T) std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<T>>(#T))
#define ADPT_DESC_TWO(T, I) \
  std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<T>>(#T), std::make_shared<OpAdapter<I>>(#I))
#define GET_MACRO(_1, _2, DESC, ...) DESC
#define ADPT_DESC(...) GET_MACRO(__VA_ARGS__, ADPT_DESC_TWO, ADPT_DESC_ONE, ...)(__VA_ARGS__)
#define REG_ADPT_DESC(name, name_str, adpt_desc) \
  static struct RegAdptDesc##name {              \
   public:                                       \
    RegAdptDesc##name() {                        \
      (void)ph_;                                 \
      OpAdapterMap::get()[name_str] = adpt_desc; \
    }                                            \
                                                 \
   private:                                      \
    int ph_{0};                                  \
  } g_reg_adpt_desc_##name;

#define DECLARE_CUST_OP_ADAPTER(T) DECLARE_OP_ADAPTER(Cust##T)
#define DECLARE_CUST_OP_USE_OUTPUT(T) DECLARE_OP_USE_OUTPUT(Cust##T)
#define DECLARE_CUST_OP_USE_INPUT_ATTR(T) DECLARE_OP_USE_INPUT_ATTR(Cust##T)
#define DECLARE_CUST_OP_USE_DYN_INPUT(T) DECLARE_OP_USE_DYN_INPUT(Cust##T)
#define DECLARE_CUST_OP_USE_DYN_OUTPUT(T) DECLARE_OP_USE_DYN_OUTPUT(Cust##T)
#define CUST_INPUT_MAP(T) INPUT_MAP(Cust##T)
#define CUST_DYN_INPUT_MAP(T) DYN_INPUT_MAP(Cust##T)
#define CUST_ATTR_MAP(T) ATTR_MAP(Cust##T)
#define CUST_INPUT_ATTR_MAP(T) INPUT_ATTR_MAP(Cust##T)
#define CUST_ATTR_INPUT_MAP(T) ATTR_INPUT_MAP(Cust##T)
#define CUST_OUTPUT_MAP(T) OUTPUT_MAP(Cust##T)
#define CUST_DYN_OUTPUT_MAP(T) DYN_OUTPUT_MAP(Cust##T)
#define CUST_ADPT_DESC(T) ADPT_DESC(Cust##T)

#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MACRO_H_
