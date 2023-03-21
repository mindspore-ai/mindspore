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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_UTIL_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_UTIL_H_

#include <string>
#include <vector>

#include "transform/graph_ir/op_adapter_base.h"

namespace mindspore {
namespace transform {
template <typename T>
inline ValuePtr GetRealValue(const T &value) {
  return MakeValue(value);
}

template <>
inline ValuePtr GetRealValue<GeDataType>(const GeDataType &value) {
  return MakeValue(static_cast<int64_t>(value));
}

template <>
inline ValuePtr GetRealValue<GeTensor>(const GeTensor &) {
  return nullptr;
}

template <typename P, typename Q>
static Q ConvertAnyUtil(const ValuePtr &value, const AnyTraits<P> &, const AnyTraits<Q> &) {
  return static_cast<Q>(GetValue<P>(value));
}

GeTensor ConvertAnyUtil(const ValuePtr &value, const AnyTraits<mindspore::tensor::Tensor> &traits);

std::vector<int64_t> ConvertAnyUtil(const ValuePtr &value, const std::string &name,
                                    const AnyTraits<std::vector<int64_t>>);

std::string ConvertAnyUtil(const ValuePtr &value, const AnyTraits<std::vector<int64_t>>, const AnyTraits<std::string>);

std::vector<float> ConvertAnyUtil(const ValuePtr &value, const AnyTraits<std::vector<float>>, const AnyTraits<float>);

std::vector<int64_t> ConvertAnyUtil(const ValuePtr &value, const std::string &format,
                                    const AnyTraits<std::vector<int64_t>>, const AnyTraits<int64_t>);

GeDataType ConvertAnyUtil(const ValuePtr &value, const AnyTraits<GEType>);

template <typename P, typename Q>
std::vector<Q> ConvertAnyUtil(const ValuePtr &value, AnyTraits<P>, const AnyTraits<std::vector<Q>>) {
  MS_EXCEPTION_IF_NULL(value);
  std::vector<Q> data;
  if (!value->isa<ValueTuple>() && !value->isa<ValueList>()) {
    MS_LOG(WARNING) << "error convert Value to vector for value: " << value->ToString()
                    << ", type: " << value->type_name() << ", value should be a tuple or list";
    data.push_back(ConvertAnyUtil(value, AnyTraits<P>(), AnyTraits<Q>()));
    return data;
  }
  auto vec = value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
  for (auto &it : vec) {
    data.push_back(ConvertAnyUtil(it, AnyTraits<P>(), AnyTraits<Q>()));
  }
  return data;
}

GeTensor ConvertAnyUtil(const ValuePtr &value, const AnyTraits<ValueAny>);

bool IsCustomPrim(const PrimitivePtr &prim);
bool IsCustomCNode(const AnfNodePtr &node);
std::string GetOpIOFormat(const AnfNodePtr &node);
}  // namespace transform
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_UTIL_H_
