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

#ifndef TRANSFORM_OP_ADAPTER_UTIL_H_
#define TRANSFORM_OP_ADAPTER_UTIL_H_

#include <string>
#include <vector>

#include "transform/op_adapter_base.h"

namespace mindspore {
namespace transform {
template <typename P, typename Q>
static Q ConvertAnyUtil(const ValuePtr& value, const AnyTraits<P>&, const AnyTraits<Q>&) {
  return static_cast<Q>(GetValue<P>(value));
}

GeTensor ConvertAnyUtil(const ValuePtr& value, const AnyTraits<mindspore::tensor::Tensor>& traits);

std::vector<int64_t> ConvertAnyUtil(const ValuePtr& value, const std::string& name,
                                    const AnyTraits<std::vector<int64_t>>);

std::string ConvertAnyUtil(const ValuePtr& value, const AnyTraits<std::vector<int64_t>>, const AnyTraits<std::string>);

std::vector<float> ConvertAnyUtil(const ValuePtr& value, const AnyTraits<std::vector<float>>, const AnyTraits<float>);

std::vector<int64_t> ConvertAnyUtil(const ValuePtr& value, const std::string& format,
                                    const AnyTraits<std::vector<int64_t>>, const AnyTraits<int64_t>);

GeDataType ConvertAnyUtil(const ValuePtr& value, const AnyTraits<GEType>);

template <typename P, typename Q>
std::vector<Q> ConvertAnyUtil(const ValuePtr& value, AnyTraits<P>, const AnyTraits<std::vector<Q>>) {
  if (!value->isa<ValueTuple>() && !value->isa<ValueList>()) {
    MS_LOG(EXCEPTION) << "error convert Value to vector for value: " << value->ToString()
                      << ", type: " << value->type_name() << ", value should be a tuple or list";
  }
  auto vec = value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
  std::vector<Q> data;
  for (auto& it : vec) {
    data.push_back(ConvertAnyUtil(it, AnyTraits<P>(), AnyTraits<Q>()));
  }
  return data;
}

GeTensor ConvertAnyUtil(const ValuePtr& value, const AnyTraits<AnyValue>);

bool IsCustomPrim(const PrimitivePtr& prim);
bool IsCustomCNode(const AnfNodePtr& node);
}  // namespace transform
}  // namespace mindspore
#endif  // TRANSFORM_OP_ADAPTER_UTIL_H_
