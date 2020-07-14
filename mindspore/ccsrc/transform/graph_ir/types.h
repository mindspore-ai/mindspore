/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef TRANSFORM_TYPES_H_
#define TRANSFORM_TYPES_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/tensor.h"

#include "graph/tensor.h"
#ifdef OPEN_SOURCE
#include "ge/client/ge_api.h"
#else
#include "external/ge/ge_api.h"
#endif
using GeTensor = ge::Tensor;

namespace mindspore {
namespace transform {
enum Status : int { SUCCESS = 0, FAILED, INVALID_ARGUMENT, ALREADY_EXISTS, NOT_FOUND };

using MeTensor = mindspore::tensor::Tensor;
using MeTensorPtr = std::shared_ptr<MeTensor>;
using MeDataType = mindspore::TypeId;
using GeDataType = ge::DataType;
using GeFormat = ge::Format;
using GeShape = ge::Shape;
using GeTensorPtr = std::shared_ptr<GeTensor>;
using GeTensorDesc = ge::TensorDesc;
using AnfGraph = FuncGraph;
using AnfGraphPtr = FuncGraphPtr;
using Operator = ge::Operator;
using OperatorPtr = std::shared_ptr<ge::Operator>;
using DfGraph = ge::Graph;
using DfGraphPtr = std::shared_ptr<DfGraph>;
using TensorMap = std::unordered_map<std::string, std::shared_ptr<MeTensor>>;
}  // namespace transform
}  // namespace mindspore

#endif  // TRANSFORM_TYPES_H_
