/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_C_API_SRC_COMMON_H_
#define MINDSPORE_CCSRC_C_API_SRC_COMMON_H_

#include <string>
#include <memory>
#include "ir/func_graph.h"
#include "ops/primitive_c.h"

using FuncGraphImpl = mindspore::FuncGraph;
using FuncGraphManagerImpl = mindspore::FuncGraphManager;
using AnfNodeImpl = mindspore::AnfNode;
using ParameterImpl = mindspore::Parameter;
using ValueNodeImpl = mindspore::ValueNode;
using CNodeImpl = mindspore::CNode;
using PrimitiveImpl = mindspore::Primitive;
using TensorImpl = mindspore::tensor::Tensor;
using ScalarImpl = mindspore::Scalar;
using TypeImpl = mindspore::Type;
using AbstractBaseImpl = mindspore::abstract::AbstractBase;
using AbstractTensorImpl = mindspore::abstract::AbstractTensor;
using AbstractScalarImpl = mindspore::abstract::AbstractScalar;
using AbstractTupleImpl = mindspore::abstract::AbstractTuple;
using AbstractTypeImpl = mindspore::abstract::AbstractType;
using ValueImpl = mindspore::Value;
using ValueTupleImpl = mindspore::ValueTuple;
using StringImmImpl = mindspore::StringImm;
using BoolImmImpl = mindspore::BoolImm;
using Int8ImmImpl = mindspore::Int8Imm;
using Int16ImmImpl = mindspore::Int16Imm;
using Int32ImmImpl = mindspore::Int32Imm;
using Int64ImmImpl = mindspore::Int64Imm;
using UInt8ImmImpl = mindspore::UInt8Imm;
using Float32ImmImpl = mindspore::FP32Imm;

using BasePtr = mindspore::BasePtr;
using ValuePtr = mindspore::ValuePtr;
using TypePtr = mindspore::TypePtr;
using ScalarPtr = mindspore::ScalarPtr;
using Int32ImmPtr = mindspore::Int32ImmPtr;
using Int64ImmPtr = mindspore::Int64ImmPtr;
using Float32ImmPtr = mindspore::FP32ImmPtr;
using BoolImmPtr = mindspore::BoolImmPtr;
using StringImmPtr = mindspore::StringImmPtr;
using ValueTuplePtr = mindspore::ValueTuplePtr;
using TensorPtr = mindspore::tensor::TensorPtr;
using PrimitivePtr = mindspore::PrimitivePtr;
using AnfNodePtr = mindspore::AnfNodePtr;
using ValueNodePtr = mindspore::ValueNodePtr;
using CNodePtr = mindspore::CNodePtr;
using ParameterPtr = mindspore::ParameterPtr;
using AbstractBasePtr = mindspore::abstract::AbstractBasePtr;
using FuncGraphPtr = mindspore::FuncGraphPtr;
using FuncGraphManagerPtr = std::shared_ptr<mindspore::FuncGraphManager>;

using AttrMap = mindspore::HashMap<std::string, ValuePtr>;

#endif  // MINDSPORE_CCSRC_C_API_SRC_COMMON_H_
