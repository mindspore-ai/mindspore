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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_META_OP_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_META_OP_H_
#include <string>
namespace mindspore::graphkernel::expander {
enum class MetaOp : int {
  Abs = 0,
  Add,
  Assign,
  BroadcastTo,
  Cast,
  Floor,
  Trunc,
  Concat,
  Cosh,
  Div,
  RealDiv,
  Equal,
  Exp,
  Gather,
  Greater,
  GreaterEqual,
  IsInf,
  IsNan,
  Less,
  LessEqual,
  Log,
  LogicalAnd,
  LogicalNot,
  LogicalOr,
  MatMul,
  Mul,
  Neg,
  Reciprocal,
  ReduceMax,
  ReduceMin,
  ReduceSum,
  Reshape,
  Rsqrt,
  Select,
  Shape,
  Sinh,
  Sqrt,
  StridedSlice,
  Sub,
  Tanh,
  TensorScatterAdd,
  Transpose,
  Maximum,
  Minimum,
  MetaOpNum  // max id
};

inline static std::string MetaOpStr[static_cast<int>(MetaOp::MetaOpNum)] = {
  "Abs",               // MetaOp::Abs
  "Add",               // MetaOp::Add
  "Assign",            // MetaOp::Assign
  "BroadcastTo",       // MetaOp::BroadcastTo
  "Cast",              // MetaOp::Cast
  "Floor",             // MetaOp::Floor
  "Trunc",             // MetaOp::Trunc
  "Concat",            // MetaOp::Concat
  "Cosh",              // MetaOp::Cosh
  "Div",               // MetaOp::Div
  "RealDiv",           // MetaOp::RealDiv
  "Equal",             // MetaOp::Equal
  "Exp",               // MetaOp::Exp
  "Gather",            // MetaOp::Gather
  "Greater",           // MetaOp::Greater
  "GreaterEqual",      // MetaOp::GreaterEqual
  "IsInf",             // MetaOp::IsInf
  "IsNan",             // MetaOp::IsNan
  "Less",              // MetaOp::Less
  "LessEqual",         // MetaOp::LessEqual
  "Log",               // MetaOp::Log
  "LogicalAnd",        // MetaOp::LogicalAnd
  "LogicalNot",        // MetaOp::LogicalNot
  "LogicalOr",         // MetaOp::LogicalOr
  "MatMul",            // MetaOp::MatMul
  "Mul",               // MetaOp::Mul
  "Neg",               // MetaOp::Neg
  "Reciprocal",        // MetaOp::Reciprocal
  "ReduceMax",         // MetaOp::ReduceMax
  "ReduceMin",         // MetaOp::ReduceMin
  "ReduceSum",         // MetaOp::ReduceSum
  "Reshape",           // MetaOp::Reshape
  "Rsqrt",             // MetaOp::Rsqrt
  "Select",            // MetaOp::Select
  "Shape",             // MetaOp::Shape
  "Sinh",              // MetaOp::Sinh
  "Sqrt",              // MetaOp::Sqrt
  "StridedSlice",      // MetaOp::StridedSlice
  "Sub",               // MetaOp::Sub
  "Tanh",              // MetaOp::Tanh
  "TensorScatterAdd",  // MetaOp::TensorScatterAdd
  "Transpose",         // MetaOp::Transpose
  "Maximum",           // MetaOp::Maximum
  "Minimum",           // MetaOp::Minimum
};
}  // namespace mindspore::graphkernel::expander
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_META_OP_H_
