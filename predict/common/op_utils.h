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

#ifndef PREDICT_COMMON_OP_UTILS_H_
#define PREDICT_COMMON_OP_UTILS_H_

#include <functional>
#include <string>
#include "schema/inner/ms_generated.h"

namespace mindspore {
namespace predict {
inline OpT GetOpType(const OpDef &opDef) { return opDef.attr_type(); }

inline OpT GetOpType(const NodeDef &nodeDef) { return GetOpType(*(nodeDef.opDef())); }

inline std::string GetOpTypeName(const NodeDef &nodeDef) { return EnumNameOpT(GetOpType(nodeDef)); }

inline std::string GetOpTypeName(const OpDef &opDef) { return EnumNameOpT(GetOpType(opDef)); }

inline OpT GetOpType(const OpDefT &opDefT) { return opDefT.attr.type; }

inline OpT GetOpType(const NodeDefT &nodeDefT) { return GetOpType(*(nodeDefT.opDef.get())); }

inline std::string GetOpTypeName(const NodeDefT &nodeDefT) { return EnumNameOpT(GetOpType(nodeDefT)); }

inline std::string GetOpTypeName(const OpDefT &opDefT) { return EnumNameOpT(GetOpType(opDefT)); }
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_COMMON_OP_UTILS_H_
