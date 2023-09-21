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
#ifndef MINDSPORE_JIT_GRAPH_TYPE_H_
#define MINDSPORE_JIT_GRAPH_TYPE_H_

namespace mindspore {
namespace jit {
namespace graph {
namespace ir {
enum Type {
  kTypeUnknown,
  kTypeVoid,
  kTypeNone,
  kTypeEllipsis,
  kTypeScalar,
  kTypeString,
  kTypeList,
  kTypeTuple,
  kTypeDict,
  kTypeTensor
};
}  // namespace ir
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_JIT_GRAPH_TYPE_H_
