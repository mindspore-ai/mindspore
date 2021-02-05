/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_FLAGS_H
#define MINDSPORE_CORE_UTILS_FLAGS_H
namespace mindspore {
extern const char GRAPH_FLAG_MIX_PRECISION_FP16[];
extern const char GRAPH_FLAG_MIX_PRECISION_FP32[];
extern const char GRAPH_FLAG_HAS_EFFECT[];
extern const char GRAPH_FLAG_EFFECT_PATIAL_ORDER[];
extern const char GRAPH_FLAG_CACHE_ENABLE[];
extern const char GRAPH_FLAG_RANDOM_EFFECT[];
extern const char GRAPH_FLAG_SIDE_EFFECT[];
extern const char GRAPH_FLAG_SIDE_EFFECT_IO[];
extern const char GRAPH_FLAG_SIDE_EFFECT_MEM[];
extern const char GRAPH_FLAG_SIDE_EFFECT_EXCEPTION[];
extern const char GRAPH_FLAG_SIDE_EFFECT_PROPAGATE[];
extern const char GRAPH_FLAG_SIDE_EFFECT_BACKPROP[];

extern const char PY_PRIM_METHOD_INFER[];
extern const char PY_PRIM_METHOD_CHECK[];
extern const char PY_PRIM_METHOD_INFER_VALUE[];

extern const char ATTR_VALUE[];
extern const char ATTR_DTYPE[];
extern const char ATTR_SHAPE[];
extern const char ATTR_MIN_SHAPE[];
extern const char ATTR_MAX_SHAPE[];
extern const char ATTR_MIN_VALUE[];
extern const char ATTR_MAX_VALUE[];
extern const char ATTR_NO_BROADEN[];
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_FLAGS_H
