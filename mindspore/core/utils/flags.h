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
// flag names
inline const char GRAPH_FLAG_MIX_PRECISION_FP16[] = "fp16";
inline const char GRAPH_FLAG_MIX_PRECISION_FP32[] = "fp32";
inline const char GRAPH_FLAG_HAS_EFFECT[] = "has_effect";
inline const char GRAPH_FLAG_EFFECT_PATIAL_ORDER[] = "_effect_patial_order";
inline const char GRAPH_FLAG_CACHE_ENABLE[] = "cache_enable";
inline const char GRAPH_FLAG_RANDOM_EFFECT[] = "_random_effect";
inline const char GRAPH_FLAG_SIDE_EFFECT[] = "_side_effect";
inline const char GRAPH_FLAG_SIDE_EFFECT_IO[] = "side_effect_io";
inline const char GRAPH_FLAG_SIDE_EFFECT_MEM[] = "side_effect_mem";
inline const char GRAPH_FLAG_SIDE_EFFECT_EXCEPTION[] = "side_effect_exception";
inline const char GRAPH_FLAG_SIDE_EFFECT_PROPAGATE[] = "side_effect_propagate";
inline const char GRAPH_FLAG_SIDE_EFFECT_BACKPROP[] = "side_effect_backprop";

// method names of python primitive called from c++ source code
// 1. infer method name of class 'PrimitiveWithInfer'
inline const char PY_PRIM_METHOD_INFER[] = "__infer__";
// 2. check method name of class 'PrimitiveWithCheck'
inline const char PY_PRIM_METHOD_CHECK[] = "__check__";
// 3. method name of class 'PrimitivePy' for constant propagation
inline const char PY_PRIM_METHOD_INFER_VALUE[] = "infer_value";

// type inference related attributes
inline const char ATTR_VALUE[] = "value";
inline const char ATTR_DTYPE[] = "dtype";
inline const char ATTR_SHAPE[] = "shape";
inline const char ATTR_MIN_SHAPE[] = "min_shape";
inline const char ATTR_MAX_SHAPE[] = "max_shape";
inline const char ATTR_MIN_VALUE[] = "min_value";
inline const char ATTR_MAX_VALUE[] = "max_value";
inline const char ATTR_NO_ELIMINATE[] = "no_eliminate";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_FLAGS_H
