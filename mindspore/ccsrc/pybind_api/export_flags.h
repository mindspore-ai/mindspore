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

#ifndef PYBIND_API_EXPORT_FLAGS_H_
#define PYBIND_API_EXPORT_FLAGS_H_

namespace mindspore {

extern const char PYTHON_PRIMITIVE_FLAG[];
extern const char PYTHON_METAFUNCGRAPH_FLAG[];
extern const char PYTHON_TENSOR_FLAG[];
extern const char PYTHON_META_TENSOR_FLAG[];
extern const char PYTHON_ENVINSTANCE_FLAG[];
extern const char PYTHON_DTYPE_FLAG[];
extern const char PYTHON_CELL_AS_LIST[];
extern const char PYTHON_DATACLASS_FIELDS[];

extern const char GRAPH_FLAG_MIX_PRECISION_FP16[];
extern const char GRAPH_FLAG_MIX_PRECISION_FP32[];
extern const char GRAPH_FLAG_LOOP_CAN_UNROLL[];
extern const char GRAPH_FLAG_HAS_EFFECT[];
extern const char GRAPH_FLAG_EFFECT_PATIAL_ORDER[];

}  // namespace mindspore

#endif  // PYBIND_API_EXPORT_FLAGS_H_
