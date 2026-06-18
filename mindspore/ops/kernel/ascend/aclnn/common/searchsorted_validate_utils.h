/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_COMMON_SEARCHSORTED_VALIDATE_UTILS_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_COMMON_SEARCHSORTED_VALIDATE_UTILS_H_

#include <string>

#include "base/base.h"
#include "mindapi/base/type_id.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace kernel {
namespace searchsorted {

inline bool SearchsortedDimsMatchedBeforeLastDim(const ShapeVector &sorted_sequence_shape,
                                                 const ShapeVector &values_shape) {
  if (sorted_sequence_shape.size() != values_shape.size()) {
    return false;
  }

  for (size_t i = 0; i + 1 < sorted_sequence_shape.size(); ++i) {
    if (sorted_sequence_shape[i] != values_shape[i]) {
      return false;
    }
  }
  return true;
}

inline void CheckSearchsortedInputShapes(const ShapeVector &sorted_sequence_shape, const ShapeVector &values_shape,
                                         const std::string &op_name) {
  if (sorted_sequence_shape.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', empty shape is not permitted in 'sorted_sequence' inputs.";
  }

  if (values_shape.empty() && sorted_sequence_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', 'values' can be a scalar only when the dimension of 'sorted_sequence' is 1.";
  }

  if (sorted_sequence_shape.size() != 1 && !SearchsortedDimsMatchedBeforeLastDim(sorted_sequence_shape, values_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the 'sorted_sequence' must be 1 dimensional or all dimensions except the last "
                                "dimension of 'sorted_sequence' must be the same as all dimensions except the last "
                                "dimension of 'values', but got shape of 'sorted_sequence': "
                             << ShapeVectorToString(sorted_sequence_shape)
                             << " and shape of 'values': " << ShapeVectorToString(values_shape) << ".";
  }
}

inline void CheckSearchsortedSorterType(TypeId sorter_type_id, const std::string &op_name) {
  if (sorter_type_id != kNumberTypeInt64) {
    MS_EXCEPTION(TypeError) << "For '" << op_name << "', the data type of 'sorter' must be int64, but got "
                            << TypeIdToString(sorter_type_id) << ".";
  }
}

inline void CheckSearchsortedSorterShape(const ShapeVector &sorted_sequence_shape, const ShapeVector &sorter_shape,
                                         const std::string &op_name) {
  if (sorted_sequence_shape != sorter_shape) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', the shape of 'sorter' must be the same as "
                             << "'sorted_sequence', but got shape of 'sorted_sequence': "
                             << ShapeVectorToString(sorted_sequence_shape)
                             << " and shape of 'sorter': " << ShapeVectorToString(sorter_shape) << ".";
  }
}

}  // namespace searchsorted
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_COMMON_SEARCHSORTED_VALIDATE_UTILS_H_
