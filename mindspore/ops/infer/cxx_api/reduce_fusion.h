/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_REDUCE_FUSION_H_
#define MINDSPORE_CORE_OPS_REDUCE_FUSION_H_
#include <map>
#include <memory>
#include <string>
#include "mindapi/base/types.h"
#include "infer/reduce.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReduceFusion = "ReduceFusion";
/// \brief ReduceFusion defined Reduce operator prototype of lite.
class OPS_API ReduceFusion : public Reduce {
 public:
  MIND_API_BASE_MEMBER(ReduceFusion);
  /// \brief Constructor.
  ReduceFusion() : Reduce(kNameReduceFusion) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] keep_dims Define a boolean value to indicate whether output dimension is kept or not.
  /// \param[in] mode Define the concrete reduction mode.
  /// \param[in] reduce_to_end Define a boolean value to indicate whether the operation need to do from the given axis
  ///            to the last.
  /// \param[in] coeff Define a size factor applied to output.
  void Init(const bool keep_dims = false, const ReduceMode mode = ReduceMode::Reduce_Mean,
            const bool reduce_to_end = false, const float coeff = 1.0);

  /// \brief Method to set keep_dims attribute.
  ///
  /// \param[in] keep_dims Define a boolean value to indicate whether output dimension is kept or not.
  void set_keep_dims(const bool keep_dims);

  /// \brief Method to set mode attribute.
  ///
  /// \param[in] mode Define the concrete reduction mode.
  void set_mode(const ReduceMode mode);

  /// \brief Method to set reduce_to_end attribute.
  ///
  /// \param[in] reduce_to_end Define a boolean value to indicate whether the operation need to do from the given axis
  ///            to the last.
  void set_reduce_to_end(const bool reduce_to_end);

  /// \brief Method to set coeff attribute.
  ///
  /// \param[in] coeff Define a size factor applied to output.
  void set_coeff(const float coeff);

  /// \brief Method to get keep_dims attribute.
  ///
  /// \return a boolean value.
  bool get_keep_dims() const;

  /// \brief Method to get mode attribute.
  ///
  /// \return reduction mode.
  ReduceMode get_mode() const;

  /// \brief Method to get reduce_to_end attribute.
  ///
  /// \return a boolean value.
  bool get_reduce_to_end() const;

  /// \brief Method to get coeff attribute.
  ///
  /// \return a size factor applied to output.
  float get_coeff() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_REDUCE_FUSION_H_
