/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_DEFORMABLE_OFFSETS_H_
#define MINDSPORE_CORE_OPS_DEFORMABLE_OFFSETS_H_

#include <string>
#include <vector>

#include "ops_utils/op_constants.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDeformableOffsets = "DeformableOffsets";
/// \brief DeformableOffsets. Refer to Python API @ref mindspore.ops.deformable_conv2d for more details.
class OPS_API DeformableOffsets : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DeformableOffsets);
  /// \brief Constructor.
  DeformableOffsets() : BaseOperator(kNameDeformableOffsets) { InitIOName({"x", "offsets"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.deformable_conv2d for the inputs.
  void Init(const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
            const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &dilations = {1, 1, 1, 1},
            const std::string &data_format = kOpFormat_NCHW, int64_t deformable_groups = 1, bool modulated = true);

  /// \brief Set strides.
  void set_strides(const std::vector<int64_t> &strides);
  /// \brief Get strides.
  ///
  /// \return strides.
  std::vector<int64_t> get_strides() const;

  /// \brief Set pads.
  void set_pads(const std::vector<int64_t> &pads);
  /// \brief Get pads.
  ///
  /// \return pads.
  std::vector<int64_t> get_pads() const;

  /// \brief Set kernel_size.
  void set_kernel_size(const std::vector<int64_t> &kernel_size);
  /// \brief Get kernel_size.
  ///
  /// \return kernel_size.
  std::vector<int64_t> get_kernel_size() const;

  /// \brief Set dilations.
  void set_dilations(const std::vector<int64_t> &dilations);
  /// \brief Get dilations.
  ///
  /// \return dilations.
  std::vector<int64_t> get_dilations() const;

  /// \brief Set format.
  void set_data_format(const std::string &data_format);
  /// \brief Get format.
  ///
  /// \return format.
  std::string get_data_format() const;

  /// \brief Set deformable_groups.
  void set_deformable_groups(int64_t deformable_groups);
  /// \brief Get deformable_groups.
  ///
  /// \return deformable_groups.
  int64_t get_deformable_groups() const;

  /// \brief Set modulated.
  void set_modulated(bool modulated);
  /// \brief Get modulated.
  ///
  /// \return modulated.
  bool get_modulated() const;
};
OPS_API abstract::AbstractBasePtr DeformableOffsetsInfer(const abstract::AnalysisEnginePtr &,
                                                         const PrimitivePtr &primitive,
                                                         const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_DEFORMABLE_OFFSETS_H_
