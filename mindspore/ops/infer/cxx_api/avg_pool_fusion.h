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

#ifndef MINDSPORE_CORE_OPS_AVG_POOL_FUSION_H_
#define MINDSPORE_CORE_OPS_AVG_POOL_FUSION_H_
#include <memory>
#include <vector>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "mindapi/base/format.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAvgPoolFusion = "AvgPoolFusion";
/// \brief AvgPoolFusion defined AvgPool operator prototype of lite.
class OPS_API AvgPoolFusion : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AvgPoolFusion);
  /// \brief Constructor.
  AvgPoolFusion() : BaseOperator(kNameAvgPoolFusion) { InitIOName({"x"}, {"output"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] kernel_size Define the size of the kernel.
  /// \param[in] stride Define the moving size of the kernel.
  /// \param[in] pad_mode Define the padding method.
  /// \param[in] format Define the format of input tensor.
  /// \param[in] pad Define the concrete padding value on each dimension
  /// \param[in] round_mode Define numerical operation mode of the output tensor.
  /// \param[in] global Define a boolean value to indicate whether to do global pooling. If true, kernel_size will be
  ///            useless.
  /// \param[in] activation_type Define the activation type.
  void Init(const std::vector<int64_t> &kernel_size = {1}, const std::vector<int64_t> &stride = {1},
            const PadMode &pad_mode = VALID, const Format &format = NCHW,
            const std::vector<int64_t> &pad = {0, 0, 0, 0}, const RoundMode &round_mode = RoundMode::FLOOR,
            const bool global = false, const ActivationType activation_type = NO_ACTIVATION);

  /// \brief Set pad_mode.
  void set_pad_mode(const int64_t &pad_mode);
  /// \brief Set kernel_size.
  void set_kernel_size(const std::vector<int64_t> &kernel_size);
  /// \brief Set strides.
  void set_strides(const std::vector<int64_t> &strides);
  /// \brief Set format.
  void set_data_format(const int64_t &data_format);
  /// \brief Set pad.
  void set_pad(const std::vector<int64_t> &pad);
  /// \brief Set round_mode.
  void set_round_mode(const int64_t &round_mode);

  /// \brief Get kernel_size.
  ///
  /// \return kernel_size.
  std::vector<int64_t> get_kernel_size() const;
  /// \brief Get strides.
  ///
  /// \return strides.
  std::vector<int64_t> get_strides() const;
  /// \brief Get pad_mode.
  ///
  /// \return pad_mode.
  int64_t get_pad_mode() const;
  /// \brief Get format.
  ///
  /// \return format.
  int64_t get_data_format() const;
  /// \brief Get pad.
  ///
  /// \return pad.
  std::vector<int64_t> get_pad() const;
  /// \brief Get round_mode.
  ///
  /// \return round_mode.
  int64_t get_round_mode() const;

  /// \brief Method to set global attribute.
  ///
  /// \param[in] global Define a boolean value to indicate whether to do global pooling. If true, kernel_size will be
  ///            useless.
  void set_global(const bool global);

  /// \brief Method to set activation type.
  ///
  /// \param[in] activation_type Define the activation type.
  void set_activation_type(const ActivationType activation_type);

  /// \brief Method to get global attribute.
  ///
  /// \return a boolean value to indicate whether to do global pooling. If true, kernel_size will be useless.
  bool get_global() const;

  /// \brief Method to get activation type.
  ///
  /// \return activation type.
  ActivationType get_activation_type() const;
};

abstract::AbstractBasePtr AvgPoolFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_AVG_POOL_FUSION_H_
