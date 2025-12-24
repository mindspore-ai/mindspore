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

#ifndef MINDSPORE_CORE_OPS_MAX_POOL_FUSION_H_
#define MINDSPORE_CORE_OPS_MAX_POOL_FUSION_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "infer/max_pool.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMaxPoolFusion = "MaxPoolFusion";
/// \brief MaxPoolFusion defined MaxPool operator prototype of lite.
class OPS_API MaxPoolFusion : public MaxPool {
 public:
  MIND_API_BASE_MEMBER(MaxPoolFusion);
  /// \brief Constructor.
  MaxPoolFusion() : MaxPool(kNameMaxPoolFusion) { InitIOName({"x"}, {"output"}); }

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

abstract::AbstractBasePtr MaxPoolFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MAX_POOL_FUSION_H_
