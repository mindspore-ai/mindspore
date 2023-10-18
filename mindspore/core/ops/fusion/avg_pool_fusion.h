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
#include "ir/anf.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAvgPoolFusion = "AvgPoolFusion";
/// \brief AvgPoolFusion defined AvgPool operator prototype of lite.
class MIND_API AvgPoolFusion : public BaseOperator {
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
            const std::vector<int64_t> &pad = {0, 0, 0, 0}, const RoundMode &round_mode = FLOOR,
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

  void set_kernel_size(const std::vector<int64_t> &kernel_size) {
    (void)this->AddAttr("kernel_size", api::MakeValue(kernel_size));
  }
  std::vector<int64_t> get_kernel_size() const { return GetValue<std::vector<int64_t>>(GetAttr("kernel_size")); }
  void set_strides(const std::vector<int64_t> &strides) { (void)this->AddAttr("strides", api::MakeValue(strides)); }
  std::vector<int64_t> get_strides() const { return GetValue<std::vector<int64_t>>(GetAttr("strides")); }
  void set_pad_mode(const int &pad_mode) { (void)this->AddAttr("pad_mode", api::MakeValue(pad_mode)); }
  int get_pad_mode() const { return GetValue<int>(GetAttr("pad_mode")); }
  void set_data_format(const int &data_format) { (void)this->AddAttr("data_format", api::MakeValue(data_format)); }
  int get_data_format() const { return GetValue<int>(GetAttr("data_format")); }
  void set_pad(const std::vector<int64_t> &pad) { (void)this->AddAttr("pad", api::MakeValue(pad)); }
  std::vector<int64_t> get_pad() const { return GetValue<std::vector<int64_t>>(GetAttr("pad")); }
  void set_round_mode(const int &round_mode) { (void)this->AddAttr("round_mode", api::MakeValue(round_mode)); }
  int get_round_mode() const { return GetValue<int>(GetAttr("round_mode")); }

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
