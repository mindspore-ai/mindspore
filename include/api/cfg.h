/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_API_CFG_H
#define MINDSPORE_INCLUDE_API_CFG_H

#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "include/api/data_type.h"
#include "include/api/dual_abi_helper.h"
#include "include/api/types.h"

namespace mindspore {
constexpr int iter_th = 1000;
class MS_API MixPrecisionCfg {
 public:
  MixPrecisionCfg() {
    this->dynamic_loss_scale_ = false;
    this->loss_scale_ = 128.0f;
    this->keep_batchnorm_fp32_ = true;
    this->num_of_not_nan_iter_th_ = iter_th;
  }
  MixPrecisionCfg(const MixPrecisionCfg &rhs) {
    this->dynamic_loss_scale_ = rhs.dynamic_loss_scale_;
    this->loss_scale_ = rhs.loss_scale_;
    this->keep_batchnorm_fp32_ = rhs.keep_batchnorm_fp32_;
    this->num_of_not_nan_iter_th_ = rhs.num_of_not_nan_iter_th_;
  }
  ~MixPrecisionCfg() = default;

  bool dynamic_loss_scale_ = false;   /**< Enable/disable dynamic loss scale during mix precision training */
  float loss_scale_;                  /**< Initial loss scale factor  */
  bool keep_batchnorm_fp32_ = true;   /**< Keep batch norm in FP32 while training */
  uint32_t num_of_not_nan_iter_th_;   /**< a threshold for modifying loss scale when dynamic loss scale is enabled */
  bool is_raw_mix_precision_ = false; /**< Is mix precision model export from mindspore  */
};

class MS_API TrainCfg {
 public:
  TrainCfg() = default;
  TrainCfg(const TrainCfg &rhs) {
    this->loss_name_ = rhs.loss_name_;
    this->mix_precision_cfg_ = rhs.mix_precision_cfg_;
    this->accumulate_gradients_ = rhs.accumulate_gradients_;
  }
  ~TrainCfg() = default;

  /// \brief obtain part of the name that identify a loss kernel.
  ///
  /// \return loss_name.
  inline std::vector<std::string> GetLossName() const;
  /// \brief Set part of the name that identify a loss kernel.
  ///
  /// \param[in] loss_name define part of the name that identify a loss kernel.
  inline void SetLossName(const std::vector<std::string> &loss_name);

  OptimizationLevel optimization_level_ = kO0;
  MixPrecisionCfg mix_precision_cfg_; /**< Mix precision configuration */
  bool accumulate_gradients_ = false;

 private:
  std::vector<std::vector<char>> loss_name_ = VectorStringToChar({"loss_fct", "_loss_fn", "SigmoidCrossEntropy"});
};

std::vector<std::string> TrainCfg::GetLossName() const { return VectorCharToString(loss_name_); }
void TrainCfg::SetLossName(const std::vector<std::string> &loss_name) { loss_name_ = VectorStringToChar(loss_name); }
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CFG_H
