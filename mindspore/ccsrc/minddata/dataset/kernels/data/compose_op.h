/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_COMPOSE_OP_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_COMPOSE_OP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
class ComposeOp : public TensorOp {
 public:
  /// constructor
  /// \param[in] ops list of TensorOps to compose into 1 TensorOp
  explicit ComposeOp(const std::vector<std::shared_ptr<TensorOp>> &ops);

  /// default destructor
  ~ComposeOp() override = default;

  /// return the number of inputs the first tensorOp in compose takes
  /// \return number of input tensors
  uint32_t NumInput() override { return ops_.front()->NumInput(); }

  /// return the number of outputs the last tensorOp in compose produces
  /// \return number of output tensors
  uint32_t NumOutput() override { return ops_.back()->NumOutput(); }

  /// \param[in] inputs
  /// \param[out] outputs
  /// \return  Status code
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  /// \param[in] inputs
  /// \param[out] outputs
  /// \return Status code
  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  /// \param[in] input
  /// \param[out] output
  /// \return Status code
  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kComposeOp; }

  // Currently maybe 910b dvpp ops & cpu ops mixed used
  bool IsMixedOps() {
    bool have_dvpp = false;
    bool have_cpu = false;
    for (auto &item : ops_) {
      if (item->IsDvppOp()) {
        have_dvpp = true;
      } else {
        have_cpu = true;
      }
    }

    if (have_dvpp && have_cpu) {
      MS_LOG(ERROR) << "Currently, it is not supported to mix DVPP transforms with CPU transforms in Compose.";
      return true;
    }
    return false;
  }

  // Check whether compose contains dvpp ops.
  virtual bool IsDvppOp() {
    bool have_dvpp = false;
    for (auto &item : ops_) {
      if (item->IsDvppOp()) {
        have_dvpp = true;
        break;
      }
    }

    if (have_dvpp) {
      return true;
    }
    return false;
  }

  std::vector<std::shared_ptr<TensorOp>> GetOps() { return ops_; }

 private:
  std::vector<std::shared_ptr<TensorOp>> ops_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_COMPOSE_OP_
