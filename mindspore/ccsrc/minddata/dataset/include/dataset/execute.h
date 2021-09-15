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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_EXECUTE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_EXECUTE_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"

namespace mindspore {
namespace dataset {
class DeviceResource;
// class to run tensor operations in eager mode
class Execute {
 public:
  /// \brief Constructor.
  /// \param[in] op TensorOperation to be applied in Eager mode, it accepts operation in type of shared pointer.
  /// \param[in] device_type Target device environment to perform operation, can be kCPU/kGPU/kAscend310 (default=kCPU).
  /// \param[in] device_id Target device ID to perform operation, only valid when device_type=kAscend310 (default=0).
  explicit Execute(std::shared_ptr<TensorOperation> op, MapTargetDevice device_type = MapTargetDevice::kCpu,
                   uint32_t device_id = 0);

  /// \brief Constructor.
  /// \param[in] op TensorTransform to be applied in Eager mode, it accepts operation in type of shared pointer.
  /// \param[in] device_type Target device environment to perform operation, can be kCPU/kGPU/kAscend310 (default=kCPU).
  /// \param[in] device_id Target device ID to perform operation, only valid when device_type=kAscend310 (default=0).
  explicit Execute(std::shared_ptr<TensorTransform> op, MapTargetDevice device_type = MapTargetDevice::kCpu,
                   uint32_t device_id = 0);

  /// \brief Constructor.
  /// \param[in] op TensorTransform to be applied in Eager mode, it accepts operation in type of reference.
  /// \param[in] device_type Target device environment to perform operation, can be kCPU/kGPU/kAscend310 (default=kCPU).
  /// \param[in] device_id Target device ID to perform operation, only valid when device_type=kAscend310 (default=0).
  explicit Execute(std::reference_wrapper<TensorTransform> op, MapTargetDevice device_type = MapTargetDevice::kCpu,
                   uint32_t device_id = 0);

  /// \brief Constructor.
  /// \param[in] op TensorTransform to be applied in Eager mode, it accepts operation in type of raw pointer.
  /// \param[in] device_type Target device environment to perform operation, can be kCPU/kGPU/kAscend310 (default=kCPU).
  /// \param[in] device_id Target device ID to perform operation, only valid when device_type=kAscend310 (default=0).
  explicit Execute(TensorTransform *op, MapTargetDevice device_type = MapTargetDevice::kCpu, uint32_t device_id = 0);

  /// \brief Constructor.
  /// \param[in] ops A vector of TensorOperations to be applied in Eager mode, it accepts operation
  ///     in type of shared pointer.
  /// \param[in] device_type Target device environment to perform operation, can be kCPU/kGPU/kAscend310 (default=kCPU).
  /// \param[in] device_id Target device ID to perform operation, only valid when device_type=kAscend310 (default=0).
  explicit Execute(std::vector<std::shared_ptr<TensorOperation>> ops,
                   MapTargetDevice device_type = MapTargetDevice::kCpu, uint32_t device_id = 0);

  /// \brief Constructor.
  /// \param[in] ops A vector of TensorTransforms to be applied in Eager mode, it accepts operation
  ///     in type of shared pointer.
  /// \param[in] device_type Target device environment to perform operation, can be kCPU/kGPU/kAscend310 (default=kCPU).
  /// \param[in] device_id Target device ID to perform operation, only valid when device_type=kAscend310 (default=0).
  explicit Execute(std::vector<std::shared_ptr<TensorTransform>> ops,
                   MapTargetDevice device_type = MapTargetDevice::kCpu, uint32_t device_id = 0);

  /// \brief Constructor.
  /// \param[in] ops A vector of TensorTransforms to be applied in Eager mode, it accepts operation
  ///     in type of raw pointer.
  /// \param[in] device_type Target device environment to perform operation, can be kCPU/kGPU/kAscend310 (default=kCPU).
  /// \param[in] device_id Target device ID to perform operation, only valid when device_type=kAscend310 (default=0).
  explicit Execute(const std::vector<std::reference_wrapper<TensorTransform>> ops,
                   MapTargetDevice device_type = MapTargetDevice::kCpu, uint32_t device_id = 0);

  /// \brief Constructor.
  /// \param[in] ops A vector of TensorTransforms to be applied in Eager mode, it accepts operation
  ///     in type of raw pointer.
  /// \param[in] device_type Target device environment to perform operation, can be kCPU/kGPU/kAscend310 (default=kCPU).
  /// \param[in] device_id Target device ID to perform operation, only valid when device_type=kAscend310 (default=0).
  explicit Execute(const std::vector<TensorTransform *> &ops, MapTargetDevice device_type = MapTargetDevice::kCpu,
                   uint32_t device_id = 0);

  /// \brief Destructor.
  ~Execute();

  /// \brief Callable function to execute the TensorTransform in eager mode.
  /// \param[in] input Tensor to be transformed.
  /// \param[out] output Transformed tensor.
  /// \return Status error code, returns OK if no error encountered.
  Status operator()(const mindspore::MSTensor &input, mindspore::MSTensor *output);

  /// \brief Callable function to execute the TensorTransform in eager mode.
  /// \param[in] input_tensor_list List of Tensor to be transformed.
  /// \param[out] out Result tensor after transform.
  /// \return Status error code, returns OK if no error encountered.
  Status operator()(const std::vector<mindspore::MSTensor> &input_tensor_list, std::vector<mindspore::MSTensor> *out);

  /// \brief Given a set of Executes, run them
  static Status Run(const std::vector<std::shared_ptr<dataset::Execute>> &data_graph,
                    const std::vector<mindspore::MSTensor> &inputs, std::vector<mindspore::MSTensor> *outputs);

  /// \brief The function to release device memory on Ascend310.
  Status DeviceMemoryRelease();

  /// \brief The function to generate AIPP configuration.
  std::string AippCfgGenerator();

 private:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  Status ParseTransforms();

  /// \brief The function to validate target device setting is valid or not.
  Status ValidateDevice();

  std::vector<std::shared_ptr<TensorTransform>> transforms_;
  std::vector<std::shared_ptr<TensorOperation>> ops_;
  MapTargetDevice device_type_;
  std::shared_ptr<DeviceResource> device_resource_;
  struct ExtraInfo;
  std::shared_ptr<ExtraInfo> info_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_EXECUTE_H_
