/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SVD_CPU_KERNEL_FUNCTION_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SVD_CPU_KERNEL_FUNCTION_H

#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"

template <typename T>
Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> SVDFloat(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix, unsigned int Options = 0);

template <typename T>
Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> SVDComplex(
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix, unsigned int Options = 0);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SVD_CPU_KERNEL_FUNCTION_H
