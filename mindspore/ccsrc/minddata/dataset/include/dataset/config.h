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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_CONFIG_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_CONFIG_H

#include <cstdint>
#include <string>
#include <vector>
#include "include/api/dual_abi_helper.h"

namespace mindspore {
namespace dataset {

// Config operations for setting and getting the configuration.
namespace config {

/// \brief A function to set the seed to be used in any random generator. This is used to produce deterministic results.
/// \param[in] seed The default seed to be used.
bool set_seed(int32_t seed);

/// \brief A function to get the seed.
/// \return The seed set in the configuration.
uint32_t get_seed();

/// \brief A function to set the number of rows to be prefetched.
/// \param[in] prefetch_size Total number of rows to be prefetched.
bool set_prefetch_size(int32_t prefetch_size);

/// \brief A function to get the prefetch size in number of rows.
/// \return Total number of rows to be prefetched.
int32_t get_prefetch_size();

/// \brief A function to set the default number of parallel workers.
/// \param[in] num_parallel_workers Number of parallel workers to be used as the default for each operation.
bool set_num_parallel_workers(int32_t num_parallel_workers);

/// \brief A function to get the default number of parallel workers.
/// \return Number of parallel workers to be used as the default for each operation.
int32_t get_num_parallel_workers();

/// \brief A function to set the default interval (in milliseconds) for monitor sampling.
/// \param[in] interval Interval (in milliseconds) to be used for performance monitor sampling.
bool set_monitor_sampling_interval(int32_t interval);

/// \brief A function to get the default interval of performance monitor sampling.
/// \return Interval (in milliseconds) for performance monitor sampling.
int32_t get_monitor_sampling_interval();

/// \brief A function to set the default timeout (in seconds) for DSWaitedCallback. In case of a deadlock, the wait
///    function will exit after the timeout period.
/// \param[in] timeout Timeout (in seconds) to be used to end the wait in DSWaitedCallback in case of a deadlock.
bool set_callback_timeout(int32_t timeout);

/// \brief A function to get the default timeout for DSWaitedCallback. In case of a deadback, the wait function
///    will exit after the timeout period.
/// \return The duration in seconds.
int32_t get_callback_timeout();

/// \brief A function to load the configuration from a file.
/// \param[in] file Path of the configuration file to be loaded.
/// \note The reason for using this API is that std::string will be constrained by the
///    compiler option '_GLIBCXX_USE_CXX11_ABI' while char is free of this restriction.
bool load(const std::vector<char> &file);

/// \brief A function to load the configuration from a file.
/// \param[in] file Path of the configuration file to be loaded.
inline bool load(std::string file) { return load(StringToChar(file)); }

}  // namespace config
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_CONFIG_H
