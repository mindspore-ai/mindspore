/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "include/api/types.h"

namespace mindspore {
namespace dataset {
// Config operations for setting and getting the configuration.
namespace config {
/// \brief A function to set the seed to be used in any random generator. This is used to produce deterministic results.
/// \param[in] seed The default seed to be used.
/// \return The seed is set successfully or not.
/// \par Example
/// \code
///     // Set a new global configuration value for the seed value.
///     // Operations with randomness will use the seed value to generate random values.
///     bool rc = config::set_seed(5);
/// \endcode
bool DATASET_API set_seed(int32_t seed);

/// \brief A function to get the seed.
/// \return The seed set in the configuration.
/// \par Example
/// \code
///     // Get the global configuration of seed.
///     // If set_seed() is never called before, the default value(std::mt19937::default_seed) will be returned.
///     uint32_t seed = config::get_seed();
/// \endcode
uint32_t DATASET_API get_seed();

/// \brief A function to set the number of rows to be prefetched.
/// \param[in] prefetch_size Total number of rows to be prefetched.
/// \return The prefetch size is set successfully or not.
/// \par Example
/// \code
///     // Set a new global configuration value for the prefetch size.
///     bool rc = config::set_prefetch_size(1000);
/// \endcode
bool DATASET_API set_prefetch_size(int32_t prefetch_size);

/// \brief A function to get the prefetch size in number of rows.
/// \return Total number of rows to be prefetched.
/// \par Example
/// \code
///     // Get the global configuration of prefetch size.
///     // If set_prefetch_size() is never called before, the default value(16) will be returned.
///     int32_t prefetch_size = config::get_prefetch_size();
/// \endcode
int32_t DATASET_API get_prefetch_size();

/// \brief A function to set the default number of parallel workers.
/// \param[in] num_parallel_workers Number of parallel workers to be used as the default for each operation.
/// \return The workers is set successfully or not.
/// \par Example
/// \code
///     // Set a new global configuration value for the number of parallel workers.
///     // Now parallel dataset operations will run with 16 workers.
///     bool rc = config::set_num_parallel_workers(16);
/// \endcode
bool DATASET_API set_num_parallel_workers(int32_t num_parallel_workers);

/// \brief A function to get the default number of parallel workers.
/// \return Number of parallel workers to be used as the default for each operation.
/// \par Example
/// \code
///     // Get the global configuration of parallel workers.
///     // If set_num_parallel_workers() is never called before, the default value(8) will be returned.
///     int32_t parallel_workers = config::get_num_parallel_workers();
/// \endcode
int32_t DATASET_API get_num_parallel_workers();

/// \brief A function to set the default interval (in milliseconds) for monitor sampling.
/// \param[in] interval Interval (in milliseconds) to be used for performance monitor sampling.
/// \return The sampling interval is set successfully or not.
/// \par Example
/// \code
///     // Set a new global configuration value for the monitor sampling interval.
///     bool rc = config::set_monitor_sampling_interval(100);
/// \endcode
bool DATASET_API set_monitor_sampling_interval(int32_t interval);

/// \brief A function to get the default interval of performance monitor sampling.
/// \return Interval (in milliseconds) for performance monitor sampling.
/// \par Example
/// \code
///     // Get the global configuration of monitor sampling interval.
///     // If set_monitor_sampling_interval() is never called before, the default value(1000) will be returned.
///     int32_t sampling_interval = config::get_monitor_sampling_interval();
/// \endcode
int32_t DATASET_API get_monitor_sampling_interval();

/// \brief A function to set the default timeout (in seconds) for DSWaitedCallback. In case of a deadlock, the wait
///    function will exit after the timeout period.
/// \param[in] timeout Timeout (in seconds) to be used to end the wait in DSWaitedCallback in case of a deadlock.
/// \return The callback timeout is set successfully or not.
/// \par Example
/// \code
///     // Set a new global configuration value for the timeout value.
///     bool rc = config::set_callback_timeout(100);
/// \endcode
bool DATASET_API set_callback_timeout(int32_t timeout);

/// \brief A function to get the default timeout for DSWaitedCallback. In case of a deadback, the wait function
///    will exit after the timeout period.
/// \return The duration in seconds.
/// \par Example
/// \code
///     // Get the global configuration of callback timeout.
///     // If set_callback_timeout() is never called before, the default value(60) will be returned.
///     int32_t callback_timeout = config::get_callback_timeout();
/// \endcode
int32_t DATASET_API get_callback_timeout();

/// \brief A function to load the configuration from a file.
/// \param[in] file Path of the configuration file to be loaded.
/// \return The config file is loaded successfully or not.
/// \note The reason for using this API is that std::string will be constrained by the
///    compiler option '_GLIBCXX_USE_CXX11_ABI' while char is free of this restriction.
///    Check API `mindspore::dataset::config::load(const std::string &file)` and find more usage.
bool DATASET_API load(const std::vector<char> &file);

/// \brief A function to load the configuration from a file.
/// \param[in] file Path of the configuration file to be loaded.
/// \return The config file is loaded successfully or not.
/// \par Example
/// \code
///     // Set new default configuration according to values in the configuration file.
///     // example config file:
///     // {
///     //     "logFilePath": "/tmp",
///     //     "numParallelWorkers": 4,
///     //     "seed": 5489,
///     //     "monitorSamplingInterval": 30
///     // }
///     std::string config_file = "/path/to/config/file";
///     bool rc = config::load(config_file);
/// \endcode
inline bool DATASET_API load(const std::string &file) { return load(StringToChar(file)); }
}  // namespace config
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_CONFIG_H
