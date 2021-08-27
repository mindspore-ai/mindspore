/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#pragma once
#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_STATISTICS_H
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_STATISTICS_H

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "minddata/mindrecord/include/common/shard_pybind.h"
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "pybind11/pybind11.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace mindrecord {
class __attribute__((visibility("default"))) Statistics {
 public:
  /// \brief save the statistic and its description
  /// \param[in] desc the statistic's description
  /// \param[in] statistics the statistic needs to be saved
  static std::shared_ptr<Statistics> Build(std::string desc, const json &statistics);

  ~Statistics() = default;

  /// \brief compare two statistics to judge if they are equal
  /// \param b another statistics to be judged
  /// \return true if they are equal,false if not
  bool operator==(const Statistics &b) const;

  /// \brief get the description
  /// \return the description
  std::string GetDesc() const;

  /// \brief get the statistic
  /// \return json format of the statistic
  json GetStatistics() const;

  /// \brief decode the bson statistics to json
  /// \param[in] encodedStatistics the bson type of statistics
  /// \return json type of statistic
  void SetStatisticsID(int64_t id);

  /// \brief get the statistics id
  /// \return the int64 statistics id
  int64_t GetStatisticsID() const;

 private:
  /// \brief validate the statistic
  /// \return true / false
  static bool Validate(const json &statistics);

  static bool LevelRecursive(json level);

  Statistics() = default;

  std::string desc_;
  json statistics_;
  int64_t statistics_id_ = -1;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_STATISTICS_H
