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

#include "mindrecord/include/shard_statistics.h"
#include "pybind11/pybind11.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

namespace mindspore {
namespace mindrecord {
std::shared_ptr<Statistics> Statistics::Build(std::string desc, const json &statistics) {
  // validate check
  if (!Validate(statistics)) {
    return nullptr;
  }
  Statistics object_statistics;
  object_statistics.desc_ = std::move(desc);
  object_statistics.statistics_ = statistics;
  object_statistics.statistics_id_ = -1;
  return std::make_shared<Statistics>(object_statistics);
}

std::shared_ptr<Statistics> Statistics::Build(std::string desc, pybind11::handle statistics) {
  // validate check
  json statistics_json = nlohmann::detail::ToJsonImpl(statistics);
  if (!Validate(statistics_json)) {
    return nullptr;
  }
  Statistics object_statistics;
  object_statistics.desc_ = std::move(desc);
  object_statistics.statistics_ = statistics_json;
  object_statistics.statistics_id_ = -1;
  return std::make_shared<Statistics>(object_statistics);
}

std::string Statistics::get_desc() const { return desc_; }

json Statistics::get_statistics() const {
  json str_statistics;
  str_statistics["desc"] = desc_;
  str_statistics["statistics"] = statistics_;
  return str_statistics;
}

pybind11::object Statistics::GetStatisticsForPython() const {
  json str_statistics = Statistics::get_statistics();
  return nlohmann::detail::FromJsonImpl(str_statistics);
}

void Statistics::set_statistics_id(int64_t id) { statistics_id_ = id; }

int64_t Statistics::get_statistics_id() const { return statistics_id_; }

bool Statistics::Validate(const json &statistics) {
  if (statistics.size() != kInt1) {
    MS_LOG(ERROR) << "Statistics object is null";
    return false;
  }
  if (statistics.find("level") == statistics.end()) {
    MS_LOG(ERROR) << "There is not 'level' object in statistic";
    return false;
  }
  return LevelRecursive(statistics["level"]);
}

bool Statistics::LevelRecursive(json level) {
  bool ini = true;
  for (json::iterator it = level.begin(); it != level.end(); ++it) {
    json a = it.value();
    if (a.size() == kInt2) {
      if ((a.find("key") == a.end()) || (a.find("count") == a.end())) {
        MS_LOG(ERROR) << "The node field is 2, but 'key'/'count' is not existed";
        return false;
      }
    } else if (a.size() == kInt3) {
      if ((a.find("key") == a.end()) || (a.find("count") == a.end()) || a.find("level") == a.end()) {
        MS_LOG(ERROR) << "The node field is 3, but 'key'/'count'/'level' is not existed";
        return false;
      } else {
        ini = LevelRecursive(a.at("level"));
      }
    } else {
      MS_LOG(ERROR) << "The node field is not equal 2/3";
      return false;
    }
  }
  return ini;
}

bool Statistics::operator==(const Statistics &b) const {
  if (this->get_statistics() != b.get_statistics()) {
    return false;
  }
  return true;
}
}  // namespace mindrecord
}  // namespace mindspore
