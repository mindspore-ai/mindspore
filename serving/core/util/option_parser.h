/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_SERVING_OPTION_PARSER_H_
#define MINDSPORE_SERVING_OPTION_PARSER_H_

#include <string>
#include <vector>
#include <memory>

namespace mindspore {
namespace serving {

struct Arguments {
  int32_t grpc_port = 5500;
  std::string grpc_socket_path;
  std::string ssl_config_file;
  int32_t poll_model_wait_seconds = 1;
  std::string model_name;
  std::string model_path;
  std::string device_type = "Ascend";
  int32_t device_id = 0;
};

class Option {
 public:
  Option(const std::string &name, int32_t *default_point, const std::string &usage);
  Option(const std::string &name, bool *default_point, const std::string &usage);
  Option(const std::string &name, std::string *default_point, const std::string &usage);
  Option(const std::string &name, float *default_point, const std::string &usage);

 private:
  friend class Options;

  bool ParseInt32(std::string *arg);
  bool ParseBool(std::string *arg);
  bool ParseString(std::string *arg);
  bool ParseFloat(std::string *arg);
  bool Parse(std::string *arg);
  std::string name_;
  enum { MS_TYPE_INT32, MS_TYPE_BOOL, MS_TYPE_STRING, MS_TYPE_FLOAT } type_;
  int32_t *int32_default_;
  bool *bool_default_;
  std::string *string_default_;
  float *float_default_;
  std::string usage_;
};

class Options {
 public:
  ~Options() = default;
  Options(const Options &) = delete;
  Options &operator=(const Options &) = delete;
  static Options &Instance();
  bool ParseCommandLine(int argc, char **argv);
  void Usage();
  std::shared_ptr<Arguments> GetArgs() { return args_; }

 private:
  Options();
  void CreateOptions();
  bool CheckOptions();
  static std::shared_ptr<Options> inst_;
  std::string usage_;
  std::vector<Option> options_;
  std::shared_ptr<Arguments> args_;
};

}  // namespace serving
}  // namespace mindspore

#endif
