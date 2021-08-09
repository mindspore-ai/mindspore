/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <string>

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

typedef enum Result {
  SUCCESS = 0,
  FAILED = 1
} Result;

/**
* Utils
*/
class Utils {
 public:
  /**
  * @brief create device buffer of file
  * @param [in] fileName: file name
  * @param [out] fileSize: size of file
  * @return device buffer of file
  */
  static void *GetDeviceBufferOfFile(std::string fileName, uint32_t *fileSize);

  /**
  * @brief create buffer of file
  * @param [in] fileName: file name
  * @param [out] fileSize: size of file
  * @return buffer of pic
  */
  static void* ReadBinFile(std::string fileName, uint32_t *fileSize);
};

#pragma once
