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
#include "../inc/utils.h"
#include "acl/acl.h"

/**
* ModelProcess
*/
class ModelProcess {
 public:
  /**
  * @brief Constructor
  */
  ModelProcess();

  /**
  * @brief Destructor
  */
  ~ModelProcess();

  /**
  * @brief load model from file with mem
  * @param [in] modelPath: model path
  * @return result
  */
  Result LoadModelFromFileWithMem(const char *modelPath);

  /**
  * @brief unload model
  */
  void Unload();

  /**
  * @brief create model desc
  * @return result
  */
  Result CreateDesc();

  /**
  * @brief destroy desc
  */
  void DestroyDesc();

  /**
  * @brief create model input
  * @param [in] inputDataBuffer: input buffer
  * @param [in] bufferSize: input buffer size
  * @return result
  */
  Result CreateInput(void *inputDataBuffer, size_t bufferSize);

  /**
  * @brief destroy input resource
  */
  void DestroyInput();

  /**
  * @brief create output buffer
  * @return result
  */
  Result CreateOutput();

  /**
  * @brief destroy output resource
  */
  void DestroyOutput();

  /**
  * @brief model execute
  * @return result
  */
  Result Execute();

  /**
  * @brief dump model output result to file
  */
  void DumpModelOutputResult(char *output_name);

  /**
  * @brief get model output result
  */
  void OutputModelResult();

 private:
  uint32_t modelId_;
  size_t modelMemSize_;
  size_t modelWeightSize_;
  void *modelMemPtr_;
  void *modelWeightPtr_;
  bool loadFlag_;  // model load flag
  aclmdlDesc *modelDesc_;
  aclmdlDataset *input_;
  aclmdlDataset *output_;
};

