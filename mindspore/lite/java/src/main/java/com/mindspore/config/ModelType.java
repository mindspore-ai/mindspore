/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

package com.mindspore.config;

/**
 * define model type
 *
 * @since v1.0
 */
public class ModelType {

    // mindir type
    public static final int MT_MINDIR = 0;

    // air type
    public static final int MT_AIR = 1;

    // om type
    public static final int MT_OM = 2;

    // onnx type
    public static final int MT_ONNX = 3;

    // mindir opt type
    public static final int MT_MINDIR_OPT = 4;
}