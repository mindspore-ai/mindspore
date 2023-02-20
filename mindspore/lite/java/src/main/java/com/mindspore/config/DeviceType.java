/*
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
package com.mindspore.config;

/**
 * define device type
 *
 * @since v1.0
 */
public class DeviceType {

    // support cpu
    public static final int DT_CPU = 0;

    // support gpu
    public static final int DT_GPU = 1;

    // support npu
    public static final int DT_NPU = 2;

    // support ascend
    public static final int DT_ASCEND = 3;
}