/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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

package com.mindspore.flclient.compression;

import java.util.HashMap;
import java.util.Map;

import static mindspore.schema.CompressType.NO_COMPRESS;
import static mindspore.schema.CompressType.QUANT;

/**
 * The compress mod.
 *
 * @since 2021-12-21
 */

public class CompressMode {
    // compress type -> num bits
    public static final Map<Byte, Integer> COMPRESS_TYPE_MAP = new HashMap<>();

    static {
        COMPRESS_TYPE_MAP.put(NO_COMPRESS, -1);
        COMPRESS_TYPE_MAP.put(QUANT, 8);
    }

}