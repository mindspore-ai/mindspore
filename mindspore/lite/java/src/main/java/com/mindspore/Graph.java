/*
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

package com.mindspore;

import com.mindspore.config.MindsporeLite;

/**
 * Graph Class
 *
 * @since v1.0
 */
public class Graph {
    static {
        MindsporeLite.init();
    }

    private long graphPtr;

    /**
     * Construct function.
     */
    public Graph() {
        this.graphPtr = 0;
    }

    /**
     * Load file.
     *
     * @param file model file.
     * @return load status.
     */
    public boolean load(String file) {
        if (file == null) {
            return false;
        }
        this.graphPtr = loadModel(file);
        return this.graphPtr != 0L;
    }

    /**
     * Get graph pointer.
     *
     * @return graph pointer.
     */
    public long getGraphPtr() {
        return this.graphPtr;
    }

    /**
     * Fre graph pointer.
     */
    public void free() {
        this.free(graphPtr);
        graphPtr = 0;
    }

    private native long loadModel(String file);

    private native boolean free(long graphPtr);
}