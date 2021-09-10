/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

package com.mindspore.flclient.cipher.struct;

/**
 * class used define new array type
 *
 * @param <T> an array
 *
 * @since 2021-8-27
 */
public class NewArray<T> {
    private int size;
    private T array;

    /**
     * get array size
     *
     * @return array size
     */
    public int getSize() {
        return size;
    }

    /**
     * set array size
     *
     * @param size array size
     */
    public void setSize(int size) {
        this.size = size;
    }

    /**
     * get array
     *
     * @return an array
     */
    public T getArray() {
        return array;
    }

    /**
     * set array
     *
     * @param array input
     */
    public void setArray(T array) {
        this.array = array;
    }
}
