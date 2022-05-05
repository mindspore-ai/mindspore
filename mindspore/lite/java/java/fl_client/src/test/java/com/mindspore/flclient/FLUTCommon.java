package com.mindspore.flclient;

/**
 * Define common functions for ut
 * @author       : zhangzhaoju
 * @since  : 2022/4/14
 */
public class FLUTCommon {
    private static String defPathRegex = ",";

    public static String getRealPathsWithSplit(String basePath, String origin, String pathRegex) {
        if (basePath == null || basePath.isEmpty() || origin == null || origin.isEmpty()) {
            return origin;
        }
        String realRegex;
        if (pathRegex == null || pathRegex.isEmpty()) {
            realRegex = defPathRegex;
        } else {
            realRegex = pathRegex;
        }

        String[] paths = origin.split(realRegex);
        StringBuilder realPath = new StringBuilder();
        for (String path : paths) {
            if (realPath.length() > 0) {
                realPath.append(pathRegex);
            }
            realPath.append(basePath).append(path);
        }
        return realPath.toString();
    }
}
