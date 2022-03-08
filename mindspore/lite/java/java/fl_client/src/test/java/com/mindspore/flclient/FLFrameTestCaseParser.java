package com.mindspore.flclient;

import com.alibaba.fastjson.JSON;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Class used to parse case file
 * @author       : zhangzhaoju
 * @since  : 2022/4/14
 */
public class FLFrameTestCaseParser {
    private List<FLFrameTestCase> testCases;
    private String caseFilePath;

    FLFrameTestCaseParser(String path) {
        caseFilePath = path;
    }

    public List<Object[]> Parser() throws IOException {
        File caseFile = new File(caseFilePath);
        String strCase = FileUtils.readFileToString(caseFile);
        testCases = JSON.parseArray(strCase, FLFrameTestCase.class);
        List<Object[]> casePairs = new ArrayList<>();
        for (FLFrameTestCase tcase : testCases) {
            Object[] objs = new Object[2];
            objs[0] = tcase.getCaseName();
            objs[1] = tcase;
            casePairs.add(objs);
        }
        return casePairs;
    }

    public void testParser() throws IOException {
        Parser();
    }
}
