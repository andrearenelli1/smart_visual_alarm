cmake_minimum_required(VERSION 3.5)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

function(add_bsp SDKCONFIG BSP)
    string(REGEX MATCH "CONFIG_${BSP}=y" REGEX_RESULT ${SDKCONFIG})
    if (REGEX_RESULT)
        set(ENV{${BSP}} 1)
    endif()
endfunction()

set(ENV{TFLITE_USE_BSP_S3_EYE} 0)
set(ENV{TFLITE_USE_BSP_KORVO_2} 0)
set(ENV{TFLITE_USE_BSP_KALUGA} 0)

if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/sdkconfig)
    file(READ ${CMAKE_CURRENT_LIST_DIR}/sdkconfig SDKCONFIG_RULE)
    add_bsp("${SDKCONFIG_RULE}" "TFLITE_USE_BSP_S3_EYE")
    add_bsp("${SDKCONFIG_RULE}" "TFLITE_USE_BSP_KORVO_2")
    add_bsp("${SDKCONFIG_RULE}" "TFLITE_USE_BSP_KALUGA")
endif()

project(smart_visual_alarm)
