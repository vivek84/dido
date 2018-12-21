if(FALSE AND ${VCA_BUILD})
set(GATHERED_INCLUDES ${GATHERED_INCLUDES} ${CURR_SRC_DIR}/VCAIface ${CURR_SRC_DIR}/DidoAnalytics ${CURR_SRC_DIR}/OVUtilities  ${CURR_SRC_DIR}/OVTrack ${CURR_SRC_DIR} ${OpenCV_INCLUDE} PARENT_SCOPE)

set(GATHERED_SOURCES ${GATHERED_SOURCES} ${CURR_SRC_DIR}/DidoAnalytics/DidoAnalytics_VCA.cpp ${CURR_SRC_DIR}/DidoAnalytics/DidoAnalytics.cpp PARENT_SCOPE)
set(GATHERED_HEADERS ${GATHERED_HEADERS} ${CURR_SRC_DIR}/DidoAnalytics/DidoAnalytics_VCA.h  ${CURR_SRC_DIR}/DidoAnalytics/DidoAnalytics.h PARENT_SCOPE)
else()
#requires everything to be in their directories under CURR_SRC_DIR
set(GATHERED_INCLUDES ${GATHERED_INCLUDES} ${CURR_SRC_DIR}/DidoAnalytics ${CURR_SRC_DIR}/OVUtilities  ${CURR_SRC_DIR}/OVTrack ${CURR_SRC_DIR} ${OpenCV_INCLUDE} PARENT_SCOPE)
 
#add_library (DidoAnalytics DidoAnalytics)
#target_link_libraries(DidoAnalytics OVUtilities ${OPENCV_LIBRARIES} PTZActuator OVTrack)

set(GATHERED_SOURCES ${GATHERED_SOURCES} ${CURR_SRC_DIR}/DidoAnalytics/DidoAnalytics.cpp ${CURR_SRC_DIR}/DidoAnalytics/DidoAnalytics_EtherSec.cpp ${CURR_SRC_DIR}/DidoAnalytics/EtherSecInterfaceProxy.cpp PARENT_SCOPE)
set(GATHERED_HEADERS ${GATHERED_HEADERS} ${CURR_SRC_DIR}/DidoAnalytics/DidoAnalytics.h ${CURR_SRC_DIR}/DidoAnalytics/DidoAnalytics_EtherSec.h ${CURR_SRC_DIR}/DidoAnalytics/EtherSecInterfaceProxy.h PARENT_SCOPE)
endif()


