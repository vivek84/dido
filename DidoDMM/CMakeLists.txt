#requires everything to be in their directories under CURR_SRC_DIR
set(GATHERED_INCLUDES ${GATHERED_INCLUDES} ${CURR_SRC_DIR}/DidoDMM ${CURR_SRC_DIR}/OVUtilities ${CURR_SRC_DIR}/PTZActuator ${CURR_SRC_DIR}/OVTrack ${CURR_SRC_DIR} ${OpenCV_INCLUDE} PARENT_SCOPE)
 
#add_library (DidoDMM DidoDMM DidoDMM_TargetSelector)
#target_link_libraries(DidoDMM OVUtilities ${OPENCV_LIBRARIES} PTZActuator OVTrack)

set(GATHERED_SOURCES ${GATHERED_SOURCES} ${CURR_SRC_DIR}/DidoDMM/DidoDMM.cpp ${CURR_SRC_DIR}/DidoDMM/DidoDMM_TargetSelector.cpp PARENT_SCOPE)

set(GATHERED_HEADERS ${GATHERED_HEADERS} ${CURR_SRC_DIR}/DidoDMM/DidoDMM.h ${CURR_SRC_DIR}/DidoDMM/DidoDMM_TargetSelector.h PARENT_SCOPE)