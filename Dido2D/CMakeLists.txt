#requires everything to be in their directories under CURR_SRC_DIR
set(GATHERED_INCLUDES ${GATHERED_INCLUDES} ${CURR_SRC_DIR}/Dido2D ${CURR_SRC_DIR}/OVUtilities  ${CURR_SRC_DIR}/OVTrack ${CURR_SRC_DIR} ${OpenCV_INCLUDE} ${CURR_SRC_DIR}/user_interface/server/ovlib/include PARENT_SCOPE)
 
#add_library (Dido2D Dido2D)
#target_link_libraries(Dido2D OVUtilities ${OPENCV_LIBRARIES} PTZActuator OVTrack)

set(GATHERED_SOURCES ${GATHERED_SOURCES} ${CURR_SRC_DIR}/Dido2D/Dido2D.cpp PARENT_SCOPE)
set(GATHERED_HEADERS ${GATHERED_HEADERS} ${CURR_SRC_DIR}/Dido2D/Dido2D.h PARENT_SCOPE)


