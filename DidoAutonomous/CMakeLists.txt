#requires everything to be in their directories under CURR_SRC_DIR
set(GATHERED_INCLUDES ${GATHERED_INCLUDES} ${CURR_SRC_DIR}/DidoAutonomous ${CURR_SRC_DIR}/OVUtilities  ${CURR_SRC_DIR}/OVTrack ${CURR_SRC_DIR} ${OpenCV_INCLUDE} ${CURR_SRC_DIR}/user_interface/server/ovlib/include PARENT_SCOPE)
#needs extra includes tbh but hey
 
#add_library (DidoAutonomous DidoAutonomous)
#target_link_libraries(DidoAutonomous OVUtilities ${OPENCV_LIBRARIES} PTZActuator OVTrack)

set(GATHERED_SOURCES ${GATHERED_SOURCES} ${CURR_SRC_DIR}/DidoAutonomous/DidoAutonomous.cpp PARENT_SCOPE)
set(GATHERED_HEADERS ${GATHERED_HEADERS} ${CURR_SRC_DIR}/DidoAutonomous/DidoAutonomous.h PARENT_SCOPE)


