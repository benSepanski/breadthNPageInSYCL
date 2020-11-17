/**
 * nvidia_selector.h
 *
 * Implements a device selector for NVIDIA GPUs
 */

#ifndef BREADTHNPAGEINSYCL_SYCLUTILS_NVIDIA_SELECTOR_
#define BREADTHNPAGEINSYCL_SYCLUTILS_NVIDIA_SELECTOR_

// custom device selector class built according to description
// in this stack overflow post:
// https://stackoverflow.com/questions/59061444/how-do-you-make-sycl-default-selector-select-an-intel-gpu-rather-than-an-nvidi
// and this example code (found from the stack overflow post):
// https://github.com/codeplaysoftware/computecpp-sdk/blob/master/samples/custom-device-selector.cpp#L46

#include <cassert>
#include <iostream>
#include <CL/sycl.hpp>

#define CL_DEVICE_PCI_BUS_ID_NV 0x4008

class NVIDIA_Selector : public cl::sycl::device_selector {
    public: 
        /**
         * @param nvidia_id the NVIDIA device id to use. Must be >= 0
         */
        NVIDIA_Selector(cl_int nvidia_id) : cl::sycl::device_selector() { 
            assert( nvidia_id >= 0);
            this->nvidia_dev_id = nvidia_id; 
        }

        /**
         * Throws an error if hits a SYCL exception or if there
         * are multiple NVIDIA platforms.
         * 
         * @return 1 iff `device` is the NVIDIA gpu with the requested nvidia id,
         *         -1 otherwise
         */
        int operator()(const cl::sycl::device &device) const override ;

    private:
        cl_int nvidia_dev_id;
};

#endif
