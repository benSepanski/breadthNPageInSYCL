/**
 * nvidia_selector.cpp
 *
 * Implements a device selector for NVIDIA GPUs
 */

#include "nvidia_selector.h"

#define CL_DEVICE_PCI_BUS_ID_NV 0x4008

int NVIDIA_Selector::operator()(const cl::sycl::device &device) const {
    // If the device is not an NVIDIA GPU, don't choose it
    if( !device.is_gpu() ) return -1;
    cl::sycl::string_class vendor_name = device.get_info<cl::sycl::info::device::vendor>().c_str();
    if( strcmp(vendor_name.c_str(), "NVIDIA Corporation") != 0 ) return -1;

    // Recover the NVIDIA order by sorting according to PCI bus ID
    //
    // We get PCI ID based on these post:
    // https://stackoverflow.com/questions/10852696/opencl-device-uniqueness
    // https://anteru.net/blog/2014/associating-opencl-device-ids-with-gpus/
    //
    // and note that PCI ID and NVIDIA GPU index have the same relative order
    // based on this post
    // https://stackoverflow.com/questions/15961878/how-do-the-nvidia-drivers-assign-device-indices-to-gpus
    int return_val = -1;  // +1 if device is requested NVIDIA ID, -1 otherwise
    bool nvidia_plat_found = false;
    for( const auto &plat : cl::sycl::platform::get_platforms() ) {
        // skip non-NVIDIA platforms
        if( strcmp("NVIDIA Corporation", plat.get_info<cl::sycl::info::platform::vendor>().c_str()) != 0 ) {
            continue;
        }
        if( nvidia_plat_found ) {
            fprintf(stderr, "Multiple \"NVIDIA Corporation\" platforms, aborting.\n");
            std::exit(1);
        }
        nvidia_plat_found = true;

        // Get this device's PCI bus id
        cl_int dev_pci_bus_id;
        try {
        int status = clGetDeviceInfo (device.get(), CL_DEVICE_PCI_BUS_ID_NV,
                                      sizeof(cl_int), &dev_pci_bus_id, NULL);
        }
        catch(const cl::sycl::exception &e) {
            std::cerr << "Caught SYCL exception:\n"
                      << e.what() << std::endl;
            std::exit(1);
        }

        // Loop through other device's PCI bus id to figure out NVIDIA
        // id of this device
        cl_int other_pci_bus_id;
        cl_int index = 0;
        for( const auto &other_dev : plat.get_devices() ) {
           try {
           int status = clGetDeviceInfo (other_dev.get(), CL_DEVICE_PCI_BUS_ID_NV,
                                         sizeof(cl_int), &other_pci_bus_id, NULL);
           }
           catch(const cl::sycl::exception &e) {
               std::cerr << "Caught SYCL exception:\n"
                         << e.what() << std::endl;
               std::exit(1);
           }
           if( other_pci_bus_id < dev_pci_bus_id ) {
               index++;
           }
        }

        return_val = (index == this->nvidia_dev_id) ? 1 : -1;
     }

    return return_val;
}
