use burn::backend::{NdArray, Wgpu};
use burn::prelude::{Backend, DeviceOps};
use wgpu::{BackendOptions, Backends, Instance, InstanceDescriptor, InstanceFlags, MemoryBudgetThresholds};

/// Generic helper that checks whether a backend exposes any devices.
///
/// Many backends provide a `device_count(type_id: u16) -> usize`
/// method on their `Device` type. Commonly `type_id == 0` is the primary
/// device-family (e.g. default GPU type). We probe a few type ids to be
/// robust to backends that distinguish device families.
fn backend_has_devices<B: Backend>() -> bool {
    // How many different type_ids to probe. Increase if you know a backend
    // uses other type_id ranges.
    const MAX_TYPE_IDS: u16 = 4;

    for type_id in 0..MAX_TYPE_IDS {
        if B::Device::device_count(type_id) > 0 {
            println!("Device has {} type {}", B::Device::device_count(type_id), type_id);
            return true;
        }
    }

    false
}

pub async fn get_backends() {

    println!("WebGPU state:");

    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::all(),
        flags: InstanceFlags::empty(), // no special flags
        memory_budget_thresholds: MemoryBudgetThresholds::default(), // default memory limits
        backend_options: BackendOptions::default(), // default backend options
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to find an appropriate adapter");

    let info = adapter.get_info();
    println!("GPU Name: {}", info.name);
    println!("Vendor ID: {:x}", info.vendor);
    println!("Device ID: {:x}", info.device);
    println!("Backend: {:?}", info.backend);

    println!("\n Backend availability (compile-time features + runtime device check):");

    // CPU backend (always available if compiled)
    // Many examples in the repo use NdArray<f32> as the concrete type.
    println!(
        "  - NdArray (CPU) -> compiled: yes, devices: {}",
        backend_has_devices::<NdArray<f32>>()
    );

    // WGPU / WebGPU backends
    println!(
        "  - WGPU -> compiled: yes, devices: {}",
        backend_has_devices::<Wgpu>()
    );

    // CUDA (NVIDIA GPUs)
    //println!(
    //    " - CUDA -> compiled: yes, devices: {}",
    //    backend_has_devices::<burn_cuda::Cuda<f32, i32>>()
    //);

    println!("\n=====\n");
}