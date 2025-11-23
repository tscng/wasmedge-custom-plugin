use burn::backend::{NdArray, Wgpu};
use burn::prelude::{Backend, DeviceOps};
use log::info;
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
            return true;
        }
    }

    false
}

pub async fn get_backends() {

    info!("=== WebGPU state:");

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
    info!("GPU Name: {}", info.name);
    info!("Vendor ID: {:x}", info.vendor);
    info!("Device ID: {:x}", info.device);
    info!("Backend: {:?}", info.backend);

    info!("=== Backend availability (compile-time features + runtime device check):");

    // CPU backend (always available if compiled)
    // Many examples in the repo use NdArray<f32> as the concrete type.
    info!(
        "  - NdArray (CPU) -> compiled: yes, devices: {}",
        backend_has_devices::<NdArray<f32>>()
    );

    // WGPU / WebGPU backends
    info!(
        "  - WGPU -> compiled: yes, devices: {}",
        backend_has_devices::<Wgpu>()
    );

    // CUDA (NVIDIA GPUs)
    //println!(
    //    " - CUDA -> compiled: yes, devices: {}",
    //    backend_has_devices::<burn_cuda::Cuda<f32, i32>>()
    //);

    info!("=== ");
}