mod backends;
mod wasi_nn;
mod helper;

use wasmedge_plugin_sdk::{
    error::CoreError,
    memory::Memory,
    module::{SyncInstanceRef},
    plugin::{register_plugin},
    types::WasmVal,
};
use wasmedge_plugin_sdk::module::PluginModule;
use wasmedge_plugin_sdk::types::ValType;

use wasmedge_wasi_nn::TensorType;
use crate::backends::get_backends;

pub enum ErrNo {
    Success = 0,              // No error occurred.
    InvalidArgument = 1,      // Caller module passed an invalid argument.
    InvalidEncoding = 2,      // Invalid encoding.
    MissingMemory = 3,        // Caller module is missing a memory export.
    Busy = 4,                 // Device or resource busy.
    RuntimeError = 5,         // Runtime Error.
    UnsupportedOperation = 6, // Unsupported Operation.
    TooLarge = 7,             // Too Large.
    NotFound = 8,             // Not Found.
}


#[derive(Debug)]
#[repr(C)]
struct WasiTensorData {
    dimens_ptr: u32,
    dimens_length: u32,
    tensor_type: TensorType,
    tensor_ptr: u32,
    tensor_length: u32,
}

pub fn create_module() -> PluginModule<()> {

    // debug backends
    futures::executor::block_on(get_backends());

    // define functions that will be accessible to call via the interface
    fn load<'a>(
        _inst_ref: &'a mut SyncInstanceRef,
        memory: &'a mut Memory,
        _data: &'a mut (),
        args: Vec<WasmVal>,
    ) -> Result<Vec<WasmVal>, CoreError> {
        if let [WasmVal::I32(data_ptr),
                WasmVal::I32(data_len),
                WasmVal::I32(encoding),
                WasmVal::I32(target),
                WasmVal::I32(graph_handle_ptr)] = &args[..]
        {
            wasi_nn::load(data_ptr, data_len, encoding, target, graph_handle_ptr, memory)
        }
        else {
            Ok(vec![WasmVal::I32(ErrNo::InvalidArgument as i32)])
        }
    }

    fn load_by_name<'a>(
        _inst_ref: &'a mut SyncInstanceRef,
        _memory: &'a mut Memory,
        _data: &'a mut (),
        _args: Vec<WasmVal>,
    ) -> Result<Vec<WasmVal>, CoreError> {
        Ok(vec![WasmVal::I32(ErrNo::UnsupportedOperation as i32)])
    }

    fn load_by_name_with_config<'a>(
        _inst_ref: &'a mut SyncInstanceRef,
        memory: &'a mut Memory,
        _data: &'a mut (),
        args: Vec<WasmVal>,
    ) -> Result<Vec<WasmVal>, CoreError> {
        Ok(vec![WasmVal::I32(ErrNo::UnsupportedOperation as i32)])
    }

    fn init_execution_context<'a>(
        _inst_ref: &'a mut SyncInstanceRef,
        memory: &'a mut Memory,
        _data: &'a mut (),
        args: Vec<WasmVal>,
    ) -> Result<Vec<WasmVal>, CoreError> {
        if let [WasmVal::I32(graph_handle), WasmVal::I32(ctx_handle_ptr)] = &args[..]
        {
            wasi_nn::init_execution_context(graph_handle, ctx_handle_ptr, memory)
        }
        else {
            Ok(vec![WasmVal::I32(ErrNo::InvalidArgument as i32)])
        }
    }

    fn set_input<'a>(
        _inst_ref: &'a mut SyncInstanceRef,
        memory: &'a mut Memory,
        _data: &'a mut (),
        args: Vec<WasmVal>,
    ) -> Result<Vec<WasmVal>, CoreError> {
        if let [WasmVal::I32(ctx_handle),
                WasmVal::I32(input_index),
                WasmVal::I32(tensor_ptr)] = &args[..]
        {
            wasi_nn::set_input(ctx_handle, input_index, tensor_ptr, memory)
        }
        else {
            Ok(vec![WasmVal::I32(ErrNo::InvalidArgument as i32)])
        }
    }

    fn get_output<'a>(
        _inst_ref: &'a mut SyncInstanceRef,
        memory: &'a mut Memory,
        _data: &'a mut (),
        args: Vec<WasmVal>,
    ) -> Result<Vec<WasmVal>, CoreError> {
        if let [WasmVal::I32(ctx_handle),
                WasmVal::I32(output_index),
                WasmVal::I32(output_ptr),
                WasmVal::I32(output_max_size),
                WasmVal::I32(output_written_len_ptr)] = &args[..]
        {
            wasi_nn::get_output(ctx_handle, output_index, output_ptr, output_max_size, output_written_len_ptr, memory)
        }
        else {
            Ok(vec![WasmVal::I32(ErrNo::InvalidArgument as i32)])
        }
    }

    fn get_output_single<'a>(
        _inst_ref: &'a mut SyncInstanceRef,
        _main_memory: &'a mut Memory,
        _data: &'a mut (),
        _args: Vec<WasmVal>,
    ) -> Result<Vec<WasmVal>, CoreError> {
        Ok(vec![WasmVal::I32(ErrNo::UnsupportedOperation as i32)])
    }

    fn compute<'a>(
        _inst_ref: &'a mut SyncInstanceRef,
        _main_memory: &'a mut Memory,
        _data: &'a mut (),
        args: Vec<WasmVal>,
    ) -> Result<Vec<WasmVal>, CoreError> {
        if let [WasmVal::I32(ctx_handle)] = &args[..]
        {
            wasi_nn::compute(ctx_handle)
        }
        else {
            Ok(vec![WasmVal::I32(ErrNo::InvalidArgument as i32)])
        }
    }

    fn compute_single<'a>(
        _inst_ref: &'a mut SyncInstanceRef,
        _main_memory: &'a mut Memory,
        _data: &'a mut (),
        _args: Vec<WasmVal>,
    ) -> Result<Vec<WasmVal>, CoreError> {
        Ok(vec![WasmVal::I32(ErrNo::UnsupportedOperation as i32)])
    }

    fn fini_single<'a>(
        _inst_ref: &'a mut SyncInstanceRef,
        _main_memory: &'a mut Memory,
        _data: &'a mut (),
        _args: Vec<WasmVal>,
    ) -> Result<Vec<WasmVal>, CoreError> {
        Ok(vec![WasmVal::I32(ErrNo::UnsupportedOperation as i32)])
    }

    fn unload<'a>(
        _inst_ref: &'a mut SyncInstanceRef,
        _main_memory: &'a mut Memory,
        _data: &'a mut (),
        _args: Vec<WasmVal>,
    ) -> Result<Vec<WasmVal>, CoreError> {
        Ok(vec![WasmVal::I32(ErrNo::UnsupportedOperation as i32)])
    }


    // create module with bound functions
    let mut module = PluginModule::create("wasi_ephemeral_nn", ()).unwrap();
    module
        .add_func(
            "load",
            (vec![ValType::I32; 5], vec![ValType::I32]),
            load
        )
        .unwrap();
    module
        .add_func(
            "unload",
            (vec![ValType::I32; 1], vec![ValType::I32]),
            unload
        )
        .unwrap();
    module
        .add_func(
            "load_by_name",
            (vec![ValType::I32; 3], vec![ValType::I32]),
            load_by_name,
        )
        .unwrap();
    module
        .add_func(
            "load_by_name_with_config",
            (vec![ValType::I32; 5], vec![ValType::I32]),
            load_by_name_with_config,
        )
        .unwrap();
    module
        .add_func(
            "init_execution_context",
            (vec![ValType::I32; 2], vec![ValType::I32]),
            init_execution_context,
        )
        .unwrap();
    module
        .add_func(
            "set_input",
            (vec![ValType::I32; 3], vec![ValType::I32]),
            set_input,
        )
        .unwrap();
    module
        .add_func(
            "compute",
            (vec![ValType::I32; 1], vec![ValType::I32]),
            compute,
        )
        .unwrap();
    module
        .add_func(
            "compute_single",
            (vec![ValType::I32; 1], vec![ValType::I32]),
            compute_single,
        )
        .unwrap();
    module
        .add_func(
            "fini_single",
            (vec![ValType::I32; 1], vec![ValType::I32]),
            fini_single,
        )
        .unwrap();
    module
        .add_func(
            "get_output",
            (vec![ValType::I32; 5], vec![ValType::I32]),
            get_output,
        )
        .unwrap();
    module
        .add_func(
            "get_output_single",
            (vec![ValType::I32; 5], vec![ValType::I32]),
            get_output_single,
        )
        .unwrap();
    module
}

register_plugin!(
    plugin_name = "wasi_nn",
    plugin_description = "Limited wasi-nn implementation for a burn backend",
    version = (0,0,0,1),
    modules = [
        {"wasi_nn", "Limited wasi-nn implementation for a burn backend", create_module}
    ]
);