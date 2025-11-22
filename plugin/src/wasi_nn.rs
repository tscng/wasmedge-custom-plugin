use wasmedge_plugin_sdk::error::CoreError;
use wasmedge_plugin_sdk::memory::Memory;
use wasmedge_plugin_sdk::types::WasmVal;
use std::mem;
use crate::{ErrNo, WasiTensorData};
use crate::helper::get_slice;

// TODO adapt to used model
pub const INPUT_DIM: usize = 4;
pub const OUTPUT_DIM: usize = 2;

pub fn load<'a>(
    data_ptr: &i32,
    data_len: &i32,
    encoding: &i32,
    target: &i32,
    graph_handle_ptr: &i32,
    memory: &'a mut Memory
) -> Result<Vec<WasmVal>, CoreError> {

    let bytes = memory
        .data_pointer(*data_ptr as usize, *data_len as usize)
        .unwrap();

    // print for debugging
    let name = String::from_utf8_lossy(&bytes);
    println!("Test bytes: {}", name);
    println!("Test graph data_ptr: {:?}", data_ptr);
    println!("Test graph data_len: {:?}", data_len);
    println!("Test graph encoding: {:?}", encoding);
    println!("Test graph target: {:?}", target);

    // TODO
    // load burn squeezenet graph, create hashmap and put handle

    // write handle to pointer
    let handle = 80085;
    memory.write_data((*graph_handle_ptr as usize).into(), handle);

    Ok(vec![WasmVal::I32(ErrNo::Success as i32)])
}

pub fn init_execution_context<'a>(
    graph_handle: &i32,
    ctx_handle_ptr: &i32,
    memory: &'a mut Memory
) -> Result<Vec<WasmVal>, CoreError> {

    // TODO
    // check if graph handle exists
    if(*graph_handle != 80085){
        Ok(vec![WasmVal::I32(ErrNo::NotFound as i32)])
    }
    else {

        // TODO
        // create execution context for graph handle, store in hashmap
        let ctx_handle = 3055;

        // write handle to pointer
        memory.write_data((*ctx_handle_ptr as usize).into(), ctx_handle);

        Ok(vec![WasmVal::I32(ErrNo::Success as i32)])
    }
}

pub fn set_input<'a>(
    ctx_handle: &i32,
    input_index: &i32,
    input_tensor_ptr: &i32,
    memory: &'a mut Memory
) -> Result<Vec<WasmVal>, CoreError> {

    match memory.get_data::<WasiTensorData>((*input_tensor_ptr as usize).into()) {
        Some(input_tensor) => {
            let raw_dimensions = get_slice!(
                            memory,
                            input_tensor.dimens_ptr,
                            INPUT_DIM * mem::size_of::<u32>(),
                            u32
                        );
            let dimensions: [usize; INPUT_DIM] = raw_dimensions
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<usize>>()
                .try_into()
                .unwrap();

            // FIXME: The type of f32 should be decided at runtime based on input_tensor.tensor_type.
            let tensor = get_slice!(
                            memory,
                            input_tensor.tensor_ptr,
                            input_tensor.tensor_length,
                            f32
                        );

            // TODO:
            // get context
            // reshape input
            // store input in hashmap index (batch)

            println!("Set input tensor context: {:?}", ctx_handle);
            println!("Set input tensor index: {:?}", input_index);
            println!("Set input tensor dimensions: {:?}", dimensions);
            println!("Set input tensor first 10 values: {:?}", &tensor[0..10]);

            Ok(vec![WasmVal::I32(ErrNo::Success as i32)])
        }
        None => Ok(vec![WasmVal::I32(ErrNo::MissingMemory as i32)]),
    }
}

pub fn compute<'a>(
    ctx_handle: &i32
) -> Result<Vec<WasmVal>, CoreError> {

    // TODO
    // get context from hashmap
    // get graph from context
    // get context inputs
    // compute graph with inputs
    // store outputs in context

    println!("Computing context: {:?}", ctx_handle);

    Ok(vec![WasmVal::I32(ErrNo::Success as i32)])
}

pub fn get_output<'a>(
    ctx_handle: &i32,
    output_index: &i32,
    output_ptr: &i32,
    output_max_size: &i32,
    output_written_len_ptr: &i32,
    memory: &'a mut Memory
) -> Result<Vec<WasmVal>, CoreError> {

    // TODO
    // get context from hashmap
    // get output at index from context
    // check length
    // write output to output_ptr
    // check written length


    let written_length = 69;
    memory.write_data((*output_written_len_ptr as usize).into(), written_length);

    Ok(vec![WasmVal::I32(ErrNo::Success as i32)])
}