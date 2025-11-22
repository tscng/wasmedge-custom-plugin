use std::collections::HashMap;
use wasmedge_plugin_sdk::error::CoreError;
use wasmedge_plugin_sdk::memory::Memory;
use wasmedge_plugin_sdk::types::WasmVal;
use std::mem;
use std::sync::Mutex;
use burn::backend::{NdArray, Wgpu};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::prelude::{Backend, DeviceOps};
use crate::{ErrNo, WasiTensorData};
use crate::helper::get_slice;
use crate::squeezenet::{SqueezenetContext, SqueezenetModel};
use crate::whisper::{WhisperContext, WhisperModel};

const INPUT_DIM: usize = 4;

type NdArrayBackend = NdArray<f32>;
type WgpuBackend = Wgpu;

pub enum Graph<B: Backend> {
    Squeezenet(SqueezenetModel<B>),
    Whisper(WhisperModel<B>),
}

pub enum GraphWithBackend {
    WithWgpu(Graph<WgpuBackend>),
    WithNdArray(Graph<NdArrayBackend>),
}

pub enum Context<B: Backend> {
    Squeezenet(SqueezenetContext<B>),
    Whisper(WhisperContext<B>),
}

pub enum ContextWithBackend {
    WithWgpu(Context<WgpuBackend>),
    WithNdArray(Context<NdArrayBackend>),
}


pub struct WasiNN {
    next_id: i32,
    graphs: Mutex<HashMap<i32, GraphWithBackend>>,
    contexts: Mutex<HashMap<i32, (ContextWithBackend, i32)>>,
}

impl WasiNN {

    pub fn new() -> Self {
        WasiNN {
            next_id: 0,
            graphs: Mutex::new(HashMap::new()),
            contexts: Mutex::new(HashMap::new()),
        }
    }

    pub fn load<'a>(
        &mut self,
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
        println!("Test graph data_ptr: {:?}", *data_ptr);
        println!("Test graph data_len: {:?}", *data_len);
        println!("Test graph encoding: {:?}", *encoding);
        println!("Test graph target: {:?}", *target);

        // must be burn encoding
        if(*encoding != 8){
            return Ok(vec![WasmVal::I32(ErrNo::InvalidEncoding as i32)]);
        }

        // init graph; only squeezenet for now
        let id = self.next_id;
        self.next_id = id + 1;

        // if target is gpu, only wgpu for now as backend
        if(*target == 1) {

            let device = WgpuDevice::DefaultDevice;

            // 0:discrete, 1:integrated, 2:virtual, 3:cpu, 4:default
            println!("Selected device: {:?}", WgpuDevice::IntegratedGpu(0).to_id());

            let graph = Graph::Squeezenet(SqueezenetModel::<WgpuBackend>::new(&device));
            self.graphs.lock().unwrap().insert(id, GraphWithBackend::WithWgpu(graph));
        }

        else if(*target == 0) {
            let device = NdArrayDevice::default();
            println!("Selected device: {:?}", device);

            let graph = Graph::Squeezenet(SqueezenetModel::<NdArrayBackend>::new(&device));
            self.graphs.lock().unwrap().insert(id, GraphWithBackend::WithNdArray(graph));

        }

        // unsupported target
        else {
            return Ok(vec![WasmVal::I32(ErrNo::InvalidArgument as i32)]);
        }

        // write handle to pointer
        memory.write_data((*graph_handle_ptr as usize).into(), id);
        println!("Created graph handle: {:?}", id);

        Ok(vec![WasmVal::I32(ErrNo::Success as i32)])
    }

    pub fn init_execution_context<'a>(
        &mut self,
        graph_handle: &i32,
        ctx_handle_ptr: &i32,
        memory: &'a mut Memory
    ) -> Result<Vec<WasmVal>, CoreError> {

        // check if graph handle exists
        if let Some(handle) = self.graphs.lock().unwrap().get(graph_handle) {

            let id = self.next_id;
            self.next_id = id + 1;

            // create context handle based on graph type
            match handle {
                GraphWithBackend::WithNdArray(graph) => {
                    match graph {
                        Graph::Squeezenet(_) => {
                            let context = SqueezenetContext::<NdArrayBackend>::new();
                            self.contexts.lock().unwrap().insert(id, (
                                ContextWithBackend::WithNdArray(Context::Squeezenet(context)), *graph_handle
                            ));
                        }
                        _ => {
                            return Ok(vec![WasmVal::I32(ErrNo::UnsupportedOperation as i32)]);
                        }
                    }
                }
                GraphWithBackend::WithWgpu(graph) => {
                    match graph {
                        Graph::Squeezenet(_) => {
                            let context = SqueezenetContext::<WgpuBackend>::new();
                            self.contexts.lock().unwrap().insert(id, (
                                ContextWithBackend::WithWgpu(Context::Squeezenet(context)), *graph_handle
                            ));
                        }
                        _ => {
                            return Ok(vec![WasmVal::I32(ErrNo::UnsupportedOperation as i32)]);
                        }
                    }
                }
            }

            // write handle to pointer
            memory.write_data((*ctx_handle_ptr as usize).into(), id);
            println!("Created context handle: {:?}", id);

            Ok(vec![WasmVal::I32(ErrNo::Success as i32)])
        }
        else {
            Ok(vec![WasmVal::I32(ErrNo::NotFound as i32)])
        }
    }

    pub fn set_input<'a>(
        &mut self,
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

                if let Some(handle) = self.contexts.lock().unwrap().get_mut(ctx_handle) {

                    match handle {
                        (ContextWithBackend::WithNdArray(context), _) => {
                            match context {
                                Context::Squeezenet(squeezenet_context) => {
                                    squeezenet_context.set_input(*input_index as u32, &tensor, dimensions);
                                }
                                _ => {
                                    return Ok(vec![WasmVal::I32(ErrNo::UnsupportedOperation as i32)]);
                                }
                            }
                        }
                        (ContextWithBackend::WithWgpu(context), _) => {
                            match context {
                                Context::Squeezenet(squeezenet_context) => {
                                    squeezenet_context.set_input(*input_index as u32, &tensor, dimensions);
                                }
                                _ => {
                                    return Ok(vec![WasmVal::I32(ErrNo::UnsupportedOperation as i32)]);
                                }
                            }
                        }
                    }
                }
                else {
                    return Ok(vec![WasmVal::I32(ErrNo::NotFound as i32)]);
                }

                println!("Set input tensor context: {:?}[{:?}] : {:?} -> {:?} ...", ctx_handle, input_index, dimensions, &tensor[0..10]);

                Ok(vec![WasmVal::I32(ErrNo::Success as i32)])
            }
            None => Ok(vec![WasmVal::I32(ErrNo::MissingMemory as i32)]),
        }
    }

    pub fn compute<'a>(
        &mut self,
        ctx_handle: &i32
    ) -> Result<Vec<WasmVal>, CoreError> {

        if let Some(handle) = self.contexts.lock().unwrap().get_mut(ctx_handle) {
            match handle {
                (ContextWithBackend::WithNdArray(context), graphHandle) => {
                    if let Some(graph) = self.graphs.lock().unwrap().get(graphHandle) {
                        match (context, graph) {
                            (Context::Squeezenet(squeezenet_context), GraphWithBackend::WithNdArray(Graph::Squeezenet(squeezenet_model))) => {
                                // get input tensor
                                let input_tensor = squeezenet_context.inputs.get(&0).unwrap();
                                // compute
                                let output_tensor = squeezenet_model.compute(input_tensor.clone());
                                // store output
                                squeezenet_context.outputs.push(output_tensor);
                            }
                            _ => {
                                return Ok(vec![WasmVal::I32(ErrNo::UnsupportedOperation as i32)]);
                            }
                        }
                    }
                }
                (ContextWithBackend::WithWgpu(context), graph_handle) => {
                    if let Some(graph) = self.graphs.lock().unwrap().get(graph_handle) {
                        match (context, graph) {
                            (Context::Squeezenet(squeezenet_context), GraphWithBackend::WithWgpu(Graph::Squeezenet(squeezenet_model))) => {
                                // get input tensor
                                let input_tensor = squeezenet_context.inputs.get(&1).unwrap();
                                // compute
                                let output_tensor = squeezenet_model.compute(input_tensor.clone());
                                // store output
                                println!("Computed output tensor: {:?}", output_tensor);
                                squeezenet_context.outputs.push(output_tensor);
                            }
                            _ => {
                                return Ok(vec![WasmVal::I32(ErrNo::UnsupportedOperation as i32)]);
                            }
                        }
                    }
                }
            }
        } else {
            return Ok(vec![WasmVal::I32(ErrNo::NotFound as i32)]);
        }

        println!("Computed context: {:?}", ctx_handle);

        Ok(vec![WasmVal::I32(ErrNo::Success as i32)])
    }

    pub fn get_output<'a>(
        &mut self,
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
}

