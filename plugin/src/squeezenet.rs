use std::collections::HashMap;
use std::marker::PhantomData;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::prelude::{Backend, DeviceOps};
use burn::Tensor;
//use squeezenet_burn::model::{squeezenet1::Model};

const INPUT_DIM: usize = 4;
const OUTPUT_DIM: usize = 2;

pub struct SqueezenetModel<B: Backend> {
    model: i32 ,//Model<B>
    _marker: PhantomData<B>,
}

pub struct SqueezenetContext<B: Backend> {
    pub inputs: HashMap<u32, Tensor<B, INPUT_DIM>>,
    pub outputs: Vec<Tensor<B, OUTPUT_DIM>>
}

impl<B: Backend> SqueezenetModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let model = 0; //Model::new(device);
        SqueezenetModel::<B> { model, _marker: Default::default() }
    }

    pub fn compute(&self, input: Tensor<B, INPUT_DIM>) -> Tensor<B, OUTPUT_DIM> {
        //self.model.forward(input)

        let shape = input.dims(); // e.g. [B, C, H, W]
        let batch_size = shape[0];
        let flatten_dim = shape[1] * shape[2] * shape[3];

        let mut thing : Tensor<B, 2> = input.clone().reshape([batch_size, flatten_dim]);

        for _ in 0..1000000 {
            // Reshape repeatedly (simulate load)
            thing = (thing * 2).clone().reshape([batch_size, flatten_dim]);
            println!("Computed tensor: {:?}", thing);
        }
        thing
    }
}

impl<B: Backend> SqueezenetContext<B> {
    pub fn new() -> Self {
        SqueezenetContext {
            inputs: HashMap::new(),
            outputs: Vec::new()
        }
    }
    pub fn set_input(&mut self, key: u32, input: &[B::FloatElem], dimens: [usize; INPUT_DIM]) {
        let device: B::Device = Default::default();
        println!("B is: {}", std::any::type_name::<B>());
        println!("Selected device: {:?}", device.to_id());

        //let device2: WgpuDevice = WgpuDevice::DiscreteGpu(0);
        //println!("total {:?}",WgpuDevice::device_count_total());
        //let tensor = Tensor::<Wgpu, 1>::from_data(&*input, &device2).reshape(dimens);
        //println!("Selected device: {:?}", device2.to_id());

        let tensor = Tensor::<B, 1>::from_data(&*input, &device).reshape(dimens);
        self.inputs.insert(key, tensor);
    }
    pub fn get_output(&mut self, key: usize) -> Vec<<B as Backend>::FloatElem> {
        self.outputs[key].clone().into_data().to_vec()
            .expect("Failed to extract output data")
    }
}
