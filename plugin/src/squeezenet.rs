use std::collections::HashMap;
use burn::prelude::Backend;
use burn::Tensor;
use squeezenet_burn::model::{squeezenet1::Model};

const INPUT_DIM: usize = 4;
const OUTPUT_DIM: usize = 2;

pub struct SqueezenetModel<B: Backend> {
    model: Model<B>
}

pub struct SqueezenetContext<B: Backend> {
    inputs: HashMap<u32, Tensor<B, 4>>,
    outputs: Vec<Tensor<B, OUTPUT_DIM>>,
    graph: SqueezenetModel<B>
}

impl<B: Backend> SqueezenetModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let model = Model::new(device);
        SqueezenetModel { model }
    }

    pub fn compute(&self, input: Tensor<B, INPUT_DIM>) -> Tensor<B, OUTPUT_DIM> {
        self.model.forward(input)
    }
}

impl<B: Backend> SqueezenetContext<B> {
    pub fn new(graph: SqueezenetModel<B>) -> Self {
        SqueezenetContext {
            inputs: HashMap::new(),
            outputs: Vec::new(),
            graph
        }
    }
    pub fn set_input(&mut self, key: u32, input: &[B::FloatElem], dimens: [usize; INPUT_DIM]) {
        let device = Default::default();
        let tensor = Tensor::<B, 1>::from_data(&*input, &device).reshape(dimens);
        self.inputs.insert(key, tensor);
    }
    pub fn get_output(&mut self, key: usize) -> Vec<<B as Backend>::FloatElem> {
        self.outputs[key].clone().into_data().to_vec()
            .expect("Failed to extract output data")
    }
}
