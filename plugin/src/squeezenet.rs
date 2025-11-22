use std::collections::HashMap;
use burn::prelude::Backend;

pub struct SqueezenetModel<B: Backend> {
    model: squeezenet_burn::model::squeezenet1::Model<B>
}

pub struct SqueezenetContext<B: Backend> {
    inputs:  HashMap<usize, burn::Tensor<B, 4>>,
    outputs: HashMap<usize, burn::Tensor<B, 2>>,
    graph: SqueezenetModel<B>
}