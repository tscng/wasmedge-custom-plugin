use std::marker::PhantomData;
use burn::prelude::Backend;

pub struct WhisperModel<B: Backend> {
    _marker: PhantomData<B>,
}

pub struct WhisperContext<B: Backend> {
    _marker: PhantomData<B>,
}