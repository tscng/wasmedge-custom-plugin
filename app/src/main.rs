
use wasmedge_wasi_nn::{GraphBuilder, GraphEncoding, ExecutionTarget, TensorType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("WASI-NN test app");

    let model_bytes: Vec<u8> = vec![1,2,3];
    let input = vec![2f32; 224 * 224 * 3];
    let input_dim = vec![1, 224, 224, 3];
    let mut output_buffer = vec![0f32; 1001];


    // example: https://docs.rs/wasmedge-wasi-nn/0.8.0/wasmedge_wasi_nn/

    // pass a short demo array as bytes
    let graph = GraphBuilder::new(GraphEncoding::Burn, ExecutionTarget::CPU).build_from_bytes([&model_bytes, &model_bytes])?;
    let mut ctx = graph.init_execution_context()?;
    ctx.set_input(1, TensorType::F32, &input_dim, &input)?;

    // Do the inference.
    ctx.compute()?;

    // Copy output to abuffer.
    let output_bytes = ctx.get_output(1, &mut output_buffer)?;
    println!("output_bytes: {:?}", output_bytes);
    assert_eq!(output_bytes, output_buffer.len() * std::mem::size_of::<f32>());

    Ok(())
}