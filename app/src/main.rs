#[link(wasm_import_module = "hello")]
extern "C" {
    fn hello();
}

fn main() {
    unsafe {
        hello();
    }
}