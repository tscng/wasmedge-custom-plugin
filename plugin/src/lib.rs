mod backends;

use wasmedge_plugin_sdk::{
    error::CoreError,
    memory::Memory,
    module::{SyncInstanceRef},
    plugin::{OptionString, register_plugin, option_string},
    types::WasmVal,
};
use wasmedge_plugin_sdk::module::PluginModule;
use crate::backends::get_backends;

pub fn create_module() -> PluginModule<()> {
    fn hello<'a>(
        _inst_ref: &'a mut SyncInstanceRef,
        _main_memory: &'a mut Memory,
        _data: &'a mut (),
        _args: Vec<WasmVal>,
    ) -> Result<Vec<WasmVal>, CoreError> {
        futures::executor::block_on(get_backends());
        Ok(vec![])
    }

    let mut module = PluginModule::create("hello", ()).unwrap();

    module.add_func("hello", (vec![], vec![]), hello).unwrap();

    module
}

register_plugin!(
    plugin_name = "hello",
    plugin_description = "burn framework adapter as wasi-nn plugin",
    version = (0,0,0,1),
    modules = [
        {"xx", "wasinn with burn backend module", create_module}
    ],
    options = [
        {
            "nn-preload",
            "Allow preload models from wasinn plugin. Each NN model can be specified as --nn-preload `COMMAND`.",
            OptionString,
            option_string!("none")
        }
    ]
);