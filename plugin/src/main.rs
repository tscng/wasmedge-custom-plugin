use crate::backends::get_backends;

mod backends;


fn main() {
    futures::executor::block_on(get_backends());
}