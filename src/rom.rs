use std::fs::File;
use std::io::prelude::*;

pub struct Rom {
    pub rom: [u8; 3584],
    pub size: usize,
}

impl Rom {
    pub fn new(filename: &str) -> Self {
        let mut f = File::open(filename).expect("file not found");
        let mut buffer = [0_u8; 3584];

        let bytes_read = if let Ok(bytes_read) = f.read(&mut buffer) {
            bytes_read
        } else {
            0
        };

        Self {
            rom: buffer,
            size: bytes_read,
        }
    }
}
