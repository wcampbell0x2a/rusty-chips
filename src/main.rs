use std::env;
use std::thread;
use std::time::Duration;

mod audio;
mod display;
mod input;
mod processor;
mod rom;

use crate::audio::{Audio, SquareWave};
use crate::display::Display;
use crate::input::Input;
use crate::processor::{Instruction, Processor};
use crate::rom::Rom;
use deku::DekuContainerRead;

fn main() {
    let sleep_duration = Duration::from_millis(2);

    let sdl_context = sdl2::init().unwrap();

    let args: Vec<String> = env::args().collect();
    let cartridge_filename = &args[1];

    let rom = Rom::new(cartridge_filename);
    let audio = Audio::new(&sdl_context);
    let mut display = Display::new(&sdl_context);
    let mut input = Input::new(&sdl_context);
    let mut processor = Processor::new();

    processor.load(&rom.rom);

    while let Ok(keypad) = input.poll() {
        let output = processor.tick(keypad);

        if output.vram_changed {
            display.draw(output.vram);
        }

        if output.beep {
            audio.start_beep();
        } else {
            audio.stop_beep();
        }

        thread::sleep(sleep_duration);
    }
}
