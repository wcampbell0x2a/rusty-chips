use deku::prelude::*;

use rand::Rng;
use std::ops::Deref;

pub const FONT_SET: [u8; 80] = [
    0xF0, 0x90, 0x90, 0x90, 0xF0, 0x20, 0x60, 0x20, 0x20, 0x70, 0xF0, 0x10, 0xF0, 0x80, 0xF0, 0xF0,
    0x10, 0xF0, 0x10, 0xF0, 0x90, 0x90, 0xF0, 0x10, 0x10, 0xF0, 0x80, 0xF0, 0x10, 0xF0, 0xF0, 0x80,
    0xF0, 0x90, 0xF0, 0xF0, 0x10, 0x20, 0x40, 0x40, 0xF0, 0x90, 0xF0, 0x90, 0xF0, 0xF0, 0x90, 0xF0,
    0x10, 0xF0, 0xF0, 0x90, 0xF0, 0x90, 0x90, 0xE0, 0x90, 0xE0, 0x90, 0xE0, 0xF0, 0x80, 0x80, 0x80,
    0xF0, 0xE0, 0x90, 0x90, 0x90, 0xE0, 0xF0, 0x80, 0xF0, 0x80, 0xF0, 0xF0, 0x80, 0xF0, 0x80, 0x80,
];

type X = u4;
type Y = u4;
type N = u4;

const OP_SIZE: usize = 2;

#[derive(Debug, PartialEq, DekuRead, DekuWrite)]
pub struct u4 {
    #[deku(bits = 4)]
    inner: usize,
}

impl Deref for u4 {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Debug, PartialEq, DekuRead, DekuWrite)]
pub struct NNN {
    #[deku(bits = 12, endian = "big")]
    inner: usize,
}

impl Deref for NNN {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Debug, PartialEq, DekuRead, DekuWrite)]
pub struct KK {
    #[deku(bits = 8, endian = "big")]
    inner: usize,
}

impl Deref for KK {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct OutputState<'a> {
    pub vram: &'a [[u8; Processor::WIDTH]; Processor::HEIGHT],
    pub vram_changed: bool,
    pub beep: bool,
}

pub enum ProgramCounter {
    Next,
    Skip,
    Jump(usize),
}

impl ProgramCounter {
    // TODO impl From
    fn skip_cond(condition: bool) -> Self {
        if condition {
            Self::Skip
        } else {
            Self::Next
        }
    }
}

pub struct Processor {
    vram: [[u8; Self::WIDTH]; Self::HEIGHT],
    vram_changed: bool,
    ram: [u8; Self::RAM],
    stack: [usize; 16],
    v: [u8; 16],
    i: usize,
    pc: usize,
    sp: usize,
    delay_timer: u8,
    sound_timer: u8,
    keypad: [bool; 16],
    keypad_waiting: bool,
    keypad_register: usize,
}

impl Processor {
    pub const WIDTH: usize = 64;
    pub const HEIGHT: usize = 32;
    pub const RAM: usize = 4096;

    pub fn new() -> Self {
        let mut ram = [0_u8; Self::RAM];
        for i in 0..FONT_SET.len() {
            ram[i] = FONT_SET[i];
        }

        Self {
            vram: [[0; Self::WIDTH]; Self::HEIGHT],
            vram_changed: false,
            ram,
            stack: [0; 16],
            v: [0; 16],
            i: 0,
            pc: 0x200,
            sp: 0,
            delay_timer: 0,
            sound_timer: 0,
            keypad: [false; 16],
            keypad_waiting: false,
            keypad_register: 0,
        }
    }

    pub fn load(&mut self, data: &[u8]) {
        for (i, &byte) in data.iter().enumerate() {
            let addr = 0x200 + i;
            if addr < 4096 {
                self.ram[0x200 + i] = byte;
            } else {
                break;
            }
        }
    }

    pub fn tick(&mut self, keypad: [bool; 16]) -> OutputState {
        self.keypad = keypad;
        self.vram_changed = false;

        if self.keypad_waiting {
            for (i, key) in keypad.iter().enumerate() {
                if *key {
                    self.keypad_waiting = false;
                    self.v[self.keypad_register] = i as u8;
                    break;
                }
            }
        } else {
            if self.delay_timer > 0 {
                self.delay_timer -= 1
            }
            if self.sound_timer > 0 {
                self.sound_timer -= 1
            }
            self.execute(
                Instruction::from_bytes((&[self.ram[self.pc], self.ram[self.pc + 1]], 0))
                    .unwrap()
                    .1,
            );
        }

        OutputState {
            vram: &self.vram,
            vram_changed: self.vram_changed,
            beep: self.sound_timer > 0,
        }
    }

    pub fn execute(&mut self, instruction: Instruction) {
        let pc = match instruction {
            Instruction::ZeroZero(_, _, ZeroZeroType::Cls) => {
                for y in 0..Self::HEIGHT {
                    for x in 0..Self::WIDTH {
                        self.vram[y][x] = 0;
                    }
                }
                self.vram_changed = true;
                ProgramCounter::Next
            }
            Instruction::ZeroZero(_, _, ZeroZeroType::Ret) => {
                self.sp -= 1;
                ProgramCounter::Jump(self.stack[self.sp])
            }
            // 1nnn
            Instruction::Jump(nnn) => ProgramCounter::Jump(*nnn),
            // 2nnn
            Instruction::Execute(nnn) => {
                self.stack[self.sp] = self.pc + OP_SIZE;
                self.sp += 1;
                ProgramCounter::Jump(*nnn)
            }
            // 3xkk
            Instruction::SkipEq(x, kk) => ProgramCounter::skip_cond(self.v[*x] == *kk as u8),
            // 4xkk
            Instruction::SkipNotEq(x, kk) => ProgramCounter::skip_cond(self.v[*x] != *kk as u8),
            // 5xy0
            Instruction::SkipEqReg(x, y, _) => ProgramCounter::skip_cond(self.v[*x] == self.v[*y]),
            // 6xkk
            Instruction::LDValue(x, kk) => {
                self.v[*x] = *kk as u8;
                ProgramCounter::Next
            }
            // 7xkk
            Instruction::ADD(x, kk) => {
                let vx = u16::from(self.v[*x]);
                let val = *kk as u16;
                let result = vx + val;
                self.v[*x] = result as u8;
                ProgramCounter::Next
            }
            // 8xyt
            Instruction::LDRegister(x, y, EightType::Store) => {
                self.v[*x] = self.v[*y];
                ProgramCounter::Next
            }
            Instruction::LDRegister(x, y, EightType::Or) => {
                self.v[*x] |= self.v[*y];
                ProgramCounter::Next
            }
            Instruction::LDRegister(x, y, EightType::And) => {
                self.v[*x] &= self.v[*y];
                ProgramCounter::Next
            }
            Instruction::LDRegister(x, y, EightType::Xor) => {
                self.v[*x] ^= self.v[*y];
                ProgramCounter::Next
            }
            Instruction::LDRegister(x, y, EightType::Add) => {
                let vx = u16::from(self.v[*x]);
                let vy = u16::from(self.v[*y]);
                let result = vx + vy;
                self.v[*x] = result as u8;
                self.v[0x0f] = if result > 0xFF { 1 } else { 0 };
                ProgramCounter::Next
            }
            Instruction::LDRegister(x, y, EightType::Sub) => {
                self.v[0x0f] = if self.v[*x] > self.v[*y] { 1 } else { 0 };
                self.v[*x] = self.v[*x].wrapping_sub(self.v[*y]);
                ProgramCounter::Next
            }
            Instruction::LDRegister(x, _, EightType::StoreShiftLeft) => {
                self.v[0x0f] = self.v[*x] & 1;
                self.v[*x] >>= 1;
                ProgramCounter::Next
            }
            Instruction::LDRegister(x, y, EightType::Set) => {
                self.v[0x0f] = if self.v[*y] > self.v[*x] { 1 } else { 0 };
                self.v[*x] = self.v[*y].wrapping_sub(self.v[*x]);
                ProgramCounter::Next
            }
            Instruction::LDRegister(x, _, EightType::StoreShiftRight) => {
                self.v[0x0f] = (self.v[*x] & 0b1000_0000) >> 7;
                self.v[*x] <<= 1;
                ProgramCounter::Next
            }
            Instruction::SNE(x, y, _) => ProgramCounter::skip_cond(self.v[*x] != self.v[*y]),
            Instruction::StoreMemInRegister(nnn) => {
                self.i = *nnn;
                ProgramCounter::Next
            }
            Instruction::JumpToAddress(nnn) => ProgramCounter::Jump((self.v[0] as usize) + *nnn),
            Instruction::SetRandomNumberWithMask(x, kk) => {
                let mut rng = rand::thread_rng();
                self.v[*x] = rng.gen::<u8>() & *kk as u8;
                ProgramCounter::Next
            }
            Instruction::Draw(x, y, n) => {
                self.v[0x0f] = 0;
                for byte in 0..*n {
                    let y = (self.v[*y] as usize + byte) % Self::HEIGHT;
                    for bit in 0..8 {
                        let x = (self.v[*x] as usize + bit) % Self::WIDTH;
                        let color = (self.ram[self.i + byte] >> (7 - bit)) & 1;
                        self.v[0x0f] |= color & self.vram[y][x];
                        self.vram[y][x] ^= color;
                    }
                }
                self.vram_changed = true;
                ProgramCounter::Next
            }
            Instruction::SKP(x, SKPType::SkipIfPressed) => {
                ProgramCounter::skip_cond(self.keypad[self.v[*x] as usize])
            }
            Instruction::SKP(x, SKPType::SkipIfNotPressed) => {
                ProgramCounter::skip_cond(!self.keypad[self.v[*x] as usize])
            }
            Instruction::LD(x, LDType::Store) => {
                self.v[*x] = self.delay_timer;
                ProgramCounter::Next
            }
            Instruction::LD(x, LDType::Wait) => {
                self.keypad_waiting = true;
                self.keypad_register = *x;
                ProgramCounter::Next
            }
            Instruction::LD(x, LDType::SetDelayTimer) => {
                self.delay_timer = self.v[*x];
                ProgramCounter::Next
            }
            Instruction::LD(x, LDType::SetSoundTimer) => {
                self.sound_timer = self.v[*x];
                ProgramCounter::Next
            }
            Instruction::LD(x, LDType::Add) => {
                self.i += self.v[*x] as usize;
                self.v[0x0f] = if self.i > 0x0F00 { 1 } else { 0 };
                ProgramCounter::Next
            }
            Instruction::LD(x, LDType::SetSprite) => {
                self.i = (self.v[*x] as usize) * 5;
                ProgramCounter::Next
            }
            Instruction::LD(x, LDType::StoreBCD) => {
                self.ram[self.i] = self.v[*x] / 100;
                self.ram[self.i + 1] = (self.v[*x] % 100) / 10;
                self.ram[self.i + 2] = self.v[*x] % 10;
                ProgramCounter::Next
            }
            Instruction::LD(x, LDType::StoreVToMem) => {
                for i in 0..=*x {
                    self.ram[self.i + i] = self.v[i];
                }
                ProgramCounter::Next
            }
            Instruction::LD(x, LDType::FillVWithMem) => {
                for i in 0..=*x {
                    self.v[i] = self.ram[self.i + i];
                }
                ProgramCounter::Next
            }
        };
        match pc {
            ProgramCounter::Next => self.pc += OP_SIZE,
            ProgramCounter::Skip => self.pc += 2 * OP_SIZE,
            ProgramCounter::Jump(addr) => self.pc = addr,
        }
    }
}

/// Parse first byte in u32 instruction
#[derive(Debug, PartialEq, DekuRead, DekuWrite)]
#[deku(type = "u8", bits = "4")]
pub enum Instruction {
    #[deku(id = "0x0")]
    ZeroZero(u4, u4, ZeroZeroType),

    #[deku(id = "0x1")]
    Jump(NNN),

    #[deku(id = "0x2")]
    Execute(NNN),

    #[deku(id = "0x3")]
    SkipEq(X, KK),

    #[deku(id = "0x4")]
    SkipNotEq(X, KK),

    #[deku(id = "0x5")]
    SkipEqReg(X, Y, u4),

    #[deku(id = "0x6")]
    LDValue(X, KK),

    #[deku(id = "0x7")]
    ADD(X, KK),

    #[deku(id = "0x8")]
    LDRegister(X, Y, EightType),

    #[deku(id = "0x9")]
    SNE(X, Y, u4),

    #[deku(id = "0xa")]
    StoreMemInRegister(NNN),

    #[deku(id = "0xb")]
    JumpToAddress(NNN),

    #[deku(id = "0xc")]
    SetRandomNumberWithMask(X, KK),

    #[deku(id = "0xd")]
    Draw(X, Y, N),

    #[deku(id = "0xe")]
    SKP(X, SKPType),

    #[deku(id = "0xf")]
    LD(X, LDType),
}

#[derive(Debug, PartialEq, DekuRead, DekuWrite)]
#[deku(type = "u8", bits = "4")]
pub enum ZeroZeroType {
    /// Clear Display
    #[deku(id = "0x0")]
    Cls,
    /// Return from subroutine
    #[deku(id = "0xe")]
    Ret,
}

#[derive(Debug, PartialEq, DekuRead, DekuWrite)]
#[deku(type = "u8", bits = "4")]
pub enum EightType {
    #[deku(id = "0x00")]
    Store,
    #[deku(id = "0x01")]
    Or,
    #[deku(id = "0x02")]
    And,
    #[deku(id = "0x03")]
    Xor,
    #[deku(id = "0x04")]
    Add,
    #[deku(id = "0x05")]
    Sub,
    #[deku(id = "0x06")]
    StoreShiftLeft,
    #[deku(id = "0x07")]
    Set,
    #[deku(id = "0x0e")]
    StoreShiftRight,
}

#[derive(Debug, PartialEq, DekuRead, DekuWrite)]
#[deku(type = "u8", endian = "big")]
pub enum SKPType {
    #[deku(id = "0x9e")]
    SkipIfPressed,
    #[deku(id = "0xa1")]
    SkipIfNotPressed,
}

#[derive(Debug, PartialEq, DekuRead, DekuWrite)]
#[deku(type = "u8", endian = "big")]
pub enum LDType {
    #[deku(id = "0x07")]
    Store,
    #[deku(id = "0x0a")]
    Wait,
    #[deku(id = "0x15")]
    SetDelayTimer,
    #[deku(id = "0x18")]
    SetSoundTimer,
    #[deku(id = "0x1e")]
    Add,
    #[deku(id = "0x29")]
    SetSprite,
    #[deku(id = "0x33")]
    StoreBCD,
    #[deku(id = "0x55")]
    StoreVToMem,
    #[deku(id = "0x65")]
    FillVWithMem,
}

#[cfg(test)]
mod tests {
    use super::*;

    const START_PC: usize = 0xF00;
    const NEXT_PC: usize = START_PC + OP_SIZE;
    const SKIPPED_PC: usize = START_PC + (2 * OP_SIZE);
    fn build_processor() -> Processor {
        let mut processor = Processor::new();
        processor.pc = START_PC;
        processor.v = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7];
        processor
    }

    #[test]
    fn test_00ee() {
        let mut processor = build_processor();
        processor.vram = [[128; Processor::WIDTH]; Processor::HEIGHT];
        processor.execute(Instruction::from_bytes((&[0x00, 0xe0], 0)).unwrap().1);

        for y in 0..Processor::HEIGHT {
            for x in 0..Processor::WIDTH {
                assert_eq!(processor.vram[y][x], 0);
            }
        }
        assert_eq!(processor.pc, NEXT_PC);
    }

    #[test]
    fn test_1nnn() {
        let mut processor = Processor::new();
        processor.sp = 5;
        processor.stack[4] = 0x6666;
        processor.execute(Instruction::from_bytes((&[0x00, 0xee], 0)).unwrap().1);
        assert_eq!(processor.sp, 4);
        assert_eq!(processor.pc, 0x6666);
    }

    #[test]
    fn test_op_1nnn() {
        let mut processor = Processor::new();
        processor.execute(Instruction::from_bytes((&[0x16, 0x66], 0)).unwrap().1);
        assert_eq!(processor.pc, 0x0666);
    }

    #[test]
    fn test_op_2nnn() {
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0x26, 0x66], 0)).unwrap().1);
        assert_eq!(processor.pc, 0x0666);
        assert_eq!(processor.sp, 1);
        assert_eq!(processor.stack[0], NEXT_PC);
    }

    // SE VX, byte
    #[test]
    fn test_op_3xkk() {
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0x32, 0x01], 0)).unwrap().1);
        assert_eq!(processor.pc, SKIPPED_PC);
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0x32, 0x00], 0)).unwrap().1);
        assert_eq!(processor.pc, NEXT_PC);
    }
    // SNE VX, byte
    #[test]
    fn test_op_4xkk() {
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0x42, 0x00], 0)).unwrap().1);
        assert_eq!(processor.pc, SKIPPED_PC);
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0x42, 0x01], 0)).unwrap().1);
        assert_eq!(processor.pc, NEXT_PC);
    }
    // SE VX, VY
    #[test]
    fn test_op_5xy0() {
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0x55, 0x40], 0)).unwrap().1);
        assert_eq!(processor.pc, SKIPPED_PC);
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0x55, 0x00], 0)).unwrap().1);
        assert_eq!(processor.pc, NEXT_PC);
    }
    // LD Vx, byte
    #[test]
    fn test_op_6xkk() {
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0x65, 0xff], 0)).unwrap().1);
        assert_eq!(processor.v[5], 0xff);
        assert_eq!(processor.pc, NEXT_PC);
    }
    // ADD Vx, byte
    #[test]
    fn test_op_7xkk() {
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0x75, 0xf0], 0)).unwrap().1);
        assert_eq!(processor.v[5], 0xf2);
        assert_eq!(processor.pc, NEXT_PC);
    }
    // LD Vx, Vy
    #[test]
    fn test_op_8xy0() {
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0x80, 0x50], 0)).unwrap().1);
        assert_eq!(processor.v[0], 0x02);
        assert_eq!(processor.pc, NEXT_PC);
    }
    fn check_math(v1: u8, v2: u8, op: u16, result: u8, vf: u8) {
        let mut processor = build_processor();
        processor.v[0] = v1;
        processor.v[1] = v2;
        processor.v[0x0f] = 0;
        let num: u16 = 0x8010 + op;
        processor.execute(Instruction::from_bytes((&num.to_be_bytes(), 0)).unwrap().1);
        assert_eq!(processor.v[0], result);
        assert_eq!(processor.v[0x0f], vf);
        assert_eq!(processor.pc, NEXT_PC);
    }
    // OR Vx, Vy
    #[test]
    fn test_op_8xy1() {
        // 0x0F or 0xF0 == 0xFF
        check_math(0x0F, 0xF0, 1, 0xFF, 0);
    }
    // AND Vx, Vy
    #[test]
    fn test_op_8xy2() {
        // 0x0F and 0xFF == 0x0F
        check_math(0x0F, 0xFF, 2, 0x0F, 0);
    }
    // XOR Vx, Vy
    #[test]
    fn test_op_8xy3() {
        // 0x0F xor 0xFF == 0xF0
        check_math(0x0F, 0xFF, 3, 0xF0, 0);
    }
    // ADD Vx, Vy
    #[test]
    fn test_op_8xy4() {
        check_math(0x0F, 0x0F, 4, 0x1E, 0);
        check_math(0xFF, 0xFF, 4, 0xFE, 1);
    }
    // SUB Vx, Vy
    #[test]
    fn test_op_8xy5() {
        check_math(0x0F, 0x01, 5, 0x0E, 1);
        check_math(0x0F, 0xFF, 5, 0x10, 0);
    }
    // SHR Vx
    #[test]
    fn test_op_8x06() {
        // 4 >> 1 == 2
        check_math(0x04, 0, 6, 0x02, 0);
        // 5 >> 1 == 2 with carry
        check_math(0x05, 0, 6, 0x02, 1);
    }
    // SUBN Vx, Vy
    #[test]
    fn test_op_8xy7() {
        check_math(0x01, 0x0F, 7, 0x0E, 1);
        check_math(0xFF, 0x0F, 7, 0x10, 0);
    }

    // SHL Vx
    #[test]
    fn test_op_8x0e() {
        check_math(0b1100_0000, 0, 0x0e, 0b1000_0000, 1);
        check_math(0b0000_0111, 0, 0x0e, 0b0000_1110, 0);
    }

    // SNE VX, VY
    #[test]
    fn test_op_9xy0() {
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0x90, 0xe0], 0)).unwrap().1);
        assert_eq!(processor.pc, SKIPPED_PC);
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0x90, 0x10], 0)).unwrap().1);
        assert_eq!(processor.pc, NEXT_PC);
    }

    // LD I, byte
    #[test]
    fn test_op_annn() {
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0xa1, 0x23], 0)).unwrap().1);
        assert_eq!(processor.i, 0x123);
    }

    // JP V0, addr
    #[test]
    fn test_op_bnnn() {
        let mut processor = build_processor();
        processor.v[0] = 3;
        processor.execute(Instruction::from_bytes((&[0xb1, 0x23], 0)).unwrap().1);
        assert_eq!(processor.pc, 0x126);
    }

    // RND Vx, byte
    // Generates random u8, then ANDs it with kk.
    // We can't test randomness, but we can test the AND.
    #[test]
    fn test_op_cxkk() {
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0xc0, 0x00], 0)).unwrap().1);
        assert_eq!(processor.v[0], 0);
        processor.execute(Instruction::from_bytes((&[0xc0, 0x0f], 0)).unwrap().1);
        assert_eq!(processor.v[0] & 0xf0, 0);
    }
    // DRW Vx, Vy, nibble
    #[test]
    fn test_op_dxyn() {
        let mut processor = build_processor();
        processor.i = 0;
        processor.ram[0] = 0b1111_1111;
        processor.ram[1] = 0b0000_0000;
        processor.vram[0][0] = 1;
        processor.vram[0][1] = 0;
        processor.vram[1][0] = 1;
        processor.vram[1][1] = 0;
        processor.v[0] = 0;
        processor.execute(Instruction::from_bytes((&[0xd0, 0x02], 0)).unwrap().1);

        assert_eq!(processor.vram[0][0], 0);
        assert_eq!(processor.vram[0][1], 1);
        assert_eq!(processor.vram[1][0], 1);
        assert_eq!(processor.vram[1][1], 0);
        assert_eq!(processor.v[0x0f], 1);
        assert!(processor.vram_changed);
        assert_eq!(processor.pc, NEXT_PC);
    }

    #[test]
    fn test_op_dxyn_wrap_horizontal() {
        let mut processor = build_processor();

        let x = Processor::WIDTH - 4;

        processor.i = 0;
        processor.ram[0] = 0b1111_1111;
        processor.v[0] = x as u8;
        processor.v[1] = 0;
        processor.execute(Instruction::from_bytes((&[0xd0, 0x11], 0)).unwrap().1);

        assert_eq!(processor.vram[0][x - 1], 0);
        assert_eq!(processor.vram[0][x], 1);
        assert_eq!(processor.vram[0][x + 1], 1);
        assert_eq!(processor.vram[0][x + 2], 1);
        assert_eq!(processor.vram[0][x + 3], 1);
        assert_eq!(processor.vram[0][0], 1);
        assert_eq!(processor.vram[0][1], 1);
        assert_eq!(processor.vram[0][2], 1);
        assert_eq!(processor.vram[0][3], 1);
        assert_eq!(processor.vram[0][4], 0);

        assert_eq!(processor.v[0x0f], 0);
    }

    // DRW Vx, Vy, nibble
    #[test]
    fn test_op_dxyn_wrap_vertical() {
        let mut processor = build_processor();
        let y = Processor::HEIGHT - 1;

        processor.i = 0;
        processor.ram[0] = 0b1111_1111;
        processor.ram[1] = 0b1111_1111;
        processor.v[0] = 0;
        processor.v[1] = y as u8;
        processor.execute(Instruction::from_bytes((&[0xd0, 0x12], 0)).unwrap().1);

        assert_eq!(processor.vram[y][0], 1);
        assert_eq!(processor.vram[0][0], 1);
        assert_eq!(processor.v[0x0f], 0);
    }

    // SKP Vx
    #[test]
    fn test_op_ex9e() {
        let mut processor = build_processor();
        processor.keypad[9] = true;
        processor.v[5] = 9;
        processor.execute(Instruction::from_bytes((&[0xe5, 0x9e], 0)).unwrap().1);
        assert_eq!(processor.pc, SKIPPED_PC);

        let mut processor = build_processor();
        processor.v[5] = 9;
        processor.execute(Instruction::from_bytes((&[0xe5, 0x9e], 0)).unwrap().1);
        assert_eq!(processor.pc, NEXT_PC);
    }

    // SKNP Vx
    #[test]
    fn test_op_exa1() {
        let mut processor = build_processor();
        processor.keypad[9] = true;
        processor.v[5] = 9;
        processor.execute(Instruction::from_bytes((&[0xe5, 0xa1], 0)).unwrap().1);
        assert_eq!(processor.pc, NEXT_PC);

        let mut processor = build_processor();
        processor.v[5] = 9;
        processor.execute(Instruction::from_bytes((&[0xe5, 0xa1], 0)).unwrap().1);
        assert_eq!(processor.pc, SKIPPED_PC);
    }

    // LD Vx, DT
    #[test]
    fn test_op_fx07() {
        let mut processor = build_processor();
        processor.delay_timer = 20;
        processor.execute(Instruction::from_bytes((&[0xf5, 0x07], 0)).unwrap().1);
        assert_eq!(processor.v[5], 20);
        assert_eq!(processor.pc, NEXT_PC);
    }

    // LD Vx, K
    #[test]
    fn test_op_fx0a() {
        let mut processor = build_processor();
        processor.execute(Instruction::from_bytes((&[0xf5, 0x0a], 0)).unwrap().1);
        assert_eq!(processor.keypad_waiting, true);
        assert_eq!(processor.keypad_register, 5);
        assert_eq!(processor.pc, NEXT_PC);

        // Tick with no keypresses doesn't do anything
        processor.tick([false; 16]);
        assert_eq!(processor.keypad_waiting, true);
        assert_eq!(processor.keypad_register, 5);
        assert_eq!(processor.pc, NEXT_PC);

        // Tick with a keypress finishes wait and loads
        // first pressed key into vx
        processor.tick([true; 16]);
        assert_eq!(processor.keypad_waiting, false);
        assert_eq!(processor.v[5], 0);
        assert_eq!(processor.pc, NEXT_PC);
    }

    // LD DT, vX
    #[test]
    fn test_op_fx15() {
        let mut processor = build_processor();
        processor.v[5] = 9;
        processor.execute(Instruction::from_bytes((&[0xf5, 0x15], 0)).unwrap().1);
        assert_eq!(processor.delay_timer, 9);
        assert_eq!(processor.pc, NEXT_PC);
    }

    // LD ST, vX
    #[test]
    fn test_op_fx18() {
        let mut processor = build_processor();
        processor.v[5] = 9;
        processor.execute(Instruction::from_bytes((&[0xf5, 0x18], 0)).unwrap().1);
        assert_eq!(processor.sound_timer, 9);
        assert_eq!(processor.pc, NEXT_PC);
    }

    // ADD I, Vx
    #[test]
    fn test_op_fx1e() {
        let mut processor = build_processor();
        processor.v[5] = 9;
        processor.i = 9;
        processor.execute(Instruction::from_bytes((&[0xf5, 0x1e], 0)).unwrap().1);
        assert_eq!(processor.i, 18);
        assert_eq!(processor.pc, NEXT_PC);
    }

    // LD F, Vx
    #[test]
    fn test_op_fx29() {
        let mut processor = build_processor();
        processor.v[5] = 9;
        processor.execute(Instruction::from_bytes((&[0xf5, 0x29], 0)).unwrap().1);
        assert_eq!(processor.i, 5 * 9);
        assert_eq!(processor.pc, NEXT_PC);
    }

    // LD B, Vx
    #[test]
    fn test_op_fx33() {
        let mut processor = build_processor();
        processor.v[5] = 123;
        processor.i = 1000;
        processor.execute(Instruction::from_bytes((&[0xf5, 0x33], 0)).unwrap().1);
        assert_eq!(processor.ram[1000], 1);
        assert_eq!(processor.ram[1001], 2);
        assert_eq!(processor.ram[1002], 3);
        assert_eq!(processor.pc, NEXT_PC);
    }

    // LD [I], Vx
    #[test]
    fn test_op_fx55() {
        let mut processor = build_processor();
        processor.i = 1000;
        processor.execute(Instruction::from_bytes((&[0xff, 0x55], 0)).unwrap().1);
        for i in 0..16 {
            assert_eq!(processor.ram[1000 + i as usize], processor.v[i]);
        }
        assert_eq!(processor.pc, NEXT_PC);
    }

    // LD Vx, [I]
    #[test]
    fn test_op_fx65() {
        let mut processor = build_processor();
        for i in 0..16_usize {
            processor.ram[1000 + i] = i as u8;
        }
        processor.i = 1000;
        processor.execute(Instruction::from_bytes((&[0xff, 0x65], 0)).unwrap().1);

        for i in 0..16_usize {
            assert_eq!(processor.v[i], processor.ram[1000 + i]);
        }
        assert_eq!(processor.pc, NEXT_PC);
    }

    #[test]
    fn test_timers() {
        let mut processor = build_processor();
        processor.delay_timer = 200;
        processor.sound_timer = 100;
        processor.tick([false; 16]);
        assert_eq!(processor.delay_timer, 199);
        assert_eq!(processor.sound_timer, 99);
    }
}
