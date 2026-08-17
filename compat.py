"""Runtime compatibility patches for legacy Mario dependencies."""

from __future__ import annotations

_PATCHED = False


def _patch_nes_py_rom():
    from nes_py._rom import ROM

    ROM.prg_rom_size = property(lambda self: 16 * int(self.header[4]))
    ROM.chr_rom_size = property(lambda self: 8 * int(self.header[5]))

    def prg_ram_size(self):
        size = int(self.header[8])
        if size == 0:
            size = 1
        return 8 * size

    ROM.prg_ram_size = property(prg_ram_size)
    ROM.prg_rom_stop = property(lambda self: self.prg_rom_start + self.prg_rom_size * 2**10)
    ROM.chr_rom_stop = property(lambda self: self.chr_rom_start + self.chr_rom_size * 2**10)


def _patch_gym_super_mario_bros():
    from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

    SuperMarioBrosEnv._level = property(
        lambda self: int(self.ram[0x075F]) * 4 + int(self.ram[0x075C])
    )
    SuperMarioBrosEnv._world = property(lambda self: int(self.ram[0x075F]) + 1)
    SuperMarioBrosEnv._stage = property(lambda self: int(self.ram[0x075C]) + 1)
    SuperMarioBrosEnv._area = property(lambda self: int(self.ram[0x0760]) + 1)
    SuperMarioBrosEnv._life = property(lambda self: int(self.ram[0x075A]))
    SuperMarioBrosEnv._x_position = property(
        lambda self: int(self.ram[0x6D]) * 0x100 + int(self.ram[0x86])
    )
    SuperMarioBrosEnv._x_position_screen = property(
        lambda self: (int(self.ram[0x86]) - int(self.ram[0x071C])) % 256
    )
    SuperMarioBrosEnv._y_pixel = property(lambda self: int(self.ram[0x03B8]))


def patch_runtime_compat():
    global _PATCHED
    if _PATCHED:
        return

    _patch_nes_py_rom()
    _patch_gym_super_mario_bros()
    _PATCHED = True
