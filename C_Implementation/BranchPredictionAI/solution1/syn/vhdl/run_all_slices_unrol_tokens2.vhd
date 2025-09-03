-- ==============================================================
-- Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.1 (64-bit)
-- Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity run_all_slices_unrol_tokens2_rom is 
    generic(
             DWIDTH     : integer := 8; 
             AWIDTH     : integer := 8; 
             MEM_SIZE    : integer := 150
    ); 
    port (
          addr0      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          q0         : out std_logic_vector(DWIDTH-1 downto 0);
          clk       : in std_logic
    ); 
end entity; 


architecture rtl of run_all_slices_unrol_tokens2_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "00000000", 1 => "00000001", 2 => "00000010", 3 => "00000011", 
    4 => "00000100", 5 => "00000101", 6 => "00000110", 7 => "00000111", 
    8 => "00001000", 9 => "00001001", 10 => "00001010", 11 => "00001011", 
    12 => "00001100", 13 => "00001101", 14 => "00001110", 15 => "00001111", 
    16 => "00010000", 17 => "00010001", 18 => "00010010", 19 => "00010011", 
    20 => "00010100", 21 => "00010101", 22 => "00010110", 23 => "00010111", 
    24 => "00011000", 25 => "00011001", 26 => "00011010", 27 => "00011011", 
    28 => "00011100", 29 => "00011101", 30 => "00011110", 31 => "00011111", 
    32 => "00100000", 33 => "00100001", 34 => "00100010", 35 => "00100011", 
    36 => "00100100", 37 => "00100101", 38 => "00100110", 39 => "00100111", 
    40 => "00101000", 41 => "00101001", 42 => "00101010", 43 => "00101011", 
    44 => "00101100", 45 => "00101101", 46 => "00101110", 47 => "00101111", 
    48 => "00110000", 49 => "00110001", 50 => "00110010", 51 => "00110011", 
    52 => "00110100", 53 => "00110101", 54 => "00110110", 55 => "00110111", 
    56 => "00111000", 57 => "00111001", 58 => "00111010", 59 => "00111011", 
    60 => "00111100", 61 => "00111101", 62 => "00111110", 63 => "00111111", 
    64 => "01000000", 65 => "01000001", 66 => "01000010", 67 => "01000011", 
    68 => "01000100", 69 => "01000101", 70 => "01000110", 71 => "01000111", 
    72 => "01001000", 73 => "01001001", 74 => "01001010", 75 => "01001011", 
    76 => "01001100", 77 => "01001101", 78 => "01001110", 79 => "01001111", 
    80 => "01010000", 81 => "01010001", 82 => "01010010", 83 => "01010011", 
    84 => "01010100", 85 => "01010101", 86 => "01010110", 87 => "01010111", 
    88 => "01011000", 89 => "01011001", 90 => "01011010", 91 => "01011011", 
    92 => "01011100", 93 => "01011101", 94 => "01011110", 95 => "01011111", 
    96 => "01100000", 97 => "01100001", 98 => "01100010", 99 => "01100011", 
    100 => "01100100", 101 => "01100101", 102 => "01100110", 103 => "01100111", 
    104 => "01101000", 105 => "01101001", 106 => "01101010", 107 => "01101011", 
    108 => "01101100", 109 => "01101101", 110 => "01101110", 111 => "01101111", 
    112 => "01110000", 113 => "01110001", 114 => "01110010", 115 => "01110011", 
    116 => "01110100", 117 => "01110101", 118 => "01110110", 119 => "01110111", 
    120 => "01111000", 121 => "01111001", 122 => "01111010", 123 => "01111011", 
    124 => "01111100", 125 => "01111101", 126 => "01111110", 127 => "01111111", 
    128 => "10000000", 129 => "10000001", 130 => "10000010", 131 => "10000011", 
    132 => "10000100", 133 => "10000101", 134 => "10000110", 135 => "10000111", 
    136 => "10001000", 137 => "10001001", 138 => "10001010", 139 => "10001011", 
    140 => "10001100", 141 => "10001101", 142 => "10001110", 143 => "10001111", 
    144 => "10010000", 145 => "10010001", 146 => "10010010", 147 => "10010011", 
    148 => "10010100", 149 => "10010101" );


begin 


memory_access_guard_0: process (addr0) 
begin
      addr0_tmp <= addr0;
--synthesis translate_off
      if (CONV_INTEGER(addr0) > mem_size-1) then
           addr0_tmp <= (others => '0');
      else 
           addr0_tmp <= addr0;
      end if;
--synthesis translate_on
end process;

p_rom_access: process (clk)  
begin 
    if (clk'event and clk = '1') then
        if (ce0 = '1') then 
            q0 <= mem(CONV_INTEGER(addr0_tmp)); 
        end if;
    end if;
end process;

end rtl;

Library IEEE;
use IEEE.std_logic_1164.all;

entity run_all_slices_unrol_tokens2 is
    generic (
        DataWidth : INTEGER := 8;
        AddressRange : INTEGER := 150;
        AddressWidth : INTEGER := 8);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of run_all_slices_unrol_tokens2 is
    component run_all_slices_unrol_tokens2_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    run_all_slices_unrol_tokens2_rom_U :  component run_all_slices_unrol_tokens2_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


