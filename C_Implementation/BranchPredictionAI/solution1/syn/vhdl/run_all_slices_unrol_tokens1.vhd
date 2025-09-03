-- ==============================================================
-- Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.1 (64-bit)
-- Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity run_all_slices_unrol_tokens1_rom is 
    generic(
             DWIDTH     : integer := 7; 
             AWIDTH     : integer := 7; 
             MEM_SIZE    : integer := 78
    ); 
    port (
          addr0      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          q0         : out std_logic_vector(DWIDTH-1 downto 0);
          clk       : in std_logic
    ); 
end entity; 


architecture rtl of run_all_slices_unrol_tokens1_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "0000000", 1 => "0000001", 2 => "0000010", 3 => "0000011", 
    4 => "0000100", 5 => "0000101", 6 => "0000110", 7 => "0000111", 
    8 => "0001000", 9 => "0001001", 10 => "0001010", 11 => "0001011", 
    12 => "0001100", 13 => "0001101", 14 => "0001110", 15 => "0001111", 
    16 => "0010000", 17 => "0010001", 18 => "0010010", 19 => "0010011", 
    20 => "0010100", 21 => "0010101", 22 => "0010110", 23 => "0010111", 
    24 => "0011000", 25 => "0011001", 26 => "0011010", 27 => "0011011", 
    28 => "0011100", 29 => "0011101", 30 => "0011110", 31 => "0011111", 
    32 => "0100000", 33 => "0100001", 34 => "0100010", 35 => "0100011", 
    36 => "0100100", 37 => "0100101", 38 => "0100110", 39 => "0100111", 
    40 => "0101000", 41 => "0101001", 42 => "0101010", 43 => "0101011", 
    44 => "0101100", 45 => "0101101", 46 => "0101110", 47 => "0101111", 
    48 => "0110000", 49 => "0110001", 50 => "0110010", 51 => "0110011", 
    52 => "0110100", 53 => "0110101", 54 => "0110110", 55 => "0110111", 
    56 => "0111000", 57 => "0111001", 58 => "0111010", 59 => "0111011", 
    60 => "0111100", 61 => "0111101", 62 => "0111110", 63 => "0111111", 
    64 => "1000000", 65 => "1000001", 66 => "1000010", 67 => "1000011", 
    68 => "1000100", 69 => "1000101", 70 => "1000110", 71 => "1000111", 
    72 => "1001000", 73 => "1001001", 74 => "1001010", 75 => "1001011", 
    76 => "1001100", 77 => "1001101" );

attribute syn_rom_style : string;
attribute syn_rom_style of mem : signal is "select_rom";
attribute ROM_STYLE : string;
attribute ROM_STYLE of mem : signal is "distributed";

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

entity run_all_slices_unrol_tokens1 is
    generic (
        DataWidth : INTEGER := 7;
        AddressRange : INTEGER := 78;
        AddressWidth : INTEGER := 7);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of run_all_slices_unrol_tokens1 is
    component run_all_slices_unrol_tokens1_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    run_all_slices_unrol_tokens1_rom_U :  component run_all_slices_unrol_tokens1_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


