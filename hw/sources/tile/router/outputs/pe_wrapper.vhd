----------------------------------------------------------------------------------
-- Company: Universidade de Brasília
-- Engineer: Bruno Almeida
-- 
-- Create Date: 01/21/2020 11:45:29 AM
-- Design Name: 
-- Module Name: pe_wrapper - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity pe_wrapper is
    generic (
            x_init : integer;
            y_init : integer;
            ram_config          : string;
            bit_width           : natural;
            memory_length       : natural;
            instruction_length  : natural;
            isf_opcode          : natural;
            regfile_size        : natural;
            isf_x               : natural;
            isf_y               : natural;
            isf_s               : natural;
            isf_f               : natural;
            isf_px              : natural;
            NOP                 : std_logic_vector(4-1 downto 0);
            GPX                 : std_logic_vector(4-1 downto 0);
            SPX                 : std_logic_vector(4-1 downto 0);
            ADD                 : std_logic_vector(4-1 downto 0);
            SUB                 : std_logic_vector(4-1 downto 0);
            MUL                 : std_logic_vector(4-1 downto 0);
            DIV                 : std_logic_vector(4-1 downto 0);
            AND1                : std_logic_vector(4-1 downto 0);
            OR1                 : std_logic_vector(4-1 downto 0);
            XOR1                : std_logic_vector(4-1 downto 0);
            NOT1                : std_logic_vector(4-1 downto 0);
            BGT                 : std_logic_vector(4-1 downto 0);
            BST                 : std_logic_vector(4-1 downto 0);
            BEQ                 : std_logic_vector(4-1 downto 0);
            JMP                 : std_logic_vector(4-1 downto 0);
            ENDPGR              : std_logic_vector(4-1 downto 0);
            addr_size           : natural;
            img_width           : natural;
            img_height          : natural;
            img_px_width        : natural;
            img_px_height       : natural;
            subimg_width        : natural;
            subimg_height       : natural;
            n_frames            : natural;
            n_steps             : natural;
            pix_depth           : natural;
            px_true_size        : natural
            );
    port(
        clk, reset : in STD_LOGIC;
        o_endpgr_flag : out STD_LOGIC;
    
    -- New Local target ports - the PE requests values using these ports.
        i_PE_pixel  : out STD_LOGIC_VECTOR(pix_depth-1 downto 0);
        i_PE_x_dest : out STD_LOGIC_VECTOR(img_width-1 downto 0);
        i_PE_y_dest : out STD_LOGIC_VECTOR(img_height-1 downto 0);
        i_PE_step   : out STD_LOGIC_VECTOR(n_steps-1 downto 0);
        i_PE_frame  : out STD_LOGIC_VECTOR(n_frames-1 downto 0);
        i_PE_x_orig : out STD_LOGIC_VECTOR(img_width-1 downto 0);
        i_PE_y_orig : out STD_LOGIC_VECTOR(img_height-1 downto 0);
        i_PE_fb     : out STD_LOGIC; -- message forward/backward
        i_PE_req    : inout STD_LOGIC; -- message request
       --iPEN_busy   : in std_logic; -- router is busy
        i_PE_ack    : in STD_LOGIC := '0'; -- message acknowledge  
            
        t_PE_pixel  : in STD_LOGIC_VECTOR(pix_depth-1 downto 0);
        t_PE_x_dest : in STD_LOGIC_VECTOR(img_width-1 downto 0);
        t_PE_y_dest : in STD_LOGIC_VECTOR(img_height-1 downto 0);
        t_PE_step   : in STD_LOGIC_VECTOR(n_steps-1 downto 0);
        t_PE_frame  : in STD_LOGIC_VECTOR(n_frames-1 downto 0);
        t_PE_x_orig : in STD_LOGIC_VECTOR(img_width-1 downto 0);
        t_PE_y_orig : in STD_LOGIC_VECTOR(img_height-1 downto 0);
        t_PE_fb     : in STD_LOGIC; -- message forward/backward
        t_PE_req    : in STD_LOGIC; -- message request
        --t_PE_busy   : out std_logic; -- router is busy
        t_PE_ack    : out STD_LOGIC; -- message acknowledge 
        
        s_reg_out       : out STD_LOGIC_VECTOR(2**bit_width-1 downto 0);
        o_noc_clk_count : out unsigned(31 downto 0);
        o_pm_clk_count  : out unsigned(31 downto 0);
        
        -- AXI/Instruction Memory interface
        i_axi_clk       : in STD_LOGIC;
        i_axi_we        : in STD_LOGIC;
        i_axi_addr      : in STD_LOGIC_VECTOR(31 downto 0);
        i_axi_data0     : in STD_LOGIC_VECTOR(31 downto 0);
        i_axi_data1     : in STD_LOGIC_VECTOR(31 downto 0);
        o_axi_imok      : out STD_LOGIC;
        
        -- PE/PM interface
        i_pm_done       : in STD_LOGIC;
        i_pm_px         : in UNSIGNED(px_true_size-1 downto 0);
        
        o_pm_en         : out STD_LOGIC;
        o_pm_wen        : out STD_LOGIC;
        o_pm_x          : out UNSIGNED(img_width-1 downto 0);
        o_pm_y          : out UNSIGNED(img_height-1 downto 0);
        o_pm_s          : out UNSIGNED(n_steps-1 downto 0);
        o_pm_f          : out UNSIGNED(n_frames-1 downto 0);
        o_pm_px         : out UNSIGNED(px_true_size-1 downto 0)
    );
end pe_wrapper;

architecture Behavioral of pe_wrapper is

component processing_element is
    generic(
            x_init              : integer;
            y_init              : integer;
            ram_config          : string;
            bit_width           : natural;
            instruction_length  : natural;
            memory_length       : natural;
            isf_opcode          : natural;
            regfile_size        : natural;
            isf_x               : natural;
            isf_y               : natural;
            isf_s               : natural;
            isf_f               : natural;
            isf_px              : natural;
            NOP                 : std_logic_vector(4-1 downto 0);
            GPX                 : std_logic_vector(4-1 downto 0);
            SPX                 : std_logic_vector(4-1 downto 0);
            ADD                 : std_logic_vector(4-1 downto 0);
            SUB                 : std_logic_vector(4-1 downto 0);
            MUL                 : std_logic_vector(4-1 downto 0);
            DIV                 : std_logic_vector(4-1 downto 0);
            AND1                : std_logic_vector(4-1 downto 0);
            OR1                 : std_logic_vector(4-1 downto 0);
            XOR1                : std_logic_vector(4-1 downto 0);
            NOT1                : std_logic_vector(4-1 downto 0);
            BGT                 : std_logic_vector(4-1 downto 0);
            BST                 : std_logic_vector(4-1 downto 0);
            BEQ                 : std_logic_vector(4-1 downto 0);
            JMP                 : std_logic_vector(4-1 downto 0);
            ENDPGR              : std_logic_vector(4-1 downto 0);
            addr_size           : natural;
            img_px_width        : natural;
            img_px_height       : natural;
            subimg_width        : natural;
            subimg_height       : natural
            );
    port(
        clk : in STD_LOGIC := '0'; -- processor clock
        
        -- P-type instuctions interface
        i_mem_gpx_f, i_reset : in STD_LOGIC;
        mem_px : in UNSIGNED(2**bit_width-1 downto 0);
        x_out : out UNSIGNED(2**bit_width-1 downto 0);
        y_out : out UNSIGNED(2**bit_width-1 downto 0);
        f_out : out UNSIGNED(2**bit_width-1 downto 0);
        s_out : out UNSIGNED(2**bit_width-1 downto 0);
        px_out : out UNSIGNED(2**bit_width-1 downto 0);
        s_reg_out : out STD_LOGIC_VECTOR(2**bit_width-1 downto 0);
        o_mem_wr_en, o_mem_gpx_f, o_endpgr_flag : out STD_LOGIC;
        
        -- AXI/Instruction Memory interface
        i_axi_clk       : in STD_LOGIC;
        i_axi_we        : in STD_LOGIC;
        i_axi_addr      : in STD_LOGIC_VECTOR(31 downto 0);
        i_axi_data0     : in STD_LOGIC_VECTOR(31 downto 0);
        i_axi_data1     : in STD_LOGIC_VECTOR(31 downto 0);
        o_axi_imok      : out STD_LOGIC
        );
end component processing_element;

signal si_mem_gpx_f : STD_LOGIC;
signal mem_px       : UNSIGNED(2**bit_width-1 downto 0);
signal x_out        : UNSIGNED(2**bit_width-1 downto 0);
signal y_out        : UNSIGNED(2**bit_width-1 downto 0);
signal f_out        : UNSIGNED(2**bit_width-1 downto 0);
signal s_out        : UNSIGNED(2**bit_width-1 downto 0);
signal px_out       : UNSIGNED(2**bit_width-1 downto 0);
signal s_mem_wr_en, so_mem_gpx_f, s_endpgr_flag : STD_LOGIC;

signal s_pm_x       : UNSIGNED(31 downto 0);
signal s_pm_y       : UNSIGNED(31 downto 0);
signal s_x_init     : UNSIGNED(31 downto 0); 
signal s_y_init     : UNSIGNED(31 downto 0);
signal s_x_final    : UNSIGNED(31 downto 0); 
signal s_y_final    : UNSIGNED(31 downto 0);

-- define FSM types
type state_type is (SPX_INIT, SPX_WAIT, SPX_END, SPX_PM_END,
                    GPX_INIT, GPX_WAIT, GPX_RECV, GPX_END, GPX_ERR,
                    GPX_RST, GPX_PM_RCV, BUSY, IDLE, NONE);
signal s_state : state_type := BUSY;

-- i_mem_gpx_f = t_PE_req
-- o_mem_gpx_f = i_PE_req
-- ACK is used as guarantee
signal s_x_dest                         : STD_LOGIC_VECTOR(img_width-1 downto 0);
signal s_y_dest                         : STD_LOGIC_VECTOR(img_height-1 downto 0);
signal s_step                           : STD_LOGIC_VECTOR(n_steps-1 downto 0);
signal s_frame                          : STD_LOGIC_VECTOR(n_frames-1 downto 0);
signal s_x_orig                         : STD_LOGIC_VECTOR(img_width-1 downto 0);
signal s_y_orig                         : STD_LOGIC_VECTOR(img_height-1 downto 0);
signal pixel_coord_in, pixel_coord_out  : STD_LOGIC_VECTOR(2*img_width+2*img_height+n_steps+n_frames-1 downto 0);
signal mode                             : STD_LOGIC_VECTOR(2 downto 0);
signal s_noc_clk_count                  : unsigned(o_noc_clk_count'length-1 downto 0);
signal s_pm_clk_count                   : unsigned(o_noc_clk_count'length-1 downto 0);
signal clk_cnt_nocf                     : std_logic := '0';
signal clk_cnt_pmf                      : std_logic := '0';
signal clk_count_sync, clk_count_wait   : integer := 0;

function get_nstate(i_mode : STD_LOGIC_VECTOR(2 downto 0)) return state_type is
begin
    case i_mode is
        when "011" =>
            return SPX_INIT;
        when "001" =>
            return GPX_INIT;
        when "100" =>
            return IDLE;
        when others =>
            return BUSY;
    end case;
end function;

begin

proc_element : processing_element   generic map(
                                                x_init              => x_init            ,
                                                y_init              => y_init            ,
                                                ram_config          => ram_config        ,
                                                bit_width           => bit_width         ,
                                                memory_length       => memory_length     ,
                                                instruction_length  => instruction_length,
                                                isf_opcode          => isf_opcode        ,
                                                regfile_size        => regfile_size      ,
                                                isf_x               => isf_x             ,
                                                isf_y               => isf_y             ,
                                                isf_s               => isf_s             ,
                                                isf_f               => isf_f             ,
                                                isf_px              => isf_px            ,
                                                NOP                 => NOP               ,
                                                GPX                 => GPX               ,
                                                SPX                 => SPX               ,
                                                ADD                 => ADD               ,
                                                SUB                 => SUB               ,
                                                MUL                 => MUL               ,
                                                DIV                 => DIV               ,
                                                AND1                => AND1              ,
                                                OR1                 => OR1               ,
                                                XOR1                => XOR1              ,
                                                NOT1                => NOT1              ,
                                                BGT                 => BGT               ,
                                                BST                 => BST               ,
                                                BEQ                 => BEQ               ,
                                                JMP                 => JMP               ,
                                                ENDPGR              => ENDPGR            ,
                                                addr_size           => addr_size         ,
                                                img_px_width        => img_px_width      ,
                                                img_px_height       => img_px_height     ,
                                                subimg_width        => subimg_width      ,
                                                subimg_height       => subimg_height
                                            ) 
                                    port map(
                                            clk             => clk,
                                            i_mem_gpx_f     => si_mem_gpx_f,
                                            mem_px          => mem_px,
                                            i_reset         => reset,
                                            x_out           => x_out,
                                            y_out           => y_out,
                                            f_out           => f_out,
                                            s_out           => s_out,
                                            px_out          => px_out,
                                            s_reg_out       => s_reg_out,
                                            o_mem_wr_en     => s_mem_wr_en,
                                            o_mem_gpx_f     => so_mem_gpx_f,
                                            o_endpgr_flag   => s_endpgr_flag,
                                            i_axi_clk       => i_axi_clk,
                                            i_axi_we        => i_axi_we,
                                            i_axi_addr      => i_axi_addr,
                                            i_axi_data0     => i_axi_data0,
                                            i_axi_data1     => i_axi_data1,
                                            o_axi_imok      => o_axi_imok
                                            );

s_pm_x          <=  resize(x_out, s_pm_x'length);      
s_pm_y          <=  resize(y_out, s_pm_y'length);
s_x_init        <=  to_unsigned(x_init, s_x_init'length);      
s_y_init        <=  to_unsigned(y_init, s_y_init'length);
s_x_final       <=  to_unsigned(x_init + (subimg_width-1), s_x_final'length);      
s_y_final       <=  to_unsigned(y_init + (subimg_height-1), s_y_final'length);

s_x_dest        <=  STD_LOGIC_VECTOR(to_unsigned(to_integer(x_out), i_PE_x_dest'length));
s_y_dest        <=  STD_LOGIC_VECTOR(to_unsigned(to_integer(y_out), i_PE_y_dest'length));
s_step          <=  STD_LOGIC_VECTOR(to_unsigned(to_integer(s_out), i_PE_step'length));
s_frame         <=  STD_LOGIC_VECTOR(to_unsigned(to_integer(f_out), i_PE_frame'length));
s_x_orig        <=  STD_LOGIC_VECTOR(to_unsigned(x_init, i_PE_x_orig'length));
s_y_orig        <=  STD_LOGIC_VECTOR(to_unsigned(y_init, i_PE_y_orig'length));

mode            <= s_endpgr_flag & s_mem_wr_en & so_mem_gpx_f;

pixel_coord_in  <= t_PE_x_dest & t_PE_y_dest & t_PE_step & t_PE_frame & t_PE_x_orig & t_PE_y_orig;
pixel_coord_out <= s_x_dest & s_y_dest & s_step & s_frame & s_x_orig & s_y_orig;

o_noc_clk_count <= s_noc_clk_count;
o_pm_clk_count  <= s_pm_clk_count;
process(reset, clk, clk_cnt_nocf, clk_cnt_pmf)
begin
    if reset = '1' then
        s_noc_clk_count     <= to_unsigned(0, s_noc_clk_count'length);
        s_pm_clk_count      <= to_unsigned(0, s_pm_clk_count'length);
    else
        if RISING_EDGE(clk) then
            if clk_cnt_nocf = '1' then
                s_noc_clk_count     <= s_noc_clk_count + 1;
            else
                s_noc_clk_count     <= s_noc_clk_count;
            end if;
            if clk_cnt_pmf = '1' then
                s_pm_clk_count      <= s_pm_clk_count + 1;
            else
                s_pm_clk_count      <= s_pm_clk_count;
            end if;
        end if;
    end if;
end process;

process(clk, reset, mode, s_x_dest, s_y_dest, s_step, s_frame, s_x_orig,
        s_y_orig, pixel_coord_in, pixel_coord_out, t_PE_req, i_PE_ack)
begin
    if reset = '1' then
        clk_cnt_nocf    <= '0';
        clk_cnt_pmf     <= '0';
        mem_px          <= (others => '0');
        i_PE_pixel      <= (others => '0');
        i_PE_x_dest     <= (others => '0');
        i_PE_y_dest     <= (others => '0');
        i_PE_step       <= (others => '0');
        i_PE_frame      <= (others => '0');
        i_PE_x_orig     <= (others => '0');
        i_PE_y_orig     <= (others => '0');
        i_PE_fb         <= '0'; -- Foward (not wait for pixel)
        i_PE_req        <= '0'; -- Pe wants to give px to Router    
        t_PE_ack        <= '0';
        si_mem_gpx_f    <= '0';
        o_endpgr_flag   <= '0';
        s_state         <= BUSY;
    else
        if RISING_EDGE(clk) then
            case s_state is
                when SPX_INIT =>
                    if  (s_pm_x >= s_x_init and s_pm_x <= s_x_final) and
                        (s_pm_y >= s_y_init and s_pm_y <= s_y_final) then
                        clk_cnt_nocf    <= '0';
                        clk_cnt_pmf     <= '1';
                        o_pm_en         <= '1';
                        o_pm_wen        <= '1';
                        o_pm_x          <= UNSIGNED(s_x_dest);
                        o_pm_y          <= UNSIGNED(s_y_dest);
                        o_pm_s          <= UNSIGNED(s_step);
                        o_pm_f          <= UNSIGNED(s_frame);
                        o_pm_px         <= to_unsigned(to_integer(px_out), o_pm_px'length);
                        si_mem_gpx_f    <= '1'; -- Lock Program Counter to wait handshake
                        s_state         <= SPX_PM_END;
                   else
                        if t_PE_req = '0' and i_PE_ack = '0' then
                            clk_cnt_nocf    <= '1';
                            clk_cnt_pmf     <= '0';
                            mem_px          <= (others => '0');
                            i_PE_pixel      <= "0" & STD_LOGIC_VECTOR(to_unsigned(to_integer(px_out), i_PE_pixel'length-1));
                            i_PE_x_dest     <= s_x_dest;
                            i_PE_y_dest     <= s_y_dest;
                            i_PE_step       <= s_step;  
                            i_PE_frame      <= s_frame; 
                            i_PE_x_orig     <= s_x_orig;
                            i_PE_y_orig     <= s_y_orig;
                            i_PE_fb         <= '0'; -- Foward -> send to PM
                            i_PE_req        <= '1'; -- Pe wants to give px to Router
                            t_PE_ack        <= '1';
                            si_mem_gpx_f    <= '0'; -- Lock Program Counter to wait handshake
                            o_endpgr_flag   <= '0';
                            s_state         <= SPX_WAIT;
                        end if;
                    end if;
                when SPX_WAIT =>
                    if i_PE_ack = '1' then
                        clk_cnt_nocf    <= '1';
                        clk_cnt_pmf     <= '0';
                        mem_px          <= (others => '0');
                        i_PE_pixel      <= (others => '0');
                        i_PE_x_dest     <= (others => '0');
                        i_PE_y_dest     <= (others => '0');
                        i_PE_step       <= (others => '0');
                        i_PE_frame      <= (others => '0');
                        i_PE_x_orig     <= (others => '0');
                        i_PE_y_orig     <= (others => '0');
                        i_PE_fb         <= '0'; -- Foward (not wait for pixel)
                        i_PE_req        <= '0';   
                        t_PE_ack        <= '0';
                        si_mem_gpx_f    <= '1';
                        o_endpgr_flag   <= '0';
                        s_state         <= SPX_END;
                    end if;
                when SPX_END =>
                    clk_cnt_nocf    <= '0';
                    clk_cnt_pmf     <= '0';
                    i_PE_pixel      <= (others => '0');
                    i_PE_x_dest     <= (others => '0');
                    i_PE_y_dest     <= (others => '0');
                    i_PE_step       <= (others => '0');
                    i_PE_frame      <= (others => '0');
                    i_PE_x_orig     <= (others => '0');
                    i_PE_y_orig     <= (others => '0');
                    i_PE_fb         <= '0'; -- Foward (not wait for pixel)
                    i_PE_req        <= '0'; -- Pe wants to give px to Router    
                    t_PE_ack        <= '0';
                    si_mem_gpx_f    <= '0';
                    o_endpgr_flag   <= '0';
                    s_state         <= BUSY;
                when SPX_PM_END =>
                    if i_pm_done = '1' then
                        clk_cnt_nocf    <= '0';
                        clk_cnt_pmf     <= '0';
                        o_pm_en         <= '0';
                        o_pm_wen        <= '0';
                        o_pm_x          <= (others => '0');
                        o_pm_y          <= (others => '0');
                        o_pm_s          <= (others => '0');
                        o_pm_f          <= (others => '0');
                        o_pm_px         <= (others => '0');
                        si_mem_gpx_f    <= '0';
                        s_state         <= BUSY;
                    end if;
                when GPX_INIT =>
                    if  (s_pm_x >= s_x_init and s_pm_x <= s_x_final) and
                        (s_pm_y >= s_y_init and s_pm_y <= s_y_final) then
                        clk_cnt_nocf    <= '0';
                        clk_cnt_pmf     <= '1';
                        o_pm_en         <= '1';
                        o_pm_wen        <= '0';
                        o_pm_x          <= UNSIGNED(s_x_dest);
                        o_pm_y          <= UNSIGNED(s_y_dest);
                        o_pm_s          <= UNSIGNED(s_step);
                        o_pm_f          <= UNSIGNED(s_frame);
                        o_pm_px         <= to_unsigned(to_integer(px_out), o_pm_px'length);
                        si_mem_gpx_f    <= '0'; -- Lock Program Counter to wait handshake
                        s_state         <= GPX_PM_RCV;
                    else
                        if t_PE_req = '0' and i_PE_ack = '0' then
                            clk_cnt_nocf    <= '1';
                            clk_cnt_pmf     <= '0';
                            mem_px          <= (others => '0');
                            i_PE_pixel      <= '1' & (pix_depth - 2 downto 0 => '0');
                            i_PE_x_dest     <= s_x_dest;
                            i_PE_y_dest     <= s_y_dest;
                            i_PE_step       <= s_step;  
                            i_PE_frame      <= s_frame; 
                            i_PE_x_orig     <= s_x_orig;
                            i_PE_y_orig     <= s_y_orig;
                            i_PE_fb         <= '0'; -- Foward -> send to PM
                            i_PE_req        <= '1'; -- Pe wants to give px to Router
                            t_PE_ack        <= '1';
                            si_mem_gpx_f    <= '0';
                            o_endpgr_flag   <= '0';
                            clk_count_wait  <= 0;
                            s_state         <= GPX_WAIT;
                        end if;
                    end if;
                when GPX_WAIT => -- wait for response from router
                    if i_PE_ack = '1' then
                        clk_cnt_nocf    <= '1';
                        clk_cnt_pmf     <= '0';
                        mem_px          <= (others => '0');
                        i_PE_pixel      <= (others => '0');
                        i_PE_x_dest     <= (others => '0');
                        i_PE_y_dest     <= (others => '0');
                        i_PE_step       <= (others => '0');
                        i_PE_frame      <= (others => '0');
                        i_PE_x_orig     <= (others => '0');
                        i_PE_y_orig     <= (others => '0');
                        i_PE_fb         <= '0'; -- Foward (not wait for pixel)
                        i_PE_req        <= '0';   
                        t_PE_ack        <= '0';
                        si_mem_gpx_f    <= '0';
                        o_endpgr_flag   <= '0';
                        s_state         <= GPX_RECV;
                    end if;
                when GPX_RECV => -- save pixel data and answer router with ack
                    if t_PE_req = '1' then --means that the router wants to give a pixel to the processor
                        clk_cnt_nocf    <= '1';
                        clk_cnt_pmf     <= '0';
                        i_PE_pixel      <= (others => '0');
                        i_PE_x_dest     <= (others => '0');
                        i_PE_y_dest     <= (others => '0');
                        i_PE_step       <= (others => '0');
                        i_PE_frame      <= (others => '0');
                        i_PE_x_orig     <= (others => '0');
                        i_PE_y_orig     <= (others => '0');
                        i_PE_fb         <= '0'; -- Foward (not wait for pixel)
                        i_PE_req        <= '0';   
                        t_PE_ack        <= '1';
                        o_endpgr_flag   <= '0';
                        
                        if pixel_coord_in = pixel_coord_out then -- valid pixel
                            if (t_PE_pixel = (pix_depth-1 downto 0 => '1')) then -- PM is locked
                                mem_px          <= (others => '0');
                                si_mem_gpx_f    <= '0';
                                clk_count_sync  <= 0;
                                s_state         <= GPX_RST;
                            else
                                mem_px          <=  (2**bit_width-1 downto pix_depth-1 => t_PE_pixel(pix_depth-2)) 
                                                    & UNSIGNED(t_PE_pixel(pix_depth-2 downto 0));
                                si_mem_gpx_f    <= '1'; -- Send flag to unlock PC as gpx is enabled
                                s_state         <= GPX_END;
                            end if;
                        else
                            mem_px          <= (others => '0');
                            si_mem_gpx_f    <= '0';
                            s_state         <= GPX_ERR;
                        end if;
                    else
                        mem_px          <= (others => '0');
                        si_mem_gpx_f    <= '0';
                        clk_count_wait  <= clk_count_wait + 1;
                        
                        if clk_count_wait > 100 then
                            s_state     <= GPX_ERR;
                        else
                            s_state     <= GPX_RECV;
                        end if;
                    end if;
                when GPX_RST =>
                    clk_cnt_nocf    <= '1';
                    clk_cnt_pmf     <= '0';
                    mem_px          <= (others => '0');
                    i_PE_pixel      <= (others => '0');
                    i_PE_x_dest     <= (others => '0');
                    i_PE_y_dest     <= (others => '0');
                    i_PE_step       <= (others => '0');
                    i_PE_frame      <= (others => '0');
                    i_PE_x_orig     <= (others => '0');
                    i_PE_y_orig     <= (others => '0');
                    i_PE_fb         <= '0'; -- Foward (not wait for pixel)
                    i_PE_req        <= '0';   
                    t_PE_ack        <= '0';
                    si_mem_gpx_f    <= '0';
                    o_endpgr_flag   <= '0';
                    clk_count_sync  <= clk_count_sync + 1;

                    if clk_count_sync > 1000 then
                        s_state     <= GPX_INIT;
                    else
                        s_state     <= GPX_RST;
                    end if;
                when GPX_ERR =>
                    clk_cnt_nocf    <= '1';
                    clk_cnt_pmf     <= '0';
                    i_PE_pixel      <= (others => '0');
                    i_PE_x_dest     <= (others => '0');
                    i_PE_y_dest     <= (others => '0');
                    i_PE_step       <= (others => '0');
                    i_PE_frame      <= (others => '0');
                    i_PE_x_orig     <= (others => '0');
                    i_PE_y_orig     <= (others => '0');
                    i_PE_fb         <= '0'; -- Foward (not wait for pixel)
                    i_PE_req        <= '0'; -- Pe wants to give px to Router    
                    t_PE_ack        <= '0';
                    si_mem_gpx_f    <= '0';
                    o_endpgr_flag   <= '0';
                    s_state         <= GPX_INIT;
                when GPX_PM_RCV =>
                    if i_pm_done = '1' then
                        mem_px          <=  (2**bit_width-1 downto i_pm_px'length => i_pm_px(px_true_size-1)) & i_pm_px;                        
                        clk_cnt_nocf    <= '0';
                        clk_cnt_pmf     <= '1';
                        o_pm_en         <= '0';
                        o_pm_wen        <= '0';
                        o_pm_x          <= (others => '0');
                        o_pm_y          <= (others => '0');
                        o_pm_s          <= (others => '0');
                        o_pm_f          <= (others => '0');
                        o_pm_px         <= (others => '0');
                        si_mem_gpx_f    <= '1';
                        s_state         <= GPX_END;
                    end if;
                when GPX_END =>
                    clk_cnt_nocf    <= '0';
                    clk_cnt_pmf     <= '0';
                    i_PE_pixel      <= (others => '0');
                    i_PE_x_dest     <= (others => '0');
                    i_PE_y_dest     <= (others => '0');
                    i_PE_step       <= (others => '0');
                    i_PE_frame      <= (others => '0');
                    i_PE_x_orig     <= (others => '0');
                    i_PE_y_orig     <= (others => '0');
                    i_PE_fb         <= '0'; -- Foward (not wait for pixel)
                    i_PE_req        <= '0'; -- Pe wants to give px to Router    
                    t_PE_ack        <= '0';
                    si_mem_gpx_f    <= '0';
                    o_endpgr_flag   <= '0';
                    s_state         <= BUSY;
                when IDLE =>
                    clk_cnt_nocf    <= '0';
                    clk_cnt_pmf     <= '0';
                    i_PE_pixel      <= (others => '0');
                    i_PE_x_dest     <= (others => '0');
                    i_PE_y_dest     <= (others => '0');
                    i_PE_step       <= (others => '0');
                    i_PE_frame      <= (others => '0');
                    i_PE_x_orig     <= (others => '0');
                    i_PE_y_orig     <= (others => '0');
                    i_PE_fb         <= '0'; -- Foward (not wait for pixel)
                    i_PE_req        <= '0'; -- Pe wants to give px to Router    
                    t_PE_ack        <= '0';
                    si_mem_gpx_f    <= '0';
                    o_endpgr_flag   <= '1';
                    s_state         <= get_nstate(mode);
                when BUSY =>
                    clk_cnt_nocf    <= '0';
                    clk_cnt_pmf     <= '0';
                    i_PE_pixel      <= (others => '0');
                    i_PE_x_dest     <= (others => '0');
                    i_PE_y_dest     <= (others => '0');
                    i_PE_step       <= (others => '0');
                    i_PE_frame      <= (others => '0');
                    i_PE_x_orig     <= (others => '0');
                    i_PE_y_orig     <= (others => '0');
                    i_PE_fb         <= '0'; -- Foward (not wait for pixel)
                    i_PE_req        <= '0'; -- Pe wants to give px to Router    
                    si_mem_gpx_f    <= '0';
                    o_endpgr_flag   <= '0';
                    s_state         <= get_nstate(mode);
                    if t_PE_req = '0' then
                        t_PE_ack        <= '0';
                    else
                        t_PE_ack        <= '1';
                    end if;
                when NONE =>
            end case;
        end if;
    end if;
end process;

end Behavioral;
