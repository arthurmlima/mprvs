-- file: pixel_mem.vhd
-- description: pixel array per processing element
-- pixel indexing - x coordinate: column
--		  - y coordinate: row
--                - intermediary image index: step
--                - current frame: frame

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity out_N_controller is

generic(
    x_init          : integer := 0;
    y_init          : integer := 0;
    img_width       : natural;
    img_height      : natural;
    n_frames        : natural;
    n_steps         : natural;
    pix_depth       : natural
);

port(
	clk : in std_logic;
	reset : in std_logic;
	-- connections with the output controller
    oc_N_pixel  : in std_logic_vector(pix_depth-1 downto 0);
    oc_N_x_dest : in std_logic_vector(img_width-1 downto 0);
    oc_N_y_dest : in std_logic_vector(img_height-1 downto 0);
    oc_N_step   : in std_logic_vector(n_steps-1 downto 0);
    oc_N_frame  : in std_logic_vector(n_frames-1 downto 0);
    oc_N_x_orig : in std_logic_vector(img_width-1 downto 0);
    oc_N_y_orig : in std_logic_vector(img_height-1 downto 0);
    oc_N_fb     : in std_logic; -- message forward/backward
    oc_N_new_msg: in std_logic;
    oc_N_ack    : out  std_logic;
    -- connections to the next router
    i_N_pixel  : out std_logic_vector(pix_depth-1 downto 0);
    i_N_x_dest : out std_logic_vector(img_width-1 downto 0);
    i_N_y_dest : out std_logic_vector(img_height-1 downto 0);
    i_N_step   : out std_logic_vector(n_steps-1 downto 0);
    i_N_frame  : out std_logic_vector(n_frames-1 downto 0);
    i_N_x_orig : out std_logic_vector(img_width-1 downto 0);
    i_N_y_orig : out std_logic_vector(img_height-1 downto 0);
    i_N_fb     : out std_logic; -- message forward/backward
    i_N_req    : inout std_logic;
    i_N_ack    : in  std_logic   
    
);
end entity out_N_controller;

architecture behavioral of out_N_controller is

signal mode : integer := 0;
    
begin
    
    
process(clk,reset) is
variable aux1 : std_logic := '0';
variable aux2 : std_logic := '0';
begin
    if(reset='1')then
        i_N_pixel   <= (others => '0');
        i_N_x_dest  <= (others => '0');
        i_N_y_dest  <= (others => '0');
        i_N_step    <= (others => '0');
        i_N_frame   <= (others => '0');
        i_N_x_orig  <= (others => '0');
        i_N_y_orig  <= (others => '0');
        i_N_fb      <= '0';  
        i_N_req <= '0';
        oc_N_ack <= '0';
        mode <= 0;
    elsif(rising_edge(clk))then
        case mode is
            when 0 =>
                if(oc_N_new_msg='1')then
                    if(i_N_ack='0' and i_N_req='0')then
                        i_N_pixel   <= oc_N_pixel ;
                        i_N_x_dest  <= oc_N_x_dest;
                        i_N_y_dest  <= oc_N_y_dest;
                        i_N_step    <= oc_N_step  ;
                        i_N_frame   <= oc_N_frame ;
                        i_N_x_orig  <= oc_N_x_orig;
                        i_N_y_orig  <= oc_N_y_orig;
                        i_N_fb      <= oc_N_fb    ;
                        i_N_req     <= '1';
                        oc_N_ack    <='1';
                        mode <= 1;
                    end if;
                end if;
             when 1 =>
                if(i_N_ack='1')then
                    i_N_req <= '0';
                     aux1:='1';
                end if;
                if(oc_N_new_msg='0')then 
                 oc_N_ack<='0';
                 aux2:='1';               
                end if;    
                            
                if( aux1='1' and aux2='1')then 
                mode<=0;
                aux1:='0';
                aux2:='0';
                end if;

            when others =>
        
        end case;
        end if;


end process;

end architecture behavioral;
