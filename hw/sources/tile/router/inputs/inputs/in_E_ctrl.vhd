-- file: pixel_mem.vhd
-- description: pixel array PMr processing element
-- pixel indexing - x coordinate: column
--		  - y coordinate: row
--                - intermediary image index: step
--                - current frame: frame

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity in_E_controller is

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
    oc_E_pixel  : in std_logic_vector(pix_depth-1 downto 0);
    oc_E_x_dest : in std_logic_vector(img_width-1 downto 0);
    oc_E_y_dest : in std_logic_vector(img_height-1 downto 0);
    oc_E_step   : in std_logic_vector(n_steps-1 downto 0);
    oc_E_frame  : in std_logic_vector(n_frames-1 downto 0);
    oc_E_x_orig : in std_logic_vector(img_width-1 downto 0);
    oc_E_y_orig : in std_logic_vector(img_height-1 downto 0);
    oc_E_fb     : in std_logic; -- message forward/backward
    oc_E_new_msg: in std_logic;
    oc_E_ack    : out  std_logic;
    -- connections to the next router
    i_E_pixel  : out std_logic_vector(pix_depth-1 downto 0);
    i_E_x_dest : out std_logic_vector(img_width-1 downto 0);
    i_E_y_dest : out std_logic_vector(img_height-1 downto 0);
    i_E_step   : out std_logic_vector(n_steps-1 downto 0);
    i_E_frame  : out std_logic_vector(n_frames-1 downto 0);
    i_E_x_orig : out std_logic_vector(img_width-1 downto 0);
    i_E_y_orig : out std_logic_vector(img_height-1 downto 0);
    i_E_fb     : out std_logic; -- message forward/backward
    i_E_req    : inout std_logic;
    i_E_ack    : in  std_logic   
    
);
end entity in_E_controller;

architecture behavioral of in_E_controller is

signal mode : integer := 0; 
begin
    
    
process(clk,reset) is
variable aux1 : std_logic := '0';
variable aux2 : std_logic := '0';
begin
    if(reset='1')then
        i_E_pixel   <= (others => '0');
        i_E_x_dest  <= (others => '0');
        i_E_y_dest  <= (others => '0');
        i_E_step    <= (others => '0');
        i_E_frame   <= (others => '0');
        i_E_x_orig  <= (others => '0');
        i_E_y_orig  <= (others => '0');
        i_E_fb      <= '0';  
        i_E_req <= '0';
        oc_E_ack <= '0';
        mode <= 0;
    elsif(rising_edge(clk))then
        case mode is
            when 0 =>
                if(oc_E_new_msg='1')then
                    if(i_E_ack='0' and i_E_req='0')then
                        i_E_pixel   <= oc_E_pixel ;
                        i_E_x_dest  <= oc_E_x_dest;
                        i_E_y_dest  <= oc_E_y_dest;
                        i_E_step    <= oc_E_step  ;
                        i_E_frame   <= oc_E_frame ;
                        i_E_x_orig  <= oc_E_x_orig;
                        i_E_y_orig  <= oc_E_y_orig;
                        i_E_fb      <= oc_E_fb    ;
                        i_E_req     <= '1';
                        oc_E_ack    <='1';
                        mode <= 1;
                    end if;
                end if;
                
                
            when 1 =>
                 if(i_E_ack='1' )then 
                       i_E_req <= '0';
                       aux1:='1';
              end if;
   
              if(oc_E_new_msg='0')then
                oc_E_ack<='0';
                 aux2:='1';

              end if;
              
              if(aux1='1' and aux2 ='1')then 
              mode<=0;
                aux1:='0';
               aux2:='0';
              end if;
            
            when others =>
        
        end case;
        end if;


end process;

end architecture behavioral;
