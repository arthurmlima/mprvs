-- file: pixel_mem.vhd
-- description: pixel array PMr processing element
-- pixel indexing - x coordinate: column
--		  - y coordinate: row
--                - intermediary image index: step
--                - current frame: frame

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity in_PE_controller is

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
    oc_PE_pixel  : in std_logic_vector(pix_depth-1 downto 0);
    oc_PE_x_dest : in std_logic_vector(img_width-1 downto 0);
    oc_PE_y_dest : in std_logic_vector(img_height-1 downto 0);
    oc_PE_step   : in std_logic_vector(n_steps-1 downto 0);
    oc_PE_frame  : in std_logic_vector(n_frames-1 downto 0);
    oc_PE_x_orig : in std_logic_vector(img_width-1 downto 0);
    oc_PE_y_orig : in std_logic_vector(img_height-1 downto 0);
    oc_PE_fb     : in std_logic; -- message forward/backward
    oc_PE_new_msg: in std_logic;
    oc_PE_ack    : out  std_logic;
    -- connections to the next router
    i_PE_pixel  : out std_logic_vector(pix_depth-1 downto 0);
    i_PE_x_dest : out std_logic_vector(img_width-1 downto 0);
    i_PE_y_dest : out std_logic_vector(img_height-1 downto 0);
    i_PE_step   : out std_logic_vector(n_steps-1 downto 0);
    i_PE_frame  : out std_logic_vector(n_frames-1 downto 0);
    i_PE_x_orig : out std_logic_vector(img_width-1 downto 0);
    i_PE_y_orig : out std_logic_vector(img_height-1 downto 0);
    i_PE_fb     : out std_logic; -- message forward/backward
    i_PE_req    : inout std_logic;
    i_PE_ack    : in  std_logic   
    
);
end entity in_PE_controller;

architecture behavioral of in_PE_controller is

signal mode : integer := 0; 
begin
    
    
process(clk,reset) is
variable aux1 : std_logic := '0';
variable aux2 : std_logic := '0';
begin
    if(reset='1')then
        i_PE_pixel   <= (others => '0');
        i_PE_x_dest  <= (others => '0');
        i_PE_y_dest  <= (others => '0');
        i_PE_step    <= (others => '0');
        i_PE_frame   <= (others => '0');
        i_PE_x_orig  <= (others => '0');
        i_PE_y_orig  <= (others => '0');
        i_PE_fb      <= '0';  
        i_PE_req <= '0';
        oc_PE_ack <= '0';
        mode <= 0;
    elsif(rising_edge(clk))then
        case mode is
            when 0 =>
                if(oc_PE_new_msg='1')then
                    if(i_PE_ack='0' and i_PE_req='0')then
                        i_PE_pixel   <= oc_PE_pixel ;
                        i_PE_x_dest  <= oc_PE_x_dest;
                        i_PE_y_dest  <= oc_PE_y_dest;
                        i_PE_step    <= oc_PE_step  ;
                        i_PE_frame   <= oc_PE_frame ;
                        i_PE_x_orig  <= oc_PE_x_orig;
                        i_PE_y_orig  <= oc_PE_y_orig;
                        i_PE_fb      <= oc_PE_fb    ;
                        i_PE_req     <= '1';
                        oc_PE_ack    <='1';
                        mode <= 1;
                    end if;
                end if;
                
                
            when 1 =>
                 if(i_PE_ack='1' )then 
                       i_PE_req <= '0';
                       aux1:='1';
              end if;
   
              if(oc_PE_new_msg='0')then
                oc_PE_ack<='0';
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
