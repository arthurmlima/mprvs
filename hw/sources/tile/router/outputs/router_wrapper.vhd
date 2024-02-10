----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 08/05/2022 06:13:20 AM
-- Design Name: 
-- Module Name: router_wrapper - Behavioral
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
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity router_wrapper is
generic(
    x_init           : integer := 0;
    y_init           : integer := 0;
    pix_depth  	     : natural; 
    img_width	     : natural ;
    img_height 	     : natural ;
    subimg_width 	 : natural; 
    subimg_height	 : natural ;
    n_steps 	     : natural ;
    n_frames   	     : natural ;
    buffer_length    : natural
);
Port ( 
        clk          : in std_logic;  
        reset          : in std_logic;
        --PM
    in_router_out_pm_data   : in std_logic_vector(63  downto 0);
    out_router_in_pm_ack    : out  std_logic;   
    
    out_router_in_pm_data   : out std_logic_vector(63  downto 0);
    in_router_out_pm_ack    : in  std_logic;
    
    in_router_out_pe_data   : in std_logic_vector(63   downto 0);
    out_router_in_pe_ack    : out  std_logic;   
        
    out_router_in_pe_data   : out std_logic_vector(63   downto 0); --i_data
    in_router_out_pe_ack    : in  std_logic;
        
    in_router_out_n_data   : in std_logic_vector(63  downto 0);
    out_router_in_n_ack    : out std_logic; 
      
    out_router_in_n_data   : out std_logic_vector(63  downto 0);
    in_router_out_n_ack    : in  std_logic;
        
    in_router_out_s_data   : in std_logic_vector(63  downto 0);
    out_router_in_s_ack    : out std_logic;    
    
    out_router_in_s_data   : out std_logic_vector(63   downto 0);
    in_router_out_s_ack    : in  std_logic;
        
    in_router_out_e_data   : in std_logic_vector(63   downto 0);
    out_router_in_e_ack    : out std_logic;
    
    out_router_in_e_data   : out std_logic_vector(63  downto 0);
    in_router_out_e_ack    : in  std_logic;
    
    in_router_out_w_data   : in std_logic_vector(63   downto 0);
    out_router_in_w_ack   :  out std_logic;
    
    out_router_in_w_data   : out std_logic_vector(63   downto 0);
    in_router_out_w_ack    : in  std_logic; 
    
    i_PE_pixel  : out std_logic_vector(pix_depth-1 downto 0);
    i_PE_x_dest : out std_logic_vector(img_width-1 downto 0);
    i_PE_y_dest : out std_logic_vector(img_height-1 downto 0);
    i_PE_step   : out std_logic_vector(n_steps-1 downto 0);
    i_PE_frame  : out std_logic_vector(n_frames-1 downto 0);
    i_PE_x_orig : out std_logic_vector(img_width-1 downto 0);
    i_PE_y_orig : out std_logic_vector(img_height-1 downto 0);
    i_PE_fb     : out std_logic; -- identify if the it is a set_pixel or a set_pixel message.
    i_PE_req    : inout std_logic;
    i_PE_ack    : in std_logic
    



);
end router_wrapper;

architecture Behavioral of router_wrapper is



component router_new is

generic(
    x_init           : natural ;
    y_init           : natural ;
    pix_depth  	     : natural; 
    img_width	     : natural ;
    img_height 	     : natural ;
    subimg_width 	 : natural; 
    subimg_height	 : natural ;
    n_steps 	     : natural ;
    n_frames   	     : natural ;
    buffer_length    : natural
);
port(

    clk          : in std_logic;
    reset          : in std_logic;
    
    t_PM_pixel     : in std_logic_vector(pix_depth-1 downto 0);
    t_PM_x_dest    : in std_logic_vector(img_width-1 downto 0);
    t_PM_y_dest    : in std_logic_vector(img_height-1 downto 0);
    t_PM_step      : in std_logic_vector(n_steps-1 downto 0);
    t_PM_frame     : in std_logic_vector(n_frames-1 downto 0);
    t_PM_x_orig    : in std_logic_vector(img_width-1 downto 0);
    t_PM_y_orig    : in std_logic_vector(img_height-1 downto 0);
    t_PM_fb        : in std_logic; -- identify if the it is a set_pixel or a set_pixel message.
   
    t_PM_req       : in std_logic;
    t_PM_ack       : out std_logic;
    
    
    
    t_PE_pixel   : in std_logic_vector(pix_depth-1 downto 0);
    t_PE_x_dest  : in std_logic_vector(img_width-1 downto 0);
    t_PE_y_dest  : in std_logic_vector(img_height-1 downto 0);
    t_PE_step    : in std_logic_vector(n_steps-1 downto 0);
    t_PE_frame   : in std_logic_vector(n_frames-1 downto 0);
    t_PE_x_orig  : in std_logic_vector(img_width-1 downto 0);
    t_PE_y_orig  : in std_logic_vector(img_height-1 downto 0);
    t_PE_fb      : in std_logic; -- identify if the it is a set_pixel or a set_pixel message. 
    t_PE_req     : in std_logic;  
    t_PE_ack     : out std_logic;   
    
    t_N_pixel     : in   std_logic_vector(pix_depth-1 downto 0);
    t_N_x_dest    : in   std_logic_vector(img_width-1 downto 0);
    t_N_y_dest    : in   std_logic_vector(img_height-1 downto 0);
    t_N_step      : in   std_logic_vector(n_steps-1 downto 0);
    t_N_frame     : in   std_logic_vector(n_frames-1 downto 0);
    t_N_x_orig    : in   std_logic_vector(img_width-1 downto 0);
    t_N_y_orig    : in   std_logic_vector(img_height-1 downto 0);
    t_N_fb        : in   std_logic; -- message forward/backward
    t_N_req       : in std_logic;    
    t_N_ack       : out std_logic;
    
    t_S_pixel      : in   std_logic_vector(pix_depth-1 downto 0);
    t_S_x_dest     : in   std_logic_vector(img_width-1 downto 0);
    t_S_y_dest     : in   std_logic_vector(img_height-1 downto 0);
    t_S_step       : in   std_logic_vector(n_steps-1 downto 0);
    t_S_frame      : in   std_logic_vector(n_frames-1 downto 0);
    t_S_x_orig     : in   std_logic_vector(img_width-1 downto 0);
    t_S_y_orig     : in   std_logic_vector(img_height-1 downto 0);
    t_S_fb         : in   std_logic; -- message forward/backward
    t_S_req        : in std_logic;    
    t_S_ack        : out std_logic;
    
    t_E_pixel      : in  std_logic_vector(pix_depth-1 downto 0);
    t_E_x_dest     : in  std_logic_vector(img_width-1 downto 0);
    t_E_y_dest     : in  std_logic_vector(img_height-1 downto 0);
    t_E_step       : in  std_logic_vector(n_steps-1 downto 0);
    t_E_frame      : in  std_logic_vector(n_frames-1 downto 0);
    t_E_x_orig     : in  std_logic_vector(img_width-1 downto 0);
    t_E_y_orig     : in  std_logic_vector(img_height-1 downto 0);
    t_E_fb         : in  std_logic; -- message forward/backward
    t_E_req        : in std_logic;
    t_E_ack        : out std_logic;
    
    t_W_pixel      : in  std_logic_vector(pix_depth-1 downto 0);
    t_W_x_dest     : in  std_logic_vector(img_width-1 downto 0);
    t_W_y_dest     : in  std_logic_vector(img_height-1 downto 0);
    t_W_step       : in  std_logic_vector(n_steps-1 downto 0);
    t_W_frame      : in  std_logic_vector(n_frames-1 downto 0);
    t_W_x_orig     : in  std_logic_vector(img_width-1 downto 0);
    t_W_y_orig     : in  std_logic_vector(img_height-1 downto 0);
    t_W_fb         : in  std_logic; -- message forward/backward    
    t_W_req        : in std_logic;
    t_W_ack        : out std_logic;
    IN_i_W_busy    : out std_logic;



    i_PM_pixel     : out std_logic_vector(pix_depth-1 downto 0);
    i_PM_x_dest    : out std_logic_vector(img_width-1 downto 0);
    i_PM_y_dest    : out std_logic_vector(img_height-1 downto 0);
    i_PM_step      : out std_logic_vector(n_steps-1 downto 0);
    i_PM_frame     : out std_logic_vector(n_frames-1 downto 0);
    i_PM_x_orig    : out std_logic_vector(img_width-1 downto 0);
    i_PM_y_orig    : out std_logic_vector(img_height-1 downto 0);
    i_PM_fb        : out std_logic; -- identify if the it is a set_pixel or a set_pixel message.    
    i_PM_req       : inout std_logic;
    i_PM_ack       : in std_logic;
    
    
    i_PE_pixel  : out std_logic_vector(pix_depth-1 downto 0);
    i_PE_x_dest : out std_logic_vector(img_width-1 downto 0);
    i_PE_y_dest : out std_logic_vector(img_height-1 downto 0);
    i_PE_step   : out std_logic_vector(n_steps-1 downto 0);
    i_PE_frame  : out std_logic_vector(n_frames-1 downto 0);
    i_PE_x_orig : out std_logic_vector(img_width-1 downto 0);
    i_PE_y_orig : out std_logic_vector(img_height-1 downto 0);
    i_PE_fb     : out std_logic; -- identify if the it is a set_pixel or a set_pixel message.
    i_PE_req    : inout std_logic;
    i_PE_ack    : in std_logic;
   -- O_i_PE_busy   : in std_logic;
    
    
    
    i_N_pixel   : out std_logic_vector(pix_depth-1 downto 0);
    i_N_x_dest  : out std_logic_vector(img_width-1 downto 0);
    i_N_y_dest  : out std_logic_vector(img_height-1 downto 0);
    i_N_step    : out std_logic_vector(n_steps-1 downto 0);
    i_N_frame   : out std_logic_vector(n_frames-1 downto 0);
    i_N_x_orig  : out std_logic_vector(img_width-1 downto 0);
    i_N_y_orig  : out std_logic_vector(img_height-1 downto 0);
    i_N_fb      : out std_logic; -- message forward/backward    
    i_N_req    : inout std_logic;
    i_N_ack    : in std_logic;
   -- O_i_N_busy    : in std_logic;
    
    i_S_pixel   : out std_logic_vector(pix_depth-1 downto 0);
    i_S_x_dest  : out std_logic_vector(img_width-1 downto 0);
    i_S_y_dest  : out std_logic_vector(img_height-1 downto 0);
    i_S_step    : out std_logic_vector(n_steps-1 downto 0);
    i_S_frame   : out std_logic_vector(n_frames-1 downto 0);
    i_S_x_orig  : out std_logic_vector(img_width-1 downto 0);
    i_S_y_orig  : out std_logic_vector(img_height-1 downto 0);
    i_S_fb      : out std_logic; -- message forward/backward 
    i_S_req    : inout std_logic;
    i_S_ack    : in std_logic;
    --O_i_S_busy    : in std_logic;
    
    
    
    i_E_pixel   : out std_logic_vector(pix_depth-1 downto 0);
    i_E_x_dest  : out std_logic_vector(img_width-1 downto 0);
    i_E_y_dest  : out std_logic_vector(img_height-1 downto 0);
    i_E_step    : out std_logic_vector(n_steps-1 downto 0);
    i_E_frame   : out std_logic_vector(n_frames-1 downto 0);
    i_E_x_orig  : out std_logic_vector(img_width-1 downto 0);
    i_E_y_orig  : out std_logic_vector(img_height-1 downto 0);
    i_E_fb      : out std_logic; -- message forward/backward
    i_E_req     : inout std_logic;
    i_E_ack     : in std_logic;
    --O_i_E_busy    : in std_logic;
    
    
    
    i_W_pixel   : out std_logic_vector(pix_depth-1 downto 0);
    i_W_x_dest  : out std_logic_vector(img_width-1 downto 0);
    i_W_y_dest  : out std_logic_vector(img_height-1 downto 0);
    i_W_step    : out std_logic_vector(n_steps-1 downto 0);
    i_W_frame   : out std_logic_vector(n_frames-1 downto 0);
    i_W_x_orig  : out std_logic_vector(img_width-1 downto 0);
    i_W_y_orig  : out std_logic_vector(img_height-1 downto 0);
    i_W_fb      : out std_logic; -- message forward/backward      
    i_W_req     : inout std_logic;
    i_W_ack     : in std_logic
   -- O_i_W_busy    : in  std_logic
    
   -- reset_output_pe_ctrl : in std_logic
);    
end component router_new;
   


signal    signal_t_PM_pixel            :       std_logic_vector(pix_depth-1 downto 0);
signal    signal_t_PM_x_dest           :       std_logic_vector(img_width-1 downto 0);
signal    signal_t_PM_y_dest           :       std_logic_vector(img_height-1 downto 0);
signal    signal_t_PM_step             :       std_logic_vector(n_steps-1 downto 0);
signal    signal_t_PM_frame            :       std_logic_vector(n_frames-1 downto 0);
signal    signal_t_PM_x_orig           :       std_logic_vector(img_width-1 downto 0);
signal    signal_t_PM_y_orig           :       std_logic_vector(img_height-1 downto 0);
signal    signal_t_PM_fb               :       std_logic; -- identify if the it is a set_pixel or a set_pixel message.  
signal    signal_t_PM_req              :       std_logic;
signal    signal_t_PM_ack              :       std_logic;


signal    signal_t_PE_pixel            :       std_logic_vector(pix_depth-1 downto 0);
signal    signal_t_PE_x_dest           :       std_logic_vector(img_width-1 downto 0);
signal    signal_t_PE_y_dest           :       std_logic_vector(img_height-1 downto 0);
signal    signal_t_PE_step             :       std_logic_vector(n_steps-1 downto 0);
signal    signal_t_PE_frame            :       std_logic_vector(n_frames-1 downto 0);
signal    signal_t_PE_x_orig           :       std_logic_vector(img_width-1 downto 0);
signal    signal_t_PE_y_orig           :       std_logic_vector(img_height-1 downto 0);
signal    signal_t_PE_fb               :       std_logic; -- identify if the it is a set_pixel or a set_pixel message. 
signal    signal_t_PE_req              :       std_logic;  
signal    signal_t_PE_ack              :       std_logic;   
signal    signal_t_N_pixel             :       std_logic_vector(pix_depth-1 downto 0);
signal    signal_t_N_x_dest            :       std_logic_vector(img_width-1 downto 0);
signal    signal_t_N_y_dest            :       std_logic_vector(img_height-1 downto 0);
signal    signal_t_N_step              :       std_logic_vector(n_steps-1 downto 0);
signal    signal_t_N_frame             :       std_logic_vector(n_frames-1 downto 0);
signal    signal_t_N_x_orig            :       std_logic_vector(img_width-1 downto 0);
signal    signal_t_N_y_orig            :       std_logic_vector(img_height-1 downto 0);
signal    signal_t_N_fb                :       std_logic; -- message forward/backward
signal    signal_t_N_req               :       std_logic;    
signal    signal_t_N_ack               :       std_logic;
signal    signal_t_S_pixel             :       std_logic_vector(pix_depth-1 downto 0);
signal    signal_t_S_x_dest            :       std_logic_vector(img_width-1 downto 0);
signal    signal_t_S_y_dest            :       std_logic_vector(img_height-1 downto 0);
signal    signal_t_S_step              :       std_logic_vector(n_steps-1 downto 0);
signal    signal_t_S_frame             :       std_logic_vector(n_frames-1 downto 0);
signal    signal_t_S_x_orig            :       std_logic_vector(img_width-1 downto 0);
signal    signal_t_S_y_orig            :       std_logic_vector(img_height-1 downto 0);
signal    signal_t_S_fb                :       std_logic; -- message forward/backward
signal    signal_t_S_req               :       std_logic;    
signal    signal_t_S_ack               :       std_logic;
signal    signal_t_E_pixel             :       std_logic_vector(pix_depth-1 downto 0);
signal    signal_t_E_x_dest            :       std_logic_vector(img_width-1 downto 0);
signal    signal_t_E_y_dest            :       std_logic_vector(img_height-1 downto 0);
signal    signal_t_E_step              :       std_logic_vector(n_steps-1 downto 0);
signal    signal_t_E_frame             :       std_logic_vector(n_frames-1 downto 0);
signal    signal_t_E_x_orig            :       std_logic_vector(img_width-1 downto 0);
signal    signal_t_E_y_orig            :       std_logic_vector(img_height-1 downto 0);
signal    signal_t_E_fb                :       std_logic; -- message forward/backward
signal    signal_t_E_req               :       std_logic;
signal    signal_t_E_ack               :       std_logic;
signal    signal_t_W_pixel             :       std_logic_vector(pix_depth-1 downto 0);
signal    signal_t_W_x_dest            :       std_logic_vector(img_width-1 downto 0);
signal    signal_t_W_y_dest            :       std_logic_vector(img_height-1 downto 0);
signal    signal_t_W_step              :       std_logic_vector(n_steps-1 downto 0);
signal    signal_t_W_frame             :       std_logic_vector(n_frames-1 downto 0);
signal    signal_t_W_x_orig            :       std_logic_vector(img_width-1 downto 0);
signal    signal_t_W_y_orig            :       std_logic_vector(img_height-1 downto 0);
signal    signal_t_W_fb                :       std_logic; -- message forward/backward    
signal    signal_t_W_req               :       std_logic;
signal    signal_t_W_ack               :       std_logic;
signal    signal_i_PM_pixel            :       std_logic_vector(pix_depth-1 downto 0);
signal    signal_i_PM_x_dest           :       std_logic_vector(img_width-1 downto 0);
signal    signal_i_PM_y_dest           :       std_logic_vector(img_height-1 downto 0);
signal    signal_i_PM_step             :       std_logic_vector(n_steps-1 downto 0);
signal    signal_i_PM_frame            :       std_logic_vector(n_frames-1 downto 0);
signal    signal_i_PM_x_orig           :       std_logic_vector(img_width-1 downto 0);
signal    signal_i_PM_y_orig           :       std_logic_vector(img_height-1 downto 0);
signal    signal_i_PM_fb               :       std_logic; -- identify if the it is a set_pixel or a set_pixel message.    
signal    signal_i_PM_req              :       std_logic;
signal    signal_i_PM_ack              :       std_logic;
signal    signal_i_PE_pixel            :       std_logic_vector(pix_depth-1 downto 0);
signal    signal_i_PE_x_dest           :       std_logic_vector(img_width-1 downto 0);
signal    signal_i_PE_y_dest           :       std_logic_vector(img_height-1 downto 0);
signal    signal_i_PE_step             :       std_logic_vector(n_steps-1 downto 0);
signal    signal_i_PE_frame            :       std_logic_vector(n_frames-1 downto 0);
signal    signal_i_PE_x_orig           :       std_logic_vector(img_width-1 downto 0);
signal    signal_i_PE_y_orig           :       std_logic_vector(img_height-1 downto 0);
signal    signal_i_PE_fb               :       std_logic; -- identify if the it is a set_pixel or a set_pixel message.
signal    signal_i_PE_req              :       std_logic;
signal    signal_i_PE_ack              :       std_logic;
signal    signal_i_N_pixel             :       std_logic_vector(pix_depth-1 downto 0);
signal    signal_i_N_x_dest            :       std_logic_vector(img_width-1 downto 0);
signal    signal_i_N_y_dest            :       std_logic_vector(img_height-1 downto 0);
signal    signal_i_N_step              :       std_logic_vector(n_steps-1 downto 0);
signal    signal_i_N_frame             :       std_logic_vector(n_frames-1 downto 0);
signal    signal_i_N_x_orig            :       std_logic_vector(img_width-1 downto 0);
signal    signal_i_N_y_orig            :       std_logic_vector(img_height-1 downto 0);
signal    signal_i_N_fb                :       std_logic; -- message forward/backward    
signal    signal_i_N_req               :       std_logic;
signal    signal_i_N_ack               :       std_logic;
signal    signal_i_S_pixel             :       std_logic_vector(pix_depth-1 downto 0);
signal    signal_i_S_x_dest            :       std_logic_vector(img_width-1 downto 0);
signal    signal_i_S_y_dest            :       std_logic_vector(img_height-1 downto 0);
signal    signal_i_S_step              :       std_logic_vector(n_steps-1 downto 0);
signal    signal_i_S_frame             :       std_logic_vector(n_frames-1 downto 0);
signal    signal_i_S_x_orig            :       std_logic_vector(img_width-1 downto 0);
signal    signal_i_S_y_orig            :       std_logic_vector(img_height-1 downto 0);
signal    signal_i_S_fb                :       std_logic; -- message forward/backward 
signal    signal_i_S_req               :       std_logic;
signal    signal_i_S_ack               :       std_logic;
signal    signal_i_E_pixel             :       std_logic_vector(pix_depth-1 downto 0);
signal    signal_i_E_x_dest            :       std_logic_vector(img_width-1 downto 0);
signal    signal_i_E_y_dest            :       std_logic_vector(img_height-1 downto 0);
signal    signal_i_E_step              :       std_logic_vector(n_steps-1 downto 0);
signal    signal_i_E_frame             :       std_logic_vector(n_frames-1 downto 0);
signal    signal_i_E_x_orig            :       std_logic_vector(img_width-1 downto 0);
signal    signal_i_E_y_orig            :       std_logic_vector(img_height-1 downto 0);
signal    signal_i_E_fb                :       std_logic; -- message forward/backward
signal    signal_i_E_req               :       std_logic;
signal    signal_i_E_ack               :       std_logic;
signal    signal_i_W_pixel             :       std_logic_vector(pix_depth-1 downto 0);
signal    signal_i_W_x_dest            :       std_logic_vector(img_width-1 downto 0);
signal    signal_i_W_y_dest            :       std_logic_vector(img_height-1 downto 0);
signal    signal_i_W_step              :       std_logic_vector(n_steps-1 downto 0);
signal    signal_i_W_frame             :       std_logic_vector(n_frames-1 downto 0);
signal    signal_i_W_x_orig            :       std_logic_vector(img_width-1 downto 0);
signal    signal_i_W_y_orig            :       std_logic_vector(img_height-1 downto 0);
signal    signal_i_W_fb                :       std_logic; -- message forward/backward      
signal    signal_i_W_req               :       std_logic;
signal    signal_i_W_ack               :       std_logic    ;































begin
wrp_router: router_new
    generic map(
    x_init                => x_init         , 
    y_init                => y_init         ,
    pix_depth  	         => pix_depth  	   ,
    img_width	         => img_width	   ,  
    img_height 	         => img_height 	   ,
    subimg_width          => subimg_width   ,
    subimg_height         => subimg_height  ,
    n_steps 	             => n_steps 	   ,   
    n_frames   	         => n_frames   	   ,
    buffer_length         => buffer_length  
    )
    port map(
    clk=>clk,
    reset=>reset,
    t_PM_pixel =>signal_t_PM_pixel  ,   
    t_PM_x_dest=>signal_t_PM_x_dest ,   
    t_PM_y_dest=>signal_t_PM_y_dest ,   
    t_PM_step  =>signal_t_PM_step   ,   
    t_PM_frame =>signal_t_PM_frame  ,   
    t_PM_x_orig=>signal_t_PM_x_orig ,   
    t_PM_y_orig=>signal_t_PM_y_orig ,   
    t_PM_fb    =>signal_t_PM_fb     ,   
    t_PM_req   =>signal_t_PM_req    ,   
    t_PM_ack   =>out_router_in_pm_ack    ,       
    
    t_PE_pixel =>signal_t_PE_pixel   ,
    t_PE_x_dest=>signal_t_PE_x_dest  ,
    t_PE_y_dest=>signal_t_PE_y_dest  ,
    t_PE_step  =>signal_t_PE_step    ,
    t_PE_frame =>signal_t_PE_frame   ,
    t_PE_x_orig=>signal_t_PE_x_orig  ,
    t_PE_y_orig=>signal_t_PE_y_orig  ,
    t_PE_fb    =>signal_t_PE_fb      ,
    t_PE_req   =>signal_t_PE_req     ,
    t_PE_ack   =>out_router_in_pe_ack  ,
    
    t_N_pixel  =>signal_t_N_pixel    ,
    t_N_x_dest =>signal_t_N_x_dest    ,
    t_N_y_dest =>signal_t_N_y_dest    ,
    t_N_step   =>signal_t_N_step      ,
    t_N_frame  =>signal_t_N_frame     ,
    t_N_x_orig =>signal_t_N_x_orig    ,
    t_N_y_orig =>signal_t_N_y_orig    ,
    t_N_fb     =>signal_t_N_fb        ,
    t_N_req    =>signal_t_N_req       ,
    t_N_ack    =>out_router_in_n_ack       ,
    
    t_S_pixel  =>signal_t_S_pixel      ,
    t_S_x_dest =>signal_t_S_x_dest     ,
    t_S_y_dest =>signal_t_S_y_dest     ,
    t_S_step   =>signal_t_S_step       ,
    t_S_frame  =>signal_t_S_frame      ,
    t_S_x_orig =>signal_t_S_x_orig     ,
    t_S_y_orig =>signal_t_S_y_orig     ,
    t_S_fb     =>signal_t_S_fb         ,
    t_S_req    =>signal_t_S_req        ,
    t_S_ack    =>out_router_in_s_ack        ,  
                       
    t_E_pixel  =>signal_t_E_pixel       ,
    t_E_x_dest =>signal_t_E_x_dest      ,
    t_E_y_dest =>signal_t_E_y_dest      ,
    t_E_step   =>signal_t_E_step        ,
    t_E_frame  =>signal_t_E_frame       ,
    t_E_x_orig =>signal_t_E_x_orig      ,
    t_E_y_orig =>signal_t_E_y_orig      ,
    t_E_fb     =>signal_t_E_fb          ,
    t_E_req    =>signal_t_E_req         ,
    t_E_ack    =>out_router_in_e_ack    ,
    
    t_W_pixel  =>signal_t_W_pixel       ,
    t_W_x_dest =>signal_t_W_x_dest      ,
    t_W_y_dest =>signal_t_W_y_dest      ,
    t_W_step   =>signal_t_W_step        ,
    t_W_frame  =>signal_t_W_frame       ,
    t_W_x_orig =>signal_t_W_x_orig      ,
    t_W_y_orig =>signal_t_W_y_orig      ,
    t_W_fb     =>signal_t_W_fb          ,
    t_W_req    =>signal_t_W_req         ,
    t_W_ack    =>out_router_in_w_ack    ,
    
    i_PM_pixel =>signal_i_PM_pixel   ,
    i_PM_x_dest=>signal_i_PM_x_dest  ,
    i_PM_y_dest=>signal_i_PM_y_dest  ,
    i_PM_step  =>signal_i_PM_step    ,
    i_PM_frame =>signal_i_PM_frame   ,
    i_PM_x_orig=>signal_i_PM_x_orig  ,
    i_PM_y_orig=>signal_i_PM_y_orig  ,
    i_PM_fb    =>signal_i_PM_fb      ,
    i_PM_req   =>signal_i_PM_req     ,
    i_PM_ack   =>in_router_out_pm_ack,
    
    i_PE_pixel =>signal_i_PE_pixel  ,
    i_PE_x_dest=>signal_i_PE_x_dest ,
    i_PE_y_dest=>signal_i_PE_y_dest ,
    i_PE_step  =>signal_i_PE_step   ,
    i_PE_frame =>signal_i_PE_frame  ,
    i_PE_x_orig=>signal_i_PE_x_orig ,
    i_PE_y_orig=>signal_i_PE_y_orig ,
    i_PE_fb    =>signal_i_PE_fb     ,
    i_PE_req   =>signal_i_PE_req    ,
    i_PE_ack   =>in_router_out_pe_ack ,
    
    i_N_pixel  =>signal_i_N_pixel   ,
    i_N_x_dest =>signal_i_N_x_dest  ,
    i_N_y_dest =>signal_i_N_y_dest  ,
    i_N_step   =>signal_i_N_step    ,
    i_N_frame  =>signal_i_N_frame   ,
    i_N_x_orig =>signal_i_N_x_orig  ,
    i_N_y_orig =>signal_i_N_y_orig  ,
    i_N_fb     =>signal_i_N_fb      ,
    i_N_req    =>signal_i_N_req     ,
    i_N_ack    =>in_router_out_n_ack     ,
    
    i_S_pixel  =>signal_i_S_pixel   ,
    i_S_x_dest =>signal_i_S_x_dest  ,
    i_S_y_dest =>signal_i_S_y_dest  ,
    i_S_step   =>signal_i_S_step    ,
    i_S_frame  =>signal_i_S_frame   ,
    i_S_x_orig =>signal_i_S_x_orig  ,
    i_S_y_orig =>signal_i_S_y_orig  ,
    i_S_fb     =>signal_i_S_fb      ,
    i_S_req    =>signal_i_S_req     ,
    i_S_ack    =>in_router_out_s_ack     ,
    
    i_E_pixel  =>signal_i_E_pixel   ,
    i_E_x_dest =>signal_i_E_x_dest  ,
    i_E_y_dest =>signal_i_E_y_dest  ,
    i_E_step   =>signal_i_E_step    ,
    i_E_frame  =>signal_i_E_frame   ,
    i_E_x_orig =>signal_i_E_x_orig  ,
    i_E_y_orig =>signal_i_E_y_orig  ,
    i_E_fb     =>signal_i_E_fb      ,
    i_E_req    =>signal_i_E_req     ,
    i_E_ack    =>in_router_out_e_ack     ,
    
    i_W_pixel  =>signal_i_W_pixel   ,
    i_W_x_dest =>signal_i_W_x_dest  ,
    i_W_y_dest =>signal_i_W_y_dest  ,
    i_W_step   =>signal_i_W_step    ,
    i_W_frame  =>signal_i_W_frame   ,
    i_W_x_orig =>signal_i_W_x_orig  ,
    i_W_y_orig =>signal_i_W_y_orig  ,
    i_W_fb     =>signal_i_W_fb      ,
    i_W_req    =>signal_i_W_req     ,
    i_W_ack    =>in_router_out_w_ack     
);

-- slice dos pacotes
--PM escreve no roteado
signal_t_PM_pixel   <=in_router_out_pm_data(pix_depth + img_width + img_width +  n_steps+ n_frames + img_width+ img_width  +2  downto  img_width + img_width +  n_steps+ n_frames + img_width+ img_width +2 +1); 
signal_t_PM_x_dest  <=in_router_out_pm_data(img_width + img_width +  n_steps+ n_frames + img_width+ img_width +2 downto img_width +  n_steps+ n_frames + img_width+ img_width +2 +1);                                          
signal_t_PM_y_dest  <=in_router_out_pm_data(img_width +  n_steps+ n_frames + img_width+ img_width +2 downto n_steps+ n_frames + img_width+ img_width +2+1);                                                               
signal_t_PM_step    <=in_router_out_pm_data(n_steps+ n_frames + img_width+ img_width +2 downto n_frames+img_width+img_width +2 +1);                                                                                 
signal_t_PM_frame   <=in_router_out_pm_data(n_frames + img_width+ img_width +2 downto img_width+img_width +2+1);                                                                                                       
signal_t_PM_x_orig  <=in_router_out_pm_data(img_width+ img_width +2   downto img_width +2+1);                                                                                                                        
signal_t_PM_y_orig  <=in_router_out_pm_data(img_width +2  downto +2+1);                                                                                                                                         
signal_t_PM_fb      <=in_router_out_pm_data(1+1);                                                                                                                                                              
signal_t_PM_req     <=in_router_out_pm_data(0+1); --req                                                                                                                                                        (0); --req

signal_t_PE_pixel   <=in_router_out_pe_data(pix_depth + img_width + img_width +  n_steps+ n_frames + img_width+ img_width  +2  downto  img_width + img_width +  n_steps+ n_frames + img_width+ img_width +2 +1);         
signal_t_PE_x_dest  <=in_router_out_pe_data(img_width + img_width +  n_steps+ n_frames + img_width+ img_width +2 downto img_width +  n_steps+ n_frames + img_width+ img_width +2 +1);                                    
signal_t_PE_y_dest  <=in_router_out_pe_data(img_width +  n_steps+ n_frames + img_width+ img_width +2 downto n_steps+ n_frames + img_width+ img_width +2+1);                                                              
signal_t_PE_step    <=in_router_out_pe_data(n_steps+ n_frames + img_width+ img_width +2 downto n_frames+img_width+img_width +2 +1);                                                                                      
signal_t_PE_frame   <=in_router_out_pe_data(n_frames + img_width+ img_width +2 downto img_width+img_width +2+1);                                                                                                         
signal_t_PE_x_orig  <=in_router_out_pe_data(img_width+ img_width +2   downto img_width +2+1);                                                                                                                            
signal_t_PE_y_orig  <=in_router_out_pe_data(img_width +2  downto +2+1);                                                                                                                                                  
signal_t_PE_fb      <=in_router_out_pe_data(1+1);                                                                                                                                                                        
signal_t_PE_req     <=in_router_out_pe_data(0+1); --req                                                                                                                                                        (0); --req

signal_t_N_pixel   <=in_router_out_n_data(pix_depth + img_width + img_width +  n_steps+ n_frames + img_width+ img_width  +2  downto  img_width + img_width +  n_steps+ n_frames + img_width+ img_width +2 +1);         
signal_t_N_x_dest  <=in_router_out_n_data(img_width + img_width +  n_steps+ n_frames + img_width+ img_width +2 downto img_width +  n_steps+ n_frames + img_width+ img_width +2 +1);                                    
signal_t_N_y_dest  <=in_router_out_n_data(img_width +  n_steps+ n_frames + img_width+ img_width +2 downto n_steps+ n_frames + img_width+ img_width +2+1);                                                              
signal_t_N_step    <=in_router_out_n_data(n_steps+ n_frames + img_width+ img_width +2 downto n_frames+img_width+img_width +2 +1);                                                                                      
signal_t_N_frame   <=in_router_out_n_data(n_frames + img_width+ img_width +2 downto img_width+img_width +2+1);                                                                                                         
signal_t_N_x_orig  <=in_router_out_n_data(img_width+ img_width +2   downto img_width +2+1);                                                                                                                            
signal_t_N_y_orig  <=in_router_out_n_data(img_width +2  downto +2+1);                                                                                                                                                  
signal_t_N_fb      <=in_router_out_n_data(1+1);                                                                                                                                                                        
signal_t_N_req     <=in_router_out_n_data(0+1); --req                                                                                                                                                        (0); --req

signal_t_S_pixel   <=in_router_out_s_data(pix_depth + img_width + img_width +  n_steps+ n_frames + img_width+ img_width  +2  downto  img_width + img_width +  n_steps+ n_frames + img_width+ img_width +2 +1);         
signal_t_S_x_dest  <=in_router_out_s_data(img_width + img_width +  n_steps+ n_frames + img_width+ img_width +2 downto img_width +  n_steps+ n_frames + img_width+ img_width +2 +1);                                    
signal_t_S_y_dest  <=in_router_out_s_data(img_width +  n_steps+ n_frames + img_width+ img_width +2 downto n_steps+ n_frames + img_width+ img_width +2+1);                                                              
signal_t_S_step    <=in_router_out_s_data(n_steps+ n_frames + img_width+ img_width +2 downto n_frames+img_width+img_width +2 +1);                                                                                      
signal_t_S_frame   <=in_router_out_s_data(n_frames + img_width+ img_width +2 downto img_width+img_width +2+1);                                                                                                         
signal_t_S_x_orig  <=in_router_out_s_data(img_width+ img_width +2   downto img_width +2+1);                                                                                                                            
signal_t_S_y_orig  <=in_router_out_s_data(img_width +2  downto +2+1);                                                                                                                                                  
signal_t_S_fb      <=in_router_out_s_data(1+1);                                                                                                                                                                          
signal_t_S_req     <=in_router_out_s_data(0+1); --req                                                                                                                                                        (0); --req  


i_PE_pixel  <=signal_i_PE_pixel  ;
i_PE_x_dest <=signal_i_PE_x_dest ;
i_PE_y_dest <=signal_i_PE_y_dest ;
i_PE_step   <=signal_i_PE_step   ;
i_PE_frame  <=signal_i_PE_frame  ;
i_PE_x_orig <=signal_i_PE_x_orig ;
i_PE_y_orig <=signal_i_PE_y_orig ;
i_PE_fb     <=signal_i_PE_fb     ;





signal_t_E_pixel   <=in_router_out_e_data(pix_depth + img_width + img_width +  n_steps+ n_frames + img_width+ img_width  +2  downto  img_width + img_width +  n_steps+ n_frames + img_width+ img_width +2 +1);         
signal_t_E_x_dest  <=in_router_out_e_data(img_width + img_width +  n_steps+ n_frames + img_width+ img_width +2 downto img_width +  n_steps+ n_frames + img_width+ img_width +2 +1);                                    
signal_t_E_y_dest  <=in_router_out_e_data(img_width +  n_steps+ n_frames + img_width+ img_width +2 downto n_steps+ n_frames + img_width+ img_width +2+1);                                                              
signal_t_E_step    <=in_router_out_e_data(n_steps+ n_frames + img_width+ img_width +2 downto n_frames+img_width+img_width +2 +1);                                                                                      
signal_t_E_frame   <=in_router_out_e_data(n_frames + img_width+ img_width +2 downto img_width+img_width +2+1);                                                                                                         
signal_t_E_x_orig  <=in_router_out_e_data(img_width+ img_width +2   downto img_width +2+1);                                                                                                                            
signal_t_E_y_orig  <=in_router_out_e_data(img_width +2  downto +2+1);                                                                                                                                                  
signal_t_E_fb      <=in_router_out_e_data(1+1);                                                                                                                                                                        
signal_t_E_req     <=in_router_out_e_data(0+1); --req                                                                                                                                                        (0); --req

signal_t_W_pixel   <=in_router_out_w_data(pix_depth + img_width + img_width +  n_steps+ n_frames + img_width+ img_width  +2  downto  img_width + img_width +  n_steps+ n_frames + img_width+ img_width +2 +1);         
signal_t_W_x_dest  <=in_router_out_w_data(img_width + img_width +  n_steps+ n_frames + img_width+ img_width +2 downto img_width +  n_steps+ n_frames + img_width+ img_width +2 +1);                                    
signal_t_W_y_dest  <=in_router_out_w_data(img_width +  n_steps+ n_frames + img_width+ img_width +2 downto n_steps+ n_frames + img_width+ img_width +2+1);                                                              
signal_t_W_step    <=in_router_out_w_data(n_steps+ n_frames + img_width+ img_width +2 downto n_frames+img_width+img_width +2 +1);                                                                                      
signal_t_W_frame   <=in_router_out_w_data(n_frames + img_width+ img_width +2 downto img_width+img_width +2+1);                                                                                                         
signal_t_W_x_orig  <=in_router_out_w_data(img_width+ img_width +2   downto img_width +2+1);                                                                                                                            
signal_t_W_y_orig  <=in_router_out_w_data(img_width +2  downto +2+1);                                                                                                                                                  
signal_t_W_fb      <=in_router_out_w_data(1+1);                                                                                                                                                                        
signal_t_W_req     <=in_router_out_w_data(0+1); --req                                                                                                                                                        (0); --req

out_router_in_pm_data(63 downto 0)   <=  signal_i_PM_pixel & signal_i_PM_x_dest & signal_i_PM_y_dest & signal_i_PM_step &  signal_i_PM_frame &  signal_i_PM_x_orig &  signal_i_PM_y_orig &  signal_i_PM_fb &  signal_i_PM_req & in_router_out_pm_ack  ;
out_router_in_pe_data(63 downto 0)   <=  signal_i_PE_pixel & signal_i_PE_x_dest & signal_i_PE_y_dest & signal_i_PE_step &  signal_i_PE_frame &  signal_i_PE_x_orig &  signal_i_PE_y_orig &  signal_i_PE_fb &  signal_i_PE_req & in_router_out_pe_ack  ;
out_router_in_n_data (63 downto 0)   <=  signal_i_N_pixel  & signal_i_N_x_dest  & signal_i_N_y_dest  & signal_i_N_step  &  signal_i_N_frame  &  signal_i_N_x_orig  &  signal_i_N_y_orig  &  signal_i_N_fb  &  signal_i_N_req  & in_router_out_n_ack ;
out_router_in_s_data (63 downto 0)   <=  signal_i_S_pixel  & signal_i_S_x_dest  & signal_i_S_y_dest  & signal_i_S_step  &  signal_i_S_frame  &  signal_i_S_x_orig  &  signal_i_S_y_orig  &  signal_i_S_fb  &  signal_i_S_req  & in_router_out_s_ack ;
out_router_in_e_data (63 downto 0)   <=  signal_i_E_pixel  & signal_i_E_x_dest  & signal_i_E_y_dest  & signal_i_E_step  &  signal_i_E_frame  &  signal_i_E_x_orig  &  signal_i_E_y_orig  &  signal_i_E_fb  &  signal_i_E_req  & in_router_out_e_ack ;
out_router_in_w_data (63 downto 0)   <=  signal_i_W_pixel  & signal_i_W_x_dest  & signal_i_W_y_dest  & signal_i_W_step  &  signal_i_W_frame  &  signal_i_W_x_orig  &  signal_i_W_y_orig  &  signal_i_W_fb  &  signal_i_W_req  & in_router_out_w_ack ;


end Behavioral;
