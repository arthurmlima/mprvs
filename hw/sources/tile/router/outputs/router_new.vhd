----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 09/06/2019 08:22:01 PM
-- Design Name: 
-- Module Name: router - Behavioral
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
use ieee.numeric_std.all;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity router_new is

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
port(

    clk          : in std_logic;
    reset          : in std_logic;
    
    
   -- R_Wrapper_PM_ready_EB_CF : in std_LOGIC;
   -- R_Wrapper_PE_ready_EB_CF : in std_LOGIC;
   -- R_Wrapper_N_ready_EB_CF  : in std_LOGIC;
  --  R_Wrapper_S_ready_EB_CF  : in std_LOGIC;
   -- R_Wrapper_W_ready_EB_CF  : in std_LOGIC;
   -- R_Wrapper_E_ready_EB_CF  : in std_LOGIC;
    
    
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
    IN_i_PM_busy   : inout std_logic;
    
    
    
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
    IN_i_PE_busy : out std_logic;
    
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
    IN_i_N_busy   : out  std_logic;
    
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
    IN_i_S_busy    : out std_logic;
    
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
    IN_i_E_busy    : out std_logic;
    
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
   -- O_i_PM_busy    : in std_logic;
    
    
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
end router_new;

architecture Behavioral of router_new is

-- signal IN_i_PM_busy   : std_logic;

  signal deadlock_PM : std_logic;
  signal deadlock_PE : std_logic;
  signal deadlock_N : std_logic;
  signal deadlock_S : std_logic; 
  signal deadlock_E : std_logic;
  signal deadlock_W : std_logic;

  signal                 w_PM_in_pixel        :  std_logic_vector(pix_depth-1 downto 0);       
  signal                 w_PM_in_x_dest       :  std_logic_vector(img_width-1 downto 0);       
  signal                 w_PM_in_y_dest       :  std_logic_vector(img_height-1 downto 0);      
  signal                 w_PM_in_step         :  std_logic_vector(n_steps-1 downto 0);         
  signal                 w_PM_in_frame        :  std_logic_vector(n_frames-1 downto 0);        
  signal                 w_PM_in_x_orig       :  std_logic_vector(img_width-1 downto 0);       
  signal                 w_PM_in_y_orig       :  std_logic_vector(img_height-1 downto 0);      
  signal                 w_PM_in_fb           :  std_logic; -- message forward/backward  
        
  signal                 w_PM_stall_in_EB     : std_logic;        
  signal                 w_PM_stall_out_EB    : std_logic;        
  signal                 w_PM_stall_in_DEC    : std_logic;        
  signal                 w_PM_stall_out_DEC    : std_logic;
                  
  signal                 w_PM_in_direction    : std_logic_vector(5 downto 0);
                 
  signal                 w_PE_in_pixel        :   std_logic_vector(pix_depth-1 downto 0); 
  signal                 w_PE_in_x_dest       :   std_logic_vector(img_width-1 downto 0); 
  signal                 w_PE_in_y_dest       :   std_logic_vector(img_height-1 downto 0);
  signal                 w_PE_in_step         :   std_logic_vector(n_steps-1 downto 0);   
  signal                 w_PE_in_frame        :   std_logic_vector(n_frames-1 downto 0);  
  signal                 w_PE_in_x_orig       :   std_logic_vector(img_width-1 downto 0); 
  signal                 w_PE_in_y_orig       :   std_logic_vector(img_height-1 downto 0);
  signal                 w_PE_in_fb           :   std_logic; -- message forward/backward  
                 
  signal                 w_PE_stall_in_EB     :  std_logic;   
  signal                 w_PE_stall_out_EB    :  std_logic;   
  signal                 w_PE_stall_in_DEC    :  std_logic;   
  signal                 w_PE_stall_out_DEC    :  std_logic; 
    
  signal                 w_PE_in_direction    :  std_logic_vector(5 downto 0);
                
  signal                 w_N_in_pixel         :  std_logic_vector(pix_depth-1 downto 0); 
  signal                 w_N_in_x_dest        :  std_logic_vector(img_width-1 downto 0); 
  signal                 w_N_in_y_dest        :  std_logic_vector(img_height-1 downto 0);
  signal                 w_N_in_step          :  std_logic_vector(n_steps-1 downto 0);   
  signal                 w_N_in_frame         :  std_logic_vector(n_frames-1 downto 0);  
  signal                 w_N_in_x_orig        :  std_logic_vector(img_width-1 downto 0); 
  signal                 w_N_in_y_orig        :  std_logic_vector(img_height-1 downto 0);
  signal                 w_N_in_fb            :  std_logic; -- message forward/backward 
   
  signal                 w_N_stall_in_EB      : std_logic;     
  signal                 w_N_stall_out_EB     : std_logic;     
  signal                 w_N_stall_in_DEC     : std_logic;     
  signal                 w_N_stall_out_DEC    : std_logic;     
  
  signal                 w_N_in_direction     : std_logic_vector(5 downto 0);
                  
  signal                 w_S_in_pixel         : std_logic_vector(pix_depth-1 downto 0);              
  signal                 w_S_in_x_dest        : std_logic_vector(img_width-1 downto 0);              
  signal                 w_S_in_y_dest        : std_logic_vector(img_height-1 downto 0);             
  signal                 w_S_in_step          : std_logic_vector(n_steps-1 downto 0);                
  signal                 w_S_in_frame         : std_logic_vector(n_frames-1 downto 0);               
  signal                 w_S_in_x_orig        : std_logic_vector(img_width-1 downto 0);              
  signal                 w_S_in_y_orig        : std_logic_vector(img_height-1 downto 0);             
  signal                 w_S_in_fb            : std_logic; -- message forward/backward  
               
  signal                 w_S_stall_in_EB      : std_logic;
  signal                 w_S_stall_out_EB     : std_logic;
  signal                 w_S_stall_in_DEC     : std_logic;
  signal                 w_S_stall_out_DEC    : std_logic;
                                           
  signal                 w_S_in_direction     :   std_logic_vector(5 downto 0);
               
  signal                 w_E_in_pixel         :   std_logic_vector(pix_depth-1 downto 0);               
  signal                 w_E_in_x_dest        :   std_logic_vector(img_width-1 downto 0);               
  signal                 w_E_in_y_dest        :   std_logic_vector(img_height-1 downto 0);              
  signal                 w_E_in_step          :   std_logic_vector(n_steps-1 downto 0);                 
  signal                 w_E_in_frame         :   std_logic_vector(n_frames-1 downto 0);                
  signal                 w_E_in_x_orig        :   std_logic_vector(img_width-1 downto 0);               
  signal                 w_E_in_y_orig        :   std_logic_vector(img_height-1 downto 0);              
  signal                 w_E_in_fb            :   std_logic; -- message forward/backward     
             
  signal                 w_E_stall_in_EB      :  std_logic;          
  signal                 w_E_stall_out_EB     :  std_logic;          
  signal                 w_E_stall_in_DEC     :  std_logic;          
  signal                 w_E_stall_out_DEC    :  std_logic; 
           
  signal                 w_E_in_direction     : std_logic_vector(5 downto 0);
                 
  signal                 w_W_in_pixel         : std_logic_vector(pix_depth-1 downto 0); 
  signal                 w_W_in_x_dest        : std_logic_vector(img_width-1 downto 0); 
  signal                 w_W_in_y_dest        : std_logic_vector(img_height-1 downto 0);
  signal                 w_W_in_step          : std_logic_vector(n_steps-1 downto 0);   
  signal                 w_W_in_frame         : std_logic_vector(n_frames-1 downto 0);  
  signal                 w_W_in_x_orig        : std_logic_vector(img_width-1 downto 0); 
  signal                 w_W_in_y_orig        : std_logic_vector(img_height-1 downto 0);
  signal                 w_W_in_fb            : std_logic; -- message forward/backward  
  
  signal                 w_W_stall_in_EB      : std_logic;       
  signal                 w_W_stall_out_EB     : std_logic;       
  signal                 w_W_stall_in_DEC     : std_logic;       
  signal                 w_W_stall_out_DEC    : std_logic; 
        
  signal                 w_W_in_direction     :  std_logic_vector(5 downto 0);              
                 
  signal                  w_o_i_PM_req         : std_logic;             
  signal                  w_o_i_PM_ack         : std_logic;
                          
  signal                  w_o_i_PE_ack         : std_logic;             
  signal                  w_o_i_PE_req         : std_logic;
                         
  signal                  w_o_i_N_req         : std_logic;          
  signal                  w_o_i_N_ack         : std_logic;   
  
  signal                  w_o_i_S_req         : std_logic;          
  signal                  w_o_i_S_ack         : std_logic;   
                         
  signal                  w_o_i_E_ack         : std_logic;       
  signal                  w_o_i_E_req         : std_logic;
                          
  signal                  w_o_i_W_ack         : std_logic;             
  signal                  w_o_i_W_req         : std_logic; 
  
  signal                 PM_input_CTRL_pixel         : std_logic_vector(pix_depth-1 downto 0); 
  signal                 PM_input_CTRL_x_dest        : std_logic_vector(img_width-1 downto 0); 
  signal                 PM_input_CTRL_y_dest        : std_logic_vector(img_height-1 downto 0);
  signal                 PM_input_CTRL_step          : std_logic_vector(n_steps-1 downto 0);   
  signal                 PM_input_CTRL_frame         : std_logic_vector(n_frames-1 downto 0);  
  signal                 PM_input_CTRL_x_orig        : std_logic_vector(img_width-1 downto 0); 
  signal                 PM_input_CTRL_y_orig        : std_logic_vector(img_height-1 downto 0);
  signal                 PM_input_CTRL_fb            : std_logic; -- message forward/backward  
  signal                 PM_input_CTRL_ack            : std_logic ;
  signal                 PM_input_CTRL_req            : std_logic; -- message forward/backward  
              
  signal                 PE_input_CTRL_pixel         : std_logic_vector(pix_depth-1 downto 0); 
  signal                 PE_input_CTRL_x_dest        : std_logic_vector(img_width-1 downto 0); 
  signal                 PE_input_CTRL_y_dest        : std_logic_vector(img_height-1 downto 0);
  signal                 PE_input_CTRL_step          : std_logic_vector(n_steps-1 downto 0);   
  signal                 PE_input_CTRL_frame         : std_logic_vector(n_frames-1 downto 0);  
  signal                 PE_input_CTRL_x_orig        : std_logic_vector(img_width-1 downto 0); 
  signal                 PE_input_CTRL_y_orig        : std_logic_vector(img_height-1 downto 0);
  signal                 PE_input_CTRL_fb            : std_logic; -- message forward/backward  
  signal                 PE_input_CTRL_ack           : std_logic ;
  signal                 PE_input_CTRL_req           : std_logic; -- message forward/backward  
  

  signal                 N_input_CTRL_pixel         : std_logic_vector(pix_depth-1 downto 0); 
  signal                 N_input_CTRL_x_dest        : std_logic_vector(img_width-1 downto 0); 
  signal                 N_input_CTRL_y_dest        : std_logic_vector(img_height-1 downto 0);
  signal                 N_input_CTRL_step          : std_logic_vector(n_steps-1 downto 0);   
  signal                 N_input_CTRL_frame         : std_logic_vector(n_frames-1 downto 0);  
  signal                 N_input_CTRL_x_orig        : std_logic_vector(img_width-1 downto 0); 
  signal                 N_input_CTRL_y_orig        : std_logic_vector(img_height-1 downto 0);
  signal                 N_input_CTRL_fb            : std_logic; -- message forward/backward  
  signal                 N_input_CTRL_ack           : std_logic ;
  signal                 N_input_CTRL_req           : std_logic; -- message forward/backward  

  signal                 S_input_CTRL_pixel         : std_logic_vector(pix_depth-1 downto 0); 
  signal                 S_input_CTRL_x_dest        : std_logic_vector(img_width-1 downto 0); 
  signal                 S_input_CTRL_y_dest        : std_logic_vector(img_height-1 downto 0);
  signal                 S_input_CTRL_step          : std_logic_vector(n_steps-1 downto 0);   
  signal                 S_input_CTRL_frame         : std_logic_vector(n_frames-1 downto 0);  
  signal                 S_input_CTRL_x_orig        : std_logic_vector(img_width-1 downto 0); 
  signal                 S_input_CTRL_y_orig        : std_logic_vector(img_height-1 downto 0);
  signal                 S_input_CTRL_fb            : std_logic; -- message forward/backward  
  signal                 S_input_CTRL_ack           : std_logic ;
  signal                 S_input_CTRL_req           : std_logic; -- message forward/backward  

  signal                 E_input_CTRL_pixel         : std_logic_vector(pix_depth-1 downto 0); 
  signal                 E_input_CTRL_x_dest        : std_logic_vector(img_width-1 downto 0); 
  signal                 E_input_CTRL_y_dest        : std_logic_vector(img_height-1 downto 0);
  signal                 E_input_CTRL_step          : std_logic_vector(n_steps-1 downto 0);   
  signal                 E_input_CTRL_frame         : std_logic_vector(n_frames-1 downto 0);  
  signal                 E_input_CTRL_x_orig        : std_logic_vector(img_width-1 downto 0); 
  signal                 E_input_CTRL_y_orig        : std_logic_vector(img_height-1 downto 0);
  signal                 E_input_CTRL_fb            : std_logic; -- message forward/backward  
  signal                 E_input_CTRL_ack           : std_logic ;
  signal                 E_input_CTRL_req           : std_logic; -- message forward/backward  

  signal                 W_input_CTRL_pixel         : std_logic_vector(pix_depth-1 downto 0); 
  signal                 W_input_CTRL_x_dest        : std_logic_vector(img_width-1 downto 0); 
  signal                 W_input_CTRL_y_dest        : std_logic_vector(img_height-1 downto 0);
  signal                 W_input_CTRL_step          : std_logic_vector(n_steps-1 downto 0);   
  signal                 W_input_CTRL_frame         : std_logic_vector(n_frames-1 downto 0);  
  signal                 W_input_CTRL_x_orig        : std_logic_vector(img_width-1 downto 0); 
  signal                 W_input_CTRL_y_orig        : std_logic_vector(img_height-1 downto 0);
  signal                 W_input_CTRL_fb            : std_logic; -- message forward/backward  
  signal                 W_input_CTRL_ack           : std_logic ;
  signal                 W_input_CTRL_req           : std_logic; -- message forward/backward  
  
      
      
         
    
  
        
begin
Control_flux : entity work.cf
generic map(
    x_init => x_init,
    y_init => y_init, 
    pix_depth  	     =>pix_depth  	   ,
    img_width	     =>img_width	   , 
    img_height 	     =>img_height 	   ,
    subimg_width 	 =>subimg_width 	, 
    subimg_height	 =>subimg_height	, 
    n_steps 	     =>n_steps 	     ,
    n_frames   	     =>n_frames   	   ,
    buffer_length    =>buffer_length   
)
port map (

   clk => clk ,
   reset => reset,      
   PM_in_pixel       =>  w_PM_in_pixel      ,
   PM_in_x_dest      =>  w_PM_in_x_dest     ,
   PM_in_y_dest      =>  w_PM_in_y_dest     ,
   PM_in_step        =>  w_PM_in_step       ,
   PM_in_frame       =>  w_PM_in_frame      ,
   PM_in_x_orig      =>  w_PM_in_x_orig     ,
   PM_in_y_orig      =>  w_PM_in_y_orig     ,
   PM_in_fb          =>  w_PM_in_fb         ,  
   PM_stall_in_EB    =>  w_PM_stall_in_EB   ,
   PM_stall_out_EB   =>  w_PM_stall_out_EB  ,
   PM_stall_in_DEC   =>  w_PM_stall_in_DEC  ,
   PM_stall_out_DEC  =>  w_PM_stall_out_DEC ,                   
   PM_in_direction   =>  w_PM_in_direction  ,
                                             
   PE_in_pixel       =>  w_PE_in_pixel      ,
   PE_in_x_dest      =>  w_PE_in_x_dest     ,
   PE_in_y_dest      =>  w_PE_in_y_dest     ,
   PE_in_step        =>  w_PE_in_step       ,
   PE_in_frame       =>  w_PE_in_frame      ,
   PE_in_x_orig      =>  w_PE_in_x_orig     ,
   PE_in_y_orig      =>  w_PE_in_y_orig     ,
   PE_in_fb          =>  w_PE_in_fb         , 
   PE_stall_in_EB    =>  w_PE_stall_in_EB   ,
   PE_stall_out_EB   =>  w_PE_stall_out_EB  ,
   PE_stall_in_DEC   =>  w_PE_stall_in_DEC  ,
   PE_stall_out_DEC  =>  w_PE_stall_out_DEC ,                  
   PE_in_direction   =>  w_PE_in_direction  ,    

   N_in_pixel        =>  w_N_in_pixel       ,  
   N_in_x_dest       =>  w_N_in_x_dest      ,  
   N_in_y_dest       =>  w_N_in_y_dest      ,  
   N_in_step         =>  w_N_in_step        ,  
   N_in_frame        =>  w_N_in_frame       ,  
   N_in_x_orig       =>  w_N_in_x_orig      ,  
   N_in_y_orig       =>  w_N_in_y_orig      ,  
   N_in_fb           =>  w_N_in_fb          ,    
   N_stall_in_EB     =>  w_N_stall_in_EB    ,  
   N_stall_out_EB    =>  w_N_stall_out_EB   ,  
   N_stall_in_DEC    =>  w_N_stall_in_DEC   ,  
   N_stall_out_DEC   =>  w_N_stall_out_DEC  ,                                        
   N_in_direction    =>  w_N_in_direction   ,

   S_in_pixel        =>  w_S_in_pixel       ,
   S_in_x_dest       =>  w_S_in_x_dest      ,
   S_in_y_dest       =>  w_S_in_y_dest      ,
   S_in_step         =>  w_S_in_step        ,
   S_in_frame        =>  w_S_in_frame       ,
   S_in_x_orig       =>  w_S_in_x_orig      ,
   S_in_y_orig       =>  w_S_in_y_orig      ,
   S_in_fb           =>  w_S_in_fb          ,                 
   S_stall_in_EB     =>  w_S_stall_in_EB    ,
   S_stall_out_EB    =>  w_S_stall_out_EB   ,
   S_stall_in_DEC    =>  w_S_stall_in_DEC   ,
   S_stall_out_DEC   =>  w_S_stall_out_DEC  ,                                       
   S_in_direction    =>  w_S_in_direction   ,                  

   E_in_pixel        =>  w_E_in_pixel       ,
   E_in_x_dest       =>  w_E_in_x_dest      ,
   E_in_y_dest       =>  w_E_in_y_dest      ,
   E_in_step         =>  w_E_in_step        ,
   E_in_frame        =>  w_E_in_frame       ,
   E_in_x_orig       =>  w_E_in_x_orig      ,
   E_in_y_orig       =>  w_E_in_y_orig      ,
   E_in_fb           =>  w_E_in_fb          ,
   E_stall_in_EB     =>  w_E_stall_in_EB    ,
   E_stall_out_EB    =>  w_E_stall_out_EB   ,
   E_stall_in_DEC    =>  w_E_stall_in_DEC   ,
   E_stall_out_DEC   =>  w_E_stall_out_DEC  ,                                 
   E_in_direction    =>  w_E_in_direction   ,

   W_in_pixel        =>  w_W_in_pixel       ,
   W_in_x_dest       =>  w_W_in_x_dest      ,
   W_in_y_dest       =>  w_W_in_y_dest      ,
   W_in_step         =>  w_W_in_step        ,
   W_in_frame        =>  w_W_in_frame       ,
   W_in_x_orig       =>  w_W_in_x_orig      ,
   W_in_y_orig       =>  w_W_in_y_orig      ,
   W_in_fb           =>  w_W_in_fb          , 
   W_stall_in_EB     =>  w_W_stall_in_EB    ,
   W_stall_out_EB    =>  w_W_stall_out_EB   ,
   W_stall_in_DEC    =>  w_W_stall_in_DEC   ,
   W_stall_out_DEC   =>  w_W_stall_out_DEC  ,             
   W_in_direction    =>  w_W_in_direction   ,
   --------------------------------------------------------------  
   -----OUTPUT_messages
   -----
   ---------------------------------------------------------------                                                      
   i_PM_pixel       =>    PM_input_CTRL_pixel  ,
   i_PM_x_dest      =>    PM_input_CTRL_x_dest ,
   i_PM_y_dest      =>    PM_input_CTRL_y_dest ,
   i_PM_step        =>    PM_input_CTRL_step   ,
   i_PM_frame       =>    PM_input_CTRL_frame  ,
   i_PM_x_orig      =>    PM_input_CTRL_x_orig ,
   i_PM_y_orig      =>    PM_input_CTRL_y_orig ,
   i_PM_fb          =>    PM_input_CTRL_fb     ,
   i_PM_busy        =>    deadlock_PM,    -- inputs
  

                     
                                   
  i_PE_pixel      =>   PE_input_CTRL_pixel  ,
  i_PE_x_dest     =>   PE_input_CTRL_x_dest ,
  i_PE_y_dest     =>   PE_input_CTRL_y_dest ,
  i_PE_step       =>   PE_input_CTRL_step   ,
  i_PE_frame      =>   PE_input_CTRL_frame  ,
  i_PE_x_orig     =>   PE_input_CTRL_x_orig ,
  i_PE_y_orig     =>   PE_input_CTRL_y_orig ,
  i_PE_fb         =>   PE_input_CTRL_fb     ,
  i_PE_busy       =>    deadlock_PE,  
                  
  i_N_pixel      =>   N_input_CTRL_pixel  ,
  i_N_x_dest     =>   N_input_CTRL_x_dest ,
  i_N_y_dest     =>   N_input_CTRL_y_dest ,
  i_N_step       =>   N_input_CTRL_step   ,
  i_N_frame      =>   N_input_CTRL_frame  ,
  i_N_x_orig     =>   N_input_CTRL_x_orig ,
  i_N_y_orig     =>   N_input_CTRL_y_orig ,
  i_N_fb         =>   N_input_CTRL_fb     ,
  --i_N_busy       =>   deadlock_PE,  

  i_S_pixel      =>   S_input_CTRL_pixel  ,
  i_S_x_dest     =>   S_input_CTRL_x_dest ,
  i_S_y_dest     =>   S_input_CTRL_y_dest ,
  i_S_step       =>   S_input_CTRL_step   ,
  i_S_frame      =>   S_input_CTRL_frame  ,
  i_S_x_orig     =>   S_input_CTRL_x_orig ,
  i_S_y_orig     =>   S_input_CTRL_y_orig ,
  i_S_fb         =>   S_input_CTRL_fb     ,
  --i_S_busy       =>    deadlock_PE,     

  i_E_pixel      =>   E_input_CTRL_pixel  ,
  i_E_x_dest     =>   E_input_CTRL_x_dest ,
  i_E_y_dest     =>   E_input_CTRL_y_dest ,
  i_E_step       =>   E_input_CTRL_step   ,
  i_E_frame      =>   E_input_CTRL_frame  ,
  i_E_x_orig     =>   E_input_CTRL_x_orig ,
  i_E_y_orig     =>   E_input_CTRL_y_orig ,
  i_E_fb         =>   E_input_CTRL_fb     ,
  --i_E_busy       =>    deadlock_PE,  

  i_W_pixel      =>   W_input_CTRL_pixel  ,
  i_W_x_dest     =>   W_input_CTRL_x_dest ,
  i_W_y_dest     =>   W_input_CTRL_y_dest ,
  i_W_step       =>   W_input_CTRL_step   ,
  i_W_frame      =>   W_input_CTRL_frame  ,
  i_W_x_orig     =>   W_input_CTRL_x_orig ,
  i_W_y_orig     =>   W_input_CTRL_y_orig ,
  i_W_fb         =>   W_input_CTRL_fb     ,
  --i_W_busy       =>    deadlock_PE,  




   
   
    i_PM_req    =>              PM_input_CTRL_req   ,
    i_PM_ack    =>              PM_input_CTRL_ack   ,                                   
    i_PE_ack    =>              PE_input_CTRL_ack  ,
    i_PE_req    =>              PE_input_CTRL_req  ,                                
    i_N_req     =>               N_input_CTRL_req   ,
    i_N_ack     =>               N_input_CTRL_ack   , 
    i_S_req     =>               S_input_CTRL_req   ,
    i_S_ack     =>               S_input_CTRL_ack   ,                                      
    i_E_ack     =>               E_input_CTRL_ack   ,
    i_E_req     =>               E_input_CTRL_req   ,         
    i_W_ack     =>               W_input_CTRL_ack   ,
    i_W_req     =>               W_input_CTRL_req   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   


);


PM_WRAPPER : entity work.PM_wrapper_EB_DEC
generic map(
    x_init => x_init,
    y_init => y_init, 
    pix_depth  	     =>pix_depth  	   ,
    img_width	     =>img_width	   , 
    img_height 	     =>img_height 	   ,
    subimg_width 	 =>subimg_width 	, 
    subimg_height	 =>subimg_height	, 
    n_steps 	     =>n_steps 	     ,
    n_frames   	     =>n_frames   	   ,
    buffer_length    =>buffer_length  
)
port map(
         
         clk => clk ,
          reset => reset,      
P_PM_ready_EB_CF   => deadlock_PM,     
P_PM_stall_in_EB   => w_PM_stall_in_EB ,      
P_PM_stall_out_EB  => w_PM_stall_out_EB,    
---------handshakes na s
    
P_PM_stall_out_DEC  => w_PM_stall_out_DEC,
P_PM_stall_in_DEC   => w_PM_stall_in_DEC,    
---------------------
P_PM_data_in_pixel   => t_PM_pixel  ,
P_PM_data_in_x_dest  => t_PM_x_dest ,
P_PM_data_in_y_dest  => t_PM_y_dest ,
P_PM_data_in_step    => t_PM_step   ,
P_PM_data_in_frame   => t_PM_frame  ,
P_PM_data_in_x_orig  => t_PM_x_orig ,
P_PM_data_in_y_orig  => t_PM_y_orig ,
P_PM_data_in_fb      => t_PM_fb     ,
                      
P_PM_data_out_pixel  =>  w_PM_in_pixel ,
P_PM_data_out_x_dest =>  w_PM_in_x_dest,
P_PM_data_out_y_dest =>  w_PM_in_y_dest,
P_PM_data_out_step   =>  w_PM_in_step  ,
P_PM_data_out_frame  =>  w_PM_in_frame ,
P_PM_data_out_x_orig =>  w_PM_in_x_orig,
P_PM_data_out_y_orig =>  w_PM_in_y_orig,
P_PM_data_out_fb     =>  w_PM_in_fb    ,
--Wrapper_PM_ready_EB_CF => R_Wrapper_PM_ready_EB_CF,
                       
P_PM_out_direction  =>     w_PM_in_direction,
  P_PM_req           =>      t_PM_req, 
  P_PM_ack           =>      t_PM_ack 
);



PE_WRAPPER : entity work.PE_wrapper_EB_DEC
generic map(
    x_init           =>   x_init,
    y_init           =>   y_init, 
    pix_depth  	     =>  pix_depth,
    img_width	     =>  img_width , 
    img_height 	     =>  img_height 	   ,
    subimg_width 	 =>  subimg_width 	, 
    subimg_height	 =>  subimg_height	, 
    n_steps 	     =>  n_steps 	     ,
    n_frames   	     =>  n_frames   	   ,
    buffer_length    =>  buffer_length   
)
port map(
clk => clk ,
reset => reset,                        
P_PE_ready_EB_CF   => deadlock_PE,     
P_PE_stall_in_EB   => w_PE_stall_in_EB ,      
P_PE_stall_out_EB  => w_PE_stall_out_EB,    
---------handshakes na s   
P_PE_stall_out_DEC  => w_PE_stall_out_DEC,
P_PE_stall_in_DEC   => w_PE_stall_in_DEC,    
---------------------
P_PE_data_in_pixel   =>  t_PE_pixel  ,
P_PE_data_in_x_dest  =>  t_PE_x_dest ,
P_PE_data_in_y_dest  =>  t_PE_y_dest ,
P_PE_data_in_step    =>  t_PE_step   ,
P_PE_data_in_frame   =>  t_PE_frame  ,
P_PE_data_in_x_orig  =>  t_PE_x_orig ,
P_PE_data_in_y_orig  =>  t_PE_y_orig ,
P_PE_data_in_fb      =>  t_PE_fb     ,
P_PE_data_out_pixel  =>  w_PE_in_pixel ,
P_PE_data_out_x_dest =>  w_PE_in_x_dest,
P_PE_data_out_y_dest =>  w_PE_in_y_dest,
P_PE_data_out_step   =>  w_PE_in_step  ,
P_PE_data_out_frame  =>  w_PE_in_frame ,
P_PE_data_out_x_orig =>  w_PE_in_x_orig,
P_PE_data_out_y_orig =>  w_PE_in_y_orig,
P_PE_data_out_fb     =>  w_PE_in_fb    ,
--Wrapper_PE_ready_EB_CF => R_Wrapper_PE_ready_EB_CF,
                       
P_PE_out_direction   =>    w_PE_in_direction,
P_PE_req             =>    t_PE_req, 
P_PE_ack             =>    t_PE_ack 

);

N_WRAPPER : entity work.N_wrapper_EB_DEC
generic map(
    x_init => x_init,
    y_init => y_init, 
    pix_depth  	     =>pix_depth  	   ,
    img_width	     =>img_width	   , 
    img_height 	     =>img_height 	   ,
    subimg_width 	 =>subimg_width 	, 
    subimg_height	 =>subimg_height	, 
    n_steps 	     =>n_steps 	     ,
    n_frames   	     =>n_frames   	   ,
    buffer_length    =>buffer_length   
)
port map(
   clk => clk ,
   reset => reset,                   
   P_N_ready_EB_CF   => IN_i_N_busy,     
   P_N_stall_in_EB   => w_N_stall_in_EB ,      
   P_N_stall_out_EB  => w_N_stall_out_EB,    
   P_N_stall_out_DEC  => w_N_stall_out_DEC,
   P_N_stall_in_DEC   => w_N_stall_in_DEC,    
   P_N_data_in_pixel   => t_N_pixel  ,
   P_N_data_in_x_dest  => t_N_x_dest ,
   P_N_data_in_y_dest  => t_N_y_dest ,
   P_N_data_in_step    => t_N_step   ,
   P_N_data_in_frame   => t_N_frame  ,
   P_N_data_in_x_orig  => t_N_x_orig ,
   P_N_data_in_y_orig  => t_N_y_orig ,
   P_N_data_in_fb      => t_N_fb     ,
   P_N_data_out_pixel  =>  w_N_in_pixel ,
   P_N_data_out_x_dest =>  w_N_in_x_dest,
   P_N_data_out_y_dest =>  w_N_in_y_dest,
   P_N_data_out_step   =>  w_N_in_step  ,
   P_N_data_out_frame  =>  w_N_in_frame ,
   P_N_data_out_x_orig =>  w_N_in_x_orig,
   P_N_data_out_y_orig =>  w_N_in_y_orig,
   P_N_data_out_fb     =>  w_N_in_fb    ,
   P_N_out_direction  =>     w_N_in_direction,
   P_N_req           =>      t_N_req,
   P_N_ack           =>      t_N_ack 
);

S_WRAPPER : entity work.S_wrapper_EB_DEC
generic map(
    x_init => x_init,
    y_init => y_init, 
    pix_depth  	     =>pix_depth  	   ,
    img_width	     =>img_width	   , 
    img_height 	     =>img_height 	   ,
    subimg_width 	 =>subimg_width 	, 
    subimg_height	 =>subimg_height	, 
    n_steps 	     =>n_steps 	     ,
    n_frames   	     =>n_frames   	   ,
    buffer_length    =>buffer_length   
)
port map(
         
                  clk => clk ,
          reset => reset,
                
        P_S_ready_EB_CF   => IN_i_S_busy,     
        P_S_stall_in_EB   => w_S_stall_in_EB ,      
        P_S_stall_out_EB  => w_S_stall_out_EB,    
        ---------handshakes na s
            
        P_S_stall_out_DEC  => w_S_stall_out_DEC,
        P_S_stall_in_DEC   => w_S_stall_in_DEC,    
        ---------------------
        P_S_data_in_pixel   => t_S_pixel  ,
        P_S_data_in_x_dest  => t_S_x_dest ,
        P_S_data_in_y_dest  => t_S_y_dest ,
        P_S_data_in_step    => t_S_step   ,
        P_S_data_in_frame   => t_S_frame  ,
        P_S_data_in_x_orig  => t_S_x_orig ,
        P_S_data_in_y_orig  => t_S_y_orig ,
        P_S_data_in_fb      => t_S_fb     ,
                              
        P_S_data_out_pixel  =>  w_S_in_pixel ,
        P_S_data_out_x_dest =>  w_S_in_x_dest,
        P_S_data_out_y_dest =>  w_S_in_y_dest,
        P_S_data_out_step   =>  w_S_in_step  ,
        P_S_data_out_frame  =>  w_S_in_frame ,
        P_S_data_out_x_orig =>  w_S_in_x_orig,
        P_S_data_out_y_orig =>  w_S_in_y_orig,
        P_S_data_out_fb     =>  w_S_in_fb    ,
        --Wrapper_S_ready_EB_CF => R_Wrapper_S_ready_EB_CF,
        P_S_out_direction  =>     w_S_in_direction,
          P_S_req           =>      t_S_req,
          P_S_ack           =>      t_S_ack 
);

E_WRAPPER : entity work.E_wrapper_EB_DEC
generic map(
    x_init => x_init,
    y_init => y_init, 
    pix_depth  	     =>pix_depth  	   ,
    img_width	     =>img_width	   , 
    img_height 	     =>img_height 	   ,
    subimg_width 	 =>subimg_width 	, 
    subimg_height	 =>subimg_height	, 
    n_steps 	     =>n_steps 	     ,
    n_frames   	     =>n_frames   	   ,
    buffer_length    =>buffer_length  
)
port map(
          
  clk => clk ,
  reset => reset,
                        
  P_E_ready_EB_CF   => IN_i_E_busy,     
  P_E_stall_in_EB   => w_E_stall_in_EB ,      
  P_E_stall_out_EB  => w_E_stall_out_EB,    
  ---------handshakes na s
      
  P_E_stall_out_DEC  => w_E_stall_out_DEC,
  P_E_stall_in_DEC   => w_E_stall_in_DEC,    
  ---------------------
  P_E_data_in_pixel   => t_E_pixel  ,
  P_E_data_in_x_dest  => t_E_x_dest ,
  P_E_data_in_y_dest  => t_E_y_dest ,
  P_E_data_in_step    => t_E_step   ,
  P_E_data_in_frame   => t_E_frame  ,
  P_E_data_in_x_orig  => t_E_x_orig ,
  P_E_data_in_y_orig  => t_E_y_orig ,
  P_E_data_in_fb      => t_E_fb     ,
                        
  P_E_data_out_pixel  =>  w_E_in_pixel ,
  P_E_data_out_x_dest =>  w_E_in_x_dest,
  P_E_data_out_y_dest =>  w_E_in_y_dest,
  P_E_data_out_step   =>  w_E_in_step  ,
  P_E_data_out_frame  =>  w_E_in_frame ,
  P_E_data_out_x_orig =>  w_E_in_x_orig,
  P_E_data_out_y_orig =>  w_E_in_y_orig,
  P_E_data_out_fb     =>  w_E_in_fb    ,
  -- Wrapper_E_ready_EB_CF => R_Wrapper_E_ready_EB_CF,                      
  P_E_out_direction  =>     w_E_in_direction,
    P_E_req           =>      t_E_req,
    P_E_ack           =>      t_E_ack 
);



W_WRAPPER : entity work.W_wrapper_EB_DEC
generic map(
    x_init => x_init,
    y_init => y_init, 
    pix_depth  	     =>pix_depth  	   ,
    img_width	     =>img_width	   , 
    img_height 	     =>img_height 	   ,
    subimg_width 	 =>subimg_width 	, 
    subimg_height	 =>subimg_height	, 
    n_steps 	     =>n_steps 	     ,
    n_frames   	     =>n_frames   	   ,
    buffer_length    =>buffer_length   
)
port map(
          
          clk => clk ,
          reset => reset,
          
    P_W_ready_EB_CF   => IN_i_W_busy,     
    P_W_stall_in_EB   => w_W_stall_in_EB ,      
    P_W_stall_out_EB  => w_W_stall_out_EB,    
    ---------handshakes na s
        
    P_W_stall_out_DEC  => w_W_stall_out_DEC,
    P_W_stall_in_DEC   => w_W_stall_in_DEC,    
    ---------------------
    P_W_data_in_pixel   => t_W_pixel  ,
    P_W_data_in_x_dest  => t_W_x_dest ,
    P_W_data_in_y_dest  => t_W_y_dest ,
    P_W_data_in_step    => t_W_step   ,
    P_W_data_in_frame   => t_W_frame  ,
    P_W_data_in_x_orig  => t_W_x_orig ,
    P_W_data_in_y_orig  => t_W_y_orig ,
    P_W_data_in_fb      => t_W_fb     ,
                          
    P_W_data_out_pixel  =>  w_W_in_pixel ,
    P_W_data_out_x_dest =>  w_W_in_x_dest,
    P_W_data_out_y_dest =>  w_W_in_y_dest,
    P_W_data_out_step   =>  w_W_in_step  ,
    P_W_data_out_frame  =>  w_W_in_frame ,
    P_W_data_out_x_orig =>  w_W_in_x_orig,
    P_W_data_out_y_orig =>  w_W_in_y_orig,
    P_W_data_out_fb     =>  w_W_in_fb    ,
    --Wrapper_W_ready_EB_CF => R_Wrapper_W_ready_EB_CF,             
    P_W_out_direction  =>     w_W_in_direction,
    
      P_W_req           =>      t_W_req,
      P_W_ack           =>      t_W_ack 
);


out_pm_controller_inst : entity work.out_pm_controller
generic map(
    x_init      => x_init,
    y_init      => y_init,
    img_width   => img_width ,
    img_height  => img_height,
    n_frames    => n_frames  ,
    n_steps     => n_steps   ,
    pix_depth   => pix_depth
    )
port map(
	clk => clk,
	reset => reset,
	-- connections with the output controller
    oc_PM_pixel   => PM_input_CTRL_pixel  ,  
    oc_PM_x_dest  => PM_input_CTRL_x_dest ,  
    oc_PM_y_dest  => PM_input_CTRL_y_dest ,  
    oc_PM_step    => PM_input_CTRL_step   ,  
    oc_PM_frame   => PM_input_CTRL_frame  ,  
    oc_PM_x_orig  => PM_input_CTRL_x_orig ,  
    oc_PM_y_orig  => PM_input_CTRL_y_orig ,  
    oc_PM_fb      => PM_input_CTRL_fb     , 
     
    oc_PM_new_msg => PM_input_CTRL_req    ,  
    oc_PM_ack     => PM_input_CTRL_ack    ,
    input_deadlockPM => deadlock_PM,

    -- connections to the next router
    i_PM_pixel  => i_PM_pixel,
    i_PM_x_dest => i_PM_x_dest,
    i_PM_y_dest => i_PM_y_dest,
    i_PM_step   => i_PM_step,
    i_PM_frame  => i_PM_frame,
    i_PM_x_orig => i_PM_x_orig,
    i_PM_y_orig => i_PM_y_orig,
    i_PM_fb     => i_PM_fb,
    i_PM_req    => i_PM_req,
    i_PM_ack    => i_PM_ack
    
);









out_pe_controller_inst : entity work.out_pe_controller
generic map(
    x_init      => x_init,
    y_init      => y_init,
    img_width   => img_width ,
    img_height  => img_height,
    n_frames    => n_frames  ,
    n_steps     => n_steps   ,
    pix_depth   => pix_depth
    )
port map(
	clk => clk,
	reset => reset,
	-- connections with the output controller
    oc_PE_pixel   => PE_input_CTRL_pixel  ,  
    oc_PE_x_dest  => PE_input_CTRL_x_dest ,  
    oc_PE_y_dest  => PE_input_CTRL_y_dest ,  
    oc_PE_step    => PE_input_CTRL_step   ,  
    oc_PE_frame   => PE_input_CTRL_frame  ,  
    oc_PE_x_orig  => PE_input_CTRL_x_orig ,  
    oc_PE_y_orig  => PE_input_CTRL_y_orig ,  
    oc_PE_fb      => PE_input_CTRL_fb     , 
     
    oc_PE_new_msg => PE_input_CTRL_req    ,  
    oc_PE_ack     => PE_input_CTRL_ack    ,
       
    -- connections to the next router
    i_PE_pixel  => i_PE_pixel,
    i_PE_x_dest => i_PE_x_dest,
    i_PE_y_dest => i_PE_y_dest,
    i_PE_step   => i_PE_step,
    i_PE_frame  => i_PE_frame,
    i_PE_x_orig => i_PE_x_orig,
    i_PE_y_orig => i_PE_y_orig,
    i_PE_fb     => i_PE_fb,
    i_PE_req    => i_PE_req,
    i_PE_ack    => i_PE_ack
    
);


out_N_controller_inst : entity work.out_N_controller
generic map(
    x_init      => x_init,
    y_init      => y_init,
    img_width   => img_width ,
    img_height  => img_height,
    n_frames    => n_frames  ,
    n_steps     => n_steps   ,
    pix_depth   => pix_depth
    )
port map(
	clk => clk,
	reset => reset,
	-- connections with the output controller
    oc_N_pixel   => N_input_CTRL_pixel  ,  
    oc_N_x_dest  => N_input_CTRL_x_dest ,  
    oc_N_y_dest  => N_input_CTRL_y_dest ,  
    oc_N_step    => N_input_CTRL_step   ,  
    oc_N_frame   => N_input_CTRL_frame  ,  
    oc_N_x_orig  => N_input_CTRL_x_orig ,  
    oc_N_y_orig  => N_input_CTRL_y_orig ,  
    oc_N_fb      => N_input_CTRL_fb     , 
     
    oc_N_new_msg => N_input_CTRL_req    ,  
    oc_N_ack     => N_input_CTRL_ack    ,
       
    -- connections to the next router
    i_N_pixel  => i_N_pixel,
    i_N_x_dest => i_N_x_dest,
    i_N_y_dest => i_N_y_dest,
    i_N_step   => i_N_step,
    i_N_frame  => i_N_frame,
    i_N_x_orig => i_N_x_orig,
    i_N_y_orig => i_N_y_orig,
    i_N_fb     => i_N_fb,
    i_N_req    => i_N_req,
    i_N_ack    => i_N_ack
    
);


out_S_controller_inst : entity work.out_S_controller
generic map(
    x_init      => x_init,
    y_init      => y_init,
    img_width   => img_width ,
    img_height  => img_height,
    n_frames    => n_frames  ,
    n_steps     => n_steps   ,
    pix_depth   => pix_depth
    )
port map(
	clk => clk,
	reset => reset,
	-- connections with the output controller
    oc_S_pixel   => S_input_CTRL_pixel  ,  
    oc_S_x_dest  => S_input_CTRL_x_dest ,  
    oc_S_y_dest  => S_input_CTRL_y_dest ,  
    oc_S_step    => S_input_CTRL_step   ,  
    oc_S_frame   => S_input_CTRL_frame  ,  
    oc_S_x_orig  => S_input_CTRL_x_orig ,  
    oc_S_y_orig  => S_input_CTRL_y_orig ,  
    oc_S_fb      => S_input_CTRL_fb     , 
     
    oc_S_new_msg => S_input_CTRL_req    ,  
    oc_S_ack     => S_input_CTRL_ack    ,
       
    -- connections to the next router
    i_S_pixel  => i_S_pixel,
    i_S_x_dest => i_S_x_dest,
    i_S_y_dest => i_S_y_dest,
    i_S_step   => i_S_step,
    i_S_frame  => i_S_frame,
    i_S_x_orig => i_S_x_orig,
    i_S_y_orig => i_S_y_orig,
    i_S_fb     => i_S_fb,
    i_S_req    => i_S_req,
    i_S_ack    => i_S_ack
    
);


out_E_controller_inst : entity work.out_E_controller
generic map(
    x_init      => x_init,
    y_init      => y_init,
    img_width   => img_width ,
    img_height  => img_height,
    n_frames    => n_frames  ,
    n_steps     => n_steps   ,
    pix_depth   => pix_depth
    )
port map(
	clk => clk,
	reset => reset,
	-- connections with the output controller
    oc_E_pixel   => E_input_CTRL_pixel  ,  
    oc_E_x_dest  => E_input_CTRL_x_dest ,  
    oc_E_y_dest  => E_input_CTRL_y_dest ,  
    oc_E_step    => E_input_CTRL_step   ,  
    oc_E_frame   => E_input_CTRL_frame  ,  
    oc_E_x_orig  => E_input_CTRL_x_orig ,  
    oc_E_y_orig  => E_input_CTRL_y_orig ,  
    oc_E_fb      => E_input_CTRL_fb     , 
     
    oc_E_new_msg => E_input_CTRL_req    ,  
    oc_E_ack     => E_input_CTRL_ack    ,
       
    -- connections to the next router
    i_E_pixel  => i_E_pixel,
    i_E_x_dest => i_E_x_dest,
    i_E_y_dest => i_E_y_dest,
    i_E_step   => i_E_step,
    i_E_frame  => i_E_frame,
    i_E_x_orig => i_E_x_orig,
    i_E_y_orig => i_E_y_orig,
    i_E_fb     => i_E_fb,
    i_E_req    => i_E_req,
    i_E_ack    => i_E_ack
    
);

out_W_controller_inst : entity work.out_W_controller
generic map(
    x_init      => x_init,
    y_init      => y_init,
    img_width   => img_width ,
    img_height  => img_height,
    n_frames    => n_frames  ,
    n_steps     => n_steps   ,
    pix_depth   => pix_depth
    )
port map(
	clk => clk,
	reset => reset,
	-- connections with the output controller
    oc_W_pixel   => W_input_CTRL_pixel  ,  
    oc_W_x_dest  => W_input_CTRL_x_dest ,  
    oc_W_y_dest  => W_input_CTRL_y_dest ,  
    oc_W_step    => W_input_CTRL_step   ,  
    oc_W_frame   => W_input_CTRL_frame  ,  
    oc_W_x_orig  => W_input_CTRL_x_orig ,  
    oc_W_y_orig  => W_input_CTRL_y_orig ,  
    oc_W_fb      => W_input_CTRL_fb     , 
     
    oc_W_new_msg => W_input_CTRL_req    ,  
    oc_W_ack     => W_input_CTRL_ack    ,
       
    -- connections to the next router
    i_W_pixel  => i_W_pixel,
    i_W_x_dest => i_W_x_dest,
    i_W_y_dest => i_W_y_dest,
    i_W_step   => i_W_step,
    i_W_frame  => i_W_frame,
    i_W_x_orig => i_W_x_orig,
    i_W_y_orig => i_W_y_orig,
    i_W_fb     => i_W_fb,
    i_W_req    => i_W_req,
    i_W_ack    => i_W_ack
    
);






end Behavioral;
